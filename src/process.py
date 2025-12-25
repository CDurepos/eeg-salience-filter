#!/usr/bin/env python3
"""
Generate Salience Maps and Filter Data (PyTorch + Captum)

This script:
1. Loads a trained teacher model
2. Generates salience maps using Captum's Integrated Gradients
3. Creates a binary mask from salience maps
4. Applies neutral fill masking (per-channel mean, not hard zeros)
5. Saves filtered data for student training

IMPORTANT: Only uses TRAINING data for salience maps to avoid data leakage.

Uses Captum (PyTorch's official interpretability library) for reliable
gradient-based attribution methods.

Usage:
  python src/process.py --teacher eegnet
  python src/process.py --teacher resnet --threshold 0.5
  python src/process.py --teacher tst
"""

import os
import argparse
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from braindecode.models import EEGNetv4
from tsai.models.ResNet import ResNet
from tsai.models.TST import TST
from captum.attr import IntegratedGradients, NoiseTunnel
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

UNFILTERED_DATA_DIR = 'data/unfiltered'

# Salience parameters (reduced for memory efficiency)
IG_STEPS = 20            # Integrated gradients steps
SMOOTHGRAD_SAMPLES = 10  # SmoothGrad noise samples
SMOOTHGRAD_NOISE = 0.15
DEFAULT_THRESHOLD = 0.95

VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
NB_CLASSES = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# MODEL LOADERS
# =============================================================================

def load_teacher_model(teacher_type, n_channels, n_samples):
    """Load the trained teacher model."""
    model_path = f'models/{teacher_type}/teacher.pt'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Teacher not found: {model_path}\n"
            f"Train it: python src/train/teacher.py --model {teacher_type}"
        )
    
    # Build model architecture
    if teacher_type == 'eegnet':
        model = EEGNetv4(
            n_chans=n_channels,
            n_outputs=NB_CLASSES,
            n_times=n_samples,
            final_conv_length='auto',
            drop_prob=0.5
        )
    elif teacher_type == 'resnet':
        model = ResNet(c_in=n_channels, c_out=NB_CLASSES)
    elif teacher_type == 'tst':
        model = TST(
            c_in=n_channels, c_out=NB_CLASSES, seq_len=n_samples,
            n_layers=4, n_heads=4, d_model=64, d_ff=128,
            dropout=0.1, fc_dropout=0.3
        )
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    return model


# =============================================================================
# SALIENCE COMPUTATION (using Captum)
# =============================================================================

def compute_salience_maps(model, X_data, batch_size=8, max_samples=2000):
    """
    Compute salience maps using Captum's Integrated Gradients with SmoothGrad.
    
    Uses:
    - Integrated Gradients: Reliable gradient-based attribution
    - NoiseTunnel (SmoothGrad): Reduces noise by averaging over noisy inputs
    
    Note: Uses a subset of samples for efficiency. Salience maps are averaged
    to create a global importance mask.
    """
    # Subsample for efficiency
    if len(X_data) > max_samples:
        idx = np.random.choice(len(X_data), max_samples, replace=False)
        X_data = X_data[idx]
        print(f"[Salience] Subsampled to {max_samples} samples for efficiency")
    
    print(f"[Salience] Computing attributions for {len(X_data)} samples...")
    
    # Set up Captum attribution methods
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    
    # Compute attributions in batches to manage memory
    all_attributions = []
    n_batches = (len(X_data) + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_data))
        
        X_batch = torch.FloatTensor(X_data[start:end]).to(DEVICE)
        
        # Get targets for this batch
        with torch.no_grad():
            outputs = model(X_batch)
            targets_batch = outputs.argmax(dim=1)
        
        # Compute attributions with SmoothGrad noise
        attributions = nt.attribute(
            X_batch,
            nt_type='smoothgrad',
            nt_samples=SMOOTHGRAD_SAMPLES,
            stdevs=SMOOTHGRAD_NOISE,
            baselines=torch.zeros_like(X_batch),
            target=targets_batch,
            n_steps=IG_STEPS
        )
        
        all_attributions.append(attributions.cpu().numpy())
        del X_batch, attributions  # Free memory
        torch.cuda.empty_cache()
        
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Batch {i+1}/{n_batches} complete")
    
    # Concatenate all batches
    salience_maps = np.concatenate(all_attributions, axis=0)
    
    print(f"[Salience] Shape: {salience_maps.shape}")
    print(f"[Salience] Range: [{salience_maps.min():.6f}, {salience_maps.max():.6f}]")
    
    return salience_maps


# =============================================================================
# MASK APPLICATION
# =============================================================================

def create_global_mask(salience_maps, threshold):
    """
    Create a GLOBAL binary mask from salience maps.
    
    Instead of thresholding per-sample (which loses global patterns when averaged),
    we first aggregate salience across samples, then threshold once.
    
    Args:
        salience_maps: (n_samples, n_channels, n_times) attribution values
        threshold: percentile of data to MASK (e.g., 0.25 = mask bottom 25%)
    
    Returns:
        mask: (1, n_channels, n_times) binary mask where 1 = keep, 0 = mask
    """
    # Average absolute salience across all samples to get global importance
    abs_salience = np.abs(salience_maps)
    global_importance = np.mean(abs_salience, axis=0)  # (channels, times)
    
    # Threshold: keep top (1-threshold) percent
    thresh_val = np.percentile(global_importance, threshold * 100)
    mask = (global_importance >= thresh_val).astype(np.float32)
    
    kept = np.sum(mask) / mask.size * 100
    print(f"[Mask] Threshold={threshold:.0%} → Kept: {kept:.1f}%, Masked: {100-kept:.1f}%")
    
    return mask[np.newaxis, :, :]  # (1, channels, times)


def apply_mask(data, mask):
    """
    Apply mask using neutral fill (per-channel mean, not zeros).
    
    Args:
        data: (n_samples, n_channels, n_times) EEG data
        mask: (1, n_channels, n_times) binary mask (1=keep, 0=mask)
    
    Returns:
        filtered: (n_samples, n_channels, n_times) filtered data
    """
    # Ensure 3D
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
    
    # Broadcast mask to all samples
    mask_broadcast = np.broadcast_to(mask, data.shape)
    
    # Per-channel mean as fill value (computed per sample, per channel)
    channel_means = np.mean(data, axis=2, keepdims=True)
    fill_values = np.broadcast_to(channel_means, data.shape)
    
    # Apply mask: keep original where mask=1, use channel mean where mask=0
    filtered = np.where(mask_broadcast > 0.5, data, fill_values)
    
    return filtered


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate salience maps and filter data')
    parser.add_argument('--teacher', type=str, required=True, choices=['eegnet', 'resnet', 'tst'])
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    
    teacher_type = args.teacher
    filtered_dir = f'data/filtered_{teacher_type}'
    output_dir = f'outputs/{teacher_type}'
    
    print("=" * 60)
    print(f"Processing: {teacher_type.upper()} Teacher → Filtered Data")
    print("=" * 60)
    print(f"[Device] Using: {DEVICE}")
    print(f"[Config] Threshold: {args.threshold:.0%}")
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n[Data] Loading...")
    
    X = np.load(os.path.join(UNFILTERED_DATA_DIR, 'epochs.npy'))
    y = np.load(os.path.join(UNFILTERED_DATA_DIR, 'y_labels.npy'))
    subject_ids = np.load(os.path.join(UNFILTERED_DATA_DIR, 'subject_ids.npy'), allow_pickle=True)
    
    # Ensure shape is (batch, channels, time)
    if len(X.shape) == 4:
        X = X[:, :, :, 0]
    
    n_channels, n_samples = X.shape[1], X.shape[2]
    print(f"[Data] Shape: {X.shape}")
    
    # -------------------------------------------------------------------------
    # Split (same as training) - use only TRAIN data for salience
    # -------------------------------------------------------------------------
    unique_subjects = np.unique(subject_ids)
    subject_labels = [y[subject_ids == s][0] for s in unique_subjects]
    
    train_subj, _ = train_test_split(
        unique_subjects, test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE, stratify=subject_labels
    )
    
    train_mask = np.isin(subject_ids, train_subj)
    X_train = X[train_mask]
    
    print(f"[Data] Using {X_train.shape[0]} TRAINING samples for salience (no leakage)")
    
    # -------------------------------------------------------------------------
    # Load teacher
    # -------------------------------------------------------------------------
    print(f"\n[Model] Loading {teacher_type.upper()} teacher...")
    model = load_teacher_model(teacher_type, n_channels, n_samples)
    
    # -------------------------------------------------------------------------
    # Generate salience maps
    # -------------------------------------------------------------------------
    print("\n[Salience] Computing attributions with Captum (IG + SmoothGrad)...")
    salience_maps = compute_salience_maps(model, X_train)
    
    # -------------------------------------------------------------------------
    # Create mask and filter
    # -------------------------------------------------------------------------
    print("\n[Filter] Creating mask and filtering data...")
    mask = create_global_mask(salience_maps, args.threshold)
    X_filtered = apply_mask(X, mask)
    
    print(f"[Filter] Original - mean: {np.mean(X):.4f}, std: {np.std(X):.4f}")
    print(f"[Filter] Filtered - mean: {np.mean(X_filtered):.4f}, std: {np.std(X_filtered):.4f}")
    
    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    print("\n[Save] Writing outputs...")
    
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtered data
    np.save(f'{filtered_dir}/epochs.npy', X_filtered)
    np.save(f'{filtered_dir}/y_labels.npy', y)
    np.save(f'{filtered_dir}/subject_ids.npy', subject_ids)
    print(f"[Save] Filtered data → {filtered_dir}/")
    
    # Salience artifacts
    np.save(f'{output_dir}/salience_maps.npy', salience_maps)
    np.save(f'{output_dir}/binary_mask.npy', mask)
    
    # Metadata
    meta = {
        'teacher': teacher_type,
        'threshold': args.threshold,
        'kept_percent': float(np.sum(mask) / mask.size * 100),
        'ig_steps': IG_STEPS,
        'smoothgrad_samples': SMOOTHGRAD_SAMPLES,
        'smoothgrad_noise': SMOOTHGRAD_NOISE,
        'n_train_samples': len(X_train),
        'salience_shape': list(salience_maps.shape)
    }
    with open(f'{output_dir}/processing_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"[Save] Salience maps → {output_dir}/")
    print("\n[Done]")
    print("=" * 60)


if __name__ == '__main__':
    main()
