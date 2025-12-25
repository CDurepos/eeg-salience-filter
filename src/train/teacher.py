#!/usr/bin/env python3
"""
Train Teacher Model for Salience Map Generation (PyTorch)

This script trains a teacher model on unfiltered EEG data. The teacher is
used to generate salience maps that identify discriminative regions of the
signal, which are then used to filter data for student training.

Supports three architectures:
  - eegnet: braindecode's EEGNetv4
  - resnet: tsai's ResNet1d
  - tst: tsai's TST (Time Series Transformer)

Usage:
  python src/train/teacher.py --model eegnet
  python src/train/teacher.py --model resnet
  python src/train/teacher.py --model tst
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from braindecode.models import EEGNetv4
from tsai.models.ResNet import ResNet
from tsai.models.TST import TST
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

UNFILTERED_DATA_DIR = 'data/unfiltered'

NB_CLASSES = 2
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_eegnet(n_channels, n_samples):
    """EEGNet using braindecode's implementation."""
    return EEGNetv4(
        n_chans=n_channels,
        n_outputs=NB_CLASSES,
        n_times=n_samples,
        final_conv_length='auto',
        drop_prob=0.5
    )


def build_resnet(n_channels, n_samples):
    """
    ResNet for time-series using tsai.
    
    tsai expects input shape (batch, channels, seq_len) which matches our EEG format.
    Note: tsai's ResNet infers seq_len from the input, so we only pass c_in and c_out.
    """
    return ResNet(
        c_in=n_channels,
        c_out=NB_CLASSES
    )


def build_tst(n_channels, n_samples):
    """
    Time Series Transformer using tsai.
    
    TST uses attention mechanisms to capture long-range dependencies in
    time-series data.
    """
    return TST(
        c_in=n_channels,
        c_out=NB_CLASSES,
        seq_len=n_samples,
        n_layers=4,
        n_heads=4,
        d_model=64,
        d_ff=128,
        dropout=0.1,
        fc_dropout=0.3
    )


MODELS = {
    'eegnet': build_eegnet,
    'resnet': build_resnet,
    'tst': build_tst
}


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
    
    return total_loss / total, correct / total


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Teacher Model')
    parser.add_argument('--model', type=str, required=True, choices=['eegnet', 'resnet', 'tst'])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    
    model_type = args.model
    model_dir = f'models/{model_type}'
    model_path = f'{model_dir}/teacher.pt'
    
    print("=" * 60)
    print(f"Training {model_type.upper()} Teacher Model (PyTorch)")
    print("=" * 60)
    print(f"[Device] Using: {DEVICE}")
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    print("\n[Data] Loading unfiltered data...")
    
    X = np.load(os.path.join(UNFILTERED_DATA_DIR, 'epochs.npy'))
    y = np.load(os.path.join(UNFILTERED_DATA_DIR, 'y_labels.npy'))
    subject_ids = np.load(os.path.join(UNFILTERED_DATA_DIR, 'subject_ids.npy'), allow_pickle=True)
    
    # Ensure shape is (batch, channels, time)
    if len(X.shape) == 4:
        X = X[:, :, :, 0]
    
    n_channels, n_samples = X.shape[1], X.shape[2]
    print(f"[Data] Shape: {X.shape}, Subjects: {len(np.unique(subject_ids))}")
    
    # -------------------------------------------------------------------------
    # Split by subject
    # -------------------------------------------------------------------------
    unique_subjects = np.unique(subject_ids)
    subject_labels = [y[subject_ids == s][0] for s in unique_subjects]
    
    train_subj, test_subj = train_test_split(
        unique_subjects, test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE, stratify=subject_labels
    )
    
    train_mask = np.isin(subject_ids, train_subj)
    test_mask = np.isin(subject_ids, test_subj)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"[Data] Train: {len(train_subj)} subjects, {X_train.shape[0]} segments")
    print(f"[Data] Test: {len(test_subj)} subjects, {X_test.shape[0]} segments")
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=args.batch_size
    )
    
    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    print(f"\n[Model] Building {model_type.upper()}...")
    
    model = MODELS[model_type](n_channels, n_samples).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {n_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)
    
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n[Train] Training for up to {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        
        scheduler.step(val_loss)
        
        history['loss'].append(train_loss)
        history['accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Early stopping & checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
                'n_channels': n_channels,
                'n_samples': n_samples
            }, model_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={train_loss:.4f}, acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # -------------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------------
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    print(f"\n[Result] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    np.save(f'{model_dir}/teacher_history.npy', history)
    np.save(f'{model_dir}/teacher_meta.npy', {
        'model_type': model_type,
        'n_channels': n_channels,
        'n_samples': n_samples,
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'n_params': n_params
    })
    
    print(f"\n[Done] Saved to {model_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
