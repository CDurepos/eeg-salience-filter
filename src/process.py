"""
Complete processing pipeline: Generate salience maps, create mask, and filter data.

This script:
1. Loads the Teacher model
2. Generates salience maps using Integrated Gradients and SmoothGrad
3. Creates a binary mask from salience maps
4. Applies the mask to filter the data
5. Saves filtered data to data/filtered/
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from dotenv import load_dotenv

# Add arl-eegmodels to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'arl-eegmodels'))
from EEGModels import EEGNet

# Load environment variables
load_dotenv()

# Configuration
# DATA_PATH from .env points to downloaded dataset directory
# Processed data goes to project root data/ directories
UNFILTERED_DATA_DIR = 'data/unfiltered'
FILTERED_DATA_DIR = 'data/filtered'
MODEL_DIR = 'models/teacher'
OUTPUT_DIR = 'outputs'

EPOCHS_FILE = 'epochs.npy'
LABELS_FILE = 'y_labels.npy'

# Model parameters (will be determined from data/model)
NB_CLASSES = 2
DROPOUT_RATE = 0.5
KERN_LENGTH = 64
F1 = 8
D = 2

# Salience map parameters
IG_STEPS = 50
SMOOTHGRAD_SAMPLES = 50
SMOOTHGRAD_NOISE_STD = 0.15
N_SAMPLES_FOR_SALIENCE = None  # None = use all data, or specify number

# Mask creation parameters
SALIENCE_METHOD = 'both'  # 'ig', 'smoothgrad', or 'both'
THRESHOLD_METHOD = 'percentile'  # 'percentile' or 'absolute'
THRESHOLD_VALUE = 0.75  # For percentile: top 25% (0.75), for absolute: threshold value
MIN_SALIENCE = 0.0  # Minimum salience value to consider


def load_teacher_model(n_samples, n_channels):
    """Load the trained Teacher model."""
    model_path = os.path.join(MODEL_DIR, 'model.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Teacher model not found: {model_path}\n"
            "Please train the teacher model first: python src/train/teacher.py"
        )
    
    # Create and load model
    model = EEGNet(
        nb_classes=NB_CLASSES,
        Chans=n_channels,
        Samples=n_samples,
        dropoutRate=DROPOUT_RATE,
        kernLength=KERN_LENGTH,
        F1=F1,
        D=D,
        dropoutType='Dropout'
    )
    
    model.load_weights(model_path)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Loaded Teacher model from {model_path}")
    return model


def integrated_gradients(model, inputs, target_class=None, steps=50):
    """Compute Integrated Gradients attributions."""
    batch_size = inputs.shape[0]
    baseline = np.zeros_like(inputs)
    
    if target_class is None:
        predictions = model(inputs, training=False)
        target_class = tf.argmax(predictions, axis=1).numpy()
    
    alphas = np.linspace(0.0, 1.0, steps + 1)
    attributions = np.zeros_like(inputs)
    
    for i in range(steps):
        alpha = alphas[i]
        interpolated = baseline + alpha * (inputs - baseline)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            outputs = model(interpolated, training=False)
            
            if isinstance(target_class, np.ndarray):
                target_outputs = tf.gather_nd(
                    outputs,
                    tf.stack([tf.range(batch_size), target_class], axis=1)
                )
            else:
                target_outputs = outputs[:, target_class]
        
        gradients = tape.gradient(target_outputs, interpolated)
        attributions += gradients.numpy() / steps
    
    attributions = attributions * (inputs - baseline)
    return attributions


def smoothgrad(model, inputs, target_class=None, num_samples=50, noise_std=0.15):
    """Compute SmoothGrad attributions."""
    batch_size = inputs.shape[0]
    input_range = np.max(inputs) - np.min(inputs)
    noise_scale = noise_std * input_range
    
    if target_class is None:
        predictions = model(inputs, training=False)
        target_class = tf.argmax(predictions, axis=1).numpy()
    
    all_gradients = []
    
    for _ in range(num_samples):
        noise = np.random.normal(0, noise_scale, size=inputs.shape).astype(np.float32)
        noisy_inputs = inputs + noise
        
        with tf.GradientTape() as tape:
            tape.watch(noisy_inputs)
            outputs = model(noisy_inputs, training=False)
            
            if isinstance(target_class, np.ndarray):
                target_outputs = tf.gather_nd(
                    outputs,
                    tf.stack([tf.range(batch_size), target_class], axis=1)
                )
            else:
                target_outputs = outputs[:, target_class]
        
        gradients = tape.gradient(target_outputs, noisy_inputs)
        all_gradients.append(gradients.numpy())
    
    attributions = np.mean(all_gradients, axis=0)
    attributions = attributions * inputs
    
    return attributions


def create_binary_mask(salience_maps, threshold_method, threshold_value):
    """Create binary mask from salience maps."""
    print(f"\nCreating binary mask using {threshold_method} method...")
    
    if threshold_method == 'percentile':
        threshold = np.percentile(salience_maps, threshold_value * 100)
        print(f"Percentile threshold ({threshold_value * 100}%): {threshold:.6f}")
    else:
        threshold = threshold_value
        print(f"Absolute threshold: {threshold:.6f}")
    
    mask = (salience_maps >= threshold).astype(np.float32)
    
    if MIN_SALIENCE > 0:
        mask = mask * (salience_maps >= MIN_SALIENCE).astype(np.float32)
    
    total_elements = mask.size
    important_elements = np.sum(mask)
    percentage = (important_elements / total_elements) * 100
    
    print(f"Mask statistics:")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Important (1): {important_elements:,} ({percentage:.2f}%)")
    print(f"  Unimportant (0): {total_elements - important_elements:,} ({100 - percentage:.2f}%)")
    
    return mask


def apply_mask_to_data(data, mask):
    """Apply binary mask to filter data."""
    print("\nApplying mask to data...")
    
    if len(data.shape) == 4:
        data_3d = data[:, :, :, 0]
    else:
        data_3d = data
    
    if mask.shape != data_3d.shape:
        if mask.shape[0] < data_3d.shape[0]:
            n_repeat = data_3d.shape[0] // mask.shape[0]
            mask = np.tile(mask, (n_repeat + 1, 1, 1))[:data_3d.shape[0]]
            print(f"Expanded mask from {mask.shape} to {data_3d.shape}")
    
    filtered_data = data_3d * mask
    
    if len(data.shape) == 4:
        filtered_data = filtered_data[:, :, :, np.newaxis]
    
    print(f"Filtered data shape: {filtered_data.shape}")
    return filtered_data


def main():
    print("=" * 60)
    print("Processing Pipeline: Salience Maps → Mask → Filtered Data")
    print("=" * 60)
    
    # Load data first to get dimensions
    print("\nLoading data to determine dimensions...")
    epochs_path = os.path.join(UNFILTERED_DATA_DIR, EPOCHS_FILE)
    if not os.path.exists(epochs_path):
        raise FileNotFoundError(f"Epochs file not found: {epochs_path}")
    
    X_sample = np.load(epochs_path)
    if len(X_sample.shape) == 3:
        n_channels = X_sample.shape[1]
        n_samples = X_sample.shape[2]
    else:
        n_channels = X_sample.shape[1]
        n_samples = X_sample.shape[2]
    
    print(f"Data dimensions: {n_channels} channels, {n_samples} time points")
    
    # Load Teacher model
    print("\nLoading Teacher model...")
    model = load_teacher_model(n_samples, n_channels)
    
    # Load unfiltered data
    print("\nLoading unfiltered data...")
    epochs_path = os.path.join(UNFILTERED_DATA_DIR, EPOCHS_FILE)
    labels_path = os.path.join(UNFILTERED_DATA_DIR, LABELS_FILE)
    
    if not os.path.exists(epochs_path):
        raise FileNotFoundError(f"Epochs file not found: {epochs_path}")
    
    X = np.load(epochs_path)
    y = np.load(labels_path)
    
    # Ensure correct shape
    if len(X.shape) == 3:
        X = X[:, :, :, np.newaxis]
    
    print(f"Data shape: {X.shape}")
    
    # Select samples for salience map generation
    if N_SAMPLES_FOR_SALIENCE is not None and N_SAMPLES_FOR_SALIENCE < len(X):
        indices = np.random.choice(len(X), N_SAMPLES_FOR_SALIENCE, replace=False)
        X_salience = X[indices]
        print(f"Using {N_SAMPLES_FOR_SALIENCE} samples for salience map generation")
    else:
        X_salience = X
        print("Using all samples for salience map generation")
    
    # Generate salience maps
    print("\n" + "=" * 60)
    print("Generating Salience Maps")
    print("=" * 60)
    
    ig_attributions = None
    sg_attributions = None
    
    if SALIENCE_METHOD in ['ig', 'both']:
        print("\nComputing Integrated Gradients...")
        ig_attributions = integrated_gradients(model, X_salience, steps=IG_STEPS)
        if ig_attributions.shape[-1] == 1:
            ig_attributions = ig_attributions[:, :, :, 0]
        print(f"IG attributions shape: {ig_attributions.shape}")
    
    if SALIENCE_METHOD in ['smoothgrad', 'both']:
        print("\nComputing SmoothGrad...")
        sg_attributions = smoothgrad(
            model, X_salience,
            num_samples=SMOOTHGRAD_SAMPLES,
            noise_std=SMOOTHGRAD_NOISE_STD
        )
        if sg_attributions.shape[-1] == 1:
            sg_attributions = sg_attributions[:, :, :, 0]
        print(f"SmoothGrad attributions shape: {sg_attributions.shape}")
    
    # Combine salience maps
    if SALIENCE_METHOD == 'both' and ig_attributions is not None and sg_attributions is not None:
        salience_maps = (np.abs(ig_attributions) + np.abs(sg_attributions)) / 2.0
        print("\nCombined IG and SmoothGrad salience maps (averaged)")
    elif ig_attributions is not None:
        salience_maps = np.abs(ig_attributions)
    else:
        salience_maps = np.abs(sg_attributions)
    
    # Expand salience maps to full dataset if needed
    if salience_maps.shape[0] < X.shape[0]:
        # Create full salience map by repeating or interpolating
        # For now, we'll use the average salience map for all samples
        avg_salience = np.mean(salience_maps, axis=0, keepdims=True)
        salience_maps = np.tile(avg_salience, (X.shape[0], 1, 1))
        print(f"Expanded salience maps to full dataset: {salience_maps.shape}")
    
    # Create binary mask
    print("\n" + "=" * 60)
    print("Creating Binary Mask")
    print("=" * 60)
    mask = create_binary_mask(salience_maps, THRESHOLD_METHOD, THRESHOLD_VALUE)
    
    # Apply mask to filter data
    print("\n" + "=" * 60)
    print("Filtering Data")
    print("=" * 60)
    X_filtered = apply_mask_to_data(X, mask)
    
    # Save everything
    os.makedirs(FILTERED_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save filtered data
    if len(X_filtered.shape) == 4:
        X_filtered_3d = X_filtered[:, :, :, 0]
    else:
        X_filtered_3d = X_filtered
    
    # Load subject IDs from unfiltered data to save with filtered data
    subject_ids_path = os.path.join(UNFILTERED_DATA_DIR, 'subject_ids.npy')
    if os.path.exists(subject_ids_path):
        subject_ids = np.load(subject_ids_path, allow_pickle=True)
        np.save(os.path.join(FILTERED_DATA_DIR, 'subject_ids.npy'), subject_ids)
        print(f"Subject IDs saved to filtered data")
    
    np.save(os.path.join(FILTERED_DATA_DIR, 'epochs.npy'), X_filtered_3d)
    np.save(os.path.join(FILTERED_DATA_DIR, 'y_labels.npy'), y)
    print(f"\nFiltered data saved to: {FILTERED_DATA_DIR}/")
    
    # Save salience maps
    if ig_attributions is not None:
        np.save(os.path.join(OUTPUT_DIR, 'ig_attributions.npy'), ig_attributions)
    if sg_attributions is not None:
        np.save(os.path.join(OUTPUT_DIR, 'smoothgrad_attributions.npy'), sg_attributions)
    np.save(os.path.join(OUTPUT_DIR, 'combined_salience_maps.npy'), salience_maps)
    
    # Save mask
    np.save(os.path.join(OUTPUT_DIR, 'binary_mask.npy'), mask)
    
    # Save processing metadata
    metadata = {
        'salience_method': SALIENCE_METHOD,
        'threshold_method': THRESHOLD_METHOD,
        'threshold_value': THRESHOLD_VALUE,
        'ig_steps': IG_STEPS,
        'smoothgrad_samples': SMOOTHGRAD_SAMPLES,
        'mask_shape': list(mask.shape),
        'important_percentage': float((np.sum(mask) / mask.size) * 100),
        'filtered_data_shape': list(X_filtered_3d.shape)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'processing_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"  Filtered data: {FILTERED_DATA_DIR}/")
    print(f"  Salience maps: {OUTPUT_DIR}/")
    print(f"  Metadata: {OUTPUT_DIR}/processing_metadata.json")


if __name__ == '__main__':
    main()

