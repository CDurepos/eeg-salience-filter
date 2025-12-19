"""
Train the Teacher EEGNet model on unfiltered data.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Add arl-eegmodels to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'arl-eegmodels'))
from EEGModels import EEGNet

# Load environment variables
load_dotenv()

# Configuration
# Processed data is in project root data/ directories
UNFILTERED_DATA_DIR = 'data/unfiltered'
MODEL_DIR = 'models/teacher'
EPOCHS_FILE = 'epochs.npy'
LABELS_FILE = 'y_labels.npy'

# Model parameters
NB_CLASSES = 2
CHANS = None  # Will be determined from data
SAMPLES = 256
DROPOUT_RATE = 0.5
KERN_LENGTH = 64
F1 = 8
D = 2

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2


def load_data():
    """Load unfiltered EEG epochs."""
    epochs_path = os.path.join(UNFILTERED_DATA_DIR, EPOCHS_FILE)
    labels_path = os.path.join(UNFILTERED_DATA_DIR, LABELS_FILE)
    subject_ids_path = os.path.join(UNFILTERED_DATA_DIR, 'subject_ids.npy')
    
    if not os.path.exists(epochs_path):
        raise FileNotFoundError(
            f"Epochs file not found: {epochs_path}\n"
            "Please run preprocessing first: python src/preprocess.py"
        )
    
    X = np.load(epochs_path)
    y = np.load(labels_path)
    subject_ids = np.load(subject_ids_path, allow_pickle=True) if os.path.exists(subject_ids_path) else None
    
    # Ensure correct shape for EEGNet: (batch, channels, time, 1)
    if len(X.shape) == 3:
        X = X[:, :, :, np.newaxis]
    
    print(f"Loaded data: {X.shape}")
    print(f"Labels: {y.shape}, unique: {np.unique(y)}")
    if subject_ids is not None:
        print(f"Subject IDs: {len(np.unique(subject_ids))} unique subjects")
    
    return X, y, subject_ids


def create_model(n_samples, n_channels):
    """Create EEGNet model."""
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
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def main():
    print("=" * 60)
    print("Training Teacher EEGNet Model")
    print("=" * 60)
    
    print("\nLoading unfiltered data...")
    X, y, subject_ids = load_data()
    
    # Convert labels to categorical
    y_categorical = keras.utils.to_categorical(y, num_classes=NB_CLASSES)
    
    # Split by subject to avoid data leakage
    if subject_ids is not None:
        unique_subjects = np.unique(subject_ids)
        # Get label for each subject (should be same for all segments)
        subject_labels = np.array([y[subject_ids == subj][0] for subj in unique_subjects])
        
        train_subjects, test_subjects = train_test_split(
            unique_subjects,
            test_size=VALIDATION_SPLIT,
            random_state=42,
            stratify=subject_labels
        )
        
        # Create masks
        train_mask = np.isin(subject_ids, train_subjects)
        test_mask = np.isin(subject_ids, test_subjects)
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y_categorical[train_mask]
        y_test = y_categorical[test_mask]
        
        print(f"Split by subject: {len(train_subjects)} train subjects, {len(test_subjects)} test subjects")
    else:
        # Fallback to segment-level split if subject_ids not available
        print("Warning: subject_ids not found, splitting by segment (may cause data leakage)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical,
            test_size=VALIDATION_SPLIT,
            random_state=42,
            stratify=y
        )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Determine actual dimensions from data
    actual_channels = X.shape[1]
    actual_samples = X.shape[2]
    
    print(f"\nData dimensions: {actual_channels} channels, {actual_samples} time points")
    
    # Create model
    print("\nCreating Teacher EEGNet model...")
    model = create_model(actual_samples, actual_channels)
    model.summary()
    
    # Callbacks
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'model.h5')
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\nTraining Teacher model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating Teacher model on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Teacher Test loss: {test_loss:.4f}")
    print(f"Teacher Test accuracy: {test_acc:.4f}")
    
    # Save final model
    model.save(model_path)
    print(f"\nTeacher model saved to: {model_path}")
    
    # Save training history
    np.save(os.path.join(MODEL_DIR, 'history.npy'), history.history)
    print("Training history saved.")


if __name__ == '__main__':
    main()

