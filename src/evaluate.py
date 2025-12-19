"""
Evaluate and compare Teacher, Student, and Benchmark models at subject level.

This script aggregates segment-level predictions to subject-level predictions
using ensemble methods (averaging probabilities).
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
from collections import defaultdict
from dotenv import load_dotenv

# Add arl-eegmodels to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'arl-eegmodels'))
from EEGModels import EEGNet

# Load environment variables
load_dotenv()

# Configuration
# Processed data is in project root data/ directories
UNFILTERED_DATA_DIR = 'data/unfiltered'
FILTERED_DATA_DIR = 'data/filtered'
OUTPUT_DIR = 'outputs'

# Model directories
TEACHER_MODEL_DIR = 'models/teacher'
STUDENT_MODEL_DIR = 'models/student'
BENCHMARK_MODEL_DIR = 'models/benchmark'

EPOCHS_FILE = 'epochs.npy'
LABELS_FILE = 'y_labels.npy'
SUBJECT_IDS_FILE = 'subject_ids.npy'

# Model parameters (will be determined from data)
NB_CLASSES = 2
DROPOUT_RATE = 0.5
KERN_LENGTH = 64
F1 = 8
D = 2

VALIDATION_SPLIT = 0.2


def load_model(model_dir, n_samples, n_channels, model_name):
    """Load a trained EEGNet model."""
    model_path = os.path.join(model_dir, 'model.h5')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_name} model not found: {model_path}")
    
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
    
    print(f"Loaded {model_name} model from {model_path}")
    return model


def aggregate_predictions_by_subject(y_pred_proba, subject_ids):
    """
    Aggregate segment-level predictions to subject-level predictions.
    
    Args:
        y_pred_proba: Probability predictions for each segment (n_segments, n_classes)
        subject_ids: Subject ID for each segment (n_segments,)
        
    Returns:
        y_subject_pred: Subject-level predictions (n_subjects,)
        y_subject_proba: Subject-level probabilities (n_subjects, n_classes)
        unique_subjects: List of unique subject IDs
    """
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    n_classes = y_pred_proba.shape[1]
    
    y_subject_proba = np.zeros((n_subjects, n_classes))
    
    # Average probabilities for each subject
    for i, subj_id in enumerate(unique_subjects):
        mask = subject_ids == subj_id
        subject_probas = y_pred_proba[mask]
        y_subject_proba[i] = np.mean(subject_probas, axis=0)
    
    # Get predicted class (argmax of averaged probabilities)
    y_subject_pred = np.argmax(y_subject_proba, axis=1)
    
    return y_subject_pred, y_subject_proba, unique_subjects


def evaluate_model_subject_level(model, X_test, y_test, subject_ids_test, model_name):
    """
    Evaluate a model at subject level by aggregating segment predictions.
    
    Args:
        model: Trained Keras model
        X_test: Test data (n_segments, channels, time, 1)
        y_test: Test labels categorical (n_segments, n_classes)
        subject_ids_test: Subject IDs for each segment (n_segments,)
        model_name: Name of model for logging
        
    Returns:
        Dictionary of metrics
    """
    print(f"\nEvaluating {model_name} at subject level...")
    
    # Get segment-level predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    
    # Get true labels (segment level)
    y_true_segment = np.argmax(y_test, axis=1)
    
    # Aggregate to subject level
    y_subject_pred, y_subject_proba, unique_subjects = aggregate_predictions_by_subject(
        y_pred_proba, subject_ids_test
    )
    
    # Get true labels at subject level (should be same for all segments of a subject)
    y_true_subject = np.zeros(len(unique_subjects), dtype=int)
    for i, subj_id in enumerate(unique_subjects):
        mask = subject_ids_test == subj_id
        # All segments from same subject should have same label
        subject_labels = y_true_segment[mask]
        y_true_subject[i] = subject_labels[0]  # Take first (all should be same)
    
    # Calculate subject-level metrics
    # For loss, we'll use the average of segment-level losses
    test_loss, _ = model.evaluate(X_test, y_test, verbose=0)
    
    # Subject-level accuracy
    subject_acc = np.mean(y_subject_pred == y_true_subject)
    
    # Classification report at subject level
    report = classification_report(y_true_subject, y_subject_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix at subject level
    cm = confusion_matrix(y_true_subject, y_subject_pred)
    
    # Segment-level metrics (for reference)
    y_pred_segment = np.argmax(y_pred_proba, axis=1)
    segment_acc = np.mean(y_pred_segment == y_true_segment)
    
    metrics = {
        'test_loss': float(test_loss),  # Segment-level loss
        'subject_accuracy': float(subject_acc),
        'segment_accuracy': float(segment_acc),  # For reference
        'n_subjects': int(len(unique_subjects)),
        'n_segments': int(len(X_test)),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    print(f"  Test Loss (segment-level): {test_loss:.4f}")
    print(f"  Subject-level Accuracy: {subject_acc:.4f} ({len(unique_subjects)} subjects)")
    print(f"  Segment-level Accuracy: {segment_acc:.4f} ({len(X_test)} segments)")
    print(f"  Precision: {report['weighted avg']['precision']:.4f}")
    print(f"  Recall: {report['weighted avg']['recall']:.4f}")
    print(f"  F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    return metrics


def main():
    print("=" * 60)
    print("Model Evaluation: Teacher, Student, and Benchmark (Subject-Level)")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X_unfiltered = np.load(os.path.join(UNFILTERED_DATA_DIR, EPOCHS_FILE))
    y_unfiltered = np.load(os.path.join(UNFILTERED_DATA_DIR, LABELS_FILE))
    subject_ids_unfiltered = np.load(os.path.join(UNFILTERED_DATA_DIR, SUBJECT_IDS_FILE), allow_pickle=True)
    
    X_filtered = np.load(os.path.join(FILTERED_DATA_DIR, EPOCHS_FILE))
    y_filtered = np.load(os.path.join(FILTERED_DATA_DIR, LABELS_FILE))
    subject_ids_filtered = np.load(os.path.join(FILTERED_DATA_DIR, SUBJECT_IDS_FILE), allow_pickle=True)
    
    # Ensure correct shape
    if len(X_unfiltered.shape) == 3:
        X_unfiltered = X_unfiltered[:, :, :, np.newaxis]
    if len(X_filtered.shape) == 3:
        X_filtered = X_filtered[:, :, :, np.newaxis]
    
    y_unfiltered_cat = keras.utils.to_categorical(y_unfiltered, num_classes=NB_CLASSES)
    y_filtered_cat = keras.utils.to_categorical(y_filtered, num_classes=NB_CLASSES)
    
    # Determine dimensions from data
    if len(X_unfiltered.shape) == 4:
        n_samples = X_unfiltered.shape[2]
        n_channels = X_unfiltered.shape[1]
    else:
        n_samples = X_unfiltered.shape[2]
        n_channels = X_unfiltered.shape[1]
    
    print(f"Data dimensions: {n_channels} channels, {n_samples} time points")
    
    # Split data by subject (not by segment) to avoid data leakage
    unique_subjects = np.unique(subject_ids_unfiltered)
    train_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=[y_unfiltered[subject_ids_unfiltered == subj][0] for subj in unique_subjects]
    )
    
    # Create masks for test set
    test_mask_unfiltered = np.isin(subject_ids_unfiltered, test_subjects)
    test_mask_filtered = np.isin(subject_ids_filtered, test_subjects)
    
    X_test_unfiltered = X_unfiltered[test_mask_unfiltered]
    y_test_unfiltered = y_unfiltered_cat[test_mask_unfiltered]
    subject_ids_test_unfiltered = subject_ids_unfiltered[test_mask_unfiltered]
    
    X_test_filtered = X_filtered[test_mask_filtered]
    y_test_filtered = y_filtered_cat[test_mask_filtered]
    subject_ids_test_filtered = subject_ids_filtered[test_mask_filtered]
    
    print(f"Test set (unfiltered): {X_test_unfiltered.shape[0]} segments from {len(test_subjects)} subjects")
    print(f"Test set (filtered): {X_test_filtered.shape[0]} segments from {len(test_subjects)} subjects")
    
    # Load models
    print("\nLoading models...")
    teacher_model = load_model(TEACHER_MODEL_DIR, n_samples, n_channels, "Teacher")
    student_model = load_model(STUDENT_MODEL_DIR, n_samples, n_channels, "Student")
    benchmark_model = load_model(BENCHMARK_MODEL_DIR, n_samples, n_channels, "Benchmark")
    
    # Evaluate models at subject level
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (Subject-Level)")
    print("=" * 60)
    
    teacher_metrics = evaluate_model_subject_level(
        teacher_model, X_test_unfiltered, y_test_unfiltered, 
        subject_ids_test_unfiltered, "Teacher"
    )
    student_metrics = evaluate_model_subject_level(
        student_model, X_test_filtered, y_test_filtered,
        subject_ids_test_filtered, "Student"
    )
    benchmark_metrics = evaluate_model_subject_level(
        benchmark_model, X_test_unfiltered, y_test_unfiltered,
        subject_ids_test_unfiltered, "Benchmark"
    )
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (Subject-Level)")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Teacher':<15} {'Student':<15} {'Benchmark':<15} {'Student-Benchmark':<15}")
    print("-" * 85)
    
    # Accuracy comparison (subject-level)
    teacher_acc = teacher_metrics['subject_accuracy']
    student_acc = student_metrics['subject_accuracy']
    benchmark_acc = benchmark_metrics['subject_accuracy']
    diff = student_acc - benchmark_acc
    
    print(f"{'Subject Accuracy':<25} {teacher_acc:<15.4f} {student_acc:<15.4f} {benchmark_acc:<15.4f} {diff:+.4f}")
    
    # Loss comparison (segment-level, for reference)
    teacher_loss = teacher_metrics['test_loss']
    student_loss = student_metrics['test_loss']
    benchmark_loss = benchmark_metrics['test_loss']
    diff_loss = student_loss - benchmark_loss
    
    print(f"{'Loss (segment-level)':<25} {teacher_loss:<15.4f} {student_loss:<15.4f} {benchmark_loss:<15.4f} {diff_loss:+.4f}")
    
    # F1 comparison (subject-level)
    teacher_f1 = teacher_metrics['classification_report']['weighted avg']['f1-score']
    student_f1 = student_metrics['classification_report']['weighted avg']['f1-score']
    benchmark_f1 = benchmark_metrics['classification_report']['weighted avg']['f1-score']
    diff_f1 = student_f1 - benchmark_f1
    
    print(f"{'F1-Score (subject)':<25} {teacher_f1:<15.4f} {student_f1:<15.4f} {benchmark_f1:<15.4f} {diff_f1:+.4f}")
    
    # Save comparison results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    comparison = {
        'evaluation_level': 'subject',
        'aggregation_method': 'average_probabilities',
        'teacher_metrics': teacher_metrics,
        'student_metrics': student_metrics,
        'benchmark_metrics': benchmark_metrics,
        'summary': {
            'student_vs_benchmark': {
                'subject_accuracy_diff': float(diff),
                'loss_diff': float(diff_loss),
                'f1_diff': float(diff_f1),
                'student_better': diff > 0
            },
            'teacher_vs_benchmark': {
                'subject_accuracy_diff': float(teacher_acc - benchmark_acc),
                'loss_diff': float(teacher_loss - benchmark_loss),
                'f1_diff': float(teacher_f1 - benchmark_f1)
            }
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'evaluation_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n\nComparison results saved to: {os.path.join(OUTPUT_DIR, 'evaluation_comparison.json')}")
    
    # Print conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    if diff > 0:
        print(f"✓ Student model (filtered data) performs BETTER than Benchmark")
        print(f"  Improvement: {diff*100:.2f}% subject-level accuracy")
    elif diff < 0:
        print(f"✗ Student model (filtered data) performs WORSE than Benchmark")
        print(f"  Decrease: {abs(diff)*100:.2f}% subject-level accuracy")
    else:
        print("= Student and Benchmark models perform similarly")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
