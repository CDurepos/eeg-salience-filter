#!/usr/bin/env python3
"""
Evaluate All Models: Benchmark vs Students (PyTorch)

This script evaluates:
- Benchmark EEGNet (baseline, trained on unfiltered data)
- Student EEGNet models trained on data filtered by each teacher:
  - EEGNet teacher
  - ResNet teacher
  - TST teacher

Generates unified comparison plots and metrics.

Usage:
  python src/evaluate.py
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from braindecode.models import EEGNetv4
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

UNFILTERED_DATA_DIR = 'data/unfiltered'
TEACHER_TYPES = ['eegnet', 'resnet', 'tst']

NB_CLASSES = 2
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 128

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_benchmark(n_channels, n_samples):
    """Load benchmark EEGNet model."""
    model_path = 'models/benchmark.pt'
    if not os.path.exists(model_path):
        print(f"  ⚠ Benchmark not found: {model_path}")
        return None
    
    model = EEGNetv4(
        n_chans=n_channels, n_outputs=NB_CLASSES, n_times=n_samples,
        final_conv_length='auto', drop_prob=0.5
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_student(teacher_type, n_channels, n_samples):
    """Load student EEGNet model."""
    model_path = f'models/{teacher_type}/student.pt'
    if not os.path.exists(model_path):
        print(f"  ⚠ Student ({teacher_type}) not found: {model_path}")
        return None
    
    model = EEGNetv4(
        n_chans=n_channels, n_outputs=NB_CLASSES, n_times=n_samples,
        final_conv_length='auto', drop_prob=0.5
    )
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def get_predictions(model, X_data):
    """Get predictions and probabilities."""
    model.eval()
    X_tensor = torch.FloatTensor(X_data).to(DEVICE)
    
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), BATCH_SIZE):
            batch = X_tensor[i:i+BATCH_SIZE]
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_probs, axis=0)


def aggregate_to_subjects(y_proba, subject_ids):
    """Aggregate segment predictions to subject level."""
    unique = np.unique(subject_ids)
    proba = np.zeros((len(unique), y_proba.shape[1]))
    for i, s in enumerate(unique):
        proba[i] = np.mean(y_proba[subject_ids == s], axis=0)
    return np.argmax(proba, axis=1), proba, unique


def evaluate_model(model, X_data, y_data, subject_ids, name):
    """Evaluate model at subject level with per-class metrics."""
    if model is None:
        return None
    
    y_proba = get_predictions(model, X_data)
    y_pred, y_proba_subj, unique = aggregate_to_subjects(y_proba, subject_ids)
    y_true = np.array([y_data[subject_ids == s][0] for s in unique])
    
    acc = np.mean(y_pred == y_true)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Overall (weighted) metrics
    precision = report.get('1', report['weighted avg'])['precision']
    recall = report.get('1', report['weighted avg'])['recall']
    f1 = report.get('1', report['weighted avg'])['f1-score']
    
    # Per-class metrics
    class_metrics = {
        'control': {
            'precision': float(report.get('0', {}).get('precision', 0)),
            'recall': float(report.get('0', {}).get('recall', 0)),
            'f1': float(report.get('0', {}).get('f1-score', 0)),
            'support': int(report.get('0', {}).get('support', 0))
        },
        'adhd': {
            'precision': float(report.get('1', {}).get('precision', 0)),
            'recall': float(report.get('1', {}).get('recall', 0)),
            'f1': float(report.get('1', {}).get('f1-score', 0)),
            'support': int(report.get('1', {}).get('support', 0))
        }
    }
    
    print(f"  {name}: Acc={acc:.4f}, F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
    
    return {
        'name': name,
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'class_metrics': class_metrics,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba_subj,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


def measure_inference(model, X_data, n_runs=10):
    """Measure inference time."""
    if model is None:
        return None
    
    X_tensor = torch.FloatTensor(X_data).to(DEVICE)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(X_tensor[:BATCH_SIZE])
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            for i in range(0, len(X_tensor), BATCH_SIZE):
                _ = model(X_tensor[i:i+BATCH_SIZE])
        times.append(time.perf_counter() - start)
    
    return {
        'avg_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'throughput': float(len(X_data) / np.mean(times))
    }


# =============================================================================
# COMPUTATIONAL METRICS
# =============================================================================

def get_model_size(model_path):
    """Get model file size in KB."""
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / 1024
    return 0


def count_parameters(model):
    """Count trainable parameters."""
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_data_sparsity(X_original, X_filtered):
    """
    Compute effective data sparsity after filtering.
    
    With neutral fill (channel mean), we measure how much signal variance
    was preserved vs replaced with constant values.
    """
    # Variance preservation
    orig_var = np.var(X_original)
    filt_var = np.var(X_filtered)
    variance_retained = filt_var / orig_var if orig_var > 0 else 0
    
    # Per-sample signal energy
    orig_energy = np.mean(X_original ** 2)
    filt_energy = np.mean(X_filtered ** 2)
    
    return {
        'original_std': float(np.std(X_original)),
        'filtered_std': float(np.std(X_filtered)),
        'variance_retained_pct': float(variance_retained * 100),
        'original_energy': float(orig_energy),
        'filtered_energy': float(filt_energy)
    }


def load_training_history(history_path):
    """Load training history and compute convergence metrics."""
    if not os.path.exists(history_path):
        return None
    
    history = np.load(history_path, allow_pickle=True).item()
    
    # Find best epoch
    val_losses = history.get('val_loss', [])
    if not val_losses:
        return None
    
    best_epoch = np.argmin(val_losses) + 1
    total_epochs = len(val_losses)
    best_val_loss = min(val_losses)
    best_val_acc = max(history.get('val_accuracy', [0]))
    
    return {
        'total_epochs': total_epochs,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'best_val_accuracy': float(best_val_acc),
        'convergence_speed': float(best_epoch / total_epochs)  # Lower = faster convergence
    }


def compute_teacher_overhead():
    """Compute the computational overhead of each teacher model."""
    from tsai.models.ResNet import ResNet
    from tsai.models.TST import TST
    
    overhead = {}
    
    # EEGNet teacher
    eegnet = EEGNetv4(n_chans=15, n_outputs=2, n_times=256, 
                      final_conv_length='auto', drop_prob=0.5)
    overhead['eegnet'] = {
        'parameters': count_parameters(eegnet),
        'file_size_kb': get_model_size('models/eegnet/teacher.pt')
    }
    
    # ResNet teacher  
    resnet = ResNet(c_in=15, c_out=2)
    overhead['resnet'] = {
        'parameters': count_parameters(resnet),
        'file_size_kb': get_model_size('models/resnet/teacher.pt')
    }
    
    # TST teacher
    tst = TST(c_in=15, c_out=2, seq_len=256, n_layers=4, n_heads=4,
              d_model=64, d_ff=128, dropout=0.1, fc_dropout=0.3)
    overhead['tst'] = {
        'parameters': count_parameters(tst),
        'file_size_kb': get_model_size('models/tst/teacher.pt')
    }
    
    return overhead


# =============================================================================
# PLOTTING
# =============================================================================

def plot_confusion_matrices(results, save_path):
    """Plot confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    
    for ax, r in zip(axes, results):
        cm = np.array(r['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Control', 'ADHD'], yticklabels=['Control', 'ADHD'])
        acc = np.trace(cm) / np.sum(cm)
        ax.set_title(f"{r['name']}\nAcc: {acc:.2%}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(results, save_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for r, c in zip(results, colors):
        fpr, tpr, _ = roc_curve(r['y_true'], r['y_proba'][:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=c, lw=2, label=f"{r['name']} (AUC={roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Subject Level')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(results, save_path):
    """Bar chart comparing all models."""
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for i, (r, c) in enumerate(zip(results, colors)):
        vals = [r['accuracy'], r['f1'], r['precision'], r['recall']]
        offset = (i - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=r['name'], color=c, alpha=0.85)
        
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison (Subject-Level)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim([0, 1.15])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(results, save_path):
    """Plot per-class precision, recall, F1 for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = [r['name'] for r in results]
    x = np.arange(len(models))
    width = 0.25
    
    # Control (Class 0)
    ax = axes[0]
    control_prec = [r.get('class_metrics', {}).get('control', {}).get('precision', 0) for r in results]
    control_rec = [r.get('class_metrics', {}).get('control', {}).get('recall', 0) for r in results]
    control_f1 = [r.get('class_metrics', {}).get('control', {}).get('f1', 0) for r in results]
    
    bars1 = ax.bar(x - width, control_prec, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, control_rec, width, label='Recall', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, control_f1, width, label='F1', color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Control (Class 0) - Per-Model Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # ADHD (Class 1)
    ax = axes[1]
    adhd_prec = [r.get('class_metrics', {}).get('adhd', {}).get('precision', 0) for r in results]
    adhd_rec = [r.get('class_metrics', {}).get('adhd', {}).get('recall', 0) for r in results]
    adhd_f1 = [r.get('class_metrics', {}).get('adhd', {}).get('f1', 0) for r in results]
    
    bars1 = ax.bar(x - width, adhd_prec, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, adhd_rec, width, label='Recall', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, adhd_f1, width, label='F1', color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('ADHD (Class 1) - Per-Model Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_computational_metrics(teacher_overhead, training_metrics, data_metrics, save_path):
    """Plot computational overhead comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    teachers = list(teacher_overhead.keys())
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    # 1. Teacher Model Size (Parameters)
    ax = axes[0, 0]
    params = [teacher_overhead[t]['parameters'] / 1000 for t in teachers]
    bars = ax.bar(teachers, params, color=colors, alpha=0.8)
    ax.set_ylabel('Parameters (K)')
    ax.set_title('Teacher Model Complexity')
    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{p:.1f}K', ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, max(params) * 1.2)
    
    # 2. Teacher File Size
    ax = axes[0, 1]
    sizes = [teacher_overhead[t]['file_size_kb'] for t in teachers]
    bars = ax.bar(teachers, sizes, color=colors, alpha=0.8)
    ax.set_ylabel('File Size (KB)')
    ax.set_title('Teacher Model Storage')
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{s:.0f}KB', ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, max(sizes) * 1.2)
    
    # 3. Variance Retained After Filtering
    ax = axes[1, 0]
    if data_metrics:
        var_retained = [data_metrics.get(t, {}).get('variance_retained_pct', 0) for t in teachers]
        bars = ax.bar(teachers, var_retained, color=colors, alpha=0.8)
        ax.set_ylabel('Variance Retained (%)')
        ax.set_title('Signal Preservation After Filtering')
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Original')
        for bar, v in zip(bars, var_retained):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax.set_ylim(0, 110)
    
    # 4. Training Convergence (epochs to best)
    ax = axes[1, 1]
    if training_metrics:
        labels = ['Benchmark'] + [f'Student ({t})' for t in teachers]
        epochs = [training_metrics.get('benchmark', {}).get('best_epoch', 0)]
        epochs += [training_metrics.get(f'student_{t}', {}).get('best_epoch', 0) for t in teachers]
        
        bar_colors = ['gray'] + colors
        bars = ax.bar(labels, epochs, color=bar_colors, alpha=0.8)
        ax.set_ylabel('Epochs to Best Validation')
        ax.set_title('Training Convergence Speed')
        ax.tick_params(axis='x', rotation=15)
        for bar, e in zip(bars, epochs):
            if e > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{e}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Computational Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 60)
    print("Model Evaluation: Benchmark vs All Students (PyTorch)")
    print("=" * 60)
    print(f"[Device] Using: {DEVICE}")
    
    # -------------------------------------------------------------------------
    # Load unfiltered data
    # -------------------------------------------------------------------------
    print("\n[Data] Loading unfiltered data...")
    
    X = np.load(f'{UNFILTERED_DATA_DIR}/epochs.npy')
    y = np.load(f'{UNFILTERED_DATA_DIR}/y_labels.npy')
    subject_ids = np.load(f'{UNFILTERED_DATA_DIR}/subject_ids.npy', allow_pickle=True)
    
    if len(X.shape) == 4:
        X = X[:, :, :, 0]
    
    n_channels, n_samples = X.shape[1], X.shape[2]
    
    # Split by subject
    unique_subjects = np.unique(subject_ids)
    subject_labels = [y[subject_ids == s][0] for s in unique_subjects]
    _, test_subj = train_test_split(
        unique_subjects, test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE, stratify=subject_labels
    )
    
    test_mask = np.isin(subject_ids, test_subj)
    X_test = X[test_mask]
    y_test = y[test_mask]
    subject_ids_test = subject_ids[test_mask]
    
    print(f"[Data] Test: {len(test_subj)} subjects, {X_test.shape[0]} segments")
    
    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    print("\n[Models] Loading...")
    
    benchmark = load_benchmark(n_channels, n_samples)
    if benchmark:
        print(f"  ✓ Benchmark loaded")
    
    students = {}
    filtered_data = {}
    
    for t in TEACHER_TYPES:
        students[t] = load_student(t, n_channels, n_samples)
        if students[t]:
            print(f"  ✓ Student ({t}) loaded")
        
        # Load filtered data
        fpath = f'data/filtered_{t}/epochs.npy'
        if os.path.exists(fpath):
            Xf = np.load(fpath)
            if len(Xf.shape) == 4:
                Xf = Xf[:, :, :, 0]
            filtered_data[t] = Xf[test_mask]
    
    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------
    print("\n[Eval] Evaluating models at subject level...")
    
    results = []
    
    # Benchmark
    r = evaluate_model(benchmark, X_test, y_test, subject_ids_test, 'Benchmark')
    if r:
        results.append(r)
    
    # Students
    for t in TEACHER_TYPES:
        if students[t] is not None and t in filtered_data:
            r = evaluate_model(students[t], filtered_data[t], y_test, subject_ids_test, f'Student ({t})')
            if r:
                results.append(r)
    
    if not results:
        print("\n⚠ No models to evaluate!")
        return
    
    # -------------------------------------------------------------------------
    # Inference timing
    # -------------------------------------------------------------------------
    print("\n[Timing] Measuring inference speed...")
    
    bench_time = measure_inference(benchmark, X_test)
    student_times = {}
    for t in TEACHER_TYPES:
        if students[t] is not None and t in filtered_data:
            student_times[t] = measure_inference(students[t], filtered_data[t])
            if student_times[t]:
                print(f"  Student ({t}): {student_times[t]['avg_ms']:.1f}ms")
    
    if bench_time:
        print(f"  Benchmark: {bench_time['avg_ms']:.1f}ms")
    
    # -------------------------------------------------------------------------
    # Computational Metrics
    # -------------------------------------------------------------------------
    print("\n[Compute] Analyzing computational overhead...")
    
    # Teacher model overhead
    teacher_overhead = compute_teacher_overhead()
    for t, metrics in teacher_overhead.items():
        print(f"  Teacher ({t}): {metrics['parameters']:,} params, {metrics['file_size_kb']:.0f}KB")
    
    # Data efficiency (variance retained after filtering)
    data_metrics = {}
    for t in TEACHER_TYPES:
        if t in filtered_data:
            data_metrics[t] = compute_data_sparsity(X_test, filtered_data[t])
            print(f"  Filter ({t}): {data_metrics[t]['variance_retained_pct']:.1f}% variance retained")
    
    # Training convergence
    training_metrics = {}
    
    # Benchmark history
    bench_hist = load_training_history('models/benchmark_history.npy')
    if bench_hist:
        training_metrics['benchmark'] = bench_hist
        print(f"  Benchmark: converged at epoch {bench_hist['best_epoch']}")
    
    # Student histories
    for t in TEACHER_TYPES:
        hist = load_training_history(f'models/{t}/student_history.npy')
        if hist:
            training_metrics[f'student_{t}'] = hist
            print(f"  Student ({t}): converged at epoch {hist['best_epoch']}")
    
    # -------------------------------------------------------------------------
    # Generate plots
    # -------------------------------------------------------------------------
    print("\n[Plots] Generating visualizations...")
    
    os.makedirs('plots', exist_ok=True)
    
    plot_confusion_matrices(results, 'plots/confusion_matrices.png')
    print("  ✓ Confusion matrices")
    
    plot_roc_curves(results, 'plots/roc_curves.png')
    print("  ✓ ROC curves")
    
    plot_metrics_comparison(results, 'plots/metrics_comparison.png')
    print("  ✓ Metrics comparison")
    
    plot_per_class_metrics(results, 'plots/per_class_metrics.png')
    print("  ✓ Per-class metrics")
    
    plot_computational_metrics(teacher_overhead, training_metrics, data_metrics, 
                               'plots/computational_analysis.png')
    print("  ✓ Computational analysis")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 73)
    for r in results:
        print(f"{r['name']:<25} {r['accuracy']:<12.4f} {r['f1']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f}")
    
    # Compare to benchmark
    if len(results) > 1:
        bench_acc = results[0]['accuracy']
        print("\n[vs Benchmark]")
        for r in results[1:]:
            diff = r['accuracy'] - bench_acc
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  {r['name']}: {diff:+.4f} ({symbol})")
    
    # Per-class metrics
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)
    
    print("\n[Control (Class 0)]")
    print(f"{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 71)
    for r in results:
        cm = r.get('class_metrics', {}).get('control', {})
        print(f"{r['name']:<25} {cm.get('precision', 0):<12.4f} {cm.get('recall', 0):<12.4f} "
              f"{cm.get('f1', 0):<12.4f} {cm.get('support', 0):<10}")
    
    print("\n[ADHD (Class 1)]")
    print(f"{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 71)
    for r in results:
        cm = r.get('class_metrics', {}).get('adhd', {})
        print(f"{r['name']:<25} {cm.get('precision', 0):<12.4f} {cm.get('recall', 0):<12.4f} "
              f"{cm.get('f1', 0):<12.4f} {cm.get('support', 0):<10}")
    
    # Computational summary
    print("\n" + "=" * 60)
    print("COMPUTATIONAL SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Teacher':<12} {'Params':<15} {'Size':<12} {'Var. Retained':<15}")
    print("-" * 54)
    for t in TEACHER_TYPES:
        params = f"{teacher_overhead[t]['parameters']:,}"
        size = f"{teacher_overhead[t]['file_size_kb']:.0f}KB"
        var_ret = f"{data_metrics.get(t, {}).get('variance_retained_pct', 0):.1f}%"
        print(f"{t:<12} {params:<15} {size:<12} {var_ret:<15}")
    
    print(f"\n{'Model':<25} {'Inference (ms)':<15} {'Throughput':<15}")
    print("-" * 55)
    if bench_time:
        print(f"{'Benchmark':<25} {bench_time['avg_ms']:<15.1f} {bench_time['throughput']:<15.0f}")
    for t in TEACHER_TYPES:
        if t in student_times and student_times[t]:
            print(f"{f'Student ({t})':<25} {student_times[t]['avg_ms']:<15.1f} {student_times[t]['throughput']:<15.0f}")
    
    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    os.makedirs('outputs', exist_ok=True)
    
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k not in ['y_true', 'y_pred', 'y_proba']}
        json_results.append(jr)
    
    # Convert numpy types to Python native types for JSON
    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    output = {
        'results': json_results,
        'benchmark_inference': bench_time,
        'student_inference': {k: v for k, v in student_times.items() if v},
        'teacher_overhead': to_python(teacher_overhead),
        'data_metrics': to_python(data_metrics),
        'training_metrics': to_python(training_metrics)
    }
    
    with open('outputs/evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[Save] Results → outputs/evaluation_results.json")
    print(f"[Save] Plots → plots/")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
