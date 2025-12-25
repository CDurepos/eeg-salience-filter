#!/usr/bin/env python3
"""
Generate Research Summary and Visualizations

This script generates:
- Salience map visualizations for each teacher
- Training history plots
- Data distribution comparisons
- Channel importance rankings
- Comprehensive JSON summary

Usage:
  python src/summary.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

UNFILTERED_DIR = 'data/unfiltered'
TEACHER_TYPES = ['eegnet', 'resnet', 'tst']
PLOTS_DIR = 'plots'
OUTPUTS_DIR = 'outputs'

CHANNEL_NAMES = [
    'Fz', 'Cz', 'Pz', 'C3', 'T3', 'C4', 'T4', 'Fp1', 'Fp2',
    'F3', 'F4', 'F7', 'F8', 'P3', 'P4', 'T5', 'T6', 'O1', 'O2'
]


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_salience_heatmaps(save_path):
    """Plot salience heatmaps for all teachers."""
    available = [t for t in TEACHER_TYPES if os.path.exists(f'outputs/{t}/salience_maps.npy')]
    
    if not available:
        print("  ⚠ No salience maps found")
        return
    
    fig, axes = plt.subplots(1, len(available), figsize=(6*len(available), 6))
    if len(available) == 1:
        axes = [axes]
    
    for ax, t in zip(axes, available):
        salience = np.load(f'outputs/{t}/salience_maps.npy')
        # Take absolute value and average across samples
        avg = np.mean(np.abs(salience), axis=0)
        
        n_ch = min(len(CHANNEL_NAMES), avg.shape[0])
        sns.heatmap(avg[:n_ch], cmap='hot', ax=ax, xticklabels=50,
                    yticklabels=CHANNEL_NAMES[:n_ch])
        ax.set_xlabel('Time')
        ax.set_ylabel('Channel')
        ax.set_title(f'{t.upper()} Teacher')
    
    plt.suptitle('Salience Maps by Teacher Architecture', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Salience heatmaps")


def plot_channel_importance(save_path):
    """Plot channel importance for each teacher."""
    available = [t for t in TEACHER_TYPES if os.path.exists(f'outputs/{t}/salience_maps.npy')]
    
    if not available:
        return
    
    fig, axes = plt.subplots(1, len(available), figsize=(5*len(available), 8))
    if len(available) == 1:
        axes = [axes]
    
    for ax, t in zip(axes, available):
        salience = np.load(f'outputs/{t}/salience_maps.npy')
        avg = np.mean(np.abs(salience), axis=0)
        
        # Channel importance = mean salience across time
        importance = np.mean(avg, axis=1)
        n_ch = min(len(CHANNEL_NAMES), len(importance))
        
        # Sort by importance
        idx = np.argsort(importance[:n_ch])[::-1]
        names = [CHANNEL_NAMES[i] for i in idx]
        vals = importance[idx] / importance.max()
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vals)))
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Normalized Importance')
        ax.set_title(f'{t.upper()}')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Channel Importance by Teacher', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Channel importance")


def plot_training_histories(save_path):
    """Plot training histories for all models."""
    histories = {}
    
    # Benchmark
    bp = 'models/benchmark_history.npy'
    if os.path.exists(bp):
        histories['Benchmark'] = np.load(bp, allow_pickle=True).item()
    
    # Teachers and Students
    for t in TEACHER_TYPES:
        tp = f'models/{t}/teacher_history.npy'
        sp = f'models/{t}/student_history.npy'
        if os.path.exists(tp):
            histories[f'Teacher ({t})'] = np.load(tp, allow_pickle=True).item()
        if os.path.exists(sp):
            histories[f'Student ({t})'] = np.load(sp, allow_pickle=True).item()
    
    if not histories:
        print("  ⚠ No training histories found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, h in histories.items():
        axes[0].plot(h['loss'], label=f'{name} (train)', alpha=0.8)
        axes[0].plot(h['val_loss'], '--', label=f'{name} (val)', alpha=0.8)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)
    
    for name, h in histories.items():
        axes[1].plot(h['accuracy'], label=f'{name} (train)', alpha=0.8)
        axes[1].plot(h['val_accuracy'], '--', label=f'{name} (val)', alpha=0.8)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.suptitle('Training Histories', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Training histories")


def plot_data_distributions(save_path):
    """Plot data distributions before/after filtering."""
    X_unfilt = np.load(f'{UNFILTERED_DIR}/epochs.npy')
    
    available = [t for t in TEACHER_TYPES if os.path.exists(f'data/filtered_{t}/epochs.npy')]
    
    n_plots = 1 + len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    
    # Unfiltered
    axes[0].hist(X_unfilt.flatten(), bins=100, density=True, alpha=0.7, color='blue')
    axes[0].set_title('Unfiltered')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].axvline(np.mean(X_unfilt), color='red', linestyle='--', label=f'μ={np.mean(X_unfilt):.2f}')
    axes[0].legend(fontsize=8)
    
    # Filtered for each teacher
    for i, t in enumerate(available):
        X_filt = np.load(f'data/filtered_{t}/epochs.npy')
        axes[i+1].hist(X_filt.flatten(), bins=100, density=True, alpha=0.7, color='green')
        axes[i+1].axvline(np.mean(X_filt), color='red', linestyle='--', label=f'μ={np.mean(X_filt):.2f}')
        axes[i+1].legend(fontsize=8)
        axes[i+1].set_title(f'Filtered ({t})')
        axes[i+1].set_xlabel('Value')
    
    plt.suptitle('Data Distributions', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Data distributions")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 60)
    print("Generating Summary & Visualizations")
    print("=" * 60)
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Generate plots
    # -------------------------------------------------------------------------
    print("\n[Plots] Generating...")
    
    plot_salience_heatmaps(f'{PLOTS_DIR}/salience_heatmaps.png')
    plot_channel_importance(f'{PLOTS_DIR}/channel_importance.png')
    plot_training_histories(f'{PLOTS_DIR}/training_histories.png')
    plot_data_distributions(f'{PLOTS_DIR}/data_distributions.png')
    
    # -------------------------------------------------------------------------
    # Build summary
    # -------------------------------------------------------------------------
    print("\n[Summary] Building...")
    
    summary = {
        'generated_at': datetime.now().isoformat(),
        'study': 'EEG Salience-Based Filtering for ADHD Classification',
        'framework': 'PyTorch + braindecode + tsai',
        'teachers': TEACHER_TYPES,
        'results': {}
    }
    
    # Load evaluation results
    eval_path = f'{OUTPUTS_DIR}/evaluation_results.json'
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            summary['evaluation'] = json.load(f)
    
    # Load processing metadata for each teacher
    for t in TEACHER_TYPES:
        meta_path = f'outputs/{t}/processing_meta.json'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                summary['results'][t] = json.load(f)
    
    # Save summary
    with open(f'{OUTPUTS_DIR}/research_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESEARCH SUMMARY")
    print("=" * 60)
    
    if 'evaluation' in summary and 'results' in summary['evaluation']:
        print(f"\n{'Model':<25} {'Accuracy':<12} {'F1':<12}")
        print("-" * 49)
        for r in summary['evaluation']['results']:
            print(f"{r['name']:<25} {r['accuracy']:<12.4f} {r['f1']:<12.4f}")
    
    print(f"\n[Save] Summary → {OUTPUTS_DIR}/research_summary.json")
    print(f"[Save] Plots → {PLOTS_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
