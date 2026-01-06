"""Benchmark script: Train EEGNet on raw preprocessed EEG data (no masking/salience filtering).

This script trains a baseline EEGNet model on the raw preprocessed data to serve as a
comparison baseline against models trained on salience-filtered data.

Usage:
    python src/benchmark/raw_eeg.py --data data/epochs_subjectsplit.npz --out outputs/benchmark_raw_eeg
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Add arl-eegmodels to path
EEGMODELS_DIR = ROOT / "arl-eegmodels"
sys.path.insert(0, str(EEGMODELS_DIR))

try:
    from EEGModels import EEGNet
except Exception:
    import importlib.util
    eegmodels_path = EEGMODELS_DIR / "EEGModels.py"
    if not eegmodels_path.exists():
        raise FileNotFoundError(f"Could not find EEGModels.py at {eegmodels_path}")
    spec = importlib.util.spec_from_file_location("EEGModels", str(eegmodels_path))
    EEGModels = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(EEGModels)
    EEGNet = EEGModels.EEGNet

# Import plotting utilities from eval.py
sys.path.insert(0, str(ROOT / "src"))
from eval import (
    plot_confusion_matrix,
    plot_per_class_bars,
    plot_learning_curves,
    plot_roc_curve,
    ensure_zero_based_labels,
)


def set_tf_runtime():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def infer_nb_classes(y: np.ndarray) -> int:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"Expected y to be 1D int labels, got shape {y.shape}")
    return int(np.max(y)) + 1


def main():
    parser = argparse.ArgumentParser(
        description="Train EEGNet baseline on raw preprocessed EEG data"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/epochs_subjectsplit.npz",
        help="Path to preprocessed .npz with X_train/y_train/X_val/y_val"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/run/benchmark_raw_eeg",
        help="Output directory for model + evaluation artifacts"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # EEGNet architecture parameters
    parser.add_argument("--F1", type=int, default=16, help="Number of temporal filters")
    parser.add_argument("--D", type=int, default=4, help="Depth multiplier")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--kernLength", type=int, default=32, help="Temporal kernel length")

    args = parser.parse_args()

    set_tf_runtime()

    # Resolve paths
    data_path = (ROOT / args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load raw preprocessed dataset
    # ----------------------------
    print(f"[INFO] Loading data from: {data_path}")
    d = np.load(data_path, allow_pickle=True)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(int)
    X_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(int)

    # Ensure labels are zero-based
    y_train, train_map = ensure_zero_based_labels(y_train)
    y_val, _ = ensure_zero_based_labels(y_val)

    # Shape validation
    if X_train.ndim != 4 or X_train.shape[-1] != 1:
        raise ValueError(f"Expected X_train shape (N,Chans,Samples,1), got {X_train.shape}")
    if X_val.ndim != 4 or X_val.shape[-1] != 1:
        raise ValueError(f"Expected X_val shape (N,Chans,Samples,1), got {X_val.shape}")

    # NaN/Inf checks
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN/Inf")
    if not np.isfinite(X_val).all():
        raise ValueError("X_val contains NaN/Inf")

    Chans = int(X_train.shape[1])
    Samples = int(X_train.shape[2])
    nb_classes = infer_nb_classes(y_train)

    if args.kernLength > Samples:
        print(f"[WARN] kernLength ({args.kernLength}) > Samples ({Samples}). Clamping to {Samples}.")
        args.kernLength = Samples

    print(f"[INFO] X_train: {X_train.shape}  y_train: {y_train.shape}  classes={nb_classes}")
    print(f"[INFO] X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print("[INFO] Train class counts:", Counter(y_train))
    print("[INFO] Val class counts:  ", Counter(y_val))
    print("[INFO] Label map (train):", train_map)

    # Class names for plots
    class_names = [str(i) for i in range(nb_classes)]

    # ----------------------------
    # Build EEGNet model
    # ----------------------------
    print("\n[INFO] Building EEGNet model...")
    model = EEGNet(
        nb_classes=nb_classes,
        Chans=Chans,
        Samples=Samples,
        dropoutRate=args.dropout,
        kernLength=args.kernLength,
        F1=args.F1,
        D=args.D,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    model.summary()

    # ----------------------------
    # Training callbacks
    # ----------------------------
    best_path = out_dir / "model_best.keras"
    final_path = out_dir / "model_final.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=False),
    ]

    # ----------------------------
    # Train model
    # ----------------------------
    print("\n[INFO] Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    model.save(final_path)
    print(f"[INFO] Saved best model:  {best_path}")
    print(f"[INFO] Saved final model: {final_path}")

    # ----------------------------
    # Evaluation
    # ----------------------------
    print("\n[INFO] Evaluating model...")
    start_time = time.time()
    probs = model.predict(X_val, batch_size=args.batch_size, verbose=0)
    inference_time = time.time() - start_time
    inference_time_per_sample = inference_time / len(X_val)

    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(nb_classes)))
    report = classification_report(y_val, y_pred, target_names=class_names, digits=4)

    # Add inference time to classification report
    report_with_time = report + f"\n\nInference Time:\n"
    report_with_time += f"  Total time: {inference_time:.4f} seconds\n"
    report_with_time += f"  Time per sample: {inference_time_per_sample*1000:.4f} ms\n"
    report_with_time += f"  Number of samples: {len(X_val)}\n"

    print("\n[VAL] accuracy:", acc)
    print("\n[VAL] classification report:\n", report_with_time)
    print(f"[VAL] Inference time: {inference_time:.4f}s ({inference_time_per_sample*1000:.4f} ms per sample)")

    # ----------------------------
    # Save metrics and artifacts
    # ----------------------------
    # Save training history
    with open(out_dir / "history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    # Save evaluation metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "val_accuracy": float(acc),
            "inference_time_total_seconds": float(inference_time),
            "inference_time_per_sample_ms": float(inference_time_per_sample * 1000),
            "Chans": Chans,
            "Samples": Samples,
            "nb_classes": nb_classes,
            "kernLength": args.kernLength,
            "F1": args.F1,
            "D": args.D,
            "dropout": args.dropout,
        }, f, indent=2)

    # Save training arguments
    with open(out_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save classification report
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report_with_time)

    # Save confusion matrix
    np.save(out_dir / "confusion_matrix.npy", cm)

    # ----------------------------
    # Generate visualizations
    # ----------------------------
    print("\n[INFO] Generating visualizations...")
    try:
        plot_learning_curves({k: v for k, v in history.history.items()}, out_dir / "learning_curves.png")
        print("  ✓ Learning curves saved")
    except Exception as e:
        print(f"  ✗ Failed to plot learning curves: {e}")

    try:
        plot_roc_curve(y_val, probs, class_names, out_dir / "roc_curve.png")
        print("  ✓ ROC curve saved")
    except Exception as e:
        print(f"  ✗ Failed to plot ROC curve: {e}")

    try:
        plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
        print("  ✓ Confusion matrix saved")
    except Exception as e:
        print(f"  ✗ Failed to plot confusion matrix: {e}")

    try:
        plot_per_class_bars(cm, class_names, out_dir / "per_class_accuracy.png")
        print("  ✓ Per-class accuracy plot saved")
    except Exception as e:
        print(f"  ✗ Failed to plot per-class accuracy: {e}")

    print(f"\n[DONE] Benchmark complete! Results saved to: {out_dir}")
    print(f"[DONE] Best model:  {best_path}")
    print(f"[DONE] Final model: {final_path}")


if __name__ == "__main__":
    main()
