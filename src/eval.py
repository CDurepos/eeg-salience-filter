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
from typing import List, Dict



# Utilities

def set_tf_runtime():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def add_arl_eegmodels_to_path() -> Path:
    root = Path(__file__).resolve().parents[1]  # project root
    eegmodels_dir = root/ "arl-eegmodels"
    if not eegmodels_dir.exists():
        raise FileNotFoundError(
            f"Could not find arl-eegmodels at {eegmodels_dir}. "
            "Adjust the path if you cloned it somewhere else."
        )
    sys.path.insert(0, str(eegmodels_dir))
    return root

#sanity check
def assert_eegnet_shape(X: np.ndarray, chans: int | None = None, samples: int | None = None):
    if X.ndim != 4 or X.shape[-1] != 1:
        raise ValueError(f"Expected X shape (N,Chans,Samples,1), got {X.shape}")
    if chans is not None and X.shape[1] != chans:
        raise ValueError(f"Expected Chans={chans}, got {X.shape[1]}")
    if samples is not None and X.shape[2] != samples:
        raise ValueError(f"Expected Samples={samples}, got {X.shape[2]}")


def ensure_zero_based_labels(y: np.ndarray):
    """If labels are {1,2} or non-contiguous, remap to 0..K-1 safely."""
    y = np.asarray(y).reshape(-1)
    uniq = np.unique(y)
    mapping = {lab: i for i, lab in enumerate(uniq)}
    y2 = np.array([mapping[v] for v in y], dtype=int)
    return y2, mapping

#when plot is done without any processingg it ends up prodicing the row wise confusion matrix and when we want the subject wise confusion matrix we use the function below
def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_png(cm: np.ndarray, class_names: list[str], out_path: Path, title: str):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def subjectwise_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subject_ids: np.ndarray,
    out_dir: str | Path,
    labels: list[int] = (0, 1),
    class_names: list[str] | None = None,
    prefix: str = "subjectwise"
):
    """
    Groups rows by subject_id, does majority-vote per subject, computes confusion matrix.
    Saves:
      - {prefix}_confusion_matrix_subject.npy
      - {prefix}_confusion_matrix_subject.png
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    subject_ids = np.asarray(subject_ids).reshape(-1)

    if not (len(y_true) == len(y_pred) == len(subject_ids)):
        raise ValueError("y_true, y_pred, subject_ids must be the same length and row-aligned.")

    subjects = np.unique(subject_ids)
    subj_true = []
    subj_pred = []

    for s in subjects:
        idx = (subject_ids == s)

        # true label per subject (should be consistent; use vote to be safe)
        t = y_true[idx]
        true_label = np.bincount(t).argmax()

        # predicted label per subject
        p = y_pred[idx]
        pred_label = np.bincount(p).argmax()

        subj_true.append(true_label)
        subj_pred.append(pred_label)

    subj_true = np.array(subj_true, dtype=int)
    subj_pred = np.array(subj_pred, dtype=int)

    cm = confusion_matrix(subj_true, subj_pred, labels=list(labels))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{prefix}_confusion_matrix_subject.npy", cm)

    if class_names is None:
        class_names = [str(l) for l in labels]

    save_confusion_matrix_png(
        cm,
        class_names=class_names,
        out_path=out_dir / f"{prefix}_confusion_matrix_subject.png",
        title="Subject-wise Confusion Matrix"
    )

    return cm, subj_true, subj_pred, subjects

def plot_per_class_bars(cm: np.ndarray, class_names: list[str], out_path: Path):
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    per_class_acc = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support != 0)

    fig = plt.figure()
    x = np.arange(len(class_names))

    # Bar: per-class accuracy
    plt.bar(x, per_class_acc)
    plt.ylim(0.0, 1.0)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Per-class Accuracy (Val)")

    # Add support labels on top
    for i, (acc, sup) in enumerate(zip(per_class_acc, support)):
        plt.text(i, acc + 0.02, f"n={int(sup)}", ha="center", va="bottom")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_learning_curves(history: Dict[str, List[float]], out_path: Path):
    """Plot train/val loss and accuracy from a Keras history-like dict.

    `history` is expected to be a dict with keys like 'loss','val_loss','accuracy','val_accuracy'.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])
    acc = history.get('accuracy', []) or history.get('acc', [])
    val_acc = history.get('val_accuracy', [])

    epochs = range(1, max(len(loss), len(val_loss), len(acc), len(val_acc)) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    if loss or val_loss:
        axes[0].plot(epochs[:len(loss)], loss, label='train_loss')
        axes[0].plot(epochs[:len(val_loss)], val_loss, label='val_loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

    # Accuracy
    if acc or val_acc:
        axes[1].plot(epochs[:len(acc)], acc, label='train_acc')
        axes[1].plot(epochs[:len(val_acc)], val_acc, label='val_acc')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, class_names: list[str], out_path: Path):
    """Plot ROC curve(s) for binary or multi-class classification.

    For binary classification: plots single ROC curve.
    For multi-class: plots one-vs-rest ROC curves for each class.

    Args:
        y_true: True labels (shape: [n_samples])
        y_probs: Predicted probabilities (shape: [n_samples, n_classes])
        class_names: Names for each class
        out_path: Path to save the plot
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    y_true = np.asarray(y_true).reshape(-1)
    y_probs = np.asarray(y_probs)

    if y_probs.shape[1] != n_classes:
        raise ValueError(f"y_probs shape {y_probs.shape} doesn't match n_classes={n_classes}")

    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2:
        # Binary classification: single ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        auc = roc_auc_score(y_true, y_probs[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f'{class_names[1]} (AUC = {auc:.3f})')
    else:
        # Multi-class: one-vs-rest approach
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        
        for i, class_name in enumerate(class_names):
            if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class has both positive and negative samples
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {auc:.3f})')

    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



# Main for running the eegnet teacher student model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="outputs/filtered/epochs_subjectsplit_masked.npz",
                        help="Masked dataset npz: X_train,y_train,X_val,y_val")
    parser.add_argument("--out", type=str, default="outputs/student_eval",
                        help="Output folder for model + plots")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Student capacity knobs (smaller than teacher)
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--kernLength", type=int, default=32)

    args = parser.parse_args()

    set_tf_runtime()
    root = add_arl_eegmodels_to_path()

    # Import EEGModels
    from EEGModels import EEGNet

    data_path = (root / args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Masked dataset not found: {data_path}")

    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

   
    # Load filtered / masked dataset

    d = np.load(data_path, allow_pickle=True)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(int)
    X_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(int)

    # Labels safety (won't change anything if already 0/1)
    y_train, train_map = ensure_zero_based_labels(y_train)
    y_val, _ = ensure_zero_based_labels(y_val)

    # Shape checks
    assert_eegnet_shape(X_train)
    assert_eegnet_shape(X_val, chans=X_train.shape[1], samples=X_train.shape[2])

    # NaN/Inf checks
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN/Inf")
    if not np.isfinite(X_val).all():
        raise ValueError("X_val contains NaN/Inf")

    Chans = int(X_train.shape[1])
    Samples = int(X_train.shape[2])
    nb_classes = int(np.max(y_train)) + 1

    if args.kernLength > Samples:
        print(f"[WARN] kernLength ({args.kernLength}) > Samples ({Samples}). Clamping to {Samples}.")
        args.kernLength = Samples

    print(f"[INFO] X_train: {X_train.shape} y_train: {y_train.shape} classes={nb_classes}")
    print(f"[INFO] X_val:   {X_val.shape} y_val:   {y_val.shape}")
    print("[INFO] Train class counts:", Counter(y_train))
    print("[INFO] Val class counts:  ", Counter(y_val))
    print("[INFO] Label map (train):", train_map)

    # Class names for plots
    class_names = [str(i) for i in range(nb_classes)]


    # Build + Train student EEGNet

    student = EEGNet(
        nb_classes=nb_classes,
        Chans=Chans,
        Samples=Samples,
        dropoutRate=args.dropout,
        kernLength=args.kernLength,
        F1=args.F1,
        D=args.D,
    )

    student.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    best_path = out_dir / "student_best.keras"
    final_path = out_dir / "student_final.keras"

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

    history = student.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    student.save(final_path)

    # Predicts and Metrics with inference time measurement
    start_time = time.time()
    probs = student.predict(X_val, batch_size=args.batch_size, verbose=0)
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

    # Save metrics + history
    with open(out_dir / "history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

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

    # Additional plots: learning curves + ROC curve
    try:
        plot_learning_curves({k: v for k, v in history.history.items()}, out_dir / "learning_curves.png")
    except Exception:
        pass

    try:
        plot_roc_curve(y_val, probs, class_names, out_dir / "roc_curve.png")
    except Exception as e:
        print(f"[WARN] Failed to plot ROC curve: {e}")

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report_with_time)

    np.save(out_dir / "confusion_matrix.npy", cm)


    # Visualizations
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    plot_per_class_bars(cm, class_names, out_dir / "per_class_accuracy.png")

    print(f"\n[DONE] Saved student model + eval artifacts to: {out_dir}")
    print(f"[DONE] Best:  {best_path}")
    print(f"[DONE] Final: {final_path}")

#main for running the eegnet resnet model
def main_resnet():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="outputs/filtered_resnet/epochs_resnet_subjectsplit_masked.npz",
                        help="Masked dataset npz: X_train,y_train,X_val,y_val")
    parser.add_argument("--out", type=str, default="outputs/student_Resnet_eval",
                        help="Output folder for resnet model + plots")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Student capacity knobs (smaller than teacher)
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--kernLength", type=int, default=32)

    args = parser.parse_args()

    set_tf_runtime()
    root = add_arl_eegmodels_to_path()

    # Import EEGModels
    from EEGModels import EEGNet

    data_path = (root / args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Masked dataset not found: {data_path}")

    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

   
    # Load filtered / masked dataset

    d = np.load(data_path, allow_pickle=True)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(int)
    X_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(int)

    # Labels safety (won't change anything if already 0/1)
    y_train, train_map = ensure_zero_based_labels(y_train)
    y_val, _ = ensure_zero_based_labels(y_val)

    # Shape checks
    assert_eegnet_shape(X_train)
    assert_eegnet_shape(X_val, chans=X_train.shape[1], samples=X_train.shape[2])

    # NaN/Inf checks
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN/Inf")
    if not np.isfinite(X_val).all():
        raise ValueError("X_val contains NaN/Inf")

    Chans = int(X_train.shape[1])
    Samples = int(X_train.shape[2])
    nb_classes = int(np.max(y_train)) + 1

    if args.kernLength > Samples:
        print(f"[WARN] kernLength ({args.kernLength}) > Samples ({Samples}). Clamping to {Samples}.")
        args.kernLength = Samples

    print(f"[INFO] X_train: {X_train.shape} y_train: {y_train.shape} classes={nb_classes}")
    print(f"[INFO] X_val:   {X_val.shape} y_val:   {y_val.shape}")
    print("[INFO] Train class counts:", Counter(y_train))
    print("[INFO] Val class counts:  ", Counter(y_val))
    print("[INFO] Label map (train):", train_map)

    # Class names for plots
    class_names = [str(i) for i in range(nb_classes)]


    # Build + Train student EEGNet

    student = EEGNet(
        nb_classes=nb_classes,
        Chans=Chans,
        Samples=Samples,
        dropoutRate=args.dropout,
        kernLength=args.kernLength,
        F1=args.F1,
        D=args.D,
    )

    student.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    best_path = out_dir / "student_best.keras"
    final_path = out_dir / "student_final.keras"

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
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_resnet_log.csv"), append=False),
    ]

    history = student.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    student.save(final_path)

    # Predicts and Metrics with inference time measurement
    start_time = time.time()
    probs = student.predict(X_val, batch_size=args.batch_size, verbose=0)
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

    # Save metrics + history
    with open(out_dir / "history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

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

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report_with_time)

    np.save(out_dir / "confusion_matrix.npy", cm)


    # Visualizations
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    subjectwise_confusion_matrix(
        y_true=y_val,
        y_pred=y_pred,
        subject_ids=d["subject_ids_val"],
        out_dir=out_dir,
        class_names=class_names,
        prefix="subjectwise"
    )

    plot_per_class_bars(cm, class_names, out_dir / "per_class_accuracy.png")

    # Additional plots: learning curves + ROC curve
    try:
        plot_learning_curves({k: v for k, v in history.history.items()}, out_dir / "learning_curves.png")
    except Exception:
        pass

    try:
        plot_roc_curve(y_val, probs, class_names, out_dir / "roc_curve.png")
    except Exception as e:
        print(f"[WARN] Failed to plot ROC curve: {e}")

    print(f"\n[DONE] Saved student model + eval artifacts to: {out_dir}")
    print(f"[DONE] Best:  {best_path}")
    print(f"[DONE] Final: {final_path}")

#main for running the eegnet Transformer model
def main_Transformer():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="outputs/filtered_transformer/epochs_transformer_subjectsplit_masked.npz",
                        help="Masked dataset npz: X_train,y_train,X_val,y_val")
    parser.add_argument("--out", type=str, default="outputs/student_Transformer_eval",
                        help="Output folder for Transformer model + plots")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Student capacity knobs (smaller than teacher)
    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--kernLength", type=int, default=32)

    args = parser.parse_args()

    set_tf_runtime()
    root = add_arl_eegmodels_to_path()

    # Import EEGModels
    from EEGModels import EEGNet

    data_path = (root / args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Masked dataset not found: {data_path}")

    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

   
    # Load filtered / masked dataset

    d = np.load(data_path, allow_pickle=True)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(int)
    X_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(int)

    # Labels safety (won't change anything if already 0/1)
    y_train, train_map = ensure_zero_based_labels(y_train)
    y_val, _ = ensure_zero_based_labels(y_val)

    # Shape checks
    assert_eegnet_shape(X_train)
    assert_eegnet_shape(X_val, chans=X_train.shape[1], samples=X_train.shape[2])

    # NaN/Inf checks
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN/Inf")
    if not np.isfinite(X_val).all():
        raise ValueError("X_val contains NaN/Inf")

    Chans = int(X_train.shape[1])
    Samples = int(X_train.shape[2])
    nb_classes = int(np.max(y_train)) + 1

    if args.kernLength > Samples:
        print(f"[WARN] kernLength ({args.kernLength}) > Samples ({Samples}). Clamping to {Samples}.")
        args.kernLength = Samples

    print(f"[INFO] X_train: {X_train.shape} y_train: {y_train.shape} classes={nb_classes}")
    print(f"[INFO] X_val:   {X_val.shape} y_val:   {y_val.shape}")
    print("[INFO] Train class counts:", Counter(y_train))
    print("[INFO] Val class counts:  ", Counter(y_val))
    print("[INFO] Label map (train):", train_map)

    # Class names for plots
    class_names = [str(i) for i in range(nb_classes)]


    # Build + Train student EEGNet

    student = EEGNet(
        nb_classes=nb_classes,
        Chans=Chans,
        Samples=Samples,
        dropoutRate=args.dropout,
        kernLength=args.kernLength,
        F1=args.F1,
        D=args.D,
    )

    student.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    best_path = out_dir / "student_best.keras"
    final_path = out_dir / "student_final.keras"

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
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_resnet_log.csv"), append=False),
    ]

    history = student.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    student.save(final_path)

    # Predicts and Metrics with inference time measurement
    start_time = time.time()
    probs = student.predict(X_val, batch_size=args.batch_size, verbose=0)
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

    # Save metrics + history
    with open(out_dir / "history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

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

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report_with_time)

    np.save(out_dir / "confusion_matrix.npy", cm)


    # Visualizations
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    subjectwise_confusion_matrix(
        y_true=y_val,
        y_pred=y_pred,
        subject_ids=d["subject_ids_val"],
        out_dir=out_dir,
        class_names=class_names,
        prefix="subjectwise"
    )

    plot_per_class_bars(cm, class_names, out_dir / "per_class_accuracy.png")

    # Additional plots: learning curves + ROC curve
    try:
        plot_learning_curves({k: v for k, v in history.history.items()}, out_dir / "learning_curves.png")
    except Exception:
        pass

    try:
        plot_roc_curve(y_val, probs, class_names, out_dir / "roc_curve.png")
    except Exception as e:
        print(f"[WARN] Failed to plot ROC curve: {e}")


    print(f"\n[DONE] Saved student model + eval artifacts to: {out_dir}")
    print(f"[DONE] Best:  {best_path}")
    print(f"[DONE] Final: {final_path}")

if __name__ == "__main__":
    # Allow selecting which student eval to run when called from the orchestrator.
    import argparse

    top = argparse.ArgumentParser(add_help=False)
    top.add_argument('--which', choices=['eegnet', 'resnet', 'transformer', 'all'], default='all',
                     help='Which student eval to run')
    ns, remaining = top.parse_known_args()

    import sys
    sys_argv_backup = sys.argv
    sys.argv = [sys.argv[0]] + remaining

    try:
        if ns.which in ('transformer', 'all'):
            main_Transformer()
        if ns.which in ('resnet', 'all'):
            main_resnet()
        if ns.which in ('eegnet', 'all'):
            main()
    finally:
        sys.argv = sys_argv_backup
