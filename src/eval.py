from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt



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



# Main

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

    # Predicts and Metrics

    probs = student.predict(X_val, batch_size=args.batch_size, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(nb_classes)))
    report = classification_report(y_val, y_pred, target_names=class_names, digits=4)

    print("\n[VAL] accuracy:", acc)
    print("\n[VAL] classification report:\n", report)

    # Save metrics + history
    with open(out_dir / "history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "val_accuracy": float(acc),
            "Chans": Chans,
            "Samples": Samples,
            "nb_classes": nb_classes,
            "kernLength": args.kernLength,
            "F1": args.F1,
            "D": args.D,
            "dropout": args.dropout,
        }, f, indent=2)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    np.save(out_dir / "confusion_matrix.npy", cm)


    # Visualizations

    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    plot_per_class_bars(cm, class_names, out_dir / "per_class_accuracy.png")

    print(f"\n[DONE] Saved student model + eval artifacts to: {out_dir}")
    print(f"[DONE] Best:  {best_path}")
    print(f"[DONE] Final: {final_path}")


if __name__ == "__main__":
    main()
