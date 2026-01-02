from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np 
import tensorflow as tf
from collections import Counter

# its not packaged as python mdule by default
import os, sys

ROOT = Path(__file__).resolve().parents[2]
EEGMODELS_DIR = ROOT / "arl-eegmodels"
# Prefer adding the arl-eegmodels dir to sys.path so `from EEGModels import EEGNet` works
sys.path.insert(0, str(EEGMODELS_DIR))

try:
    from EEGModels import EEGNet
except Exception:
    # Robust fallback: load the EEGModels.py file directly by path
    import importlib.util

    eegmodels_path = EEGMODELS_DIR / "EEGModels.py"
    if not eegmodels_path.exists():
        raise FileNotFoundError(f"Could not find EEGModels.py at {eegmodels_path}")

    spec = importlib.util.spec_from_file_location("EEGModels", str(eegmodels_path))
    EEGModels = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(EEGModels)
    EEGNet = EEGModels.EEGNet

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))



# ----------------------------
# Training the model: Now we can actually train the teacher model
# ----------------------------

def set_tf_runtime():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # GPU memory growth (safe even if you don't have a GPU)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/epochs_subjectsplit.npz",
                        help="Path to .npz with X_train/y_train/X_val/y_val")
    parser.add_argument("--out", type=str, default="outputs/eegnet_teacher",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Teacher capacity knobs
    parser.add_argument("--F1", type=int, default=16)
    parser.add_argument("--D", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--kernLength", type=int, default=32) # with a sample of 60 its rec that klen is 16 or 32: due to potential areas with the shape

    args = parser.parse_args()

    set_tf_runtime()

    data_path = (ROOT / args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load dataset
    # ----------------------------
    d = np.load(data_path, allow_pickle=True)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(int)
    X_val   = d["X_val"].astype(np.float32)
    y_val   = d["y_val"].astype(int)

    # We have an earlier condition method that checks the shape of the output from the preprocessing step. so we can skip this step // Redundant code
    # Shape sanity
    if X_train.ndim != 4 or X_train.shape[-1] != 1:
        raise ValueError(f"Expected X_train (N,Chans,Samples,1), got {X_train.shape}")
    if X_val.ndim != 4 or X_val.shape[-1] != 1:
        raise ValueError(f"Expected X_val (N,Chans,Samples,1), got {X_val.shape}")

    Chans = int(X_train.shape[1])
    Samples = int(X_train.shape[2])
    nb_classes = infer_nb_classes(y_train)

    # val sanity: all val classes exist in train, not completely necessay 
    train_classes = set(np.unique(y_train))
    val_classes = set(np.unique(y_val))
    if not val_classes.issubset(train_classes):
        raise ValueError(f"Val has unseen classes: {val_classes - train_classes}")

    print(f"[INFO] X_train: {X_train.shape}  y_train: {y_train.shape}  classes={nb_classes}")
    print(f"[INFO] X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print("[INFO] Train class counts:", Counter(y_train))
    print("[INFO] Val class counts:  ", Counter(y_val))

    # ----------------------------
    # Build teacher EEGNet
    # ----------------------------
    teacher = EEGNet(
        nb_classes=nb_classes,
        Chans=Chans,
        Samples=Samples,
        dropoutRate=args.dropout,
        kernLength=args.kernLength,
        F1=args.F1,
        D=args.D,
    )

    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # ----------------------------
    # Callbacks
    # ----------------------------
    best_path = out_dir / "teacher_best.keras"
    final_path = out_dir / "teacher_final.keras"

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
    # Train
    # ----------------------------
    history = teacher.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # Save final model (best weights already restored by EarlyStopping)
    teacher.save(final_path)

    # Save history + config
    with open(out_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(out_dir / "history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    meta = {
        "Chans": Chans,
        "Samples": Samples,
        "nb_classes": nb_classes,
        "X_train_shape": list(X_train.shape),
        "X_val_shape": list(X_val.shape),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Saved best:  {best_path}")
    print(f"[DONE] Saved final: {final_path}")
    print(f"[DONE] Logs in:     {out_dir}")


if __name__ == "__main__":
    main()
