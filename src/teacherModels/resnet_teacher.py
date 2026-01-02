
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from pathlib import Path
import argparse
import os, sys

ROOT = Path(__file__).resolve().parents[1]

def set_tf_runtime():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/epochs_subjectsplit.npz",
                        help="Path to .npz with X_train/y_train/X_val/y_val")
parser.add_argument("--out", type=str, default="outputs/resnet_teacher",
                        help="Output directory")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)

args = parser.parse_args()

data_path = (ROOT / args.data).resolve()
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

d = np.load(data_path, allow_pickle=True)
X_train = d["X_train"].astype(np.float32)
y_train = d["y_train"].astype(int)
X_val   = d["X_val"].astype(np.float32)
y_val   = d["y_val"].astype(int)

if X_train.ndim != 4 or X_train.shape[-1] != 1:
    raise ValueError(f"Expected X_train (N,Chans,Samples,1), got {X_train.shape}")
if X_val.ndim != 4 or X_val.shape[-1] != 1:
    raise ValueError(f"Expected X_val (N,Chans,Samples,1), got {X_val.shape}")

def infer_nb_classes(y: np.ndarray) -> int:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"Expected y to be 1D int labels, got shape {y.shape}")
    return int(np.max(y)) + 1

Chans = int(X_train.shape[1])
Samples = int(X_train.shape[2])
nb_classes = infer_nb_classes(y_train)


def residual_block(x, filters, stride=(1, 1), name="res"):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    # Project shortcut if shape changed
    if shortcut.shape[-1] != filters or stride != (1, 1):
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False, name=f"{name}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_out")(x)
    return x

def build_resnet_teacher(input_shape=(15, 60, 1), nb_classes=2):
    inp = layers.Input(shape=input_shape, name="eeg_input")

    # Small stem (avoid aggressive downsampling)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False, name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Residual stages
    x = residual_block(x, 32, stride=(1, 1), name="s1_b1")
    x = residual_block(x, 32, stride=(1, 1), name="s1_b2")

    # Downsample ONLY along the 60-axis (time/features axis), keep height=15 intact
    x = residual_block(x, 64, stride=(1, 2), name="s2_b1")
    x = residual_block(x, 64, stride=(1, 1), name="s2_b2")

    x = residual_block(x, 128, stride=(1, 2), name="s3_b1")
    x = residual_block(x, 128, stride=(1, 1), name="s3_b2")

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # LOGITS (no softmax here)
    logits = layers.Dense(nb_classes, activation=None, name="logits")(x)

    return Model(inputs=inp, outputs=logits, name="ResNetTeacherEEG")

teacher = build_resnet_teacher(input_shape=(Chans, Samples, 1), nb_classes=nb_classes)
teacher.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

out_dir = (ROOT / args.out).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
best_path = out_dir / "ResNet_best.keras"
final_path = out_dir / "ResNet_final.keras"


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
        tf.keras.callbacks.CSVLogger(str(out_dir / "Resnet_train_log.csv"), append=False),
    ]

history = teacher.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

teacher.save(final_path)
print(f"Model saved to: {final_path}")

