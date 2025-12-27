# Time series transformer teacher model
from __future__ import annotations

import os, json, argparse
from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model





def set_tf_runtime():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def infer_nb_classes(y: np.ndarray) -> int:
    y = np.asarray(y).reshape(-1)
    return int(np.max(y)) + 1


def to_transformer_shape(X_eegnet: np.ndarray) -> np.ndarray:
    """
    Convert (N,Chans,Samples,1) -> (N,Samples,Chans)
    """
    if X_eegnet.ndim != 4 or X_eegnet.shape[-1] != 1:
        raise ValueError(f"Expected (N,Chans,Samples,1), got {X_eegnet.shape}")
    X = np.squeeze(X_eegnet, axis=-1)        # (N,Chans,Samples)
    X = np.transpose(X, (0, 2, 1))           # (N,Samples,Chans)
    return X.astype(np.float32)


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, mlp_dim: int, dropout: float, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.drop1 = layers.Dropout(dropout)

        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        h = self.ln1(x)
        h = self.mha(h, h, training=training)
        h = self.drop1(h, training=training)
        x = x + h

        h = self.ln2(x)
        h = self.mlp(h, training=training)
        h = self.drop2(h, training=training)
        return x + h

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        })
        return config


def build_time_series_transformer_teacher(
    seq_len: int,
    feature_dim: int,
    nb_classes: int,
    d_model: int = 64,
    num_heads: int = 4,
    depth: int = 4,
    mlp_dim: int = 128,
    dropout: float = 0.2,
    pooling: str = "mean",  # "mean" or "cls"
):
    """
    Input: (batch, seq_len, feature_dim)
    Output: logits (batch, nb_classes)
    """
    inp = layers.Input(shape=(seq_len, feature_dim), name="ts_input")

    # Project feature_dim -> d_model
    x = layers.Dense(d_model, name="input_projection")(inp)

    # Positional embedding (learnable)
    pos = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=d_model, name="pos_embedding")(pos)
    x = x + pos_emb

    x = layers.Dropout(dropout, name="emb_dropout")(x)

    # Encoder stack
    for i in range(depth):
        x = TransformerEncoderBlock(d_model, num_heads, mlp_dim, dropout, name=f"enc_{i}")(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

    if pooling == "mean":
        x = layers.GlobalAveragePooling1D(name="gap1d")(x)
    elif pooling == "cls":
        # Add a CLS token for pooling (optional)
        # Simple approach: take first token (not a true CLS token unless you add one)
        x = x[:, 0, :]
    else:
        raise ValueError("pooling must be 'mean' or 'cls'")

    logits = layers.Dense(nb_classes, activation=None, name="logits")(x)
    return Model(inp, logits, name="TimeSeriesTransformerTeacher")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/epochs_subjectsplit.npz")
    parser.add_argument("--out", type=str, default="outputs/teacher_transformer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)

    # Transformer knobs
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])

    args = parser.parse_args()

    set_tf_runtime()
    ROOT = Path(__file__).resolve().parents[1]

    data_path = (ROOT / args.data).resolve()
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(data_path, allow_pickle=True)
    X_train_eegnet = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(int)
    X_val_eegnet = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(int)

    # Convert to Transformer shape: (N, seq_len, feature_dim)
    X_train = to_transformer_shape(X_train_eegnet)  # (N, 60, 15)
    X_val = to_transformer_shape(X_val_eegnet)

    seq_len = X_train.shape[1]       # 60
    feature_dim = X_train.shape[2]   # 15
    nb_classes = infer_nb_classes(y_train)

    print("[INFO] X_train:", X_train.shape, "y_train:", y_train.shape, "classes:", nb_classes)
    print("[INFO] X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
    print("[INFO] Train counts:", Counter(y_train))
    print("[INFO] Val counts:  ", Counter(y_val))

    teacher = build_time_series_transformer_teacher(
        seq_len=seq_len,
        feature_dim=feature_dim,
        nb_classes=nb_classes,
        d_model=args.d_model,
        num_heads=args.num_heads,
        depth=args.depth,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        pooling=args.pooling,
    )

    teacher.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    best_path = out_dir / "teacher_best.keras"
    final_path = out_dir / "teacher_final.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy", save_best_only=True, mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"), append=False),
    ]

    history = teacher.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    teacher.save(final_path)

    with open(out_dir / "history.json", "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    print("[DONE] Saved best:", best_path)
    print("[DONE] Saved final:", final_path)
    print("[DONE] Outputs:", out_dir)


if __name__ == "__main__":
    main()