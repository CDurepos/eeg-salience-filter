from __future__ import annotations

import os, sys, argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from teacherModels.tst_teacher import TransformerEncoderBlock

# This script will filter all the models 

def add_arl_eegmodels_to_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    eegmodels_dir = root / "external" / "arl-eegmodels"
    if eegmodels_dir.exists():
        sys.path.insert(0, str(eegmodels_dir))
    return root



def set_tf_runtime():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# Sanity check for input shape
def assert_eegnet_shape(X: np.ndarray, chans=15, samples=60):
    if X.ndim != 4 or X.shape[1:] != (chans, samples, 1):
        raise ValueError(f"Expected X shape (N,{chans},{samples},1), got {X.shape}")


def build_logits_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    extracts the pre-softmax logits from a model.
    Prefer pre-softmax logits rather than softmax probabilities.
    EEGNet in arl-eegmodels usually has a Dense layer named 'dense' before softmax,
    but we add a fallback if the name differs.
    """
    # Try the common case first
    try:
        dense = model.get_layer("dense")
        return tf.keras.Model(inputs=model.input, outputs=dense.output)
    except Exception:
        pass

    # Fallback: find last Dense layer before softmax
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    if not dense_layers:
        raise ValueError("Could not find a Dense layer to use as logits output.")
    return tf.keras.Model(inputs=model.input, outputs=dense_layers[-1].output)


def saliency_for_batch(
    x_batch: np.ndarray,
    logits_model: tf.keras.Model,
    y_batch: np.ndarray | None = None,
    mode: str = "pred",  # "pred" or "true"
) -> np.ndarray:
    """
    Returns |d logit_class / d input| for each sample in batch.
    Shape: (B, Chans, Samples, 1)
    """
    x = tf.convert_to_tensor(x_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = logits_model(x, training=False)  # disable dropout etc.

        if mode == "pred":
            cls = tf.argmax(logits, axis=1, output_type=tf.int32)
        elif mode == "true":
            if y_batch is None:
                raise ValueError("mode='true' requires y_batch.")
            cls = tf.convert_to_tensor(y_batch, dtype=tf.int32)
            cls = tf.reshape(cls, [-1])
        else:
            raise ValueError("mode must be 'pred' or 'true'.")

        idx = tf.stack([tf.range(tf.shape(logits)[0]), cls], axis=1)
        score = tf.gather_nd(logits, idx)  # (B,)

    grads = tape.gradient(score, x)            # (B,Chans,Samples,1)
    sal = tf.abs(grads)

    # Normalize per-trial to [0,1] to make thresholding more stable
    reduce_axes = list(range(1, len(sal.shape)))
    sal = sal / (tf.reduce_max(sal, axis=reduce_axes, keepdims=True) + 1e-8)
    return sal.numpy().astype(np.float32)


def make_mask_from_saliency(
    saliency: np.ndarray,
    keep_ratio: float = 0.70,
    per_trial: bool = False
) -> np.ndarray:
    """
    Convert saliency -> binary mask.

    keep_ratio=0.70 means keep top 70% salient elements.

    per_trial=False: one global threshold across all samples (stable overall sparsity)
    per_trial=True: threshold separately per trial (each trial keeps the same %)
    """
    if not (0.0 < keep_ratio < 1.0):
        raise ValueError("keep_ratio must be between 0 and 1")

    if per_trial:
        # threshold per sample
        flat = saliency.reshape(saliency.shape[0], -1)
        thr = np.quantile(flat, 1.0 - keep_ratio, axis=1)  # (N,)
        thr = thr.reshape(-1, 1, 1, 1)
    else:
        thr = np.quantile(saliency, 1.0 - keep_ratio)

    mask = (saliency >= thr).astype(np.float32)
    return mask

def ts_to_eeg(X_ts: np.ndarray) -> np.ndarray:
    """(N, Samples, Chans) -> (N, Chans, Samples, 1)"""
    if X_ts.ndim != 3:
        raise ValueError(f"Expected TS shape (N,Samples,Chans), got {X_ts.shape}")
    X = np.transpose(X_ts, (0, 2, 1))       # (N, Chans, Samples)
    return X[..., None]  

def eeg_to_ts(X_eeg: np.ndarray) -> np.ndarray:
    """(N, Chans, Samples, 1) -> (N, Samples, Chans)"""
    if X_eeg.ndim != 4 or X_eeg.shape[-1] != 1:
        raise ValueError(f"Expected EEG shape (N,Chans,Samples,1), got {X_eeg.shape}")
    X = np.squeeze(X_eeg, axis=-1)          # (N, Chans, Samples)
    return np.transpose(X, (0, 2, 1)) 

def process_split(
    X: np.ndarray,
    y: np.ndarray,
    logits_model: tf.keras.Model,
    batch_size: int,
    mode: str,
    keep_ratio: float,
    per_trial: bool,
    save_saliency: bool,
    split_name: str,
    out_dir: Path
):
    N = X.shape[0]
    all_sal = [] if save_saliency else None

    # Compute saliency in batches
    sal_chunks = []
    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size] if mode == "true" else None
        sal_b = saliency_for_batch(xb, logits_model, yb, mode=mode)
        sal_chunks.append(sal_b)

    sal = np.concatenate(sal_chunks, axis=0)  # (N,Chans,Samples,1)

    mask = make_mask_from_saliency(sal, keep_ratio=keep_ratio, per_trial=per_trial)
    X_masked = X * mask

    # Save arrays for this split
    np.save(out_dir / f"{split_name}_mask.npy", mask)
    np.save(out_dir / f"{split_name}_X_masked.npy", X_masked)

    if save_saliency:
        np.save(out_dir / f"{split_name}_saliency.npy", sal)

    return X_masked, mask

# Main function to run the script for filtering the EEg student model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/epochs_subjectsplit.npz")
    parser.add_argument("--teacher", type=str, default="outputs/teacher_best.keras")
    parser.add_argument("--out", type=str, default="outputs/filtered")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--keep_ratio", type=float, default=0.70, help="Keep top fraction of salient values current default is 70%")
    parser.add_argument("--per_trial", action="store_true", help="Threshold per trial instead of global")

    parser.add_argument("--mode", type=str, default="pred", choices=["pred", "true"],
                        help="Use predicted class or true label when computing gradients")

    parser.add_argument("--save_saliency", action="store_true", help="Also save full saliency arrays (large files!)")

    # you can change these if your shapes differ
    parser.add_argument("--chans", type=int, default=15)
    parser.add_argument("--samples", type=int, default=60)

    args = parser.parse_args()

    set_tf_runtime()
    root = add_arl_eegmodels_to_path()

    

    data_path = (root / args.data).resolve()
    teacher_path = (root / args.teacher).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(data_path, allow_pickle=True)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(int)
    X_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(int)

    # Shape checks (your target: (N,15,60,1))
    assert_eegnet_shape(X_train, chans=args.chans, samples=args.samples)
    assert_eegnet_shape(X_val, chans=args.chans, samples=args.samples)

    # Load teacher + build logits model
    teacher = tf.keras.models.load_model(teacher_path)
    logits_model = build_logits_model(teacher)

    # Build filtered splits
    X_train_masked, train_mask = process_split(
        X_train, y_train, logits_model,
        batch_size=args.batch_size,
        mode=args.mode,
        keep_ratio=args.keep_ratio,
        per_trial=args.per_trial,
        save_saliency=args.save_saliency,
        split_name="train",
        out_dir=out_dir
    )

    X_val_masked, val_mask = process_split(
        X_val, y_val, logits_model,
        batch_size=args.batch_size,
        mode=args.mode,
        keep_ratio=args.keep_ratio,
        per_trial=args.per_trial,
        save_saliency=args.save_saliency, 
        split_name="val",
        out_dir=out_dir
    )

    subject_ids_train = d["subject_ids_train"]
    subject_ids_val   = d["subject_ids_val"]

    # Save an NPZ ready for student training
    out_npz = out_dir / "epochs_subjectsplit_masked.npz"
    np.savez(
        out_npz,
        X_train=X_train_masked, y_train=y_train, subject_ids_train=subject_ids_train,
        X_val=X_val_masked, y_val=y_val, subject_ids_val=subject_ids_val,
        keep_ratio=args.keep_ratio,
        per_trial=args.per_trial,
        mode=args.mode,
    )

    print("[DONE] Saved masked dataset:", out_npz)
    print("[INFO] Example masked shape:", X_train_masked.shape)
    print("[INFO] Mask sparsity (train):", float(1.0 - train_mask.mean()))
    print("[INFO] Mask sparsity (val):  ", float(1.0 - val_mask.mean()))




# Main function for running the Resnet model
def main_resnet():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/epochs_subjectsplit.npz")
    parser.add_argument("--teacher", type=str, default="outputs/ResNet_best.keras")
    parser.add_argument("--out", type=str, default="outputs/filtered_resnet")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--keep_ratio", type=float, default=0.70, help="Keep top fraction of salient values, default is 70%")
    parser.add_argument("--per_trial", action="store_true", help="Threshold per trial instead of global")

    parser.add_argument("--mode", type=str, default="pred", choices=["pred", "true"],
                        help="Use predicted class or true label when computing gradients")

    parser.add_argument("--save_saliency", action="store_true", help="Also save full saliency arrays (large files!)")

    # you can change these if your shapes differ
    parser.add_argument("--chans", type=int, default=15)
    parser.add_argument("--samples", type=int, default=60)

    args = parser.parse_args()

    set_tf_runtime()
    root = add_arl_eegmodels_to_path()

    

    data_path = (root / args.data).resolve()
    teacher_path = (root / args.teacher).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)


    d = np.load(data_path, allow_pickle=True)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(int)
    X_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(int)

    # Shape checks (your target: (N,15,60,1))
    assert_eegnet_shape(X_train, chans=args.chans, samples=args.samples)
    assert_eegnet_shape(X_val, chans=args.chans, samples=args.samples)

    # Load teacher + build logits model
    teacher = tf.keras.models.load_model(teacher_path)
    logits_model = build_logits_model(teacher)

    # Build filtered splits
    X_train_masked, train_mask = process_split(
        X_train, y_train, logits_model,
        batch_size=args.batch_size,
        mode=args.mode,
        keep_ratio=args.keep_ratio,
        per_trial=args.per_trial,
        save_saliency=args.save_saliency,
        split_name="train",
        out_dir=out_dir
    )

    X_val_masked, val_mask = process_split(
        X_val, y_val, logits_model,
        batch_size=args.batch_size,
        mode=args.mode,
        keep_ratio=args.keep_ratio,
        per_trial=args.per_trial,
        save_saliency=args.save_saliency,
        split_name="val",
        out_dir=out_dir
    )
    # subject ids aligned with the rows of X_train / X_val
    subject_ids_train = d["subject_ids_train"]
    subject_ids_val   = d["subject_ids_val"]

    # Save an NPZ ready for student training
    out_npz = out_dir / "epochs_resnet_subjectsplit_masked.npz"
    np.savez(
        out_npz,
        X_train=X_train_masked, y_train=y_train, subject_ids_train=subject_ids_train,
        X_val=X_val_masked, y_val=y_val, subject_ids_val=subject_ids_val,
        keep_ratio=args.keep_ratio,
        per_trial=args.per_trial,
        mode=args.mode,
    )

    print("[DONE] Saved masked dataset:", out_npz)
    print("[INFO] Example masked shape:", X_train_masked.shape)
    print("[INFO] Mask sparsity (train):", float(1.0 - train_mask.mean()))
    print("[INFO] Mask sparsity (val):  ", float(1.0 - val_mask.mean()))


def main_Transformer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/epochs_subjectsplit.npz")
    parser.add_argument("--teacher", type=str, default="outputs/teacher_transformer/teacher_best.keras")
    parser.add_argument("--out", type=str, default="outputs/filtered_transformer")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--keep_ratio", type=float, default=0.70, help="Keep top fraction of salient values, default is 70%")
    parser.add_argument("--per_trial", action="store_true", help="Threshold per trial instead of global")

    parser.add_argument("--mode", type=str, default="pred", choices=["pred", "true"],
                        help="Use predicted class or true label when computing gradients")

    parser.add_argument("--save_saliency", action="store_true", help="Also save full saliency arrays (large files!)")

    # you can change these if your shapes differ
    parser.add_argument("--chans", type=int, default=15)
    parser.add_argument("--samples", type=int, default=60)

    args = parser.parse_args()

    set_tf_runtime()
    root = add_arl_eegmodels_to_path()

    
    data_path = (root / args.data).resolve()
    teacher_path = (root / args.teacher).resolve()
    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)


    d = np.load(data_path, allow_pickle=True)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(int)
    X_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(int)


    # Load teacher + build logits model
    teacher = tf.keras.models.load_model(
    teacher_path,
    custom_objects={"TransformerEncoderBlock": TransformerEncoderBlock},
    compile=False
    )
    logits_model = build_logits_model(teacher)

    # Convert EEGNet input -> Transformer input
    X_train_ts = eeg_to_ts(X_train)   # (N, 60, 15)
    X_val_ts   = eeg_to_ts(X_val)

    # Build filtered splits
    X_train_ts_masked, train_mask = process_split(
        X_train_ts, y_train, logits_model,
        batch_size=args.batch_size,
        mode=args.mode,
        keep_ratio=args.keep_ratio,
        per_trial=args.per_trial,
        save_saliency=args.save_saliency,
        split_name="train",
        out_dir=out_dir
    )

    X_val_ts_masked, val_mask = process_split(
        X_val_ts, y_val, logits_model,
        batch_size=args.batch_size,
        mode=args.mode,
        keep_ratio=args.keep_ratio,
        per_trial=args.per_trial,
        save_saliency=args.save_saliency,
        split_name="val",
        out_dir=out_dir
    )

    # convert back to eeg shape
    X_train_masked = ts_to_eeg(X_train_ts_masked)  # (N, 15, 60, 1)
    X_val_masked   = ts_to_eeg(X_val_ts_masked)


    # subject ids aligned with the rows of X_train / X_val
    subject_ids_train = d["subject_ids_train"]
    subject_ids_val   = d["subject_ids_val"]

    # Save an NPZ ready for student training
    out_npz = out_dir / "epochs_transformer_subjectsplit_masked.npz"
    np.savez(
        out_npz,
        X_train=X_train_masked, y_train=y_train, subject_ids_train=subject_ids_train,
        X_val=X_val_masked, y_val=y_val, subject_ids_val=subject_ids_val,
        keep_ratio=args.keep_ratio,
        per_trial=args.per_trial,
        mode=args.mode,
    )

    print("[DONE] Saved masked dataset:", out_npz)
    print("[INFO] Example masked shape:", X_train_ts_masked.shape)
    print("[INFO] Mask sparsity (train):", float(1.0 - train_mask.mean()))
    print("[INFO] Mask sparsity (val):  ", float(1.0 - val_mask.mean()))
    print("X_train_masked:", X_train_masked.shape)  # should be (N, 15, 60, 1)
    print("X_val_masked:",   X_val_masked.shape)



if __name__ == "__main__":
    # Support running only a specific filter pipeline when invoked from an orchestrator.
    import argparse

    top = argparse.ArgumentParser(add_help=False)
    top.add_argument('--which', choices=['eegnet', 'resnet', 'transformer', 'all'], default='all',
                     help='Which teacher filter to run')
    ns, remaining = top.parse_known_args()

    # Preserve remaining args for the specific runner functions
    import sys
    sys_argv_backup = sys.argv
    sys.argv = [sys.argv[0]] + remaining

    try:
        if ns.which in ('resnet', 'all'):
            main_resnet()
        if ns.which in ('transformer', 'all'):
            main_Transformer()
        if ns.which in ('eegnet', 'all'):
            main()
    finally:
        sys.argv = sys_argv_backup

