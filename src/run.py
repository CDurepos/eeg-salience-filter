#!/usr/bin/env python3
"""  This is just a Simple orchestrator to run preprocessing, splitting, training, evaluation and prediction.

Usage examples:
  # Preprocess raw CSV -> feature files (saved to data/)
  python src/run.py preprocess --data-path data/adhdata.csv --out-dir data

  # Create subject split .npz (uses files in data-dir, saved to data/)
  python src/run.py split --data-dir data --out data/epochs_subjectsplit.npz

  # Train a teacher model script (path relative to repo root)
  python src/run.py train --model-script teacherModels/eeg_teacher.py --out outputs/eegnet_run

  # Run evaluation
  python src/run.py evaluate --data outputs/filtered/epochs_subjectsplit_masked.npz --out outputs/student_eval

  # Predict with a saved model on numpy data
  python src/run.py predict --model outputs/student_eval/student_best.keras --input data/X_sample.npy

  # Run full pipeline (data files -> data/, model outputs -> outputs/run)
  python src/run.py all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_preprocess(data_path: str, out_dir: str, args_extra: Optional[list] = None):
    script = REPO_ROOT / "src" / "preprocess.py"
    # Ensure output directory exists so subprocess does not fail on write
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    cmd = [sys.executable, str(script)]
    # Only pass --data-path if explicitly provided; otherwise let preprocess.py resolve defaults
    if data_path:
        cmd += ["--data-path", data_path]
    cmd += ["--out-dir", out_dir]
    if args_extra:
        cmd += args_extra
    subprocess.check_call(cmd)


def run_split(data_dir: str, out_path: str, test_size: float = 0.2, seed: int = 42):
    # call the function directly from preprocess module
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from preprocess import subject_split_and_save

    # Ensure output directory exists
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)

    out = subject_split_and_save(data_dir=data_dir, out_path=out_path, test_size=test_size, random_state=seed)
    print(f"Saved subject split to: {out}")
    return out


def run_train(model_script: str, extra_args: Optional[list] = None):
    script = REPO_ROOT / "src" / model_script
    if not script.exists():
        # try relative to repo root
        script = REPO_ROOT / model_script
    if not script.exists():
        raise FileNotFoundError(f"Model script not found: {model_script}")

    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd += extra_args
    subprocess.check_call(cmd)


def run_eval(data: str, out: str, extra_args: Optional[list] = None):
    script = REPO_ROOT / "src" / "eval.py"
    # Ensure output directory exists
    Path(out).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), "--data", data, "--out", out]
    if extra_args:
        cmd += extra_args
    subprocess.check_call(cmd)


def predict(model_path: str, input_path: str, out_path: Optional[str] = None, topk: int = 3):
    model = tf.keras.models.load_model(str(model_path))

    # load input
    ip = Path(input_path)
    if ip.suffix == ".npz":
        d = np.load(str(ip), allow_pickle=True)
        # prefer common keys
        if 'X' in d:
            X = d['X']
        elif 'X_val' in d:
            X = d['X_val']
        else:
            # take first array
            X = d[list(d.files)[0]]
    else:
        X = np.load(str(ip))

    X = np.asarray(X).astype(np.float32)

    # If single sample (1D or 2D), expand batch dim
    if X.ndim == 1:
        X = X[np.newaxis, ...]

    probs = model.predict(X)
    preds = np.argsort(probs, axis=1)[:, ::-1][:, :topk]

    results = []
    for i in range(len(probs)):
        row = {
            'preds': preds[i].tolist(),
            'probs': probs[i, preds[i]].tolist()
        }
        results.append(row)

    if out_path:
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved predictions to {out_path}")
    else:
        print(json.dumps(results, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')

    sp = sub.add_parser('preprocess')
    sp.add_argument('--data-path', required=True)
    sp.add_argument('--out-dir', required=True)

    ss = sub.add_parser('split')
    ss.add_argument('--data-dir', required=True)
    ss.add_argument('--out', required=True)
    ss.add_argument('--test-size', type=float, default=0.2)
    ss.add_argument('--seed', type=int, default=42)

    st = sub.add_parser('train')
    st.add_argument('--model-script', required=True, help='Path under src/ to model script, e.g. teacherModels/eeg_teacher.py')
    st.add_argument('--extra', nargs=argparse.REMAINDER)

    se = sub.add_parser('evaluate')
    se.add_argument('--data', required=True)
    se.add_argument('--out', required=True)
    se.add_argument('--extra', nargs=argparse.REMAINDER)

    spred = sub.add_parser('predict')
    spred.add_argument('--model', required=True)
    spred.add_argument('--input', required=True)
    spred.add_argument('--out', required=False)
    spred.add_argument('--topk', type=int, default=3)

    sall = sub.add_parser('all')
    sall.add_argument('--data-path', required=False, default=None,
                      help='Path to raw CSV (default: data/adhdata.csv if present)')
    sall.add_argument('--data-dir', required=False, default=None,
                      help='Directory for data files (preprocessing outputs, splits) (default: data)')
    sall.add_argument('--out-dir', required=False, default=None,
                      help='Output directory root for model outputs (default: outputs/run)')
    sall.add_argument('--model-script', required=False, default=None,
                      help='Model script under src/ to run (default: teacherModels/eeg_teacher.py)')

    args = p.parse_args()

    if args.cmd == 'preprocess':
        run_preprocess(args.data_path, args.out_dir)
    elif args.cmd == 'split':
        run_split(args.data_dir, args.out, test_size=args.test_size, seed=args.seed)
    elif args.cmd == 'train':
        run_train(args.model_script, extra_args=args.extra)
    elif args.cmd == 'evaluate':
        run_eval(args.data, args.out, extra_args=args.extra)
    elif args.cmd == 'predict':
        predict(args.model, args.input, out_path=args.out, topk=args.topk)
    elif args.cmd == 'all':
        # Provide sensible defaults so `python src/run.py all` works end-to-end
        # Do not force a concrete CSV path here; let src/preprocess.py resolve DATA_PATH/defaults
        data_path = args.data_path

        # Separate data directory for preprocessing outputs and splits
        data_dir = args.data_dir or str(REPO_ROOT / 'data')
        # Output directory for model outputs (trained models, evaluations, etc.)
        out_dir = args.out_dir or str(REPO_ROOT / 'outputs' / 'run')
        model_script = args.model_script or 'teacherModels/eeg_teacher.py'

        # preprocess -> split -> train -> eval
        # ensure directories exist
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        # Run preprocessing - outputs go to data_dir
        run_preprocess(data_path, data_dir)
        # Split file goes to data_dir
        split_npz = str(Path(data_dir) / 'epochs_subjectsplit.npz')
        run_split(data_dir, split_npz)

        # Train & evaluate all teacher pipelines: EEGNet, ResNet, Transformer
        teachers = [
            {
                'script': 'teacherModels/eeg_teacher.py',
                'name': 'eegnet',
                'best_name': 'teacher_best.keras',
                'filtered_dir': 'filtered',
                'masked_npz': 'epochs_subjectsplit_masked.npz',
            },
            {
                'script': 'teacherModels/resnet_teacher.py',
                'name': 'resnet',
                'best_name': 'ResNet_best.keras',
                'filtered_dir': 'filtered_resnet',
                'masked_npz': 'epochs_resnet_subjectsplit_masked.npz',
            },
            {
                'script': 'teacherModels/tst_teacher.py',
                'name': 'transformer',
                'best_name': 'teacher_best.keras',
                'filtered_dir': 'filtered_transformer',
                'masked_npz': 'epochs_transformer_subjectsplit_masked.npz',
            },
        ]

        for t in teachers:
            model_out = Path(out_dir) / f"run_model_{t['name']}"
            model_out.mkdir(parents=True, exist_ok=True)

            # Train teacher
            run_train(t['script'], extra_args=["--data", split_npz, "--out", str(model_out)])

            # Locate teacher best checkpoint
            teacher_best = model_out / t['best_name']

            # Run filter for this teacher to produce masked dataset
            filter_script = REPO_ROOT / 'src' / 'filter.py'
            filtered_out = model_out / t['filtered_dir']
            filtered_out.mkdir(parents=True, exist_ok=True)

            filter_cmd = [sys.executable, str(filter_script),
                          '--which', t['name'],
                          '--data', split_npz,
                          '--teacher', str(teacher_best),
                          '--out', str(filtered_out)]
            subprocess.check_call(filter_cmd)

            # Run evaluation on the masked dataset created by filter
            eval_data = str(filtered_out / t['masked_npz'])
            eval_out = str(model_out / 'eval')
            Path(eval_out).mkdir(parents=True, exist_ok=True)
            run_eval(eval_data, eval_out, extra_args=["--which", t['name']])
    else:
        p.print_help()


if __name__ == '__main__':
    main()
