import sys
print("interpreter path:", sys.executable)
print("Search Paths:", sys.path)




import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from dotenv import load_dotenv

import mne
from mne.preprocessing import ICA

from sklearn.preprocessing import StandardScaler

# Add tqwt_tools to path if not installed as package
_tqwt_tools_path = os.path.join(os.path.dirname(__file__), 'tqwt_tools')
if _tqwt_tools_path not in sys.path:
    sys.path.insert(0, _tqwt_tools_path)

from tqwt_tools import tqwt
import pywt


# -----------------------------
# Dataset / channel constants
# -----------------------------

SFREQ = 128.0  # Hz

METADATA_COLUMNS = ['ID', 'Class']

STANDARD_19_CH = [
    'Fz', 'Cz', 'Pz',
    'C3', 'T3', 'C4', 'T4',
    'Fp1', 'Fp2',
    'F3', 'F4', 'F7', 'F8',
    'P3', 'P4',
    'T5', 'T6',
    'O1', 'O2'
]


# -----------------------------
# Feature params
# -----------------------------

TQWT_Q = 3
TQWT_R = 3
TQWT_J = 6

WPD_WAVELET = 'db4'
WPD_LEVEL = 3


# -----------------------------
# Preprocessing config
# -----------------------------

@dataclass
class PreprocessConfig:
    l_freq: float = 1.0
    h_freq: float = 48.0
    notch_freq: float = 50.0

    window_size_sec: float = 2.0
    window_overlap: float = 0.5

    use_notch: bool = True
    use_ica: bool = False
    use_car: bool = True
    use_zscore: bool = True

    ica_random_state: int = 97
    ica_max_iter: str = 'auto'
    ica_eog_channels: Tuple[str, str] = ('Fp1', 'Fp2')


# -----------------------------
# Helper: channel detection
# -----------------------------

def detect_eeg_channels(df: pd.DataFrame, prefer_standard_19: bool = True) -> List[str]:
    cols = [c for c in df.columns if c not in METADATA_COLUMNS]

    if prefer_standard_19:
        # Use intersection in the correct standard order
        present = [ch for ch in STANDARD_19_CH if ch in df.columns]
        if len(present) >= 10:
            # If most expected channels exist, trust the whitelist ordering
            return present

    # Fallback: filter out obvious non-signal columns
    # Drop unnamed index-like columns
    cols = [c for c in cols if not c.lower().startswith('unnamed')]
    return cols


# -----------------------------
# Raw construction
# -----------------------------

def build_raw_from_df(df_subj: pd.DataFrame, eeg_channels: List[str]) -> mne.io.Raw:
    data = df_subj[eeg_channels].to_numpy().T  # (n_channels, n_times)

    info = mne.create_info(
        ch_names=eeg_channels,
        sfreq=SFREQ,
        ch_types='eeg'
    )

    raw = mne.io.RawArray(data, info)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    return raw


# -----------------------------
# Preprocessing
# -----------------------------

def preprocess_raw(raw: mne.io.Raw, cfg: PreprocessConfig) -> mne.io.Raw:
    # Copy to avoid side effects
    raw = raw.copy()

    # Band-pass
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, fir_design='firwin')

    # Notch (only if it makes sense)
    if cfg.use_notch:
        # If low-pass already kills the notch band, skip it
        if cfg.h_freq is None or cfg.h_freq > (cfg.notch_freq - 1.0):
            raw.notch_filter(freqs=[cfg.notch_freq])
        else:
            # Auto-skip to avoid redundant filtering
            pass

    # ICA (optional)
    if cfg.use_ica:
        n_channels = len(raw.ch_names)
        n_components = max(2, min(15, n_channels - 1))

        ica = ICA(
            n_components=n_components,
            random_state=cfg.ica_random_state,
            max_iter=cfg.ica_max_iter
        )
        ica.fit(raw)

        eog_inds = []
        for ch in cfg.ica_eog_channels:
            if ch in raw.ch_names:
                inds, _scores = ica.find_bads_eog(raw, ch_name=ch)
                eog_inds.extend(inds)

        ica.exclude = sorted(set(eog_inds))
        if ica.exclude:
            ica.apply(raw)

    # Re-reference (optional)
    if cfg.use_car:
        raw.set_eeg_reference('average')

    # Per-channel z-score on the continuous signal (optional)
    if cfg.use_zscore:
        data = raw.get_data()  # (n_channels, n_times)
        data_t = data.T  # (n_times, n_channels)

        scaler = StandardScaler()
        data_z = scaler.fit_transform(data_t).T

        # RawArray is preloaded; safe to overwrite internal buffer
        raw._data = data_z

    return raw


# -----------------------------
# Epoching
# -----------------------------

def make_epochs(raw: mne.io.Raw, cfg: PreprocessConfig) -> mne.Epochs:
    return mne.make_fixed_length_epochs(
        raw,
        duration=cfg.window_size_sec,
        overlap=cfg.window_size_sec * cfg.window_overlap,
        preload=True
    )


# -----------------------------
# Feature extraction
# -----------------------------

def extract_tqwt_features(signal_1d: np.ndarray) -> np.ndarray:
    subbands = tqwt(signal_1d, TQWT_Q, TQWT_R, TQWT_J)

    feats = []
    for band in subbands:
        band = np.asarray(band)
        feats.extend([
            float(band.mean()),
            float(band.std()),
            float(np.mean(np.abs(band))),
            float(np.sum(band ** 2) / max(1, len(band))),
        ])
    return np.asarray(feats, dtype=np.float32)


def extract_wpd_features(signal_1d: np.ndarray) -> np.ndarray:
    wp = pywt.WaveletPacket(
        data=signal_1d,
        wavelet=WPD_WAVELET,
        mode='symmetric',
        maxlevel=WPD_LEVEL
    )
    nodes = wp.get_level(WPD_LEVEL, order='freq')

    feats = []
    for n in nodes:
        c = np.asarray(n.data)
        feats.extend([
            float(c.mean()),
            float(c.std()),
            float(np.mean(np.abs(c))),
            float(np.sum(c ** 2) / max(1, len(c))),
        ])
    return np.asarray(feats, dtype=np.float32)


def extract_features_from_epochs(
    epochs: mne.Epochs,
    label: int,
    subject_id: str
):
    X_list, y_list, subj_list = [], [], []

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, _n_times = data.shape

    # Determine feature size per channel
    sample_sig = data[0, 0, :]
    tqwt_feats = extract_tqwt_features(sample_sig)
    wpd_feats = extract_wpd_features(sample_sig)
    features_per_channel = len(tqwt_feats) + len(wpd_feats)

    for i in range(n_epochs):
        epoch = data[i]  # (n_channels, n_times)
        epoch_feats = []

        for ch_idx in range(n_channels):
            sig = epoch[ch_idx, :]
            ch_feats = np.concatenate(
                [extract_tqwt_features(sig), extract_wpd_features(sig)],
                axis=0
            )
            epoch_feats.append(ch_feats)

        X_epoch = np.concatenate(epoch_feats, axis=0)
        X_list.append(X_epoch)
        y_list.append(label)
        subj_list.append(subject_id)

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.int64)
    subjects = np.asarray(subj_list, dtype=object)

    return X, y, subjects, features_per_channel, epochs.ch_names


def label_to_int(class_str: str) -> int:
    mapping = {'Control': 0, 'ADHD': 1}
    if class_str not in mapping:
        raise ValueError(f"Unexpected class label: {class_str}")
    return mapping[class_str]


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(
    csv_path: str,
    out_dir: str,
    cfg: PreprocessConfig,
    prefer_standard_19: bool = True
):
    df = pd.read_csv(csv_path)

    eeg_channels = detect_eeg_channels(df, prefer_standard_19=prefer_standard_19)

    print(f"Detected {len(eeg_channels)} EEG channels:")
    print(eeg_channels)

    all_X, all_y, all_subjects = [], [], []
    channel_names = None
    features_per_channel = None

    for subj_id, df_subj in df.groupby('ID'):
        subj_id = str(subj_id)
        n_samples = len(df_subj)

        if n_samples < int(cfg.window_size_sec * SFREQ):
            print(f"Skipping subject {subj_id}: too few samples ({n_samples})")
            continue

        print(f"Processing subject {subj_id}: {n_samples} samples")

        raw = build_raw_from_df(df_subj, eeg_channels)
        raw = preprocess_raw(raw, cfg)
        epochs = make_epochs(raw, cfg)

        class_str = df_subj['Class'].iloc[0]
        y_label = label_to_int(class_str)

        X_subj, y_subj, subj_vec, feats_per_ch, ch_names = extract_features_from_epochs(
            epochs, label=y_label, subject_id=subj_id
        )

        all_X.append(X_subj)
        all_y.append(y_subj)
        all_subjects.append(subj_vec)

        if channel_names is None:
            channel_names = ch_names
            features_per_channel = feats_per_ch

    if not all_X:
        raise RuntimeError("No subjects processed. Check data integrity and IDs.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_ids = np.concatenate(all_subjects, axis=0)

    print("\nFINAL SHAPES:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("subject_ids shape:", subject_ids.shape)
    print(f"Number of channels: {len(channel_names)}")
    print(f"Features per channel: {features_per_channel}")
    print(f"Total features: {len(channel_names) * features_per_channel}")
    print(f"Channel names: {channel_names}")

    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'X_tqwt_wpd.npy'), X)
    np.save(os.path.join(out_dir, 'y_labels.npy'), y)
    np.save(os.path.join(out_dir, 'subject_ids.npy'), subject_ids)
    np.save(os.path.join(out_dir, 'channel_names.npy'), np.array(channel_names, dtype=object))
    np.save(os.path.join(out_dir, 'features_per_channel.npy'), np.array(features_per_channel))

    # Save config for reproducibility
    with open(os.path.join(out_dir, 'preprocess_config.txt'), 'w') as f:
        for k, v in cfg.__dict__.items():
            f.write(f"{k} = {v}\n")

    print("\nCompleted preprocessing.")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ADHD EEG preprocessing + TQWT/WPD feature extraction")

    p.add_argument('--data-path', type=str, default=None,
                   help="Path to adhdata.csv (overrides DATA_PATH env)")
    p.add_argument('--out-dir', type=str, default='data')

    p.add_argument('--l-freq', type=float, default=1.0)
    p.add_argument('--h-freq', type=float, default=48.0)
    p.add_argument('--notch-freq', type=float, default=50.0)

    p.add_argument('--window-sec', type=float, default=2.0)
    p.add_argument('--overlap', type=float, default=0.5)

    p.add_argument('--no-notch', action='store_true')
    p.add_argument('--ica', action='store_true')
    p.add_argument('--no-car', action='store_true')
    p.add_argument('--no-zscore', action='store_true')

    p.add_argument('--no-standard-19', action='store_true',
                   help="Do not prefer the standard 19-channel whitelist")

    return p.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = parse_args()

    base_path = args.data_path
    if base_path is None:
        base_dir = os.getenv('DATA_PATH', 'data')
        base_path = os.path.join(base_dir, 'adhdata.csv')

    cfg = PreprocessConfig(
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch_freq=args.notch_freq,
        window_size_sec=args.window_sec,
        window_overlap=args.overlap,
        use_notch=not args.no_notch,
        use_ica=args.ica,
        use_car=not args.no_car,
        use_zscore=not args.no_zscore
    )

    run_pipeline(
        csv_path=base_path,
        out_dir=args.out_dir,
        cfg=cfg,
        prefer_standard_19=not args.no_standard_19
    )
