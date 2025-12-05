import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv

import mne
from mne.preprocessing import ICA

from sklearn.preprocessing import StandardScaler

from tqwt_tools import tqwt
import pywt

SFREQ = 128.0  # Hz, EEG sampling frequency

# EEG_CHANNELS will be determined dynamically from the data file columns
# Metadata columns to exclude when identifying EEG channels
METADATA_COLUMNS = ['ID', 'Class']

# Filters
L_FREQ = 1.0      # Hz, high-pass
H_FREQ = 40.0     # Hz, low-pass
NOTCH_FREQ = 50.0 # Hz, mains notch (change to 60 if you prefer)

# Windowing
WINDOW_SIZE_SEC = 2.0    # 2–4 s; here we pick 2 s
WINDOW_OVERLAP = 0.5     # 50% overlap

# TQWT params
TQWT_Q = 3
TQWT_R = 3
TQWT_J = 6    # number of subbands

# WPD params
WPD_WAVELET = 'db4'
WPD_LEVEL = 3


# helper functions
def build_raw_from_df(df_subj: pd.DataFrame, eeg_channels: list) -> mne.io.Raw:
    """Create an MNE RawArray from a single subject's dataframe."""
    data = df_subj[eeg_channels].to_numpy().T  # (n_channels, n_times)

    info = mne.create_info(
        ch_names=eeg_channels,
        sfreq=SFREQ,
        ch_types='eeg'
    )
    raw = mne.io.RawArray(data, info)

    # Set standard 10–20 montage (ignore missing channels)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    return raw


def preprocess_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply band-pass + notch filtering, ICA-based artifact reduction,
    common-average reference, and per-channel z-score normalization.
    """
    # Band-pass filter (MNE uses SciPy under the hood)
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design='firwin')

    # Notch filter
    raw.notch_filter(freqs=[NOTCH_FREQ])

    # ICA for artifact reduction (basic EOG removal using Fp1/Fp2)
    ica = ICA(n_components=15, random_state=97, max_iter='auto')
    ica.fit(raw)

    eog_inds = []
    for ch in ['Fp1', 'Fp2']:
        if ch in raw.ch_names:
            inds, scores = ica.find_bads_eog(raw, ch_name=ch)
            eog_inds.extend(inds)

    ica.exclude = list(set(eog_inds))
    if len(ica.exclude) > 0:
        ica.apply(raw)

    # Common-average reference
    raw.set_eeg_reference('average')

    # Per-channel z-score normalization with StandardScaler
    # shape: (n_channels, n_times) -> (n_times, n_channels) for sklearn
    data = raw.get_data().T
    scaler = StandardScaler()
    data_z = scaler.fit_transform(data)
    raw._data = data_z.T  # back to (n_channels, n_times)

    return raw


def make_epochs(raw: mne.io.Raw) -> mne.Epochs:
    """
    Window the continuous EEG into fixed-length epochs
    using MNE's fixed-length epoching.
    """
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=WINDOW_SIZE_SEC,
        overlap=WINDOW_SIZE_SEC * WINDOW_OVERLAP,
        preload=True
    )
    return epochs


def extract_tqwt_features(signal_1d: np.ndarray) -> np.ndarray:
    """
    Feature extraction using the TQWT implementation from tqwt_tools.
    """
    # call the library function
    # The library may return a list of subbands (check docs or repo code)
    subbands = tqwt(signal_1d, TQWT_Q, TQWT_R, TQWT_J)

    feats = []
    for band in subbands:
        band = np.asarray(band)
        feats.extend([
            band.mean(),
            band.std(),
            np.mean(np.abs(band)),
            np.sum(band ** 2) / len(band),
        ])
    return np.asarray(feats, dtype=np.float32)


def extract_wpd_features(signal_1d: np.ndarray) -> np.ndarray:
    """
    Apply Wavelet Packet Decomposition (PyWavelets) and
    summarize each node (subband) with simple statistics.
    """
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
            c.mean(),
            c.std(),
            np.mean(np.abs(c)),
            np.sum(c ** 2) / len(c),  # normalized energy
        ])
    return np.asarray(feats, dtype=np.float32)


def extract_features_from_epochs(epochs: mne.Epochs, label: int, subject_id: str):
    """
    For each epoch: for each channel, compute TQWT + WPD features,
    then concatenate across channels. Returns X (n_epochs, n_features),
    y (n_epochs,), and subjects (n_epochs,).
    """
    X_list = []
    y_list = []
    subj_list = []

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    
    # Compute features per channel (same for all channels)
    # Use first epoch, first channel to determine feature size
    sample_sig = data[0, 0, :]
    tqwt_feats = extract_tqwt_features(sample_sig)
    wpd_feats = extract_wpd_features(sample_sig)
    features_per_channel = len(tqwt_feats) + len(wpd_feats)

    for i in range(n_epochs):
        epoch = data[i]  # (n_channels, n_times)
        epoch_feats = []

        for ch_idx in range(n_channels):
            sig = epoch[ch_idx, :]
            tqwt_feats = extract_tqwt_features(sig)
            wpd_feats = extract_wpd_features(sig)
            ch_feats = np.concatenate([tqwt_feats, wpd_feats], axis=0)
            epoch_feats.append(ch_feats)

        # Concatenate features from all channels
        X_epoch = np.concatenate(epoch_feats, axis=0)
        X_list.append(X_epoch)
        y_list.append(label)
        subj_list.append(subject_id)

    X = np.vstack(X_list)
    y = np.asarray(y_list)
    subjects = np.asarray(subj_list, dtype=object)

    return X, y, subjects, features_per_channel, epochs.ch_names


def label_to_int(class_str: str) -> int:
    """Map string labels to integers."""
    mapping = {'Control': 0, 'ADHD': 1}
    return mapping[class_str]



# main

if __name__ == '__main__':
    load_dotenv()

    # Load dataset
    df = pd.read_csv(
        os.path.join(
             os.getenv('DATA_PATH', 'data'),
             'adhdata.csv'
        )
    )

    # Determine EEG channels dynamically from data columns
    # Exclude metadata columns (ID, Class, etc.)
    all_columns = df.columns.tolist()
    EEG_CHANNELS = [col for col in all_columns if col not in METADATA_COLUMNS]
    print(f'Detected {len(EEG_CHANNELS)} EEG channels: {EEG_CHANNELS}')

    all_X = []
    all_y = []
    all_subjects = []

    # Process each subject separately to avoid mixing boundaries
    for subj_id, df_subj in df.groupby('ID'):
        print(f'Processing subject {subj_id}: {len(df_subj)} samples')

        # Build MNE Raw for this subject
        raw = build_raw_from_df(df_subj, EEG_CHANNELS)

        # Preprocessing: filters, ICA, CAR, z-scoring
        raw = preprocess_raw(raw)

        # Window into fixed-length epochs
        epochs = make_epochs(raw)

        # Get class label for this subject (constant per ID)
        class_str = df_subj['Class'].iloc[0]
        y_label = label_to_int(class_str)

        # Extract time–frequency features for each epoch
        X_subj, y_subj, subj_vec, feats_per_ch, ch_names = extract_features_from_epochs(
            epochs, label=y_label, subject_id=subj_id
        )

        all_X.append(X_subj)
        all_y.append(y_subj)
        all_subjects.append(subj_vec)
        
        # Store channel info (should be same for all subjects, but we'll use first)
        if 'channel_names' not in locals():
            channel_names = ch_names
            features_per_channel = feats_per_ch

    # Stack all subjects together
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_ids = np.concatenate(all_subjects, axis=0)

    print("FINAL SHAPES:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("subject_ids shape:", subject_ids.shape)
    print(f"Number of channels: {len(channel_names)}")
    print(f"Features per channel: {features_per_channel}")
    print(f"Total features: {len(channel_names) * features_per_channel}")
    print(f"Channel names: {channel_names}")


    # Save to disk for downstream models
    out_dir = os.path.join('data')
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, 'X_tqwt_wpd.npy'), X)
    np.save(os.path.join(out_dir, 'y_labels.npy'), y)
    np.save(os.path.join(out_dir, 'subject_ids.npy'), subject_ids)
    
    # Save channel information for saliency analysis
    np.save(os.path.join(out_dir, 'channel_names.npy'), np.array(channel_names, dtype=object))
    np.save(os.path.join(out_dir, 'features_per_channel.npy'), np.array(features_per_channel))
    
    print("Completed the preprocessing")