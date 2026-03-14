"""
STEP 2: Dataset builder (windowing + spectral + time-domain features)

Matches the pipeline from your screenshots:
- Time series: accX, accY, accZ @ 62.5 Hz
- Window: 2000 ms, stride 80 ms, zero-pad
- Spectral Analysis: FFT length 16, log spectrum, overlap FFT frames
- Features: 12 time-domain (mean, std, min, max per axis) + 27 spectral (9 bins x 3 axes) = 39
- Output: X (n_samples, 39), y (0=TAP, 1=WAVE, 2=SHAKE, 3=IDEAL)
"""

import numpy as np
from data_loader import load_training_data, load_testing_data


# Fixed order: 4 classes (IDEAL = phone at rest / no gesture).
LABEL_ORDER = ("TAP", "WAVE", "SHAKE", "IDEAL")

# Pipeline constants (from your screenshots)
SAMPLE_RATE_HZ = 62.5
WINDOW_MS = 2000
STRIDE_MS = 80
FFT_LENGTH = 16
FFT_STRIDE = 8   # overlap (e.g. 50% when stride = FFT_LENGTH/2)

WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)   # 125
STRIDE_SAMPLES = int(SAMPLE_RATE_HZ * STRIDE_MS / 1000)   # 5
NUM_FFT_BINS = FFT_LENGTH // 2 + 1   # 9 for real FFT
NUM_SPECTRAL_FEATURES = NUM_FFT_BINS * 3   # 27 (9 per axis)
NUM_TIME_FEATURES = 12   # mean, std, min, max per axis
NUM_FEATURES = NUM_TIME_FEATURES + NUM_SPECTRAL_FEATURES   # 39


def label_to_index(label):
    """Convert a label string to index: TAP=0, WAVE=1, SHAKE=2, IDEAL=3."""
    label = label.upper()
    if label not in LABEL_ORDER:
        raise ValueError(f"Unknown label: {label}. Expected one of {LABEL_ORDER}")
    return LABEL_ORDER.index(label)


def index_to_label(index):
    """Convert a number (0, 1, 2, 3) back to the label string."""
    return LABEL_ORDER[index]


def _make_windows(values, window_len, stride, zero_pad=True):
    """
    Slice a recording into overlapping windows.
    values: (N, 3) array
    Returns: list of (window_len, 3) arrays
    """
    arr = np.array(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 3:
        arr = np.hstack([arr, np.zeros((len(arr), 3 - arr.shape[1]), dtype=np.float32)])
    n = len(arr)
    if n < window_len and zero_pad:
        pad = np.zeros((window_len - n, 3), dtype=np.float32)
        arr = np.vstack([arr, pad])
        n = len(arr)
    windows = []
    for start in range(0, n - window_len + 1, stride):
        windows.append(arr[start : start + window_len].copy())
    if zero_pad and n >= window_len and (n - window_len) % stride != 0:
        # last partial window: zero-pad and add
        start = (n - window_len + stride - 1) // stride * stride
        if start < n:
            w = np.zeros((window_len, 3), dtype=np.float32)
            w[: n - start] = arr[start:]
            windows.append(w)
    return windows


def _time_domain_features(window):
    """Per-axis mean, std, min, max -> 12 features."""
    feats = []
    for axis in range(3):
        col = window[:, axis]
        feats.extend([float(np.mean(col)), float(np.std(col) or 0), float(np.min(col)), float(np.max(col))])
    return np.array(feats, dtype=np.float32)


def _spectral_features(window):
    """
    FFT length 16, overlap (stride 8), log of spectrum.
    Average over frames -> 9 bins per axis -> 27 features.
    """
    frame_len = FFT_LENGTH
    stride = FFT_STRIDE
    n_axes = 3
    n_bins = NUM_FFT_BINS  # 9
    spectral = np.zeros((n_axes, n_bins), dtype=np.float32)
    counts = np.zeros((n_axes, n_bins), dtype=np.float32)
    n_samples = len(window)
    for start in range(0, n_samples - frame_len + 1, stride):
        for axis in range(n_axes):
            frame = window[start : start + frame_len, axis].ravel()
            fft = np.fft.rfft(frame)
            mag = np.abs(fft)
            log_mag = np.log1p(mag)
            spectral[axis, :] += log_mag
            counts[axis, :] += 1
    for axis in range(n_axes):
        where = counts[axis, :] > 0
        spectral[axis, where] /= counts[axis, where]
    return spectral.ravel()


def extract_features_from_window(window):
    """
    One window (125 x 3) -> 39 features: 12 time + 27 spectral.
    """
    time_feats = _time_domain_features(window)
    spec_feats = _spectral_features(window)
    return np.concatenate([time_feats, spec_feats]).astype(np.float32)


def recording_to_feature_vectors(values, label):
    """
    Split recording into windows (2000 ms, stride 80 ms, zero-pad).
    Each window -> 39 features. All windows get the same label.
    Returns: X (n_windows, 39), y (n_windows,) with same label index.
    """
    windows = _make_windows(values, WINDOW_SAMPLES, STRIDE_SAMPLES, zero_pad=True)
    if not windows:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32), np.array([], dtype=np.int32)
    X_list = []
    for w in windows:
        X_list.append(extract_features_from_window(w))
    X = np.array(X_list, dtype=np.float32)
    idx = label_to_index(label)
    y = np.full(len(X_list), idx, dtype=np.int32)
    return X, y


def records_to_dataset(records, allowed_labels=None):
    """
    Convert a list of recordings into X and y using windowing + 39 features.

    records: list of dicts with "values" and "label"
    allowed_labels: only include these (e.g. TAP, WAVE, SHAKE).

    Returns:
        X: (n_samples, 39)
        y: (n_samples,) 0, 1, or 2
        skipped: list of skipped filenames
    """
    if allowed_labels is None:
        allowed_labels = LABEL_ORDER

    X_list = []
    y_list = []
    skipped = []

    for rec in records:
        label = rec["label"].upper()
        if label not in allowed_labels:
            skipped.append(rec.get("file_name", label))
            continue
        Xw, yw = recording_to_feature_vectors(rec["values"], label)
        if len(Xw) > 0:
            X_list.append(Xw)
            y_list.append(yw)

    if not X_list:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32), np.array([], dtype=np.int32), skipped
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y, skipped


def get_training_dataset():
    """Load training data, window it, extract 39 features per window."""
    records = load_training_data()
    return records_to_dataset(records, allowed_labels=LABEL_ORDER)


def get_testing_dataset():
    """Load testing data, window it, extract 39 features per window."""
    records = load_testing_data()
    return records_to_dataset(records, allowed_labels=LABEL_ORDER)


def print_dataset_summary(X, y, name="Dataset"):
    """Print shape and label counts."""
    print(f"\n--- {name} ---")
    print(f"X shape: {X.shape}  (samples, features)")
    print(f"y shape: {y.shape}")
    for i, label in enumerate(LABEL_ORDER):
        count = np.sum(y == i)
        print(f"  {label}: {count} samples")
    print()


if __name__ == "__main__":
    print("Step 2: Dataset - windowing + spectral (39 features).\n")

    X_train, y_train, skip_train = get_training_dataset()
    X_test, y_test, skip_test = get_testing_dataset()

    print_dataset_summary(X_train, y_train, "Training")
    print_dataset_summary(X_test, y_test, "Testing")

    if skip_test:
        print(f"Skipped (not TAP/WAVE/SHAKE): {len(skip_test)} files")

    print(f"Done. X has {NUM_FEATURES} features per sample (12 time + 27 spectral).")
