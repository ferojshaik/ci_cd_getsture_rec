"""
STEP 1: Data Loader

What this module does:
- Finds all gesture JSON files in your Training/ and Testing/ folders
- Reads each file and extracts the accelerometer data (accX, accY, accZ)
- Gets the label (TAP, WAVE, or SHAKE) from the filename
- Gives you a simple way to get "all training data" or "all testing data"

We use only Python's built-in libraries here (json, pathlib) so you see
exactly how the data is structured. No ML yet — just loading.
"""

import json
from pathlib import Path


def get_project_root():
    """
    Returns the folder that contains Training/ and Testing/.
    We assume this script lives in that folder (AI_ML).
    """
    return Path(__file__).resolve().parent


def get_label_from_filename(filename):
    """
    Your files are named like: TAP.6j2esfl3.json, WAVE.6j2epvfg.json, shake.6j2ecqjv.json
    The part before the first dot is the gesture label.
    We normalize to uppercase so 'shake' and 'SHAKE' are the same.
    """
    # e.g. "TAP.6j2esfl3.json" -> "TAP", "shake.6j2ecqjv.json" -> "SHAKE"
    name_without_ext = Path(filename).stem   # "TAP.6j2esfl3"
    label = name_without_ext.split(".")[0]   # "TAP"
    return label.upper()


def load_one_recording(file_path):
    """
    Loads a single JSON file and returns:
    - values: list of [accX, accY, accZ] for each time step
    - interval_ms: time between samples (e.g. 16)
    - label: TAP, WAVE, or SHAKE (from filename)

    The JSON has a 'payload' with 'values' and 'interval_ms'.
    We ignore 'protected' and 'signature' — they are for verification.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    payload = data.get("payload", {})
    values = payload.get("values", [])
    interval_ms = payload.get("interval_ms", 16)

    label = get_label_from_filename(file_path.name)

    return {
        "values": values,
        "interval_ms": interval_ms,
        "label": label,
        "num_samples": len(values),
    }


def load_all_from_folder(folder_path):
    """
    Loads every .json file in the given folder.
    folder_path can be "Training" or "Testing" (relative to project root).

    Returns a list of dicts; each dict has "values", "interval_ms", "label", "num_samples".
    """
    root = get_project_root()
    folder = root / folder_path
    if not folder.is_dir():
        return []

    results = []
    for file_path in sorted(folder.glob("*.json")):
        try:
            rec = load_one_recording(file_path)
            rec["file_name"] = file_path.name
            results.append(rec)
        except Exception as e:
            print(f"Warning: could not load {file_path.name}: {e}")

    return results


def load_training_data():
    """Load all recordings from the Training/ folder."""
    return load_all_from_folder("Training")


def load_testing_data():
    """Load all recordings from the Testing/ folder."""
    return load_all_from_folder("Testing")


# ---------------------------------------------------------------------------
# Data validation (ML best practice: ensure no leakage, sufficient samples)
# ---------------------------------------------------------------------------

# Minimum samples per recording for a 2 s @ 62.5 Hz window (125 samples)
MIN_SAMPLES_PER_RECORDING = 125


def validate_recording(rec, min_samples=MIN_SAMPLES_PER_RECORDING):
    """
    Check a single recording for training readiness.
    Returns (is_valid, reason). Reason is empty if valid.
    """
    n = rec.get("num_samples", len(rec.get("values", [])))
    if n == 0:
        return False, "empty"
    if n < min_samples:
        return False, f"too_short ({n} < {min_samples})"
    interval = rec.get("interval_ms", 16)
    if interval <= 0:
        return False, "invalid_interval_ms"
    return True, ""


def check_train_test_leakage(train_records, test_records):
    """
    Detect if the same file (by filename) appears in both train and test.
    Such overlap would be data leakage and invalidate test metrics.
    Returns list of filenames that appear in both.
    """
    train_names = {r.get("file_name", "") for r in train_records}
    test_names = {r.get("file_name", "") for r in test_records}
    return sorted(train_names & test_names)


def validate_all_data(train_records, test_records, allowed_labels, verbose=True):
    """
    Run ML-sanity checks and return (ok, messages).
    - No train/test filename overlap (leakage)
    - Each recording has enough samples for at least one window
    - Expected labels present in training
    """
    messages = []
    ok = True

    leakage = check_train_test_leakage(train_records, test_records)
    if leakage:
        ok = False
        messages.append(f"DATA LEAKAGE: {len(leakage)} file(s) in both Training/ and Testing/: {leakage[:5]}{'...' if len(leakage) > 5 else ''}")

    short_train = []
    short_test = []
    for r in train_records:
        valid, reason = validate_recording(r)
        if not valid and reason == "too_short":
            short_train.append(r.get("file_name", "?"))
    for r in test_records:
        valid, reason = validate_recording(r)
        if not valid and reason == "too_short":
            short_test.append(r.get("file_name", "?"))

    if short_train:
        messages.append(f"Training: {len(short_train)} recording(s) shorter than {MIN_SAMPLES_PER_RECORDING} samples (will be zero-padded): {short_train[:3]}{'...' if len(short_train) > 3 else ''}")
    if short_test:
        messages.append(f"Testing: {len(short_test)} recording(s) shorter than {MIN_SAMPLES_PER_RECORDING} samples: {short_test[:3]}{'...' if len(short_test) > 3 else ''}")

    train_labels = {r["label"].upper() for r in train_records}
    for lbl in allowed_labels:
        if lbl not in train_labels:
            messages.append(f"Training has no recordings for label '{lbl}'.")

    if verbose and messages:
        print("\n--- Data validation ---")
        for m in messages:
            print("  ", m)
        if leakage:
            print("  -> Fix: remove duplicate files from one of Training/ or Testing/.")
        print()

    return ok, messages


def print_summary(records, title="Data"):
    """
    Prints a short summary so you can verify the loader works:
    - How many files per label (TAP, WAVE, SHAKE)
    - Total number of recordings
    - Example: one file's shape (number of time steps, 3 axes)
    """
    if not records:
        print(f"{title}: No records found.")
        return

    label_counts = {}
    for r in records:
        lbl = r["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"\n--- {title} ---")
    print(f"Total recordings: {len(records)}")
    print("Per label:", label_counts)
    # Show one example
    first = records[0]
    values = first["values"]
    print(f"Example: {first['file_name']} -> label={first['label']}, "
          f"num_samples={len(values)}, interval_ms={first['interval_ms']}")
    if values:
        print(f"  First row (accX, accY, accZ): {values[0]}")
    print()


# ---------------------------------------------------------------------------
# When you run this file, it loads Training and Testing and prints a summary.
# This way you can confirm the data loader works before we add the next step.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Step 1: Data Loader — loading your gesture JSON files.\n")

    training = load_training_data()
    testing = load_testing_data()

    print_summary(training, "Training")
    print_summary(testing, "Testing")

    print("Done. If you see your TAP, WAVE, and SHAKE counts above, Step 1 is OK.")
