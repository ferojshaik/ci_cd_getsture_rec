"""
MLOps Lesson 2: Data validation before training

Why: Bad or changed data causes hard-to-debug failures. Running this script
before training (or as the first step of the pipeline) catches:
- Missing or invalid JSON structure (schema)
- Train/test leakage (same file in both folders)
- Missing labels or too few samples

Run: python validate_data.py
Exit: 0 if all checks pass, 1 otherwise. Pipeline can run this automatically.
"""

import sys
from pathlib import Path

# Add project root so we can import data_loader
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import (
    get_project_root,
    load_training_data,
    load_testing_data,
    validate_all_data,
    load_one_recording,
    get_label_from_filename,
)
from dataset import LABEL_ORDER


def check_schema(records, folder_name):
    """
    Check that each recording has the expected structure (payload.values, interval_ms, numeric).
    Returns (ok, list of error messages).
    """
    errors = []
    for rec in records:
        fname = rec.get("file_name", "?")
        if "values" not in rec:
            errors.append(f"{folder_name}/{fname}: missing 'values'")
            continue
        vals = rec["values"]
        if not isinstance(vals, list) or len(vals) == 0:
            errors.append(f"{folder_name}/{fname}: 'values' must be non-empty list")
            continue
        for i, row in enumerate(vals[:3]):  # check first few rows
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                errors.append(f"{folder_name}/{fname}: row {i} must be [accX, accY, accZ]")
                break
            try:
                _ = [float(x) for x in row]
            except (TypeError, ValueError):
                errors.append(f"{folder_name}/{fname}: row {i} must be numeric")
                break
        interval = rec.get("interval_ms")
        if interval is not None and (not isinstance(interval, (int, float)) or interval <= 0):
            errors.append(f"{folder_name}/{fname}: interval_ms must be positive number")
    return len(errors) == 0, errors


def main():
    print("MLOps Lesson 2: Data validation\n")
    print("Schema: DATA_SCHEMA.md")
    print("Checks: schema, no train/test leakage, all labels present, min samples.\n")

    root = get_project_root()
    train_path = root / "Training"
    test_path = root / "Testing"

    if not train_path.is_dir():
        print(f"ERROR: Training folder not found: {train_path}")
        return 1
    if not test_path.is_dir():
        print(f"ERROR: Testing folder not found: {test_path}")
        return 1

    train_records = load_training_data()
    test_records = load_testing_data()

    if not train_records:
        print("ERROR: No training recordings found in Training/")
        return 1

    # 1. Schema check
    schema_ok_train, schema_errors_train = check_schema(train_records, "Training")
    schema_ok_test, schema_errors_test = check_schema(test_records, "Testing")
    if not schema_ok_train or not schema_ok_test:
        print("Schema validation failed:")
        for e in schema_errors_train[:5]:
            print("  ", e)
        for e in schema_errors_test[:5]:
            print("  ", e)
        if len(schema_errors_train) + len(schema_errors_test) > 10:
            print("  ... and more")
        return 1
    print("Schema: OK (payload.values, interval_ms, numeric)")

    # 2. Leakage + labels + min samples (reuse data_loader logic)
    validation_ok, messages = validate_all_data(
        train_records, test_records, LABEL_ORDER, verbose=True
    )
    if not validation_ok:
        print("Validation failed. Fix the issues above and re-run.")
        return 1

    print("\nAll data validation checks passed. Safe to run trainer.py or run_pipeline.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
