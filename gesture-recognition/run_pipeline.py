"""
MLOps Lessons 3, 4, 5: Experiment tracking + model versioning + one-command pipeline

Run: python run_pipeline.py

What it does (in order):
  1. Validate data (Lesson 2) – exit if invalid
  2. Load config from config.yaml
  3. Create a run_id (timestamp) and folder runs/run_<id>/
  4. Train (Lesson 1 config) and save SavedModel to runs/run_<id>/saved_model
  5. Save config.json and metrics.json to runs/run_<id>/ (Lesson 3 – experiment tracking)
  6. Export TFLite to runs/run_<id>/ (Lesson 4 – model versioning)
  7. Append a row to runs/registry.csv (Lesson 4 – registry)
  8. Print where everything was written and how to deploy (Lesson 6)

So one command gives you: validated data, one trained model, one run folder, and a registry row.
"""

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"


def run_id():
    """Unique run identifier (timestamp)."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def main():
    rid = run_id()
    run_dir = RUNS_DIR / f"run_{rid}"
    run_dir.mkdir(parents=True, exist_ok=True)
    saved_model_dir = run_dir / "saved_model"

    print("=" * 60)
    print("MLOps Pipeline: validate → train → export → register")
    print("=" * 60)
    print(f"Run ID: {rid}")
    print(f"Run dir: {run_dir}\n")

    # 1. Data validation (Lesson 2)
    print("[1/5] Data validation...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "validate_data.py")],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print("Pipeline stopped: data validation failed. Fix data and re-run.")
        return 1
    print("Data validation OK.\n")

    # 2. Load config and run trainer (Lesson 1 + 3)
    print("[2/5] Training (config from config.yaml)...")
    try:
        from mlops_config import load_config
        cfg = load_config()
    except FileNotFoundError:
        cfg = {}
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved to {run_dir / 'config.json'}")

    from trainer import train_and_evaluate
    model, test_acc, test_f1 = train_and_evaluate(
        verbose=1,
        config=cfg,
        save_dir=str(saved_model_dir),
    )

    metrics = {
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "run_id": rid,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {run_dir / 'metrics.json'}\n")

    # 3. Export TFLite (Lesson 4)
    print("[3/5] Exporting TFLite...")
    from export_tflite import convert_to_tflite
    convert_to_tflite(quantize=False, saved_model_dir=str(saved_model_dir), output_dir=str(run_dir))
    convert_to_tflite(quantize=True, saved_model_dir=str(saved_model_dir), output_dir=str(run_dir))
    print(f"TFLite files in {run_dir}\n")

    # 4. Registry (Lesson 4)
    print("[4/5] Updating registry...")
    registry_path = RUNS_DIR / "registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": rid,
        "saved_model_dir": str(saved_model_dir),
        "tflite_quant": str(run_dir / "gesture_model_quant.tflite"),
        "test_accuracy": f"{test_acc:.4f}",
        "test_f1": f"{test_f1:.4f}",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    file_exists = registry_path.exists()
    with open(registry_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"Registry: {registry_path}\n")

    # 5. Deploy instructions (Lesson 6)
    print("[5/5] Done.\n")
    print("=" * 60)
    print("Deploy this run (Lesson 6)")
    print("=" * 60)
    print(f"  • App assets:  copy  {run_dir / 'gesture_model_quant.tflite'}  →  GestureApp/app/src/main/assets/")
    print(f"  • OTA:         copy  {run_dir / 'gesture_model_quant.tflite'}  →  ota/model/  and bump  ota/version.json")
    print(f"  • Registry:    {registry_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
