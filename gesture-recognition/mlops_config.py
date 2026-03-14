"""
MLOps Lesson 1: Load training config from config.yaml

Why: One place for all knobs. Trainer (and later the pipeline) read from here
so we never have to guess "what settings did we use?"
"""

from pathlib import Path
import yaml


def get_project_root():
    return Path(__file__).resolve().parent


def load_config(config_path=None):
    """Load config from config.yaml. Returns a dict."""
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
