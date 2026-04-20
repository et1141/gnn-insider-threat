import yaml
from pathlib import Path


def get_project_root() -> Path:
    """Return path to project root folder"""
    return Path(__file__).resolve().parent.parent.parent


def load_config():
    """Loads global config from YAML file."""
    root = get_project_root()
    config_path = root / "configs" / "config.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
