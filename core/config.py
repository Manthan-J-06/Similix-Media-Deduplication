"""
core/config.py — Load and validate config.toml.

Uses the stdlib tomllib (Python 3.11+). Returns a plain dict; callers
access values as config["section"]["key"]. No external dependencies.

Usage:
    from core.config import load_config
    cfg = load_config()                        # uses repo-root config.toml
    cfg = load_config("/path/to/config.toml")  # explicit path
"""

import tomllib
from pathlib import Path

# Default config path: repo root / config.toml
_REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.toml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """
    Parse and return the TOML config file as a nested dict.
    Raises FileNotFoundError if the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Make sure config.toml exists in the repo root."
        )
    with open(path, "rb") as f:
        return tomllib.load(f)


def get(config: dict, *keys: str, default=None):
    """
    Safe nested key access. Example:
        get(cfg, "stage1", "hamming_threshold", default=10)
    """
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
