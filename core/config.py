"""
core/config.py — Load and validate config.toml.

Uses the stdlib tomllib (Python 3.11+). Returns a plain dict; callers
access values as config["section"]["key"]. No external dependencies.

Usage:
    from core.config import load_config
    cfg = load_config()                        # uses repo-root config.toml
    cfg = load_config("/path/to/config.toml")  # explicit path
"""

import logging
import os
import shutil
import tomllib
from pathlib import Path

logger = logging.getLogger("similix.config")

# Default config path: repo root / config.toml
_REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.toml"


def get_env(name: str, default: str | None = None) -> str | None:
    """
    Get environment variable checking SIMILIX_<name> first, then ARGUS_<name>.
    Accepts raw names ('DATA_DIR'), 'SIMILIX_DATA_DIR', or 'ARGUS_DATA_DIR'.
    """
    if name.startswith("SIMILIX_"):
        suffix = name[len("SIMILIX_"):]
        candidates = [name, f"ARGUS_{suffix}"]
    elif name.startswith("ARGUS_"):
        suffix = name[len("ARGUS_"):]
        candidates = [f"SIMILIX_{suffix}", name]
    else:
        candidates = [f"SIMILIX_{name}", f"ARGUS_{name}", name]

    for cand in candidates:
        val = os.environ.get(cand)
        if val is not None:
            return val
    return default


def resolve_data_dir(raw_dir: str | Path | None = None) -> Path:
    """
    Resolve the primary application data directory.
    Priority:
      1. Explicit argument `raw_dir` (if provided)
      2. Environment variable SIMILIX_DATA_DIR or ARGUS_DATA_DIR
      3. Default: ~/.similix
         - If ~/.similix does not exist, but legacy ~/.argus exists,
           automatically migrate contents from ~/.argus to ~/.similix.
         - Fallback to .similix in current working directory if home is read-only.
    """
    if raw_dir is not None:
        target = Path(raw_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    env_dir = get_env("DATA_DIR")
    if env_dir:
        target = Path(env_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    home_similix = Path.home() / ".similix"
    home_argus = Path.home() / ".argus"

    def _is_writable(p: Path) -> bool:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test_file = p / ".perm_check"
            test_file.touch()
            test_file.unlink()
            return True
        except (OSError, PermissionError):
            return False

    if home_argus.exists() and not home_similix.exists():
        try:
            if _is_writable(home_similix):
                shutil.copytree(home_argus, home_similix, dirs_exist_ok=True, symlinks=True)
                logger.info("Automatically migrated legacy data directory from %s to %s", home_argus, home_similix)
                return home_similix.resolve()
        except Exception as exc:
            logger.warning("Automatic migration from %s to %s failed: %s", home_argus, home_similix, exc)

    if _is_writable(home_similix):
        return home_similix.resolve()

    if home_argus.exists() and _is_writable(home_argus):
        return home_argus.resolve()

    # Fallback to current working directory
    cwd_similix = Path.cwd() / ".similix"
    cwd_argus = Path.cwd() / ".argus"
    if cwd_argus.exists() and not cwd_similix.exists():
        try:
            shutil.copytree(cwd_argus, cwd_similix, dirs_exist_ok=True, symlinks=True)
            logger.info("Migrated legacy working data directory from %s to %s", cwd_argus, cwd_similix)
        except Exception:
            pass
    cwd_similix.mkdir(parents=True, exist_ok=True)
    return cwd_similix.resolve()


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
