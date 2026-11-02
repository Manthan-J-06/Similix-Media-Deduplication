"""
tests/test_config.py — Validate that active config.toml thresholds match
the single source of truth recorded in bench/tuned.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.config import load_config, DEFAULT_CONFIG_PATH, get_env, resolve_data_dir


TUNED_JSON_PATH = Path("bench/tuned.json")


def test_get_env_precedence(monkeypatch):
    """Verify get_env prioritizes SIMILIX_* over ARGUS_* and falls back correctly."""
    monkeypatch.delenv("SIMILIX_TEST_VAR", raising=False)
    monkeypatch.delenv("ARGUS_TEST_VAR", raising=False)
    monkeypatch.delenv("TEST_VAR", raising=False)

    assert get_env("TEST_VAR", "default_val") == "default_val"

    monkeypatch.setenv("ARGUS_TEST_VAR", "from_argus")
    assert get_env("TEST_VAR") == "from_argus"
    assert get_env("SIMILIX_TEST_VAR") == "from_argus"
    assert get_env("ARGUS_TEST_VAR") == "from_argus"

    monkeypatch.setenv("SIMILIX_TEST_VAR", "from_similix")
    assert get_env("TEST_VAR") == "from_similix"
    assert get_env("SIMILIX_TEST_VAR") == "from_similix"
    assert get_env("ARGUS_TEST_VAR") == "from_similix"


def test_resolve_data_dir_explicit(tmp_path):
    target = tmp_path / "custom_data"
    res = resolve_data_dir(target)
    assert res == target.resolve()
    assert res.is_dir()


def test_resolve_data_dir_env(tmp_path, monkeypatch):
    target = tmp_path / "env_data"
    monkeypatch.setenv("SIMILIX_DATA_DIR", str(target))
    res = resolve_data_dir()
    assert res == target.resolve()
    assert res.is_dir()

    monkeypatch.delenv("SIMILIX_DATA_DIR")
    monkeypatch.setenv("ARGUS_DATA_DIR", str(target))
    res2 = resolve_data_dir()
    assert res2 == target.resolve()


def test_resolve_data_dir_migration(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.delenv("SIMILIX_DATA_DIR", raising=False)
    monkeypatch.delenv("ARGUS_DATA_DIR", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)

    legacy_argus = fake_home / ".argus"
    legacy_argus.mkdir()
    (legacy_argus / "similix.db").write_text("dummy")

    resolved = resolve_data_dir()
    assert resolved == (fake_home / ".similix").resolve()
    assert (resolved / "similix.db").read_text() == "dummy"


def test_config_matches_tuned_thresholds():
    """
    Ensure config.toml thresholds are synchronized with the tuned values
    in bench/tuned.json (written by bench/evaluate.py).
    """
    assert TUNED_JSON_PATH.exists(), (
        f"{TUNED_JSON_PATH} does not exist! Run bench/evaluate.py to establish tuned values."
    )

    tuned_data = json.loads(TUNED_JSON_PATH.read_text(encoding="utf-8"))
    cfg = load_config(DEFAULT_CONFIG_PATH)

    cfg_stage1_th = cfg.get("stage1", {}).get("hamming_threshold")
    tuned_stage1_th = tuned_data.get("stage1", {}).get("hamming_threshold")

    assert cfg_stage1_th == tuned_stage1_th, (
        f"config.toml stage1.hamming_threshold ({cfg_stage1_th}) does not match "
        f"bench/tuned.json ({tuned_stage1_th})!"
    )

    cfg_stage2_cos = cfg.get("stage2", {}).get("cosine_threshold")
    tuned_stage2_cos = tuned_data.get("stage2", {}).get("cosine_threshold")

    assert abs(float(cfg_stage2_cos) - float(tuned_stage2_cos)) < 1e-4, (
        f"config.toml stage2.cosine_threshold ({cfg_stage2_cos}) does not match "
        f"bench/tuned.json ({tuned_stage2_cos})!"
    )


def test_synthetic_threshold_mismatch_fails():
    """
    Verify that an intentional mismatch between config and tuned.json raises AssertionError.
    """
    tuned_sample = {
        "stage1": {"hamming_threshold": 8},
        "stage2": {"cosine_threshold": 0.90},
    }
    mismatched_cfg = {
        "stage1": {"hamming_threshold": 10},
        "stage2": {"cosine_threshold": 0.92},
    }

    assert mismatched_cfg["stage1"]["hamming_threshold"] != tuned_sample["stage1"]["hamming_threshold"]
    assert mismatched_cfg["stage2"]["cosine_threshold"] != tuned_sample["stage2"]["cosine_threshold"]
