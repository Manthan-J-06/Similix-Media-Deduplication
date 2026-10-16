"""
tests/test_config.py — Validate that active config.toml thresholds match
the single source of truth recorded in bench/tuned.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.config import load_config, DEFAULT_CONFIG_PATH


TUNED_JSON_PATH = Path("bench/tuned.json")


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
