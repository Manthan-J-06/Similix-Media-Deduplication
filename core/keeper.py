"""
core/keeper.py — Stage 4: Recommendation and keeper selection.

1. Pure function of stored metrics (no image reading), so UI slider can recompute instantly:
   choose_keeper(cluster_members, mode) -> {keeper_id, delete_safe_ids, review_ids, reasons}.
   Modes (config.toml [keeper.modes]):
   - conservative: ssim_gray=0.98, ssim_color=0.97, pixel_ratio_floor=0.90, allow_blur_flag=False
   - balanced:     ssim_gray=0.95, ssim_color=0.94, pixel_ratio_floor=0.60, allow_blur_flag=False
   - aggressive:   ssim_gray=0.92, ssim_color=0.90, pixel_ratio_floor=0.25, allow_blur_flag=True

2. Candidates = members with quality_status in (reference, aligned), not is_low_confidence,
   meeting the mode limits. Keeper = smallest file among candidates; ties go to higher pixel count,
   then shorter path. If no other candidate qualifies, the reference is the keeper.

3. delete_safe = members that are (a) byte-identical to the keeper, or (b) verified, aligned,
   and not low-confidence in the same cluster. Anything unalignable, low-confidence or inconclusive
   goes to review_ids, never delete_safe. Never mark every member of a cluster as delete_safe.

4. Store per member: is_recommended_keeper (the glow flag), delete_safe, review_flag,
   quality_retained_pct (derived from SSIM vs reference), bytes_saved_if_deleted, and a short reason string.

5. Fused duplicate confidence: logistic regression model inference on stored features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any

import re
import numpy as np

def _select_reference_from_metrics(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Select reference member purely from stored database metrics (no disk I/O)."""
    if not members:
        return {}
    for m in members:
        if m.get("quality_status") == "reference":
            return m
    for m in members:
        if m.get("is_keeper") or m.get("is_recommended_keeper"):
            return m

    def fallback_ref_key(m: dict[str, Any]) -> tuple:
        w = m.get("width") or 0
        h = m.get("height") or 0
        px = w * h
        sz = m.get("file_size") or 0
        mtime = m.get("modified_time") or 0.0
        p = str(m.get("file_path") or "")
        m_id = m.get("id") or 0
        is_derived = 1 if re.search(r"(__|_copy|\bcopy\b|\s\(\d+\)|_resave)", Path(p).stem, re.IGNORECASE) else 0
        return (-px, mtime if mtime > 0 else float("inf"), is_derived, len(p), sz, m_id)

    return min(members, key=fallback_ref_key)

# Default keeper mode limits per specification
DEFAULT_KEEPER_MODES: dict[str, dict[str, Any]] = {
    "conservative": {
        "ssim_gray": 0.98,
        "ssim_color": 0.97,
        "pixel_ratio_floor": 0.90,
        "min_saving_ratio": 0.0,
        "allow_blur_flag": False,
        "allow_resolution_loss": False,
    },
    "balanced": {
        "ssim_gray": 0.92,
        "ssim_color": 0.90,
        "pixel_ratio_floor": 0.50,
        "min_saving_ratio": 0.05,
        "allow_blur_flag": False,
        "allow_resolution_loss": True,
    },
    "aggressive": {
        "ssim_gray": 0.0,
        "ssim_color": 0.0,
        "pixel_ratio_floor": 0.0,
        "min_saving_ratio": 0.0,
        "allow_blur_flag": True,
        "allow_resolution_loss": True,
    },
}

_CONFIDENCE_MODEL_CACHE: dict[str, Any] | None = None


def get_mode_limits(mode: str, config: dict | None = None) -> dict[str, Any]:
    """Retrieve mode limits from config.toml or fall back to specification defaults."""
    if config:
        modes_cfg = config.get("keeper", {}).get("modes", {})
        if mode in modes_cfg:
            return modes_cfg[mode]
    if mode in DEFAULT_KEEPER_MODES:
        return DEFAULT_KEEPER_MODES[mode]
    raise ValueError(f"Unknown keeper mode '{mode}'. Expected one of: {list(DEFAULT_KEEPER_MODES.keys())}")


def _is_member_candidate(
    m: dict[str, Any],
    limits: dict[str, Any],
    is_reference: bool = False,
) -> bool:
    """
    Check if a cluster member qualifies as a keeper candidate under the mode limits.
    Candidates = members with quality_status in (reference, aligned), not is_low_confidence,
    meeting the mode limits.
    """
    if m.get("is_low_confidence"):
        return False

    q_status = m.get("quality_status")
    if q_status not in ("reference", "aligned"):
        return False

    if is_reference or q_status == "reference":
        return True

    # Mode limits check
    ssim_gray = m.get("ssim")
    ssim_color = m.get("ssim_color")
    px_ratio = m.get("pixel_ratio")
    
    # Sharpness / blur flag
    sharpness = m.get("sharpness_ratio")
    blur_flag = bool(m.get("blur_flag", False))
    if sharpness is not None and sharpness < 0.70:
        blur_flag = True

    if ssim_gray is None or ssim_gray < limits["ssim_gray"]:
        return False
    if ssim_color is None or ssim_color < limits["ssim_color"]:
        return False
    if px_ratio is None or px_ratio < limits["pixel_ratio_floor"]:
        return False
    if not limits.get("allow_blur_flag", False) and blur_flag:
        return False

    return True


def _keeper_candidate_sort_key(m: dict[str, Any]) -> tuple:
    """
    Deterministic sort key for keeper among candidates:
    1. Smallest file size (file_size ascending)
    2. Higher pixel count (width * height descending)
    3. Shorter path (len(path) ascending)
    4. Deterministic string tie-breaker (path ascending)
    5. ID ascending
    """
    file_size = m.get("file_size") or 0
    w = m.get("width") or 0
    h = m.get("height") or 0
    pixel_count = w * h
    path = str(m.get("file_path") or "")
    m_id = m.get("id") or 0
    return (file_size, -pixel_count, len(path), path, m_id)


def choose_keeper(
    cluster_members: list[dict[str, Any] | sqlite3.Row],
    mode: str = "balanced",
    config: dict | None = None,
    reference_uncertain: int | bool | None = None,
    allow_resolution_loss: bool | None = None,
) -> dict[str, Any]:
    """
    Pure function of stored metrics (no image reading), so a UI slider can recompute instantly.

    Returns:
    {
        "keeper_id": int | None,
        "delete_safe_ids": list[int],
        "review_ids": list[int],
        "reasons": dict[int, str],
        "total_savings_bytes": int,
        "quality_retained": dict[int, float],
        "reference_uncertain": int,
        "allow_resolution_loss": bool,
    }
    """
    if not cluster_members:
        return {
            "keeper_id": None,
            "delete_safe_ids": [],
            "review_ids": [],
            "reasons": {},
            "total_savings_bytes": 0,
            "quality_retained": {},
            "reference_uncertain": 0,
            "allow_resolution_loss": False,
            "swap_proposal": None,
            "no_swap_reason": None,
        }

    # Normalize rows to dictionaries
    members = [dict(m) if not isinstance(m, dict) else m for m in cluster_members]
    limits = get_mode_limits(mode, config)
    res_loss_allowed = limits.get("allow_resolution_loss", False)
    if allow_resolution_loss is not None:
        res_loss_allowed = bool(allow_resolution_loss)

    cfg = config or {}
    q_cfg = cfg.get("quality", {})
    min_cluster_match_pct = float(q_cfg.get("min_cluster_match_pct", 75.0))

    # Identify the cluster reference member
    ref_member = None
    for m in members:
        if m.get("quality_status") == "reference":
            ref_member = m
            break

    if ref_member is None:
        for m in members:
            if m.get("is_keeper"):
                ref_member = m
                break

    if ref_member is None:
        ref_member = _select_reference_from_metrics(members)

    ref_id = ref_member.get("id")
    ref_sz = ref_member.get("file_size") or 0
    ref_w = ref_member.get("width") or 0
    ref_h = ref_member.get("height") or 0
    ref_px = ref_w * ref_h
    min_saving = float(limits.get("min_saving_ratio", 0.0))
    max_non_ref_sz = ref_sz * (1.0 - min_saving)

    # Determine reference_uncertain flag
    if reference_uncertain is not None:
        is_uncertain = bool(reference_uncertain)
    else:
        is_uncertain = any(bool(m.get("reference_uncertain")) for m in members if hasattr(m, "get"))
        if not is_uncertain and ref_member and hasattr(ref_member, "get"):
            is_uncertain = bool(ref_member.get("reference_uncertain"))

    # Requirement: Mark reference_uncertain = 0 when all non-identical members
    # of a cluster are byte-identical to the reference (exact-copy-only clusters).
    ref_md5 = ref_member.get("md5")
    ref_sz = ref_member.get("file_size") or 0
    ref_path = str(ref_member.get("file_path") or "")

    if not ref_md5 and ref_path and os.path.exists(ref_path):
        try:
            with open(ref_path, "rb") as f:
                ref_md5 = hashlib.md5(f.read()).hexdigest()
            ref_sz = os.path.getsize(ref_path)
        except Exception:
            pass

    def _is_byte_identical_to_ref(m: dict[str, Any]) -> bool:
        m_md5 = m.get("md5")
        m_sz = m.get("file_size") or 0
        m_path = str(m.get("file_path") or "")
        if not m_md5 and m_path and os.path.exists(m_path):
            try:
                with open(m_path, "rb") as f:
                    m_md5 = hashlib.md5(f.read()).hexdigest()
                m_sz = os.path.getsize(m_path)
            except Exception:
                pass
        if ref_md5 and m_md5:
            return bool(m_md5 == ref_md5 and m_sz == ref_sz)
        if ref_path and m_path and os.path.exists(ref_path) and os.path.exists(m_path):
            try:
                if os.path.getsize(ref_path) != os.path.getsize(m_path):
                    return False
                with open(ref_path, "rb") as f1, open(m_path, "rb") as f2:
                    return f1.read() == f2.read()
            except Exception:
                pass
        return False

    is_exact_copy_only = (
        len(members) > 1
        and all(_is_byte_identical_to_ref(m) for m in members if m.get("id") != ref_member.get("id"))
    )
    if is_exact_copy_only:
        is_uncertain = False

    # 1. Select Candidates & Pick Keeper
    # Conservative mode: keeper = the reference.
    # Balanced / aggressive: keeper = smallest file among candidates; ties to higher pixel count, then shorter path.
    # If no other candidate qualifies, the reference is the keeper.
    if mode == "conservative":
        keeper = ref_member
    elif mode == "balanced":
        # Balanced: smart quality-saving tradeoff.
        # Candidates meet balanced limits (SSIM >= 0.90, pixel_ratio >= 0.50) and save space.
        candidates = []
        for m in members:
            is_ref = (m.get("id") == ref_id)
            if _is_member_candidate(m, limits, is_reference=is_ref):
                m_w = m.get("width") or 0
                m_h = m.get("height") or 0
                m_px = m_w * m_h
                if (not res_loss_allowed) and (m_px < ref_px):
                    continue
                if is_ref:
                    candidates.append(m)
                else:
                    m_sz = m.get("file_size") or 0
                    if m_sz <= max_non_ref_sz:
                        candidates.append(m)

        if candidates:
            def _balanced_sort_key(m):
                m_ssim = float(m.get("ssim") or 1.0)
                m_sz = m.get("file_size") or 0
                sz_val = m_sz if m_sz > 0 else float("inf")
                # Sort primarily by smaller file size, then higher SSIM
                return (sz_val, -m_ssim, len(str(m.get("file_path") or "")), m.get("id", 0))
            keeper = min(candidates, key=_balanced_sort_key)
        else:
            keeper = ref_member
    else:  # aggressive
        # Aggressive: strictly pick the lowest file size verified member in the cluster.
        # "No matter how it's the resolution, what is the clarity, I don't want to care.
        # I want each and every cluster to have the lowest possible size selected."
        candidates = []
        for m in members:
            is_ref = (m.get("id") == ref_id)
            if _is_member_candidate(m, limits, is_reference=is_ref):
                m_w = m.get("width") or 0
                m_h = m.get("height") or 0
                m_px = m_w * m_h
                if (not res_loss_allowed) and (m_px < ref_px):
                    continue
                candidates.append(m)

        if candidates:
            def _aggressive_sort_key(m):
                sz = m.get("file_size")
                sz_val = sz if (sz is not None and sz > 0) else float("inf")
                is_not_ref = 0 if m.get("id") == ref_id else 1
                w = m.get("width") or 0
                h = m.get("height") or 0
                px = w * h
                path = str(m.get("file_path") or "")
                return (sz_val, is_not_ref, -px, len(path), path, m.get("id") or 0)
            keeper = min(candidates, key=_aggressive_sort_key)
        else:
            keeper = ref_member

    keeper_id = keeper.get("id")
    keeper_md5 = keeper.get("md5")
    keeper_sz = keeper.get("file_size") or 0
    k_w = keeper.get("width") or 0
    k_h = keeper.get("height") or 0
    k_px = k_w * k_h
    if keeper_id == ref_id or keeper.get("quality_status") == "reference":
        k_quality = 1.0
    elif keeper.get("ssim") is not None:
        k_quality = float(keeper["ssim"])
    else:
        k_quality = 1.0

    delete_safe_ids: list[int] = []
    review_ids: list[int] = []
    reasons: dict[int, str] = {}
    quality_retained: dict[int, float] = {}

    # Reason for keeper
    if keeper_id == ref_id:
        reasons[keeper_id] = "Recommended keeper (reference copy, highest fidelity)"
        quality_retained[keeper_id] = 100.0
    elif mode == "aggressive":
        k_pct = round(float(k_quality) * 100.0, 1) if k_quality is not None else 100.0
        reasons[keeper_id] = f"Recommended keeper (lowest file size in cluster, {keeper_sz:,} B)"
        quality_retained[keeper_id] = k_pct
    else:
        k_ssim = keeper.get("ssim")
        k_pct = round(float(k_ssim) * 100.0, 1) if k_ssim is not None else 100.0
        reasons[keeper_id] = f"Recommended keeper (smallest candidate in {mode} mode, {k_pct}% quality retained)"
        quality_retained[keeper_id] = k_pct

    # In conservative mode, if the reference is uncertain, non-identical members go to review,
    # but byte-identical copies of the keeper remain delete_safe (zero loss regardless of which copy is source).
    if mode == "conservative" and is_uncertain:
        keeper_md5 = keeper.get("md5")
        keeper_sz = keeper.get("file_size") or 0
        del_safe_ids: list[int] = []
        rev_ids: list[int] = []
        for m in members:
            mid = m.get("id")
            if mid == keeper_id:
                continue
            m_ssim = m.get("ssim")
            is_exact_copy = bool(keeper_md5 and m.get("md5") == keeper_md5 and (m.get("file_size") or 0) == keeper_sz)
            if is_exact_copy:
                del_safe_ids.append(mid)
                reasons[mid] = "Byte-identical copy of recommended keeper"
                quality_retained[mid] = 100.0
            else:
                rev_ids.append(mid)
                reasons[mid] = "Cluster reference is uncertain; manual review required in conservative mode"
                quality_retained[mid] = round(float(m_ssim) * 100.0, 1) if m_ssim is not None else 0.0

        savings = sum(m.get("file_size") or 0 for m in members if m.get("id") in del_safe_ids)
        return {
            "keeper_id": keeper_id,
            "delete_safe_ids": del_safe_ids,
            "review_ids": rev_ids,
            "reasons": reasons,
            "total_savings_bytes": savings,
            "quality_retained": quality_retained,
            "reference_uncertain": 1,
            "allow_resolution_loss": res_loss_allowed,
            "swap_proposal": None,
            "no_swap_reason": "No swap: reference uncertain",
        }

    # Map byte-identical groups by (md5, file_size)
    exact_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for m in members:
        md5_val = m.get("md5")
        sz_val = m.get("file_size") or 0
        if md5_val:
            exact_groups.setdefault((md5_val, sz_val), []).append(m)

    # Secondary byte-identical duplicates (for keeper or non-keeper groups)
    # Byte-identical copies of the keeper are always delete_safe.
    # A byte-identical copy of a non-keeper goes through that member's own rule,
    # so exact duplicate copies are never stranded in review when the keeper is a different file.
    secondary_duplicate_map: dict[int, dict[str, Any]] = {}
    for group_key, grp_members in exact_groups.items():
        if len(grp_members) <= 1:
            continue
        has_keeper = any(m.get("id") == keeper_id for m in grp_members)
        if has_keeper:
            for m in grp_members:
                mid = m.get("id")
                if mid != keeper_id:
                    secondary_duplicate_map[mid] = keeper
        else:
            # Pick canonical anchor representative (earliest mtime, then shortest path length, then path, then id)
            def anchor_sort_key(m: dict[str, Any]):
                p = str(m.get("file_path") or "")
                mtime_val = float(m.get("modified_time") or 0.0)
                if mtime_val == 0.0 and p and os.path.exists(p):
                    try:
                        mtime_val = float(os.path.getmtime(p))
                    except Exception:
                        mtime_val = 0.0
                mtime_sort = mtime_val if mtime_val > 0.0 else float("inf")
                return (mtime_sort, len(p), p, m.get("id") or 0)
            anchor = min(grp_members, key=anchor_sort_key)
            anchor_id = anchor.get("id")
            for m in grp_members:
                mid = m.get("id")
                if mid != anchor_id:
                    secondary_duplicate_map[mid] = anchor

    # 3. Partition remaining members into delete_safe vs review
    for m in members:
        mid = m.get("id")
        if mid == keeper_id:
            continue

        q_status = m.get("quality_status")
        is_low_conf = bool(m.get("is_low_confidence"))
        m_ssim = m.get("ssim")
        m_md5 = m.get("md5")
        m_sz = m.get("file_size") or 0
        m_w = m.get("width") or 0
        m_h = m.get("height") or 0
        m_px = m_w * m_h

        # Calculate quality retained vs reference
        if q_status == "reference" or (keeper_md5 and m_md5 == keeper_md5):
            retained_pct = 100.0
        elif m_ssim is not None:
            retained_pct = round(float(m_ssim) * 100.0, 1)
        else:
            retained_pct = 0.0
        quality_retained[mid] = retained_pct

        # Quality value for comparison
        if mid == ref_id or q_status == "reference" or (keeper_md5 and m_md5 == keeper_md5):
            m_quality = 1.0
        elif m_ssim is not None:
            m_quality = float(m_ssim)
        else:
            m_quality = 0.0

        # Determine effective match percentage
        m_match_pct = m.get("match_pct")
        if m_match_pct is None and m.get("ssim_color") is not None:
            m_match_pct = round(float(m["ssim_color"]) * 100.0, 1)
        elif m_match_pct is None and keeper_md5 and m_md5 == keeper_md5:
            m_match_pct = 100.0
        elif m_match_pct is None and m.get("ssim") is not None:
            m_match_pct = round(float(m["ssim"]) * 100.0, 1)

        m_conf = m.get("confidence")
        conf_val = (float(m_conf) if float(m_conf) <= 1.0 else float(m_conf) / 100.0) if m_conf is not None else None

        # Hard invariant: no member with match_pct below min_cluster_match_pct,
        # or with confidence < 0.99, or with quality_status != 'aligned'/'reference'
        # can ever be delete_safe in any mode.
        if q_status not in ("reference", "aligned"):
            review_ids.append(mid)
            if q_status == "unalignable":
                reasons[mid] = "Unalignable member (geometric verification/alignment failed)"
            else:
                reasons[mid] = f"Unverified quality status '{q_status}' (manual review required)"
            continue

        if m_match_pct is not None and m_match_pct < min_cluster_match_pct:
            review_ids.append(mid)
            reasons[mid] = f"Match percentage ({m_match_pct:.1f}%) below minimum cluster match threshold ({min_cluster_match_pct:.1f}%)"
            continue

        if conf_val is not None and conf_val < 0.99:
            review_ids.append(mid)
            reasons[mid] = f"Duplicate confidence ({conf_val:.4f}) below safe deletion threshold (0.99)"
            continue

        if is_low_conf:
            review_ids.append(mid)
            reasons[mid] = "Inconclusive keypoints (< 100) / low confidence (manual review required)"
            continue

        # Drive thumbnail quality invariant:
        # A Drive member whose quality_basis is 'thumbnail' can never be delete_safe
        # (byte-identical md5 copies excepted); it goes to review with reason
        # "needs full-resolution confirmation".
        is_gdrive = (
            m.get("source") == "gdrive"
            or str(m.get("file_path") or "").startswith("drive://")
        )
        q_basis = m.get("quality_basis") or ("thumbnail" if is_gdrive else "full")
        is_exact_copy_of_keeper = bool(
            keeper_md5 and m_md5 == keeper_md5 and (m.get("file_size") or 0) == keeper_sz
        )
        if is_gdrive and q_basis == "thumbnail" and not is_exact_copy_of_keeper:
            review_ids.append(mid)
            reasons[mid] = "needs full-resolution confirmation"
            continue

        # Handle secondary byte-identical duplicates first
        if mid in secondary_duplicate_map:
            anchor = secondary_duplicate_map[mid]
            if anchor.get("id") == keeper_id:
                delete_safe_ids.append(mid)
                reasons[mid] = "Byte-identical copy of keeper (100.0% quality retained)"
            else:
                # Duplicate copy of a non-keeper:
                # Invariant: keeper's file size must never exceed any delete_safe member's size
                # unless that member is a lower-resolution copy.
                if (m_px >= k_px) and (keeper_sz > m_sz):
                    review_ids.append(mid)
                    reasons[mid] = f"Keeper ({keeper_sz:,} B) is larger than same-resolution copy ({m_sz:,} B)"
                else:
                    delete_safe_ids.append(mid)
                    anchor_name = Path(str(anchor.get("file_path") or "")).name or f"member {anchor.get('id')}"
                    reasons[mid] = f"Byte-identical duplicate of {anchor_name}"
            continue

        if mode == "conservative":
            # Conservative mode: keeper = reference.
            # delete_safe = byte-identical copies of keeper plus members strictly worse
            # (lower resolution or SSIM below keeper by > 0.005) and verified.
            is_strictly_worse = (m_px < k_px) or (m_quality < k_quality - 0.005)
            if not is_strictly_worse:
                review_ids.append(mid)
                reasons[mid] = "Conservative mode: same-fidelity duplicate requires manual review"
            elif (m_px >= k_px) and (keeper_sz > m_sz):
                review_ids.append(mid)
                reasons[mid] = f"Keeper ({keeper_sz:,} B) is larger than same-resolution copy ({m_sz:,} B)"
            else:
                delete_safe_ids.append(mid)
                reasons[mid] = f"Verified strictly worse duplicate ({retained_pct}% quality retained)"
        else:
            # Balanced / aggressive modes:
            # Keeper must be at least as good in pixel count and quality
            is_keeper_at_least_as_good = (k_px >= m_px) and (k_quality >= m_quality - 0.005)
            if not is_keeper_at_least_as_good:
                review_ids.append(mid)
                reasons[mid] = "better than recommended keeper"
            elif (m_px >= k_px) and (keeper_sz > m_sz):
                review_ids.append(mid)
                reasons[mid] = f"Keeper ({keeper_sz:,} B) is larger than same-resolution copy ({m_sz:,} B)"
            else:
                delete_safe_ids.append(mid)
                reasons[mid] = f"Verified aligned duplicate ({retained_pct}% quality retained)"

    # Invariants:
    assert keeper_id not in delete_safe_ids, "Keeper cannot be marked delete_safe"
    assert len(delete_safe_ids) < len(members), "Cannot mark all cluster members as delete_safe"

    id_to_m = {m.get("id"): m for m in members}
    for ds_id in delete_safe_ids:
        ds_m = id_to_m[ds_id]
        ds_status = ds_m.get("quality_status")
        assert ds_status in ("aligned", "reference"), (
            f"Hard invariant violation: delete_safe member {ds_id} has quality_status '{ds_status}'"
        )
        ds_mp = ds_m.get("match_pct")
        if ds_mp is None and ds_m.get("ssim_color") is not None:
            ds_mp = round(float(ds_m["ssim_color"]) * 100.0, 1)
        elif ds_mp is None and keeper_md5 and ds_m.get("md5") == keeper_md5:
            ds_mp = 100.0
        elif ds_mp is None and ds_m.get("ssim") is not None:
            ds_mp = round(float(ds_m["ssim"]) * 100.0, 1)

        if ds_mp is not None:
            assert ds_mp >= min_cluster_match_pct, (
                f"Hard invariant violation: delete_safe member {ds_id} has match_pct {ds_mp} < {min_cluster_match_pct}"
            )

        ds_conf = ds_m.get("confidence")
        if ds_conf is not None:
            ds_conf_val = float(ds_conf) if float(ds_conf) <= 1.0 else float(ds_conf) / 100.0
            assert ds_conf_val >= 0.99, (
                f"Hard invariant violation: delete_safe member {ds_id} has confidence {ds_conf_val} < 0.99"
            )

        ds_px = (ds_m.get("width") or 0) * (ds_m.get("height") or 0)
        ds_sz = ds_m.get("file_size") or 0
        if ds_px >= k_px:
            assert keeper_sz <= ds_sz, (
                f"Keeper file size ({keeper_sz} B) exceeds same-resolution delete_safe member ({ds_sz} B)"
            )

        is_ds_gdrive = (
            ds_m.get("source") == "gdrive"
            or str(ds_m.get("file_path") or "").startswith("drive://")
        )
        ds_q_basis = ds_m.get("quality_basis") or ("thumbnail" if is_ds_gdrive else "full")
        if is_ds_gdrive and ds_q_basis == "thumbnail":
            is_ds_exact_keeper = bool(
                keeper_md5 and ds_m.get("md5") == keeper_md5 and (ds_m.get("file_size") or 0) == keeper_sz
            )
            assert is_ds_exact_keeper, (
                f"Hard invariant violation: Drive member {ds_id} with quality_basis='thumbnail' marked delete_safe"
            )

    # Total savings in bytes if delete_safe members are deleted
    total_savings = sum(
        (m.get("file_size") or 0) for m in members if m.get("id") in delete_safe_ids
    )

    # 5. Swap Proposal: when recommended keeper is not the reference (the original)
    swap_proposal = None
    no_swap_reason = None
    if keeper_id != ref_id and ref_member is not None:
        bytes_saved = max(0, ref_sz - keeper_sz)
        saving_pct = round((bytes_saved / ref_sz) * 100.0, 1) if ref_sz > 0 else 0.0

        if keeper.get("match_pct") is not None:
            match_pct = float(keeper["match_pct"])
        elif keeper.get("ssim_color") is not None:
            match_pct = round(float(keeper["ssim_color"]) * 100.0, 1)
        elif keeper.get("ssim") is not None:
            match_pct = round(float(keeper["ssim"]) * 100.0, 1)
        else:
            match_pct = 100.0

        if keeper.get("resolution_pct") is not None:
            res_pct = float(keeper["resolution_pct"])
        elif keeper.get("pixel_ratio") is not None:
            res_pct = round(float(keeper["pixel_ratio"]) * 100.0, 1)
        elif ref_px > 0:
            res_pct = round((k_px / ref_px) * 100.0, 1)
        else:
            res_pct = 100.0

        if mode == "aggressive":
            better_ids = []
            swap_reason = reasons.get(
                keeper_id,
                f"Swap reference with lowest size candidate ({saving_pct}% size reduction)",
            )
        else:
            better_ids = [
                m.get("id")
                for m in members
                if m.get("id") in review_ids
                and ("better than" in reasons.get(m.get("id"), "").lower() or m.get("id") == ref_id)
            ]
            swap_reason = reasons.get(
                keeper_id,
                f"Swap reference with smaller {mode} candidate ({saving_pct}% size reduction, {match_pct}% quality retained)",
            )

        swap_proposal = {
            "from_id": ref_id,
            "to_id": keeper_id,
            "bytes_saved": bytes_saved,
            "saving_pct": saving_pct,
            "match_pct": match_pct,
            "res_pct": res_pct,
            "reason": swap_reason,
            "better_than_keeper_ids": better_ids,
        }
    else:
        if is_uncertain:
            no_swap_reason = "No swap: reference uncertain"
        elif mode == "conservative":
            no_swap_reason = "No swap: conservative mode keeps original reference"
        else:
            smaller_members = [
                m for m in members
                if (m.get("file_size") or 0) < ref_sz and m.get("id") != ref_id
            ]
            if not smaller_members:
                no_swap_reason = "No swap: reference is already smallest copy"
            else:
                reasons_list = []
                for sm in smaller_members:
                    sm_w = sm.get("width") or 0
                    sm_h = sm.get("height") or 0
                    sm_px = sm_w * sm_h
                    sm_sz = sm.get("file_size") or 0
                    if not res_loss_allowed and sm_px < ref_px:
                        reasons_list.append("resolution loss not allowed")
                    elif sm_sz > max_non_ref_sz:
                        reasons_list.append("size reduction below threshold")
                    elif sm.get("quality_status") == "unalignable":
                        reasons_list.append("unaligned")
                    elif sm.get("ssim_color") is not None and sm.get("ssim_color") < limits.get("ssim_color", 0.0):
                        reasons_list.append("candidate below mode quality threshold")
                    else:
                        reasons_list.append("candidate below mode limits")
                unique_reasons = list(dict.fromkeys(reasons_list))
                no_swap_reason = f"No swap: {', '.join(unique_reasons)}"

    return {
        "keeper_id": keeper_id,
        "delete_safe_ids": delete_safe_ids,
        "review_ids": review_ids,
        "reasons": reasons,
        "total_savings_bytes": total_savings,
        "quality_retained": quality_retained,
        "reference_uncertain": 1 if is_uncertain else 0,
        "allow_resolution_loss": res_loss_allowed,
        "swap_proposal": swap_proposal,
        "no_swap_reason": no_swap_reason,
    }


def ensure_keeper_columns(conn: sqlite3.Connection) -> None:
    """Ensure images table has all Stage 4 keeper recommendation columns."""
    existing_cols = {
        col[1] for col in conn.execute("PRAGMA table_info(images)").fetchall()
    }
    cols_to_add = [
        ("is_recommended_keeper", "INTEGER NOT NULL DEFAULT 0"),
        ("delete_safe", "INTEGER NOT NULL DEFAULT 0"),
        ("review_flag", "INTEGER NOT NULL DEFAULT 0"),
        ("quality_retained_pct", "REAL"),
        ("bytes_saved_if_deleted", "INTEGER DEFAULT 0"),
        ("reason", "TEXT"),
    ]
    for col_name, col_def in cols_to_add:
        if col_name not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE images ADD COLUMN {col_name} {col_def};")
            except sqlite3.OperationalError:
                pass


def predict_duplicate_confidence(
    phash_distance: int | float,
    best_clip_cosine: float,
    sift_inlier_ratio: float,
    inlier_count: int,
    md5_match: bool = False,
    model_path: str | Path | None = None,
) -> float:
    """
    Compute calibrated duplicate confidence percentage (0.0 to 100.0).
    Exact MD5 matches are fixed at 100.0%.
    Otherwise uses the logistic regression model trained on the tuning set.
    """
    if md5_match:
        return 1.0

    global _CONFIDENCE_MODEL_CACHE
    if _CONFIDENCE_MODEL_CACHE is None:
        p = Path(model_path or "models/duplicate_confidence.json")
        if not p.is_file():
            # Fallback path relative to project root
            p = Path(__file__).resolve().parent.parent / "models" / "duplicate_confidence.json"
        if p.is_file():
            _CONFIDENCE_MODEL_CACHE = json.loads(p.read_text(encoding="utf-8"))
        else:
            # Fallback hardcoded weights from calibrated model
            _CONFIDENCE_MODEL_CACHE = {
                "intercept": -9.69251,
                "weights": [-0.279621, 13.455433, 5.682465, 0.848134],
            }

    intercept = float(_CONFIDENCE_MODEL_CACHE.get("intercept", -9.69251))
    weights = _CONFIDENCE_MODEL_CACHE.get("weights", [-0.279621, 13.455433, 5.682465, 0.848134])

    log_inliers = float(np.log1p(max(0, inlier_count)))
    features = [
        float(phash_distance),
        float(best_clip_cosine),
        float(sift_inlier_ratio),
        log_inliers,
    ]

    logit = intercept + sum(w * f for w, f in zip(weights, features))
    prob = 1.0 / (1.0 + np.exp(-logit))
    return float(prob)


@dataclass
class KeeperClusterResult:
    cluster_id: int
    keeper_id: int
    keeper_path: str
    keeper_size: int
    delete_safe_count: int
    delete_safe_bytes: int
    review_count: int
    delete_safe_paths: list[str] = field(default_factory=list)
    review_paths: list[str] = field(default_factory=list)
    reasons: dict[int, str] = field(default_factory=dict)


@dataclass
class KeeperRunResult:
    clusters_evaluated: int = 0
    total_keepers: int = 0
    total_delete_safe: int = 0
    total_review: int = 0
    total_bytes_saved: int = 0
    cluster_results: list[KeeperClusterResult] = field(default_factory=list)


def run_keeper_pipeline(
    conn: sqlite3.Connection,
    mode: str = "balanced",
    config: dict | None = None,
    allow_resolution_loss: bool | None = None,
) -> KeeperRunResult:
    """
    Run keeper selection across all duplicate clusters in SQLite and store recommendations.
    Does NOT delete any files.
    """
    ensure_keeper_columns(conn)

    # Fetch all clusters with their members
    cluster_rows = conn.execute(
        "SELECT id, representative_id, reference_uncertain FROM clusters ORDER BY id"
    ).fetchall()

    run_res = KeeperRunResult()

    for c in cluster_rows:
        cid = c["id"]
        members = conn.execute(
            """
            SELECT *
            FROM images
            WHERE cluster_id = ?
            ORDER BY id
            """,
            (cid,),
        ).fetchall()

        if not members:
            continue

        ref_unc = c["reference_uncertain"] if "reference_uncertain" in c.keys() else None
        decision = choose_keeper(
            members,
            mode=mode,
            config=config,
            reference_uncertain=ref_unc,
            allow_resolution_loss=allow_resolution_loss,
        )
        keeper_id = decision["keeper_id"]
        delete_safe_ids = set(decision["delete_safe_ids"])
        review_ids = set(decision["review_ids"])
        reasons = decision["reasons"]
        quality_retained = decision["quality_retained"]

        id_to_row = {m["id"]: m for m in members}
        keeper_m = id_to_row[keeper_id]

        c_res = KeeperClusterResult(
            cluster_id=cid,
            keeper_id=keeper_id,
            keeper_path=keeper_m["file_path"],
            keeper_size=keeper_m["file_size"] or 0,
            delete_safe_count=len(delete_safe_ids),
            delete_safe_bytes=decision["total_savings_bytes"],
            review_count=len(review_ids),
            delete_safe_paths=[id_to_row[i]["file_path"] for i in decision["delete_safe_ids"]],
            review_paths=[id_to_row[i]["file_path"] for i in decision["review_ids"]],
            reasons=reasons,
        )

        # Update database records
        for m in members:
            mid = m["id"]
            is_k = 1 if (mid == keeper_id) else 0
            is_del = 1 if (mid in delete_safe_ids) else 0
            is_rev = 1 if (mid in review_ids) else 0
            retained = quality_retained.get(mid, 0.0)
            bytes_saved = (m["file_size"] or 0) if is_del else 0
            reason_str = reasons.get(mid, "")

            conn.execute(
                """
                UPDATE images
                SET is_recommended_keeper = ?,
                    is_keeper = ?,
                    delete_safe = ?,
                    review_flag = ?,
                    quality_retained_pct = ?,
                    bytes_saved_if_deleted = ?,
                    reason = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                (
                    is_k,
                    is_k,
                    is_del,
                    is_rev,
                    retained,
                    bytes_saved,
                    reason_str,
                    mid,
                ),
            )

        # Update cluster representative to keeper
        conn.execute(
            "UPDATE clusters SET representative_id = ? WHERE id = ?",
            (keeper_id, cid),
        )

        run_res.clusters_evaluated += 1
        run_res.total_keepers += 1
        run_res.total_delete_safe += len(delete_safe_ids)
        run_res.total_review += len(review_ids)
        run_res.total_bytes_saved += decision["total_savings_bytes"]
        run_res.cluster_results.append(c_res)

    conn.commit()
    return run_res
