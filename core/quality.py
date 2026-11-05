"""
core/quality.py — Stage 4: Proxy-based geometric alignment and image quality scoring.

For each duplicate cluster:
1. Chooses a reference member:
   - Primary: Highest pixel count (width * height, descending)
   - Secondary: Least compressed (file_size / (width * height) bytes/pixel, descending)
   - Tertiary: Shortest path length (len(file_path), ascending)
   - Quaternary: Deterministic tie-breaker (file_path, ascending)
2. Aligns every other member to the reference using geometric transform from Stage 3
   (cached in pair_transforms; if missing, recomputed on ~1024px proxies).
   Handles rotation and rescale. If alignment fails, marks member as 'unalignable'
   (which is never eligible for deletion).
3. Computes on ~1024px proxies:
   - SSIM via scikit-image (grayscale and color-aware variants)
   - Sharpness ratio (Laplacian variance of member / reference)
   - Pixel ratio (member pixels / reference pixels)
   - Bytes ratio (member file size / reference file size)
4. Persists scores and quality_status to SQLite images table.
   Keeper selection is deferred to Stage 4.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import hashlib
import io
import logging
import os
from pathlib import Path
import pickle
import re
import sqlite3
import time
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from core.verify import (
    cache_pair_transform,
    full_matrix_to_proxy,
    proxy_matrix_to_full,
)
from sources.base import ImageSource

logger = logging.getLogger("argus.quality")


@dataclass
class QualityMetrics:
    quality_status: str       # "reference" | "aligned" | "unalignable"
    ssim: float | None = None
    ssim_color: float | None = None
    sharpness_ratio: float = 0.0
    pixel_ratio: float = 0.0
    bytes_ratio: float = 0.0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    blur_flag: bool = False


@dataclass
class MemberQualityRecord:
    member_id: int
    cluster_id: int
    file_path: str
    is_reference: bool
    quality_status: str
    ssim: float | None
    ssim_color: float | None
    sharpness_ratio: float
    pixel_ratio: float
    bytes_ratio: float
    width: int
    height: int
    file_size: int
    source_likelihood: float | None = None
    match_pct: float | None = None
    resolution_pct: float | None = None


@dataclass
class QualityResult:
    clusters_evaluated: int = 0
    members_evaluated: int = 0
    members_aligned: int = 0
    members_unalignable: int = 0
    elapsed_seconds: float = 0.0
    member_records: list[MemberQualityRecord] = field(default_factory=list)
    slowest_clusters: list[tuple[int, float]] = field(default_factory=list)


LOSSLESS_FORMATS = frozenset({"PNG", "TIFF", "TIF", "BMP"})


def _is_lossless_member(m: sqlite3.Row | dict[str, Any]) -> bool:
    fmt = m["format"] if ("format" in m.keys() and m["format"]) else ""
    if not fmt:
        p = str(m["file_path"])
        fmt = Path(p).suffix.lstrip(".").upper()
    return fmt.upper() in LOSSLESS_FORMATS


def _member_content_key(m: sqlite3.Row | dict[str, Any]) -> Any:
    md5_val = None
    sz_val = 0
    if isinstance(m, dict):
        md5_val = m.get("md5")
        sz_val = m.get("file_size") or 0
        p = m.get("file_path")
    elif hasattr(m, "keys"):
        if "md5" in m.keys():
            md5_val = m["md5"]
        if "file_size" in m.keys():
            sz_val = m["file_size"] or 0
        p = m["file_path"] if "file_path" in m.keys() else None
    else:
        p = None

    if md5_val is None and p and os.path.exists(str(p)):
        try:
            with open(str(p), "rb") as f:
                md5_val = hashlib.md5(f.read()).hexdigest()
            sz_val = os.path.getsize(str(p))
        except Exception:
            pass

    if md5_val is not None:
        return (md5_val, sz_val)
    mid = m["id"] if ("id" in m.keys() and m["id"] is not None) else (m["file_path"] if "file_path" in m.keys() else None)
    return mid



# Standard IJG 50% luminance quantization table (8x8)
IJG_LUMINANCE_TABLE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
], dtype=np.float32)


def _generate_ijg_table(q: int) -> np.ndarray:
    s = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    t = np.floor((IJG_LUMINANCE_TABLE * s + 50.0) / 100.0)
    return np.clip(t, 1.0, 255.0)


# Precomputed (100, 64) matrix for instant vector comparison across quality levels 1-100
_IJG_TABLES_MATRIX = np.stack([_generate_ijg_table(q) for q in range(1, 101)], axis=0)


def estimate_jpeg_quality_from_table(q_table: list[int] | np.ndarray) -> int:
    """Estimate JPEG quality factor (1-100) from luminance quantization table."""
    arr = np.array(q_table[:64], dtype=np.float32)
    diffs = np.mean(np.abs(_IJG_TABLES_MATRIX - arr), axis=1)
    return int(np.argmin(diffs) + 1)


def extract_jpeg_quality_estimate(image_path: str | Path | None = None, im: Image.Image | None = None) -> int | None:
    """
    Extract estimated JPEG quality from image quantization tables.
    Returns integer 1-100 for JPEG, or None for non-JPEG/unreadable images.
    """
    try:
        if im is not None:
            if getattr(im, "format", None) not in ("JPEG", "MPO"):
                return None
            if hasattr(im, "quantization") and im.quantization and 0 in im.quantization:
                return estimate_jpeg_quality_from_table(im.quantization[0])
            return None

        if image_path is not None and os.path.exists(str(image_path)):
            with Image.open(str(image_path)) as opened_im:
                if opened_im.format not in ("JPEG", "MPO"):
                    return None
                if hasattr(opened_im, "quantization") and opened_im.quantization and 0 in opened_im.quantization:
                    return estimate_jpeg_quality_from_table(opened_im.quantization[0])
        return None
    except Exception:
        return None


def check_exif_metadata(image_path: str | Path | None = None, im: Image.Image | None = None) -> tuple[bool, bool]:
    """
    Check for presence of EXIF capture time and camera make.
    Returns (has_capture_time, has_camera_make).
    """
    try:
        exif = None
        if im is not None:
            exif = im.getexif()
        elif image_path is not None and os.path.exists(str(image_path)):
            with Image.open(str(image_path)) as opened_im:
                exif = opened_im.getexif()

        if not exif:
            return False, False

        # Primary IFD tags:
        # DateTimeOriginal (36867), DateTimeDigitized (36868), DateTime (306)
        # Make (271), Model (272)
        has_time = any(k in exif for k in (36867, 36868, 306))
        has_make = any(k in exif for k in (271, 272))

        # Check ExifOffset sub-IFD (34665)
        if hasattr(exif, "get_ifd"):
            try:
                sub_exif = exif.get_ifd(34665)
                if sub_exif:
                    if not has_time:
                        has_time = any(k in sub_exif for k in (36867, 36868, 306))
                    if not has_make:
                        has_make = any(k in sub_exif for k in (271, 272))
            except Exception:
                pass

        return bool(has_time), bool(has_make)
    except Exception:
        return False, False


def _get_exif_score(m: dict[str, Any] | sqlite3.Row) -> int:
    d = dict(m) if not isinstance(m, dict) else m
    has_time = d.get("has_exif_time")
    has_make = d.get("has_exif_make")
    p = str(d.get("file_path") or "")
    if (has_time is None or has_make is None) and p and os.path.exists(p):
        has_time, has_make = check_exif_metadata(p)
    return int(bool(has_time)) + int(bool(has_make))


def _get_mtime(m: dict[str, Any] | sqlite3.Row) -> float:
    d = dict(m) if not isinstance(m, dict) else m
    mtime = float(d["modified_time"]) if d.get("modified_time") else 0.0
    p = str(d.get("file_path") or "")
    if mtime == 0.0 and p and os.path.exists(p):
        try:
            mtime = float(os.path.getmtime(p))
        except Exception:
            mtime = 0.0
    return mtime


def _get_jpeg_quality(m: dict[str, Any] | sqlite3.Row) -> int:
    d = dict(m) if not isinstance(m, dict) else m
    q_est = d.get("jpeg_quality_est")
    p = str(d.get("file_path") or "")
    lossless = _is_lossless_member(m)
    if q_est is None and not lossless and p and os.path.exists(p):
        q_est = extract_jpeg_quality_estimate(p)
    return int(q_est) if q_est is not None else 0


def _calc_proxy_ssim(g1: np.ndarray, g2: np.ndarray) -> float:
    """Calculate SSIM between two proxy gray images, handling rotation and aspect ratio."""
    if g1.shape == g2.shape:
        return float(ssim(g1, g2, data_range=255))
    if g1.shape == g2.shape[::-1]:
        rot_90 = cv2.rotate(g2, cv2.ROTATE_90_CLOCKWISE)
        rot_270 = cv2.rotate(g2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return max(
            float(ssim(g1, rot_90, data_range=255)),
            float(ssim(g1, rot_270, data_range=255)),
        )
    cw = min(g1.shape[1], g2.shape[1])
    ch = min(g1.shape[0], g2.shape[0])
    gi_r = cv2.resize(g1, (cw, ch), interpolation=cv2.INTER_AREA)
    gj_r = cv2.resize(g2, (cw, ch), interpolation=cv2.INTER_AREA)
    return float(ssim(gi_r, gj_r, data_range=255))


def _load_image_and_qtables(
    m: dict[str, Any] | sqlite3.Row,
    source: ImageSource | None = None,
) -> tuple[Image.Image | None, dict[int, list[int]] | None]:
    """Load Pillow Image and its quantization tables for an image member."""
    d = dict(m) if not isinstance(m, dict) else m

    qtables = d.get("qtables") or d.get("quantization")
    custom_im = d.get("image")
    if isinstance(custom_im, Image.Image):
        return custom_im.convert("RGB"), qtables or getattr(custom_im, "quantization", None)

    p = str(d.get("file_path") or "")
    if not p:
        return None, None

    raw_im = None
    if source is not None:
        try:
            raw_bytes = source.get_bytes(p)
            raw_im = Image.open(io.BytesIO(raw_bytes))
        except Exception:
            raw_im = None

    if raw_im is None and os.path.exists(p):
        try:
            raw_im = Image.open(p)
        except Exception:
            raw_im = None

    if raw_im is None:
        return None, None

    if qtables is None:
        qtables = getattr(raw_im, "quantization", None)

    if not qtables:
        return None, None

    im_rgb = raw_im.convert("RGB")
    return im_rgb, qtables


def run_jpeg_derivation_test(
    m_a: dict[str, Any] | sqlite3.Row,
    m_b: dict[str, Any] | sqlite3.Row,
    source: ImageSource | None = None,
) -> tuple[float | None, float | None, str | None]:
    """
    Run JPEG derivation test between two same-resolution lossy candidates A and B:
    1. Re-encode A using B's quantization tables (via Pillow's qtables parameter) -> A_to_B.
    2. Re-encode B using A's quantization tables -> B_to_A.
    3. Measure SSIM of each re-encode against the other original file:
       score_a_to_b = SSIM(A_to_B, B)
       score_b_to_a = SSIM(B_to_A, A)
    4. If one direction reproduces the target markedly better (gap >= 0.005), that
       direction indicates "B derives from A"; A is the source.
    Returns:
       (score_a_to_b, score_b_to_a, winner) where winner is 'A', 'B', or None.
    """
    im_a, q_a = _load_image_and_qtables(m_a, source=source)
    im_b, q_b = _load_image_and_qtables(m_b, source=source)

    if im_a is None or im_b is None or not q_a or not q_b:
        return None, None, None

    if im_a.size != im_b.size:
        return None, None, None

    try:
        buf_a_to_b = io.BytesIO()
        im_a.save(buf_a_to_b, format="JPEG", qtables=q_b)
        buf_a_to_b.seek(0)
        im_a_to_b = Image.open(buf_a_to_b).convert("L")

        buf_b_to_a = io.BytesIO()
        im_b.save(buf_b_to_a, format="JPEG", qtables=q_a)
        buf_b_to_a.seek(0)
        im_b_to_a = Image.open(buf_b_to_a).convert("L")

        arr_a = np.array(im_a.convert("L"))
        arr_b = np.array(im_b.convert("L"))
        arr_a_to_b = np.array(im_a_to_b)
        arr_b_to_a = np.array(im_b_to_a)

        score_a_to_b = round(float(ssim(arr_a_to_b, arr_b, data_range=255)), 6)
        score_b_to_a = round(float(ssim(arr_b_to_a, arr_a, data_range=255)), 6)

        gap = round(score_a_to_b - score_b_to_a, 6)
        if gap >= 0.005:
            winner = "A"
        elif -gap >= 0.005:
            winner = "B"
        else:
            winner = None

        return score_a_to_b, score_b_to_a, winner
    except Exception:
        return None, None, None


def _get_proxy_gray(
    m: sqlite3.Row | dict[str, Any],
    source: ImageSource | None = None,
    proxy_size: int = 1024,
    cache: dict[Any, np.ndarray | None] | None = None,
) -> np.ndarray | None:
    """Load and return gray proxy for an image member, caching result if cache dict is provided."""
    mid = m["id"] if ("id" in m.keys() and m["id"] is not None) else m.get("file_path")
    if cache is not None and mid in cache:
        return cache[mid]
    p = m.get("file_path")
    if not p:
        if cache is not None:
            cache[mid] = None
        return None
    try:
        if source is not None or os.path.exists(str(p)):
            arr, _, _ = load_proxy_rgb(p, proxy_size=proxy_size, source=source)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            if cache is not None:
                cache[mid] = gray
            return gray
    except Exception:
        pass
    if cache is not None:
        cache[mid] = None
    return None


def compute_source_likelihood_scores(
    members: list[sqlite3.Row | dict[str, Any]],
    source: ImageSource | None = None,
    proxy_size: int = 1024,
    max_comparisons: int = 40,
    out_info: dict[str, Any] | None = None,
) -> dict[Any, float | None]:
    """
    Compute source-likelihood score for members:
    For same-resolution lossy members, compute the mean SSIM of each member against the others
    using ~1024px proxies, caching pairwise SSIM results.
    For clusters with >8 members, caps pairwise comparisons (default 40).
    Returns a mapping from member id (or file_path) to float score (or None if no same-res peers).
    """
    scores: dict[Any, float | None] = {}
    if not members:
        return scores

    # If all members already have precomputed source_likelihood, reuse it
    all_precomputed = True
    for m in members:
        mid = m["id"] if ("id" in m.keys() and m["id"] is not None) else m.get("file_path")
        val = m.get("source_likelihood") if "source_likelihood" in m.keys() else None
        if val is None:
            all_precomputed = False
            break
        scores[mid] = float(val)
    if all_precomputed:
        return scores

    # Group members by resolution (width, height)
    by_res: dict[tuple[int, int], list[sqlite3.Row | dict[str, Any]]] = {}
    for m in members:
        w = m.get("width") or 0
        h = m.get("height") or 0
        by_res.setdefault((w, h), []).append(m)

    gray_cache: dict[Any, np.ndarray | None] = {}
    ssim_cache: dict[tuple[Any, Any], float] = {}
    total_comparisons = 0

    for (w, h), group in by_res.items():
        lossy_members = [m for m in group if not _is_lossless_member(m)]

        # Collapse byte-identical members (same md5 and file_size) into distinct candidates
        content_groups: dict[Any, list[sqlite3.Row | dict[str, Any]]] = {}
        for m in lossy_members:
            ck = _member_content_key(m)
            content_groups.setdefault(ck, []).append(m)

        def candidate_rep_key(m: sqlite3.Row | dict[str, Any]) -> tuple:
            p = str(m["file_path"] if "file_path" in m.keys() else "")
            mtime_val = _get_mtime(m)
            mtime_sort = mtime_val if mtime_val > 0.0 else float("inf")
            mid = m["id"] if ("id" in m.keys() and m["id"] is not None) else 0
            return (mtime_sort, len(p), p, mid)

        group_rep_map: dict[Any, sqlite3.Row | dict[str, Any]] = {}
        distinct_lossy: list[sqlite3.Row | dict[str, Any]] = []
        for ck, grp in content_groups.items():
            rep = min(grp, key=candidate_rep_key)
            group_rep_map[ck] = rep
            distinct_lossy.append(rep)

        if len(distinct_lossy) < 2:
            for m in group:
                mid = m["id"] if ("id" in m.keys() and m["id"] is not None) else m.get("file_path")
                if m.get("source_likelihood") is not None:
                    scores[mid] = float(m["source_likelihood"])
                else:
                    scores[mid] = None
            continue

        valid_lossy: list[tuple[Any, np.ndarray]] = []
        for m in distinct_lossy:
            mid = m["id"] if ("id" in m.keys() and m["id"] is not None) else m.get("file_path")
            g = _get_proxy_gray(m, source=source, proxy_size=proxy_size, cache=gray_cache)
            if g is not None:
                valid_lossy.append((mid, g))
            else:
                existing = m.get("source_likelihood")
                scores[mid] = float(existing) if existing is not None else None

        if len(valid_lossy) >= 2:
            cap_hit = False
            for i in range(len(valid_lossy)):
                id_i, g_i = valid_lossy[i]
                for j in range(i + 1, len(valid_lossy)):
                    id_j, g_j = valid_lossy[j]
                    cache_key = (id_i, id_j)
                    if cache_key not in ssim_cache:
                        if len(members) > 8 and total_comparisons >= max_comparisons:
                            cap_hit = True
                            if out_info is not None:
                                out_info["cap_hit"] = True
                            break
                        sim = _calc_proxy_ssim(g_i, g_j)
                        ssim_cache[(id_i, id_j)] = sim
                        ssim_cache[(id_j, id_i)] = sim
                        total_comparisons += 1
                if cap_hit:
                    break

            if out_info is not None:
                out_info["total_comparisons"] = total_comparisons

            rep_scores: dict[Any, float | None] = {}
            for id_i, _ in valid_lossy:
                peer_sims = [ssim_cache[(id_i, id_j)] for id_j, _ in valid_lossy if (id_i, id_j) in ssim_cache]
                if peer_sims:
                    rep_scores[id_i] = round(float(np.mean(peer_sims)), 6)
                else:
                    rep_scores[id_i] = None

            # Distribute scores to all members in each content group
            for ck, grp in content_groups.items():
                rep = group_rep_map[ck]
                rep_id = rep["id"] if ("id" in rep.keys() and rep["id"] is not None) else rep.get("file_path")
                grp_score = rep_scores.get(rep_id)
                for m in grp:
                    mid = m["id"] if ("id" in m.keys() and m["id"] is not None) else m.get("file_path")
                    scores[mid] = grp_score
        elif len(valid_lossy) == 1:
            for ck, grp in content_groups.items():
                for m in grp:
                    mid = m["id"] if ("id" in m.keys() and m["id"] is not None) else m.get("file_path")
                    scores[mid] = None

        for m in group:
            if _is_lossless_member(m):
                mid = m["id"] if ("id" in m.keys() and m["id"] is not None) else m.get("file_path")
                scores[mid] = float(m["source_likelihood"]) if m.get("source_likelihood") is not None else None

    return scores


def select_reference_member(
    members: list[sqlite3.Row | dict[str, Any]],
    source: ImageSource | None = None,
    conn: sqlite3.Connection | None = None,
    max_comparisons: int = 40,
) -> dict[str, Any]:
    """
    Select the reference member for a duplicate cluster:
    1. Highest pixel count (width * height, descending)
    2. Lossless format only if no lossy twin in cluster of same resolution
    3. Source-likelihood score: mean SSIM of each member against the others in same-res lossy pool (descending)
    4. Presence of EXIF capture time and camera make (descending)
    5. Earliest modified time (ascending)
    6. Highest jpeg_quality_est (descending, tiebreaker)
    7. Filename rule: canonical non-derived before copy suffixes, then shortest path
    8. Smallest file size (ascending)
    9. Deterministic tie-breakers: file_path ascending, ID ascending

    If top two candidates' source-likelihood scores differ by < 0.002, or EXIF/mtime
    contradict the SSIM ranking, or comparison cap was hit on >8 members, sets reference_uncertain = 1 on the cluster.
    """
    if not members:
        raise ValueError("Cannot select reference from empty member list")

    member_dicts = [dict(m) for m in members]

    # Map each pixel count to whether there is any lossy member with that pixel count
    pixel_has_lossy: dict[int, bool] = {}
    for m in member_dicts:
        px = (m.get("width") or 0) * (m.get("height") or 0)
        if not _is_lossless_member(m):
            pixel_has_lossy[px] = True

    # 1. Compute source-likelihood scores
    sl_info: dict[str, Any] = {}
    sl_scores = compute_source_likelihood_scores(
        member_dicts,
        source=source,
        max_comparisons=max_comparisons,
        out_info=sl_info,
    )
    for m in member_dicts:
        mid = m["id"] if ("id" in m.keys() and m["id"] is not None) else m.get("file_path")
        m["source_likelihood"] = sl_scores.get(mid)

    # 2. Identify top candidate pool at max resolution
    max_px = max(((m.get("width") or 0) * (m.get("height") or 0)) for m in member_dicts)
    cands_at_max = [m for m in member_dicts if ((m.get("width") or 0) * (m.get("height") or 0)) == max_px]

    has_lossy_twin = pixel_has_lossy.get(max_px, False)
    lossless_cands = [m for m in cands_at_max if _is_lossless_member(m)]
    lossy_cands = [m for m in cands_at_max if not _is_lossless_member(m)]

    if lossless_cands and not has_lossy_twin:
        top_cands = lossless_cands
    else:
        top_cands = lossy_cands if lossy_cands else cands_at_max

    # 3. Sorting key_fn for reference selection
    def key_fn(m: dict[str, Any]) -> tuple:
        w = m.get("width") or 0
        h = m.get("height") or 0
        px = w * h
        sz = m.get("file_size") or 0
        p = str(m.get("file_path") or "")
        m_id = m.get("id") or 0

        lossless = _is_lossless_member(m)
        has_lossy = pixel_has_lossy.get(px, False)
        prefer_lossless = (lossless and not has_lossy)
        lossless_rank = 0 if prefer_lossless else 1

        src_score = float(m.get("source_likelihood") or 0.0)
        exif_score = _get_exif_score(m)
        mtime = _get_mtime(m)
        q_val = _get_jpeg_quality(m)

        stem = Path(p).stem
        is_derived = 1 if re.search(r"(__|_copy|\bcopy\b|\s\(\d+\)|_resave)", stem, re.IGNORECASE) else 0
        path_len = len(p)

        return (
            -px,
            lossless_rank,
            -src_score,
            -exif_score,
            mtime,
            -q_val,
            is_derived,
            path_len,
            sz,
            p,
            m_id,
        )

    # 4. Collapse byte-identical members into distinct candidates before uncertainty check
    top_content_groups: dict[Any, list[dict[str, Any]]] = {}
    for m in top_cands:
        ck = _member_content_key(m)
        top_content_groups.setdefault(ck, []).append(m)

    distinct_top_cands = [min(grp, key=key_fn) for grp in top_content_groups.values()]

    # 5. Check reference_uncertain on distinct candidates
    reference_uncertain = 0
    top_candidate_gap: float | None = None
    head_to_head_result: str | None = None
    is_pairwise_consistent: bool | None = None

    if len(distinct_top_cands) >= 2:
        top_candidates = sorted(distinct_top_cands, key=key_fn)
        c1 = top_candidates[0]
        c2 = top_candidates[1]
        s1 = c1.get("source_likelihood")
        s2 = c2.get("source_likelihood")

        if s1 is not None and s2 is not None:
            gap = round(abs(float(s1) - float(s2)), 6)
            top_candidate_gap = gap
            is_close_score = gap < 0.002
        else:
            gap = None
            is_close_score = False

        # Head-to-head pairwise ordering consistency against other distinct cluster members
        # A candidate beats the other if it has higher SSIM against the majority of other members
        ck1 = _member_content_key(c1)
        ck2 = _member_content_key(c2)

        other_groups: dict[Any, list[dict[str, Any]]] = {}
        for m in member_dicts:
            ck = _member_content_key(m)
            if ck != ck1 and ck != ck2:
                other_groups.setdefault(ck, []).append(m)

        other_peers = [min(grp, key=key_fn) for grp in other_groups.values()]

        c1_wins = 0
        c2_wins = 0
        peer_count = 0

        ref_gray_cache: dict[Any, np.ndarray | None] = {}
        g1 = _get_proxy_gray(c1, source=source, cache=ref_gray_cache)
        g2 = _get_proxy_gray(c2, source=source, cache=ref_gray_cache)

        if g1 is not None and g2 is not None and other_peers:
            for peer in other_peers:
                gp = _get_proxy_gray(peer, source=source, cache=ref_gray_cache)
                if gp is not None:
                    sim1 = _calc_proxy_ssim(g1, gp)
                    sim2 = _calc_proxy_ssim(g2, gp)
                    peer_count += 1
                    if sim1 > sim2 + 1e-6:
                        c1_wins += 1
                    elif sim2 > sim1 + 1e-6:
                        c2_wins += 1

        c1_name = Path(str(c1.get("file_path") or "")).name or f"id_{c1.get('id')}"
        c2_name = Path(str(c2.get("file_path") or "")).name or f"id_{c2.get('id')}"

        # Always run derivation test between top two distinct candidates (available for reporting)
        score_a_to_b, score_b_to_a, deriv_winner = run_jpeg_derivation_test(c1, c2, source=source)
        deriv_gap = round(score_a_to_b - score_b_to_a, 6) if (score_a_to_b is not None and score_b_to_a is not None) else None

        # Check structural / hardware differentiation
        res1 = ((c1.get("width") or 0), (c1.get("height") or 0))
        res2 = ((c2.get("width") or 0), (c2.get("height") or 0))
        has_diff_res = (res1 != res2)
        fmt1 = str(c1.get("format") or "").upper()
        fmt2 = str(c2.get("format") or "").upper()
        has_diff_fmt = bool(fmt1 and fmt2 and fmt1 != fmt2)
        exif1_c = _get_exif_score(c1)
        exif2_c = _get_exif_score(c2)
        has_single_exif_support = (exif1_c > 0 and exif2_c == 0)

        if deriv_winner in ("A", "B"):
            if deriv_winner == "B":
                # Swap c1 and c2 so that the true source c2 becomes c1
                c1, c2 = c2, c1
                c1_name = Path(str(c1.get("file_path") or "")).name or f"id_{c1.get('id')}"
                c2_name = Path(str(c2.get("file_path") or "")).name or f"id_{c2.get('id')}"

            # Condition 2: EXIF contradiction (c2 has EXIF while c1 lacks it)
            exif1 = _get_exif_score(c1)
            exif2 = _get_exif_score(c2)
            is_exif_contradiction = (exif2 > exif1)

            # Condition 3: mtime contradiction (c2 has earlier modified time than c1)
            mtime1 = _get_mtime(c1)
            mtime2 = _get_mtime(c2)
            is_mtime_contradiction = (mtime1 > 0.0 and mtime2 > 0.0 and mtime2 < mtime1)

            if peer_count > 0:
                is_pairwise_consistent = (c1_wins > c2_wins)
            else:
                is_pairwise_consistent = True

            if (is_close_score and not is_pairwise_consistent) or is_exif_contradiction or is_mtime_contradiction:
                reference_uncertain = 1
            else:
                reference_uncertain = 0

            head_to_head_result = (
                f"{c1_name} -> {c2_name}: "
                f"{(score_a_to_b if deriv_winner == 'A' else score_b_to_a):.4f} vs "
                f"{(score_b_to_a if deriv_winner == 'A' else score_a_to_b):.4f} "
                f"(gap: {abs(deriv_gap):+.4f}) [derivation confirmed {c1_name}]"
            )
        else:
            # Derivation test has no winner (gap < 0.005 or test unavailable)
            # Clusters where derivation gap is below 0.005 must not become confident from mtime or majority vote alone.
            if has_diff_res or has_diff_fmt or has_single_exif_support:
                exif1 = _get_exif_score(c1)
                exif2 = _get_exif_score(c2)
                is_exif_contradiction = (exif2 > exif1)

                mtime1 = _get_mtime(c1)
                mtime2 = _get_mtime(c2)
                is_mtime_contradiction = (mtime1 > 0.0 and mtime2 > 0.0 and mtime2 < mtime1)

                if peer_count > 0:
                    is_pairwise_consistent = (c1_wins > c2_wins)
                else:
                    is_pairwise_consistent = True

                if (is_close_score and not is_pairwise_consistent) or is_exif_contradiction or is_mtime_contradiction:
                    reference_uncertain = 1
                else:
                    reference_uncertain = 0
            else:
                # Same resolution, same format, no single EXIF support
                # Set reference_uncertain = 1.
                # Majority vote may still pick the reference (ref = min(member_dicts, key=key_fn)),
                # but must NOT clear the flag.
                reference_uncertain = 1
                if peer_count > 0:
                    is_pairwise_consistent = (c1_wins > c2_wins)
                else:
                    is_pairwise_consistent = False

            if peer_count > 0:
                head_to_head_result = f"{c1_name} vs {c2_name}: {c1_wins} vs {c2_wins} ({peer_count} peers)"
                if score_a_to_b is not None and score_b_to_a is not None:
                    head_to_head_result += f" [derivation gap: {deriv_gap:+.4f}, uncertain]"
            else:
                if has_single_exif_support:
                    head_to_head_result = f"{c1_name} vs {c2_name}: confident (EXIF support on single file)"
                elif score_a_to_b is not None and score_b_to_a is not None:
                    head_to_head_result = (
                        f"{c1_name} -> {c2_name}: {score_a_to_b:.4f} vs {score_b_to_a:.4f} "
                        f"(gap: {deriv_gap:+.4f}) [no derivation winner, uncertain]"
                    )
                else:
                    head_to_head_result = f"{c1_name} vs {c2_name}: no derivation data (uncertain)"

    if len(distinct_top_cands) >= 2 and deriv_winner in ("A", "B"):
        ref = c1
    else:
        ref = min(member_dicts, key=key_fn)

    if sl_info.get("cap_hit"):
        reference_uncertain = 1

    # Requirement: Mark reference_uncertain = 0 when all non-identical members
    # of a cluster are byte-identical to the reference (exact-copy-only clusters).
    ref_ck = _member_content_key(ref)
    if len(member_dicts) > 1 and all(_member_content_key(m) == ref_ck for m in member_dicts):
        reference_uncertain = 0

    ref["reference_uncertain"] = reference_uncertain
    ref["top_candidate_gap"] = top_candidate_gap
    ref["head_to_head_result"] = head_to_head_result
    ref["is_pairwise_consistent"] = is_pairwise_consistent
    ref["derivation_scores"] = (score_a_to_b, score_b_to_a) if len(distinct_top_cands) >= 2 else (None, None)
    ref["derivation_gap"] = deriv_gap if len(distinct_top_cands) >= 2 else None
    ref["derivation_winner"] = deriv_winner if len(distinct_top_cands) >= 2 else None

    # Update input dicts in-place if callers passed dict objects
    for orig_m, d in zip(members, member_dicts):
        if isinstance(orig_m, dict):
            orig_m["source_likelihood"] = d.get("source_likelihood")
            orig_m["reference_uncertain"] = reference_uncertain
            orig_m["top_candidate_gap"] = top_candidate_gap
            orig_m["head_to_head_result"] = head_to_head_result
            orig_m["is_pairwise_consistent"] = is_pairwise_consistent
            orig_m["derivation_scores"] = ref.get("derivation_scores")
            orig_m["derivation_gap"] = ref.get("derivation_gap")
            orig_m["derivation_winner"] = ref.get("derivation_winner")

    return ref


def compute_laplacian_sharpness(gray_img: np.ndarray) -> float:
    """Compute variance of the Laplacian as a measure of focus / sharpness."""
    if gray_img.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


def compute_common_scale_sharpness(
    gray_member: np.ndarray,
    gray_ref: np.ndarray,
    blur_threshold: float = 0.70,
) -> tuple[float, bool]:
    """
    Compute sharpness ratio between member and reference at a common resolution.
    Resizes both to the smaller dimension (min width, min height) using INTER_AREA,
    computes Laplacian variance for both, and caps the ratio at 1.0.
    Returns:
        (sharpness_ratio, is_blurred)
        where sharpness_ratio <= 1.0 and is_blurred is True if ratio < blur_threshold.
    """
    h_m, w_m = gray_member.shape[:2]
    h_r, w_r = gray_ref.shape[:2]
    cw = max(1, min(w_m, w_r))
    ch = max(1, min(h_m, h_r))

    m_scaled = cv2.resize(gray_member, (cw, ch), interpolation=cv2.INTER_AREA) if (w_m, h_m) != (cw, ch) else gray_member
    ref_scaled = cv2.resize(gray_ref, (cw, ch), interpolation=cv2.INTER_AREA) if (w_r, h_r) != (cw, ch) else gray_ref

    lap_m = compute_laplacian_sharpness(m_scaled)
    lap_ref = compute_laplacian_sharpness(ref_scaled)

    raw_ratio = (lap_m / lap_ref) if lap_ref > 0 else 0.0
    capped_ratio = min(1.0, raw_ratio)
    is_blurred = capped_ratio < blur_threshold
    return round(capped_ratio, 4), is_blurred


def load_proxy_rgb(
    image_or_path: str | Path | Image.Image | np.ndarray,
    proxy_size: int = 1024,
    source: ImageSource | None = None,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Load an image proxy suitable for quality and alignment evaluation.
    Returns:
        rgb_array: uint8 RGB numpy array with max dimension <= proxy_size
        scale: float scaling factor (proxy dimension / original dimension)
        orig_dims: (orig_width, orig_height)
    """
    if isinstance(image_or_path, np.ndarray):
        arr = image_or_path
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        h, w = arr.shape[:2]
        orig_dims = (w, h)
        max_dim = max(w, h)
        if max_dim > proxy_size:
            scale = proxy_size / max_dim
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
            return arr, scale, orig_dims
        return arr, 1.0, orig_dims

    if isinstance(image_or_path, Image.Image):
        im = image_or_path.convert("RGB")
        orig_dims = im.size
    else:
        file_path = str(image_or_path)
        if file_path.startswith("drive://"):
            drive_id = file_path[8:].split("/")[0]
            cache_dirs = []
            if source is not None and hasattr(source, "cache_dir") and source.cache_dir:
                cache_dirs.append(Path(source.cache_dir))
            try:
                from core.config import resolve_data_dir
                cache_dirs.append(resolve_data_dir() / "drive_cache")
            except Exception:
                pass
            cache_dirs.extend([
                Path.cwd() / ".similix" / "drive_cache",
                Path.home() / ".similix" / "drive_cache",
                Path.home() / ".argus" / "drive_cache",
            ])
            cached_file = None
            for cd in cache_dirs:
                if cd and cd.is_dir():
                    matches = sorted(cd.glob(f"{drive_id}_*.jpg"))
                    if not matches:
                        matches = sorted(cd.glob(f"{drive_id}.jpg"))
                    if matches:
                        cached_file = matches[0]
                        break

            if cached_file and cached_file.is_file():
                with Image.open(cached_file) as raw_im:
                    im = raw_im.convert("RGB")
                    orig_dims = im.size
            elif source is not None and hasattr(source, "get_thumbnail"):
                im = source.get_thumbnail(file_path, size=(proxy_size, proxy_size)).convert("RGB")
                orig_dims = im.size
            else:
                try:
                    from sources.google_drive import GoogleDriveSource
                    gdrive_source = GoogleDriveSource(stage="quality_thumbnail")
                    im = gdrive_source.get_thumbnail(file_path, size=(proxy_size, proxy_size)).convert("RGB")
                    orig_dims = im.size
                except Exception as exc:
                    raise OSError(f"Drive proxy thumbnail not available for {file_path}: {exc}")
        elif os.path.exists(file_path):
            with Image.open(file_path) as raw_im:
                orig_dims = raw_im.size
                if raw_im.format == "JPEG" and max(orig_dims) > 2 * proxy_size:
                    raw_im.draft("RGB", (proxy_size * 2, proxy_size * 2))
                im = raw_im.convert("RGB")
        elif source is not None:
            im = source.get_thumbnail(file_path, size=(proxy_size, proxy_size)).convert("RGB")
            try:
                meta = source.get_metadata(file_path)
                orig_dims = (meta.width or im.size[0], meta.height or im.size[1])
            except Exception:
                orig_dims = im.size
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    w, h = im.size
    max_dim = max(w, h)
    if max_dim > proxy_size:
        scale = proxy_size / max_dim
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)

    arr = np.array(im)
    actual_scale = max(arr.shape[:2]) / max(1, max(orig_dims))
    return arr, actual_scale, orig_dims


def detect_orthogonal_rotation(
    M_full: np.ndarray,
    tol_scale: float = 0.05,
    tol_angle_deg: float = 3.0,
) -> int | None:
    """
    Check if an affine transform matrix represents a pure 90/180/270 degree rotation
    (scale ~ 1.0, angle near 90/180/270).
    Returns cv2.ROTATE_* flag if detected, else None.
    """
    a, b = M_full[0, 0], M_full[0, 1]
    c, d = M_full[1, 0], M_full[1, 1]
    scale = float(np.sqrt(a * a + c * c))
    if abs(scale - 1.0) > tol_scale:
        return None

    angle_deg = float(np.degrees(np.arctan2(c, a))) % 360.0

    # 90° clockwise: a ~ 0, c ~ 1 -> atan2(1, 0) = 90°
    if abs(angle_deg - 90.0) <= tol_angle_deg:
        return cv2.ROTATE_90_CLOCKWISE
    # 180°: a ~ -1, c ~ 0 -> atan2(0, -1) = 180°
    elif abs(angle_deg - 180.0) <= tol_angle_deg:
        return cv2.ROTATE_180
    # 270° clockwise (= 90° CCW): a ~ 0, c ~ -1 -> atan2(-1, 0) = 270° (or -90°)
    elif abs(angle_deg - 270.0) <= tol_angle_deg:
        return cv2.ROTATE_90_COUNTERCLOCKWISE

    return None


def align_and_compute_metrics(
    member_path: str | Path | Image.Image | np.ndarray,
    reference_path: str | Path | Image.Image | np.ndarray,
    member_id: int | None = None,
    reference_id: int | None = None,
    member_orig_dims: tuple[int, int] | None = None,
    reference_orig_dims: tuple[int, int] | None = None,
    member_file_size: int | None = None,
    reference_file_size: int | None = None,
    conn: sqlite3.Connection | None = None,
    config: dict | None = None,
    source: ImageSource | None = None,
    ref_rgb_cached: np.ndarray | None = None,
    ref_sift_cached: tuple[list[cv2.KeyPoint], np.ndarray | None] | None = None,
    cached_transform: np.ndarray | None = None,
    cached_inliers: int = 0,
    cached_ratio: float = 0.0,
) -> QualityMetrics:
    """
    Align member image to reference image and compute quality metrics:
    SSIM (gray + color), sharpness ratio, pixel ratio, bytes ratio.
    """
    cfg = (config or {}).get("quality", {})
    proxy_size = int(cfg.get("proxy_size", 512))
    min_inliers = int(cfg.get("min_align_inliers", 15))
    min_inlier_ratio = float(cfg.get("min_align_inlier_ratio", 0.50))
    ransac_reproj = float(cfg.get("ransac_reproj_threshold", 4.0))
    ratio_threshold = float(cfg.get("ratio_threshold", 0.75))
    max_features = int(cfg.get("max_features", 2000))

    # Fast path: exact MD5 match skips proxy load, SIFT, RANSAC, and SSIM
    if conn is not None and member_id is not None and reference_id is not None:
        m_row = conn.execute("SELECT md5 FROM images WHERE id = ?", (member_id,)).fetchone()
        r_row = conn.execute("SELECT md5 FROM images WHERE id = ?", (reference_id,)).fetchone()
        if m_row and r_row and m_row["md5"] and m_row["md5"] == r_row["md5"]:
            return QualityMetrics(
                quality_status="aligned",
                ssim=1.0,
                ssim_color=1.0,
                sharpness_ratio=1.0,
                pixel_ratio=1.0,
                bytes_ratio=1.0,
                inlier_count=500,
                inlier_ratio=1.0,
                blur_flag=False,
            )

    # Load proxies
    try:
        if ref_rgb_cached is not None:
            ref_rgb = ref_rgb_cached
            ref_dims_detected = (ref_rgb.shape[1], ref_rgb.shape[0])
        else:
            ref_rgb, _s_ref, ref_dims_detected = load_proxy_rgb(reference_path, proxy_size=proxy_size, source=source)
        m_rgb, _s_m, m_dims_detected = load_proxy_rgb(member_path, proxy_size=proxy_size, source=source)
    except Exception as exc:
        logger.warning("Failed to load proxies for %s vs %s: %s", reference_path, member_path, exc)
        return QualityMetrics(
            quality_status="unalignable",
            ssim=None,
            ssim_color=None,
            sharpness_ratio=1.0,
            pixel_ratio=1.0,
            bytes_ratio=1.0,
            inlier_count=0,
            inlier_ratio=0.0,
            blur_flag=False,
        )

    ref_orig = reference_orig_dims or ref_dims_detected
    m_orig = member_orig_dims or m_dims_detected

    # File sizes
    if reference_file_size is not None:
        ref_sz = reference_file_size
    elif isinstance(reference_path, (str, Path)) and os.path.exists(str(reference_path)):
        ref_sz = os.path.getsize(str(reference_path))
    else:
        ref_sz = ref_rgb.nbytes

    if member_file_size is not None:
        m_sz = member_file_size
    elif isinstance(member_path, (str, Path)) and os.path.exists(str(member_path)):
        m_sz = os.path.getsize(str(member_path))
    else:
        m_sz = m_rgb.nbytes

    ref_px = max(1, ref_orig[0] * ref_orig[1])
    m_px = max(1, m_orig[0] * m_orig[1])

    pixel_ratio = round(m_px / ref_px, 4)
    bytes_ratio = round((m_sz / ref_sz) if ref_sz > 0 else 0.0, 4)

    # Sharpness ratio computed at common scale (capped at 1.0)
    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    m_gray = cv2.cvtColor(m_rgb, cv2.COLOR_RGB2GRAY)

    blur_threshold = float(cfg.get("blur_threshold", 0.70))
    sharpness_ratio, blur_flag = compute_common_scale_sharpness(
        m_gray, ref_gray, blur_threshold=blur_threshold
    )

    # ── Alignment ──────────────────────────────────────────────────────────
    # Check if cached in pair_transforms (member -> reference)
    M_full: np.ndarray | None = None
    inlier_count = 0
    inlier_ratio = 0.0

    if cached_transform is not None:
        M_full = cached_transform
        inlier_count = cached_inliers
        inlier_ratio = cached_ratio
    elif conn is not None and member_id is not None and reference_id is not None:
        # 1. Try forward transform (member -> ref)
        row = conn.execute(
            "SELECT transform_matrix, inlier_count, inlier_ratio FROM pair_transforms WHERE source_id = ? AND target_id = ?",
            (member_id, reference_id),
        ).fetchone()
        if row is not None and row["transform_matrix"] is not None:
            try:
                M_full = pickle.loads(row["transform_matrix"])
                inlier_count = row["inlier_count"]
                inlier_ratio = row["inlier_ratio"]
            except Exception:
                M_full = None

        # 2. Try inverse transform (ref -> member)
        if M_full is None:
            row_inv = conn.execute(
                "SELECT transform_matrix, inlier_count, inlier_ratio FROM pair_transforms WHERE source_id = ? AND target_id = ?",
                (reference_id, member_id),
            ).fetchone()
            if row_inv is not None and row_inv["transform_matrix"] is not None:
                try:
                    M_inv = pickle.loads(row_inv["transform_matrix"])
                    M_full = cv2.invertAffineTransform(M_inv)
                    inlier_count = row_inv["inlier_count"]
                    inlier_ratio = row_inv["inlier_ratio"]
                except Exception:
                    M_full = None

    # If missing from cache, recompute directly on proxy images
    if M_full is None:
        if ref_sift_cached is not None:
            kp_ref, des_ref = ref_sift_cached
        else:
            sift_ref = cv2.SIFT_create(max_features)
            kp_ref, des_ref = sift_ref.detectAndCompute(ref_gray, None)
        sift_m = cv2.SIFT_create(max_features)
        kp_m, des_m = sift_m.detectAndCompute(m_gray, None)

        if (
            des_m is not None
            and des_ref is not None
            and len(des_m) >= 4
            and len(des_ref) >= 4
        ):
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            raw_matches = matcher.knnMatch(des_m, des_ref, k=2)
            good_matches: list[cv2.DMatch] = []
            for match in raw_matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
                elif len(match) == 1:
                    good_matches.append(match[0])

            if len(good_matches) >= 4:
                pts_m = np.float32([kp_m[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M_proxy, mask = cv2.estimateAffinePartial2D(
                    pts_m, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=ransac_reproj
                )
                if mask is not None:
                    inlier_count = int(np.sum(mask))
                    inlier_ratio = inlier_count / len(good_matches) if good_matches else 0.0
                    if inlier_count >= min_inliers and inlier_ratio >= min_inlier_ratio:
                        M_full = proxy_matrix_to_full(
                            M_proxy,
                            m_rgb.shape[:2],
                            m_orig,
                            ref_rgb.shape[:2],
                            ref_orig,
                        )
                        if conn is not None and member_id is not None and reference_id is not None:
                            cache_pair_transform(
                                conn,
                                member_id,
                                reference_id,
                                M_full,
                                model="affine",
                                inlier_count=inlier_count,
                                inlier_ratio=round(inlier_ratio, 4),
                            )

    # If alignment failed or rejected: mark unalignable
    if M_full is None:
        return QualityMetrics(
            quality_status="unalignable",
            ssim=None,
            ssim_color=None,
            sharpness_ratio=sharpness_ratio,
            pixel_ratio=pixel_ratio,
            bytes_ratio=bytes_ratio,
            inlier_count=inlier_count,
            inlier_ratio=round(inlier_ratio, 4),
            blur_flag=blur_flag,
        )

    ref_w, ref_h = ref_rgb.shape[1], ref_rgb.shape[0]
    rot_code = detect_orthogonal_rotation(M_full)

    # 1. Exact orthogonal rotation (avoids interpolation artifacts)
    if rot_code is not None:
        rotated_rgb = cv2.rotate(m_rgb, rot_code)
        if rotated_rgb.shape == ref_rgb.shape:
            warped_rgb = rotated_rgb
        else:
            M_proxy = full_matrix_to_proxy(
                M_full,
                m_rgb.shape[:2],
                m_orig,
                ref_rgb.shape[:2],
                ref_orig,
            )
            warped_rgb = cv2.warpAffine(m_rgb, M_proxy, (ref_w, ref_h))
    else:
        # 2. Check near-identity (identical coordinates)
        a, c = M_full[0, 0], M_full[1, 0]
        scale = float(np.sqrt(a * a + c * c))
        angle_deg = float(np.degrees(np.arctan2(c, a))) % 360.0
        is_identity = (
            abs(scale - 1.0) <= 0.02
            and (angle_deg <= 1.0 or angle_deg >= 359.0)
            and m_rgb.shape == ref_rgb.shape
            and abs(M_full[0, 2]) <= 1.0
            and abs(M_full[1, 2]) <= 1.0
        )
        if is_identity:
            warped_rgb = m_rgb
        else:
            M_proxy = full_matrix_to_proxy(
                M_full,
                m_rgb.shape[:2],
                m_orig,
                ref_rgb.shape[:2],
                ref_orig,
            )
            warped_rgb = cv2.warpAffine(m_rgb, M_proxy, (ref_w, ref_h))

    warped_gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)

    # Compute SSIM
    score_gray = float(ssim(ref_gray, warped_gray, data_range=255))
    score_color = float(ssim(ref_rgb, warped_rgb, channel_axis=2, data_range=255))

    return QualityMetrics(
        quality_status="aligned",
        ssim=round(score_gray, 4),
        ssim_color=round(score_color, 4),
        sharpness_ratio=sharpness_ratio,
        pixel_ratio=pixel_ratio,
        bytes_ratio=bytes_ratio,
        inlier_count=inlier_count,
        inlier_ratio=round(inlier_ratio, 4),
        blur_flag=blur_flag,
    )


def ensure_quality_columns(conn: sqlite3.Connection) -> None:
    """Ensure images table has all Stage 4 quality columns including jpeg_quality_est and source_likelihood."""
    existing_cols = {col[1] for col in conn.execute("PRAGMA table_info(images)").fetchall()}
    cols_to_add = [
        ("ssim", "REAL"),
        ("ssim_color", "REAL"),
        ("sharpness_ratio", "REAL"),
        ("pixel_ratio", "REAL"),
        ("bytes_ratio", "REAL"),
        ("quality_status", "TEXT"),
        ("jpeg_quality_est", "INTEGER"),
        ("source_likelihood", "REAL"),
        ("match_pct", "REAL"),
        ("resolution_pct", "REAL"),
        ("quality_basis", "TEXT"),
    ]
    for col_name, col_def in cols_to_add:
        if col_name not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE images ADD COLUMN {col_name} {col_def};")
            except sqlite3.OperationalError:
                pass

    existing_cluster_cols = {col[1] for col in conn.execute("PRAGMA table_info(clusters)").fetchall()}
    if "reference_uncertain" not in existing_cluster_cols:
        try:
            conn.execute("ALTER TABLE clusters ADD COLUMN reference_uncertain INTEGER NOT NULL DEFAULT 0;")
        except sqlite3.OperationalError:
            pass


@dataclass
class _EvaluatedCluster:
    cid: int
    c_elapsed: float
    ref_id: int
    ref_uncertain: int
    ref_q_basis: str
    member_dim_updates: list[tuple[int, int, int, int]]
    member_mtime_updates: list[tuple[float, int]]
    member_jpeg_updates: list[tuple[int, int]]
    member_sl_updates: list[tuple[float | None, int]]
    member_quality_updates: list[tuple[Any, ...]]
    member_records: list[MemberQualityRecord]
    pruned_mids: list[int]
    aligned_count: int
    unalignable_count: int


def _evaluate_cluster_worker(
    cid: int,
    members: list[dict[str, Any]],
    cfg: dict,
    source: ImageSource | None,
    pt_cache: dict[tuple[int, int], tuple[np.ndarray, int, float]],
) -> _EvaluatedCluster:
    c_t0 = time.perf_counter()
    q_cfg = cfg.get("quality", {})
    min_cluster_match_pct = float(q_cfg.get("min_cluster_match_pct", 75.0))
    cluster_timeout = float(q_cfg.get("cluster_timeout_seconds", 60.0))
    max_comparisons = int(q_cfg.get("max_pairwise_comparisons", 40))

    member_dim_updates: list[tuple[int, int, int, int]] = []
    member_mtime_updates: list[tuple[float, int]] = []
    member_jpeg_updates: list[tuple[int, int]] = []
    member_sl_updates: list[tuple[float | None, int]] = []
    member_quality_updates: list[tuple[Any, ...]] = []
    member_records: list[MemberQualityRecord] = []
    pruned_mids: list[int] = []
    aligned_count = 0
    unalignable_count = 0

    member_rows: list[dict[str, Any]] = []
    for d in members:
        p = d["file_path"]
        if not d.get("width") or not d.get("height") or not d.get("file_size"):
            try:
                if source is not None:
                    meta = source.get_metadata(p)
                    d["width"] = meta.width or 0
                    d["height"] = meta.height or 0
                    d["file_size"] = meta.file_size or 0
                else:
                    with Image.open(p) as im:
                        d["width"], d["height"] = im.size
                    d["file_size"] = os.path.getsize(p)
                member_dim_updates.append((d["width"], d["height"], d["file_size"], d["id"]))
            except Exception:
                pass

        if not d.get("modified_time") and os.path.exists(p):
            try:
                mtime_val = os.path.getmtime(p)
                d["modified_time"] = mtime_val
                member_mtime_updates.append((mtime_val, d["id"]))
            except Exception:
                pass

        if d.get("jpeg_quality_est") is None and os.path.exists(p):
            q_est = extract_jpeg_quality_estimate(p)
            d["jpeg_quality_est"] = q_est
            if q_est is not None:
                member_jpeg_updates.append((q_est, d["id"]))

        has_time, has_make = check_exif_metadata(p)
        d["has_exif_time"] = has_time
        d["has_exif_make"] = has_make
        member_rows.append(d)

    ref_member = select_reference_member(member_rows, source=source, max_comparisons=max_comparisons)
    ref_id = ref_member["id"]
    ref_path = ref_member["file_path"]
    ref_orig = (ref_member["width"] or 0, ref_member["height"] or 0)
    ref_sz = ref_member["file_size"] or 0
    ref_uncertain = ref_member.get("reference_uncertain", 0)

    ref_ck = _member_content_key(ref_member)
    if len(member_rows) > 1 and all(_member_content_key(m) == ref_ck for m in member_rows):
        ref_uncertain = 0
        ref_member["reference_uncertain"] = 0

    for d in member_rows:
        member_sl_updates.append((d.get("source_likelihood"), d["id"]))

    is_ref_gdrive = (
        ref_member.get("source") == "gdrive"
        or str(ref_path).startswith("drive://")
    )
    ref_q_basis = ref_member.get("quality_basis") if ref_member.get("quality_basis") == "full" else ("thumbnail" if is_ref_gdrive else "full")

    ref_record = MemberQualityRecord(
        member_id=ref_id,
        cluster_id=cid,
        file_path=ref_path,
        is_reference=True,
        quality_status="reference",
        ssim=1.0,
        ssim_color=1.0,
        sharpness_ratio=1.0,
        pixel_ratio=1.0,
        bytes_ratio=1.0,
        width=ref_orig[0],
        height=ref_orig[1],
        file_size=ref_sz,
        source_likelihood=ref_member.get("source_likelihood"),
        match_pct=100.0,
        resolution_pct=100.0,
    )
    member_records.append(ref_record)

    # Pre-cache reference proxy and SIFT
    ref_rgb_cached = None
    ref_sift_cached = None
    proxy_sz = int(q_cfg.get("proxy_size", 512))
    max_feat = int(q_cfg.get("max_features", 2000))
    try:
        ref_rgb_cached, _, _ = load_proxy_rgb(ref_path, proxy_size=proxy_sz, source=source)
        ref_gray_cached = cv2.cvtColor(ref_rgb_cached, cv2.COLOR_RGB2GRAY)
        sift_ref_init = cv2.SIFT_create(max_feat)
        ref_sift_cached = sift_ref_init.detectAndCompute(ref_gray_cached, None)
    except Exception:
        pass

    for m in member_rows:
        if m["id"] == ref_id:
            continue
        if (time.perf_counter() - c_t0) > cluster_timeout:
            break

        mid = m["id"]
        m_path = m["file_path"]
        m_orig = (m["width"] or 0, m["height"] or 0)
        m_sz = m["file_size"] or 0

        is_exact_copy = bool(m.get("md5") and m.get("md5") == ref_member.get("md5"))
        if is_exact_copy:
            metrics = QualityMetrics(
                quality_status="aligned",
                ssim=1.0,
                ssim_color=1.0,
                sharpness_ratio=1.0,
                pixel_ratio=1.0,
                bytes_ratio=1.0,
                inlier_count=500,
                inlier_ratio=1.0,
                blur_flag=False,
            )
            match_pct = 100.0
            resolution_pct = 100.0
        else:
            cached_xf = pt_cache.get((mid, ref_id))
            cached_inv = pt_cache.get((ref_id, mid))
            inv_matrix = None
            if cached_inv and cached_inv[0] is not None:
                try:
                    inv_matrix = cv2.invertAffineTransform(cached_inv[0])
                except Exception:
                    inv_matrix = None

            metrics = align_and_compute_metrics(
                member_path=m_path,
                reference_path=ref_path,
                member_id=mid,
                reference_id=ref_id,
                member_orig_dims=m_orig,
                reference_orig_dims=ref_orig,
                member_file_size=m_sz,
                reference_file_size=ref_sz,
                conn=None,
                config=cfg,
                source=source,
                ref_rgb_cached=ref_rgb_cached,
                ref_sift_cached=ref_sift_cached,
                cached_transform=cached_xf[0] if (cached_xf and cached_xf[0] is not None) else inv_matrix,
                cached_inliers=cached_xf[1] if cached_xf else (cached_inv[1] if cached_inv else 0),
                cached_ratio=cached_xf[2] if cached_xf else (cached_inv[2] if cached_inv else 0.0),
            )
            if metrics.ssim_color is not None:
                match_pct = round(metrics.ssim_color * 100.0, 1)
            elif metrics.ssim is not None:
                match_pct = round(metrics.ssim * 100.0, 1)
            else:
                match_pct = None

            resolution_pct = round(metrics.pixel_ratio * 100.0, 1) if metrics.pixel_ratio is not None else None
            if resolution_pct is None and m_orig[0] and m_orig[1] and ref_orig[0] and ref_orig[1]:
                ref_px = max(1, ref_orig[0] * ref_orig[1])
                m_px = max(1, m_orig[0] * m_orig[1])
                resolution_pct = round((m_px / ref_px) * 100.0, 1)

            if match_pct is None and m.get("clip_vector") and ref_member.get("clip_vector"):
                try:
                    from core.embeddings import deserialize_embedding
                    v_m = deserialize_embedding(m["clip_vector"]).astype(np.float32)
                    v_r = deserialize_embedding(ref_member["clip_vector"]).astype(np.float32)
                    nm = np.linalg.norm(v_m)
                    nr = np.linalg.norm(v_r)
                    if nm > 0 and nr > 0:
                        sim = float(np.dot(v_m / nm, v_r / nr))
                        match_pct = round(max(0.0, min(100.0, sim * 100.0)), 1)
                except Exception:
                    pass
            if match_pct is None and m.get("phash") and ref_member.get("phash"):
                try:
                    h1 = int(m["phash"], 16)
                    h2 = int(ref_member["phash"], 16)
                    dist = bin(h1 ^ h2).count("1")
                    match_pct = round(max(0.0, (1.0 - (dist / 64.0)) * 100.0), 1)
                except Exception:
                    pass

        is_gdrive_member = (
            m.get("source") == "gdrive"
            or str(m_path).startswith("drive://")
        )
        existing_basis = m.get("quality_basis")
        q_basis = existing_basis if existing_basis == "full" else ("thumbnail" if is_gdrive_member else "full")

        member_quality_updates.append((
            metrics.ssim,
            metrics.ssim_color,
            metrics.sharpness_ratio,
            metrics.pixel_ratio,
            metrics.bytes_ratio,
            metrics.quality_status,
            match_pct,
            resolution_pct,
            q_basis,
            mid,
        ))

        if metrics.quality_status == "aligned":
            aligned_count += 1
        else:
            unalignable_count += 1

        member_records.append(
            MemberQualityRecord(
                member_id=mid,
                cluster_id=cid,
                file_path=m_path,
                is_reference=False,
                quality_status=metrics.quality_status,
                ssim=metrics.ssim,
                ssim_color=metrics.ssim_color,
                sharpness_ratio=metrics.sharpness_ratio,
                pixel_ratio=metrics.pixel_ratio,
                bytes_ratio=metrics.bytes_ratio,
                width=m_orig[0],
                height=m_orig[1],
                file_size=m_sz,
                source_likelihood=m.get("source_likelihood"),
                match_pct=match_pct,
                resolution_pct=resolution_pct,
            )
        )

        # Quality metrics recorded for keeper selection.
        # Members with lower match_pct or unalignable status are safely retained in the cluster
        # and marked for manual review (review_ids) by core/keeper.py, rather than being
        # destructively unclustered or falsely marked as unique.

    return _EvaluatedCluster(
        cid=cid,
        c_elapsed=time.perf_counter() - c_t0,
        ref_id=ref_id,
        ref_uncertain=ref_uncertain,
        ref_q_basis=ref_q_basis,
        member_dim_updates=member_dim_updates,
        member_mtime_updates=member_mtime_updates,
        member_jpeg_updates=member_jpeg_updates,
        member_sl_updates=member_sl_updates,
        member_quality_updates=member_quality_updates,
        member_records=member_records,
        pruned_mids=pruned_mids,
        aligned_count=aligned_count,
        unalignable_count=unalignable_count,
    )


def run_quality_pipeline(
    conn: sqlite3.Connection,
    config: dict | None = None,
    source: ImageSource | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> QualityResult:
    """
    Run quality scoring and geometric alignment across all clusters in SQLite.
    Stores per-member metrics in images table.
    """
    ensure_quality_columns(conn)
    if conn.row_factory is None:
        conn.row_factory = sqlite3.Row
    t0 = time.perf_counter()
    res = QualityResult()

    cfg = config or {}
    q_cfg = cfg.get("quality", {})

    # Pre-fetch pair transforms cache into memory for thread-safe access
    pt_rows = conn.execute(
        "SELECT source_id, target_id, transform_matrix, inlier_count, inlier_ratio FROM pair_transforms"
    ).fetchall()
    pt_cache: dict[tuple[int, int], tuple[np.ndarray, int, float]] = {}
    for r in pt_rows:
        if r["transform_matrix"]:
            try:
                pt_cache[(r["source_id"], r["target_id"])] = (
                    pickle.loads(r["transform_matrix"]),
                    r["inlier_count"],
                    r["inlier_ratio"],
                )
            except Exception:
                pass

    # Query all clusters and their members upfront
    clusters = conn.execute("SELECT id FROM clusters ORDER BY id").fetchall()
    total_clusters = len(clusters)
    cluster_times: list[tuple[int, float]] = []

    clusters_to_process: list[tuple[int, list[dict[str, Any]]]] = []
    processed_count = 0

    for c in clusters:
        cid = c["id"]
        members = conn.execute(
            """
            SELECT id, cluster_id, file_path, width, height, file_size, format, modified_time, jpeg_quality_est, source_likelihood, md5, clip_vector, phash
            FROM images
            WHERE cluster_id = ?
            ORDER BY id
            """,
            (cid,),
        ).fetchall()

        if len(members) < 2:
            cluster_times.append((cid, 0.0))
            processed_count += 1
            if progress_callback is not None:
                progress_callback(processed_count, total_clusters, cid)
            continue

        clusters_to_process.append((cid, [dict(m) for m in members]))

    def _apply_cluster_result(ev: _EvaluatedCluster):
        nonlocal processed_count
        if ev.member_dim_updates:
            conn.executemany("UPDATE images SET width = ?, height = ?, file_size = ? WHERE id = ?", ev.member_dim_updates)
        if ev.member_mtime_updates:
            conn.executemany("UPDATE images SET modified_time = ? WHERE id = ?", ev.member_mtime_updates)
        if ev.member_jpeg_updates:
            conn.executemany("UPDATE images SET jpeg_quality_est = ? WHERE id = ?", ev.member_jpeg_updates)
        if ev.member_sl_updates:
            conn.executemany("UPDATE images SET source_likelihood = ? WHERE id = ?", ev.member_sl_updates)

        conn.execute(
            "UPDATE clusters SET representative_id = ?, reference_uncertain = ? WHERE id = ?",
            (ev.ref_id, ev.ref_uncertain, ev.cid),
        )

        conn.execute(
            """
            UPDATE images
            SET ssim = 1.0,
                ssim_color = 1.0,
                sharpness_ratio = 1.0,
                pixel_ratio = 1.0,
                bytes_ratio = 1.0,
                quality_status = 'reference',
                match_pct = 100.0,
                resolution_pct = 100.0,
                quality_basis = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE id = ?
            """,
            (ev.ref_q_basis, ev.ref_id),
        )

        if ev.member_quality_updates:
            conn.executemany(
                """
                UPDATE images
                SET ssim = ?,
                    ssim_color = ?,
                    sharpness_ratio = ?,
                    pixel_ratio = ?,
                    bytes_ratio = ?,
                    quality_status = ?,
                    match_pct = ?,
                    resolution_pct = ?,
                    quality_basis = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                ev.member_quality_updates,
            )

        # Ensure cluster member_count reflects current members
        surviving = conn.execute("SELECT id FROM images WHERE cluster_id = ?", (ev.cid,)).fetchall()
        conn.execute("UPDATE clusters SET member_count = ? WHERE id = ?", (len(surviving), ev.cid))

        res.clusters_evaluated += 1
        res.members_evaluated += len(ev.member_records)
        res.members_aligned += ev.aligned_count
        res.members_unalignable += ev.unalignable_count
        res.member_records.extend(ev.member_records)
        cluster_times.append((ev.cid, ev.c_elapsed))

        processed_count += 1
        if progress_callback is not None:
            progress_callback(processed_count, total_clusters, ev.cid)

    # Process clusters with thread pool
    max_workers = min(8, os.cpu_count() or 4)
    if len(clusters_to_process) <= 1:
        for cid, mems in clusters_to_process:
            ev = _evaluate_cluster_worker(cid, mems, cfg, source, pt_cache)
            _apply_cluster_result(ev)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cid = {
                executor.submit(_evaluate_cluster_worker, cid, mems, cfg, source, pt_cache): cid
                for cid, mems in clusters_to_process
            }
            for fut in concurrent.futures.as_completed(future_to_cid):
                ev = fut.result()
                _apply_cluster_result(ev)

    conn.commit()
    res.elapsed_seconds = round(time.perf_counter() - t0, 3)
    res.slowest_clusters = sorted(cluster_times, key=lambda x: x[1], reverse=True)[:5]
    if res.slowest_clusters:
        slowest_str = ", ".join(f"#{cid} ({t:.2f}s)" for cid, t in res.slowest_clusters)
        print(f"Quality stage slowest 5 clusters: {slowest_str}")
        logger.info("Quality stage slowest 5 clusters: %s", slowest_str)
    return res
