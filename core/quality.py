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

from dataclasses import dataclass, field
import os
from pathlib import Path
import pickle
import sqlite3
import time
from typing import Any

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


@dataclass
class QualityResult:
    clusters_evaluated: int = 0
    members_evaluated: int = 0
    members_aligned: int = 0
    members_unalignable: int = 0
    elapsed_seconds: float = 0.0
    member_records: list[MemberQualityRecord] = field(default_factory=list)


def select_reference_member(members: list[sqlite3.Row | dict[str, Any]]) -> sqlite3.Row | dict[str, Any]:
    """
    Select the reference member for a duplicate cluster:
    1. Highest pixel count (width * height, descending)
    2. Least compressed (file_size / (width * height), descending)
    3. Shortest file path length (len(file_path), ascending)
    4. Deterministic tie-breaker (file_path, ascending)
    """
    if not members:
        raise ValueError("Cannot select reference from empty member list")

    def key_fn(m: sqlite3.Row | dict[str, Any]) -> tuple[int, float, int, str]:
        w = m["width"] or 0
        h = m["height"] or 0
        px = w * h
        sz = m["file_size"] or 0
        bpp = (sz / px) if px > 0 else 0.0
        p = str(m["file_path"])
        return (-px, -bpp, len(p), p)

    return min(members, key=key_fn)


def compute_laplacian_sharpness(gray_img: np.ndarray) -> float:
    """Compute variance of the Laplacian as a measure of focus / sharpness."""
    if gray_img.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var())


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
        if source is not None:
            im = source.get_thumbnail(file_path, size=(proxy_size, proxy_size)).convert("RGB")
            try:
                meta = source.get_metadata(file_path)
                orig_dims = (meta.width or im.size[0], meta.height or im.size[1])
            except Exception:
                orig_dims = im.size
        else:
            with Image.open(file_path) as raw_im:
                orig_dims = raw_im.size
                if raw_im.format == "JPEG" and max(orig_dims) > 2 * proxy_size:
                    raw_im.draft("RGB", (proxy_size * 2, proxy_size * 2))
                im = raw_im.convert("RGB")

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
) -> QualityMetrics:
    """
    Align member image to reference image and compute quality metrics:
    SSIM (gray + color), sharpness ratio, pixel ratio, bytes ratio.
    """
    cfg = (config or {}).get("quality", {})
    proxy_size = int(cfg.get("proxy_size", 1024))
    min_inliers = int(cfg.get("min_align_inliers", 15))
    min_inlier_ratio = float(cfg.get("min_align_inlier_ratio", 0.50))
    ransac_reproj = float(cfg.get("ransac_reproj_threshold", 4.0))
    ratio_threshold = float(cfg.get("ratio_threshold", 0.75))
    max_features = int(cfg.get("max_features", 2000))

    # Load 1024px proxies
    ref_rgb, _s_ref, ref_dims_detected = load_proxy_rgb(reference_path, proxy_size=proxy_size, source=source)
    m_rgb, _s_m, m_dims_detected = load_proxy_rgb(member_path, proxy_size=proxy_size, source=source)

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

    # Sharpness ratio (Laplacian variance)
    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    m_gray = cv2.cvtColor(m_rgb, cv2.COLOR_RGB2GRAY)

    ref_sharpness = compute_laplacian_sharpness(ref_gray)
    m_sharpness = compute_laplacian_sharpness(m_gray)
    sharpness_ratio = round((m_sharpness / ref_sharpness) if ref_sharpness > 0 else 0.0, 4)

    # ── Alignment ──────────────────────────────────────────────────────────
    # Check if cached in pair_transforms (member -> reference)
    M_full: np.ndarray | None = None
    inlier_count = 0
    inlier_ratio = 0.0

    if conn is not None and member_id is not None and reference_id is not None:
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
        sift = cv2.SIFT_create(max_features)
        kp_m, des_m = sift.detectAndCompute(m_gray, None)
        kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)

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
    )


def run_quality_pipeline(
    conn: sqlite3.Connection,
    config: dict | None = None,
    source: ImageSource | None = None,
) -> QualityResult:
    """
    Run quality scoring and geometric alignment across all clusters in SQLite.
    Stores per-member metrics in images table.
    """
    t0 = time.perf_counter()
    res = QualityResult()

    # Query all clusters
    clusters = conn.execute("SELECT id FROM clusters ORDER BY id").fetchall()

    for c in clusters:
        cid = c["id"]
        members = conn.execute(
            """
            SELECT id, cluster_id, file_path, width, height, file_size
            FROM images
            WHERE cluster_id = ?
            ORDER BY id
            """,
            (cid,),
        ).fetchall()

        if len(members) < 2:
            continue

        res.clusters_evaluated += 1

        # Fill missing dimensions if necessary
        member_rows: list[dict[str, Any]] = []
        for m in members:
            d = dict(m)
            if not d["width"] or not d["height"] or not d["file_size"]:
                p = d["file_path"]
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
                    conn.execute(
                        "UPDATE images SET width = ?, height = ?, file_size = ? WHERE id = ?",
                        (d["width"], d["height"], d["file_size"], d["id"]),
                    )
                except Exception:
                    pass
            member_rows.append(d)

        ref_member = select_reference_member(member_rows)
        ref_id = ref_member["id"]
        ref_path = ref_member["file_path"]
        ref_orig = (ref_member["width"] or 0, ref_member["height"] or 0)
        ref_sz = ref_member["file_size"] or 0

        # Mark reference member in DB
        conn.execute(
            """
            UPDATE images
            SET ssim = 1.0,
                ssim_color = 1.0,
                sharpness_ratio = 1.0,
                pixel_ratio = 1.0,
                bytes_ratio = 1.0,
                quality_status = 'reference',
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE id = ?
            """,
            (ref_id,),
        )
        res.members_evaluated += 1
        res.member_records.append(
            MemberQualityRecord(
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
            )
        )

        # Align and evaluate every other member
        for m in member_rows:
            if m["id"] == ref_id:
                continue

            mid = m["id"]
            m_path = m["file_path"]
            m_orig = (m["width"] or 0, m["height"] or 0)
            m_sz = m["file_size"] or 0

            metrics = align_and_compute_metrics(
                member_path=m_path,
                reference_path=ref_path,
                member_id=mid,
                reference_id=ref_id,
                member_orig_dims=m_orig,
                reference_orig_dims=ref_orig,
                member_file_size=m_sz,
                reference_file_size=ref_sz,
                conn=conn,
                config=config,
                source=source,
            )

            conn.execute(
                """
                UPDATE images
                SET ssim = ?,
                    ssim_color = ?,
                    sharpness_ratio = ?,
                    pixel_ratio = ?,
                    bytes_ratio = ?,
                    quality_status = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                (
                    metrics.ssim,
                    metrics.ssim_color,
                    metrics.sharpness_ratio,
                    metrics.pixel_ratio,
                    metrics.bytes_ratio,
                    metrics.quality_status,
                    mid,
                ),
            )

            res.members_evaluated += 1
            if metrics.quality_status == "aligned":
                res.members_aligned += 1
            else:
                res.members_unalignable += 1

            res.member_records.append(
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
                )
            )

    conn.commit()
    res.elapsed_seconds = round(time.perf_counter() - t0, 3)
    return res
