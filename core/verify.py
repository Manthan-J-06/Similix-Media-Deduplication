"""
core/verify.py — Stage 3: Geometric keypoint matching with RANSAC verification.

For each duplicate cluster, verifies every non-representative member against the
cluster representative using keypoint matching (SIFT or ORB) and robust geometric
estimation (RANSAC) on ~512px proxy thumbnails.

Features:
- Tolerates crops, rescaling, and arbitrary rotations (90°, 180°, 270°).
- Outputs inlier count and inlier ratio for each member-representative pair.
- Non-matching members (below min_inliers or min_inlier_ratio) are split out of the cluster.
- Clusters reduced to fewer than 2 members are disbanded.
- Uses SIFT (patent expired March 2020, Apache 2.0 license in OpenCV 4.4+) as the default
  matcher, with ORB as a configurable alternative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from sources.base import ImageSource


@dataclass
class MatchResult:
    """Detailed result of keypoint matching and RANSAC verification between two images."""
    verified: bool
    inlier_count: int
    inlier_ratio: float
    good_matches: int
    total_keypoints_1: int
    total_keypoints_2: int
    message: str = ""


@dataclass
class PairVerificationRecord:
    member_id: int
    representative_id: int
    member_path: str
    representative_path: str
    inlier_count: int
    inlier_ratio: float
    good_matches: int
    verified: bool


@dataclass
class Stage3Result:
    pairs_evaluated: int = 0
    pairs_verified: int = 0
    members_split: int = 0
    clusters_disbanded: int = 0
    elapsed_seconds: float = 0.0
    time_per_1000: float = 0.0
    pair_details: list[PairVerificationRecord] = field(default_factory=list)


def load_proxy_grayscale(
    image_or_path: str | Path | Image.Image | np.ndarray,
    proxy_size: int = 512,
    source: ImageSource | None = None,
) -> np.ndarray:
    """
    Load an image proxy suitable for keypoint detection.
    Scale preserves aspect ratio such that the max dimension is <= proxy_size.
    Returns a single-channel 8-bit grayscale numpy array.
    """
    if isinstance(image_or_path, np.ndarray):
        if len(image_or_path.shape) == 3:
            return cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY if image_or_path.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        return image_or_path

    if isinstance(image_or_path, Image.Image):
        im = image_or_path.copy()
    else:
        file_path = str(image_or_path)
        if source is not None:
            im = source.get_thumbnail(file_path, size=(proxy_size, proxy_size))
        else:
            p = Path(file_path)
            with Image.open(p) as raw_im:
                if raw_im.format == "JPEG":
                    raw_im.draft("RGB", (proxy_size, proxy_size))
                im = raw_im.copy()

    # Normalize color mode
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")

    # Downscale preserving aspect ratio
    w, h = im.size
    max_dim = max(w, h)
    if max_dim > proxy_size:
        scale = proxy_size / max_dim
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)

    arr = np.array(im)
    if len(arr.shape) == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


class KeypointVerifier:
    """
    Performs keypoint detection, descriptor extraction, ratio-test matching,
    and RANSAC geometric consistency verification.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        stage3_cfg = cfg.get("stage3", {})

        self.algorithm = str(stage3_cfg.get("algorithm", "sift")).lower()
        self.max_features = int(stage3_cfg.get("max_features", 2000))
        self.proxy_size = int(stage3_cfg.get("proxy_size", 512))
        self.ratio_threshold = float(stage3_cfg.get("ratio_threshold", 0.75))
        self.ransac_reproj = float(stage3_cfg.get("ransac_reproj_threshold", 4.0))
        self.min_inliers = int(stage3_cfg.get("min_inliers", 20))
        self.min_inlier_ratio = float(stage3_cfg.get("min_inlier_ratio", 0.65))
        self.model = str(stage3_cfg.get("transformation_model", "affine")).lower()

        if self.algorithm == "orb":
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            # SIFT (default): patent expired March 2020, Apache 2.0 license in OpenCV 4.4+
            self.detector = cv2.SIFT_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def extract_features(self, proxy_gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """Detect keypoints and compute descriptors for a grayscale proxy image."""
        kp, des = self.detector.detectAndCompute(proxy_gray, None)
        return kp, des

    def match_features(
        self,
        kp1: list[cv2.KeyPoint],
        des1: np.ndarray | None,
        kp2: list[cv2.KeyPoint],
        des2: np.ndarray | None,
    ) -> MatchResult:
        """
        Match extracted descriptors and compute inliers via RANSAC.
        """
        nkp1 = len(kp1)
        nkp2 = len(kp2)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return MatchResult(
                verified=False,
                inlier_count=0,
                inlier_ratio=0.0,
                good_matches=0,
                total_keypoints_1=nkp1,
                total_keypoints_2=nkp2,
                message="Insufficient descriptors (< 4)",
            )

        # KNN match with k=2 for ratio test
        raw_matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches: list[cv2.DMatch] = []
        for match in raw_matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
            elif len(match) == 1:
                good_matches.append(match[0])

        if len(good_matches) < 4:
            return MatchResult(
                verified=False,
                inlier_count=len(good_matches),
                inlier_ratio=0.0,
                good_matches=len(good_matches),
                total_keypoints_1=nkp1,
                total_keypoints_2=nkp2,
                message=f"Too few good matches after ratio test ({len(good_matches)} < 4)",
            )

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        if self.model == "homography":
            _M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj)
        else:
            # Affine partial (2D Euclidean similarity: scale, rotation, translation)
            _M, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj
            )

        if mask is None:
            return MatchResult(
                verified=False,
                inlier_count=0,
                inlier_ratio=0.0,
                good_matches=len(good_matches),
                total_keypoints_1=nkp1,
                total_keypoints_2=nkp2,
                message="RANSAC failed to estimate valid transformation",
            )

        inlier_count = int(np.sum(mask))
        inlier_ratio = inlier_count / len(good_matches) if good_matches else 0.0

        verified = (inlier_count >= self.min_inliers) and (inlier_ratio >= self.min_inlier_ratio)

        return MatchResult(
            verified=verified,
            inlier_count=inlier_count,
            inlier_ratio=round(inlier_ratio, 4),
            good_matches=len(good_matches),
            total_keypoints_1=nkp1,
            total_keypoints_2=nkp2,
            message="Verified" if verified else f"Failed thresholds (inliers={inlier_count}/{self.min_inliers}, ratio={inlier_ratio:.3f}/{self.min_inlier_ratio})",
        )


def verify_pair(
    img1: str | Path | Image.Image | np.ndarray,
    img2: str | Path | Image.Image | np.ndarray,
    config: dict | None = None,
    source: ImageSource | None = None,
    verifier: KeypointVerifier | None = None,
) -> MatchResult:
    """
    Verify two images against each other using keypoint matching and RANSAC.
    Accepts file paths, PIL Images, or proxy numpy arrays.
    """
    v = verifier or KeypointVerifier(config)
    g1 = load_proxy_grayscale(img1, proxy_size=v.proxy_size, source=source)
    g2 = load_proxy_grayscale(img2, proxy_size=v.proxy_size, source=source)

    kp1, des1 = v.extract_features(g1)
    kp2, des2 = v.extract_features(g2)

    return v.match_features(kp1, des1, kp2, des2)


def run_stage3_verification(
    conn: sqlite3.Connection,
    config: dict | None = None,
    source: ImageSource | None = None,
) -> Stage3Result:
    """
    Execute Stage 3 verification on all duplicate clusters in SQLite.

    For each cluster:
    1. Loads the cluster representative proxy thumbnail.
    2. Compares each non-representative cluster member against the representative.
    3. Members that fail geometric verification (below inlier / inlier_ratio thresholds)
       are split out of the cluster.
    4. If splitting leaves fewer than 2 members, the cluster is disbanded.
    5. Updates clusters member_count and commits changes.
    """
    t0 = time.perf_counter()
    verifier = KeypointVerifier(config)

    clusters_rows = conn.execute(
        """
        SELECT c.id, c.representative_id, i.file_path as rep_path
        FROM clusters c
        JOIN images i ON i.id = c.representative_id
        ORDER BY c.id
        """
    ).fetchall()

    res = Stage3Result()
    proxy_cache: dict[str, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}

    def get_features(path_str: str) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        if path_str not in proxy_cache:
            proxy = load_proxy_grayscale(path_str, proxy_size=verifier.proxy_size, source=source)
            proxy_cache[path_str] = verifier.extract_features(proxy)
        return proxy_cache[path_str]

    for c_row in clusters_rows:
        cluster_id = c_row["id"]
        rep_id = c_row["representative_id"]
        rep_path = c_row["rep_path"]

        rep_kp, rep_des = get_features(rep_path)

        members = conn.execute(
            """
            SELECT id, file_path
            FROM images
            WHERE cluster_id = ? AND id != ?
            ORDER BY id
            """,
            (cluster_id, rep_id),
        ).fetchall()

        members_to_split: list[int] = []

        for m_row in members:
            m_id = m_row["id"]
            m_path = m_row["file_path"]

            m_kp, m_des = get_features(m_path)
            match_res = verifier.match_features(rep_kp, rep_des, m_kp, m_des)

            res.pairs_evaluated += 1
            if match_res.verified:
                res.pairs_verified += 1
            else:
                members_to_split.append(m_id)
                res.members_split += 1

            res.pair_details.append(
                PairVerificationRecord(
                    member_id=m_id,
                    representative_id=rep_id,
                    member_path=m_path,
                    representative_path=rep_path,
                    inlier_count=match_res.inlier_count,
                    inlier_ratio=match_res.inlier_ratio,
                    good_matches=match_res.good_matches,
                    verified=match_res.verified,
                )
            )

        # Split failing members out of the cluster
        if members_to_split:
            conn.executemany(
                """
                UPDATE images
                SET cluster_id  = NULL,
                    is_keeper   = 0,
                    confidence  = NULL,
                    link_signal = NULL,
                    status      = 'embedded',
                    updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                [(mid,) for mid in members_to_split],
            )

        # Check remaining member count in this cluster
        remaining_count = conn.execute(
            "SELECT COUNT(*) FROM images WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()[0]

        if remaining_count < 2:
            # Only keeper left or empty: disband cluster
            conn.execute(
                """
                UPDATE images
                SET cluster_id  = NULL,
                    is_keeper   = 0,
                    confidence  = NULL,
                    link_signal = NULL,
                    status      = 'embedded',
                    updated_at  = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE cluster_id = ?
                """,
                (cluster_id,),
            )
            conn.execute("DELETE FROM clusters WHERE id = ?", (cluster_id,))
            res.clusters_disbanded += 1
        else:
            conn.execute(
                "UPDATE clusters SET member_count = ? WHERE id = ?",
                (remaining_count, cluster_id),
            )

    conn.commit()

    t1 = time.perf_counter()
    res.elapsed_seconds = t1 - t0
    if res.pairs_evaluated > 0:
        res.time_per_1000 = (res.elapsed_seconds / res.pairs_evaluated) * 1000.0

    return res
