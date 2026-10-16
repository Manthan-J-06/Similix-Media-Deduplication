"""
core/verify.py — Stage 3: Geometric keypoint matching with RANSAC verification.

For each duplicate cluster, verifies every non-representative member against the
cluster representative using keypoint matching (SIFT or ORB) and robust geometric
estimation (RANSAC) on ~512px proxy thumbnails.

Features:
- Tolerates crops, rescaling, and arbitrary rotations (90°, 180°, 270°).
- Outputs inlier count, inlier ratio, and status ("verified", "failed", "inconclusive").
- Inconclusive verdict when either image has fewer than min_keypoints (default 100).
  Inconclusive members keep their existing cluster, are flagged with is_low_confidence = 1,
  and are excluded from auto-delete recommendations.
- Non-matching members (below min_inliers or min_inlier_ratio) are split out of the cluster.
- Re-evaluation: members split out of a cluster are re-evaluated against other surviving
  cluster representatives and against each other to form separate clusters (never simply dropped).
- Caches keypoints and descriptors per image in SQLite (table keypoint_cache) and in-memory
  so representatives and members are never re-detected repeatedly.
- Uses SIFT (patent expired March 2020, Apache 2.0 license in OpenCV 4.4+) as the default
  matcher, with ORB as a configurable alternative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
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
    status: str  # "verified" | "failed" | "inconclusive"
    verified: bool
    inlier_count: int
    inlier_ratio: float
    good_matches: int
    total_keypoints_1: int
    total_keypoints_2: int
    message: str = ""
    transform_matrix: np.ndarray | None = None


def proxy_matrix_to_full(
    M: np.ndarray,
    p1_shape: tuple[int, int],
    o1_shape: tuple[int, int],
    p2_shape: tuple[int, int],
    o2_shape: tuple[int, int],
) -> np.ndarray:
    """
    Convert an affine transformation matrix M estimated between proxy images
    into full-resolution image coordinates.
    p1_shape, p2_shape: (height, width) of proxy images
    o1_shape, o2_shape: (width, height) of original images
    """
    s1x = p1_shape[1] / max(1, o1_shape[0])
    s1y = p1_shape[0] / max(1, o1_shape[1])
    s2x = p2_shape[1] / max(1, o2_shape[0])
    s2y = p2_shape[0] / max(1, o2_shape[1])

    M_full = np.zeros((2, 3), dtype=np.float64)
    M_full[0, 0] = M[0, 0] * (s1x / s2x)
    M_full[0, 1] = M[0, 1] * (s1y / s2x)
    M_full[0, 2] = M[0, 2] / s2x
    M_full[1, 0] = M[1, 0] * (s1x / s2y)
    M_full[1, 1] = M[1, 1] * (s1y / s2y)
    M_full[1, 2] = M[1, 2] / s2y
    return M_full


def full_matrix_to_proxy(
    M_full: np.ndarray,
    p1_shape: tuple[int, int],
    o1_shape: tuple[int, int],
    p2_shape: tuple[int, int],
    o2_shape: tuple[int, int],
) -> np.ndarray:
    """
    Convert a full-resolution affine matrix M_full into target proxy image coordinates.
    """
    s1x = p1_shape[1] / max(1, o1_shape[0])
    s1y = p1_shape[0] / max(1, o1_shape[1])
    s2x = p2_shape[1] / max(1, o2_shape[0])
    s2y = p2_shape[0] / max(1, o2_shape[1])

    M = np.zeros((2, 3), dtype=np.float64)
    M[0, 0] = M_full[0, 0] * (s2x / s1x)
    M[0, 1] = M_full[0, 1] * (s2x / s1y)
    M[0, 2] = M_full[0, 2] * s2x
    M[1, 0] = M_full[1, 0] * (s2y / s1x)
    M[1, 1] = M_full[1, 1] * (s2y / s1y)
    M[1, 2] = M_full[1, 2] * s2y
    return M


def cache_pair_transform(
    conn: sqlite3.Connection,
    source_id: int,
    target_id: int,
    M_full: np.ndarray,
    model: str = "affine",
    inlier_count: int = 0,
    inlier_ratio: float = 0.0,
) -> None:
    """
    Store bidirectional transformation matrices in SQLite pair_transforms table.
    """
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO pair_transforms (source_id, target_id, transform_matrix, model, inlier_count, inlier_ratio)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (source_id, target_id, pickle.dumps(M_full), model, inlier_count, inlier_ratio),
        )
        if model == "affine":
            M_inv = cv2.invertAffineTransform(M_full)
        else:
            M_inv = np.linalg.inv(M_full)
        conn.execute(
            """
            INSERT OR REPLACE INTO pair_transforms (source_id, target_id, transform_matrix, model, inlier_count, inlier_ratio)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (target_id, source_id, pickle.dumps(M_inv), model, inlier_count, inlier_ratio),
        )
    except Exception:
        pass


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
    status: str = "verified"


@dataclass
class Stage3Result:
    pairs_evaluated: int = 0
    pairs_verified: int = 0
    pairs_inconclusive: int = 0
    pairs_failed: int = 0
    members_split: int = 0
    clusters_disbanded: int = 0
    new_clusters_formed: int = 0
    low_confidence_members: int = 0
    elapsed_seconds: float = 0.0
    time_per_1000: float = 0.0
    pair_details: list[PairVerificationRecord] = field(default_factory=list)


def serialize_keypoints(keypoints: list[cv2.KeyPoint]) -> bytes:
    """Serialize OpenCV KeyPoints into compact float32 binary format."""
    if not keypoints:
        return b""
    arr = np.array(
        [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints],
        dtype=np.float32,
    )
    return arr.tobytes()


def deserialize_keypoints(data: bytes) -> list[cv2.KeyPoint]:
    """Deserialize binary format back into a list of OpenCV KeyPoints."""
    if not data:
        return []
    arr = np.frombuffer(data, dtype=np.float32).reshape(-1, 7)
    return [
        cv2.KeyPoint(
            x=float(r[0]),
            y=float(r[1]),
            size=float(r[2]),
            angle=float(r[3]),
            response=float(r[4]),
            octave=int(r[5]),
            class_id=int(r[6]),
        )
        for r in arr
    ]


def serialize_descriptors(descriptors: np.ndarray | None) -> bytes:
    """Serialize numpy feature descriptors into bytes."""
    if descriptors is None or len(descriptors) == 0:
        return b""
    return pickle.dumps(descriptors)


def deserialize_descriptors(data: bytes) -> np.ndarray | None:
    """Deserialize bytes back into numpy feature descriptors."""
    if not data:
        return None
    return pickle.loads(data)


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
        self.min_keypoints = int(stage3_cfg.get("min_keypoints", 100))
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
        Returns MatchResult with status in {"verified", "failed", "inconclusive"}.
        """
        nkp1 = len(kp1)
        nkp2 = len(kp2)

        # Check for insufficient keypoints -> inconclusive verdict
        if nkp1 < self.min_keypoints or nkp2 < self.min_keypoints:
            if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
                return MatchResult(
                    status="inconclusive",
                    verified=True,
                    inlier_count=0,
                    inlier_ratio=0.0,
                    good_matches=0,
                    total_keypoints_1=nkp1,
                    total_keypoints_2=nkp2,
                    message=f"Inconclusive: insufficient keypoints ({min(nkp1, nkp2)} < {self.min_keypoints})",
                )

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return MatchResult(
                status="failed",
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
            if nkp1 < self.min_keypoints or nkp2 < self.min_keypoints:
                return MatchResult(
                    status="inconclusive",
                    verified=True,
                    inlier_count=len(good_matches),
                    inlier_ratio=0.0,
                    good_matches=len(good_matches),
                    total_keypoints_1=nkp1,
                    total_keypoints_2=nkp2,
                    message=f"Inconclusive: insufficient keypoints ({min(nkp1, nkp2)} < {self.min_keypoints})",
                )
            return MatchResult(
                status="failed",
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
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj)
        else:
            M, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj
            )

        if mask is None:
            if nkp1 < self.min_keypoints or nkp2 < self.min_keypoints:
                return MatchResult(
                    status="inconclusive",
                    verified=True,
                    inlier_count=0,
                    inlier_ratio=0.0,
                    good_matches=len(good_matches),
                    total_keypoints_1=nkp1,
                    total_keypoints_2=nkp2,
                    message=f"Inconclusive: insufficient keypoints ({min(nkp1, nkp2)} < {self.min_keypoints})",
                )
            return MatchResult(
                status="failed",
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

        if nkp1 < self.min_keypoints or nkp2 < self.min_keypoints:
            return MatchResult(
                status="inconclusive",
                verified=True,
                inlier_count=inlier_count,
                inlier_ratio=round(inlier_ratio, 4),
                good_matches=len(good_matches),
                total_keypoints_1=nkp1,
                total_keypoints_2=nkp2,
                message=f"Inconclusive: insufficient keypoints ({min(nkp1, nkp2)} < {self.min_keypoints})",
                transform_matrix=M,
            )

        verified = (inlier_count >= self.min_inliers) and (inlier_ratio >= self.min_inlier_ratio)
        status = "verified" if verified else "failed"

        return MatchResult(
            status=status,
            verified=verified,
            inlier_count=inlier_count,
            inlier_ratio=round(inlier_ratio, 4),
            good_matches=len(good_matches),
            total_keypoints_1=nkp1,
            total_keypoints_2=nkp2,
            message="Verified" if verified else f"Failed thresholds (inliers={inlier_count}/{self.min_inliers}, ratio={inlier_ratio:.3f}/{self.min_inlier_ratio})",
            transform_matrix=M,
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


def get_auto_delete_candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    Return cluster members that are safe for auto-delete recommendation.
    Inconclusive / low-confidence members and unalignable members are strictly excluded.
    """
    return conn.execute(
        """
        SELECT * FROM images
        WHERE cluster_id IS NOT NULL
          AND is_keeper = 0
          AND is_low_confidence = 0
          AND (quality_status IS NULL OR quality_status != 'unalignable')
        ORDER BY cluster_id, id
        """
    ).fetchall()


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
    3. If either image has < min_keypoints, verdict is 'inconclusive'; the member keeps
       its cluster, gets is_low_confidence = 1 in the database, and is excluded from auto-delete.
    4. Members that fail geometric verification are split out of the cluster.
    5. Disbands clusters reduced to fewer than 2 members.
    6. Re-evaluates split members: matches them against other cluster representatives and
       against each other to form separate clusters (never simply dropped).
    7. Caches keypoints and descriptors in SQLite table keypoint_cache and in-memory.
    """
    t0 = time.perf_counter()
    verifier = KeypointVerifier(config)

    # Ensure keypoint cache table, pair_transforms table, and is_low_confidence column exist
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS keypoint_cache (
            image_id    INTEGER PRIMARY KEY,
            keypoints   BLOB,
            descriptors BLOB,
            created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pair_transforms (
            source_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            target_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            transform_matrix BLOB NOT NULL,
            model            TEXT NOT NULL DEFAULT 'affine',
            inlier_count     INTEGER NOT NULL DEFAULT 0,
            inlier_ratio     REAL NOT NULL DEFAULT 0.0,
            created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
            PRIMARY KEY (source_id, target_id)
        );
        """
    )
    cursor = conn.execute("PRAGMA table_info(images);")
    existing_cols = {row["name"] for row in cursor.fetchall()}
    if "is_low_confidence" not in existing_cols:
        conn.execute("ALTER TABLE images ADD COLUMN is_low_confidence INTEGER NOT NULL DEFAULT 0;")

    # Cache original dimensions for coordinate transformation
    orig_dims: dict[int, tuple[int, int]] = {}
    for r in conn.execute("SELECT id, width, height, file_path FROM images").fetchall():
        w, h = r["width"], r["height"]
        if not w or not h:
            try:
                with Image.open(r["file_path"]) as im:
                    w, h = im.size
                conn.execute("UPDATE images SET width = ?, height = ? WHERE id = ?", (w, h, r["id"]))
            except Exception:
                w, h = 1024, 768
        orig_dims[r["id"]] = (w, h)

    def get_proxy_shape(img_id: int) -> tuple[int, int]:
        w, h = orig_dims.get(img_id, (1024, 1024))
        max_dim = max(w, h)
        scale = verifier.proxy_size / max_dim if max_dim > verifier.proxy_size else 1.0
        pw = max(1, int(round(w * scale)))
        ph = max(1, int(round(h * scale)))
        return (ph, pw)

    clusters_rows = conn.execute(
        """
        SELECT c.id, c.representative_id, i.file_path as rep_path
        FROM clusters c
        JOIN images i ON i.id = c.representative_id
        ORDER BY c.id
        """
    ).fetchall()

    res = Stage3Result()
    mem_cache: dict[int, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}

    def get_features(image_id: int, file_path: str) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        if image_id in mem_cache:
            return mem_cache[image_id]

        row = conn.execute(
            "SELECT keypoints, descriptors FROM keypoint_cache WHERE image_id = ?",
            (image_id,),
        ).fetchone()

        if row is not None and row["keypoints"] is not None:
            kps = deserialize_keypoints(row["keypoints"])
            des = deserialize_descriptors(row["descriptors"])
            mem_cache[image_id] = (kps, des)
            return mem_cache[image_id]

        proxy = load_proxy_grayscale(file_path, proxy_size=verifier.proxy_size, source=source)
        kps, des = verifier.extract_features(proxy)
        conn.execute(
            "INSERT OR REPLACE INTO keypoint_cache (image_id, keypoints, descriptors) VALUES (?, ?, ?)",
            (image_id, serialize_keypoints(kps), serialize_descriptors(des)),
        )
        mem_cache[image_id] = (kps, des)
        return mem_cache[image_id]

    split_pool: list[tuple[int, str]] = []

    for c_row in clusters_rows:
        cluster_id = c_row["id"]
        rep_id = c_row["representative_id"]
        rep_path = c_row["rep_path"]

        rep_kp, rep_des = get_features(rep_id, rep_path)

        members = conn.execute(
            """
            SELECT id, file_path
            FROM images
            WHERE cluster_id = ? AND id != ?
            ORDER BY id
            """,
            (cluster_id, rep_id),
        ).fetchall()

        members_to_split: list[tuple[int, str]] = []
        cluster_has_inconclusive = (len(rep_kp) < verifier.min_keypoints)

        for m_row in members:
            m_id = m_row["id"]
            m_path = m_row["file_path"]

            m_kp, m_des = get_features(m_id, m_path)
            match_res = verifier.match_features(rep_kp, rep_des, m_kp, m_des)

            res.pairs_evaluated += 1

            if match_res.status == "inconclusive":
                res.pairs_inconclusive += 1
                cluster_has_inconclusive = True
                conn.execute("UPDATE images SET is_low_confidence = 1 WHERE id = ?", (m_id,))
                if match_res.transform_matrix is not None:
                    p1_s = get_proxy_shape(rep_id)
                    o1_s = orig_dims.get(rep_id, (1024, 1024))
                    p2_s = get_proxy_shape(m_id)
                    o2_s = orig_dims.get(m_id, (1024, 1024))
                    M_full = proxy_matrix_to_full(match_res.transform_matrix, p1_s, o1_s, p2_s, o2_s)
                    cache_pair_transform(conn, rep_id, m_id, M_full, verifier.model, match_res.inlier_count, match_res.inlier_ratio)
            elif match_res.status == "verified":
                res.pairs_verified += 1
                conn.execute("UPDATE images SET is_low_confidence = 0 WHERE id = ?", (m_id,))
                if match_res.transform_matrix is not None:
                    p1_s = get_proxy_shape(rep_id)
                    o1_s = orig_dims.get(rep_id, (1024, 1024))
                    p2_s = get_proxy_shape(m_id)
                    o2_s = orig_dims.get(m_id, (1024, 1024))
                    M_full = proxy_matrix_to_full(match_res.transform_matrix, p1_s, o1_s, p2_s, o2_s)
                    cache_pair_transform(conn, rep_id, m_id, M_full, verifier.model, match_res.inlier_count, match_res.inlier_ratio)
            else:
                res.pairs_failed += 1
                members_to_split.append((m_id, m_path))
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
                    status=match_res.status,
                )
            )

        if cluster_has_inconclusive:
            conn.execute("UPDATE images SET is_low_confidence = 1 WHERE id = ?", (rep_id,))
        else:
            conn.execute("UPDATE images SET is_low_confidence = 0 WHERE id = ?", (rep_id,))

        # Split failing members out of the cluster
        if members_to_split:
            for mid, mpath in members_to_split:
                split_pool.append((mid, mpath))
            conn.executemany(
                """
                UPDATE images
                SET cluster_id        = NULL,
                    is_keeper         = 0,
                    confidence        = NULL,
                    link_signal       = NULL,
                    status            = 'embedded',
                    is_low_confidence = 0,
                    updated_at        = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                [(mid,) for mid, _ in members_to_split],
            )

        # Check remaining member count in this cluster
        remaining_count = conn.execute(
            "SELECT COUNT(*) FROM images WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()[0]

        if remaining_count < 2:
            # Cluster reduced below 2 members: disband and add remaining rep to split_pool
            rep_row = conn.execute("SELECT id, file_path FROM images WHERE id = ?", (rep_id,)).fetchone()
            if rep_row is not None:
                split_pool.append((rep_row["id"], rep_row["file_path"]))
            conn.execute(
                """
                UPDATE images
                SET cluster_id        = NULL,
                    is_keeper         = 0,
                    confidence        = NULL,
                    link_signal       = NULL,
                    status            = 'embedded',
                    is_low_confidence = 0,
                    updated_at        = strftime('%Y-%m-%dT%H:%M:%SZ','now')
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

    # ── Re-evaluation of Split Members ─────────────────────────────────────
    # Members split out by Stage 3 must be re-evaluated:
    # 1. Matched against other surviving clusters' representatives
    # 2. Matched against each other using the same verification to form new clusters
    if split_pool:
        surviving_clusters = conn.execute(
            """
            SELECT c.id, c.representative_id, i.file_path as rep_path
            FROM clusters c
            JOIN images i ON i.id = c.representative_id
            ORDER BY c.id
            """
        ).fetchall()

        still_split: list[tuple[int, str]] = []

        # 1. Try matching against other surviving clusters
        for mid, mpath in split_pool:
            mkp, mdes = get_features(mid, mpath)
            matched_cid: int | None = None

            for sc in surviving_clusters:
                rep_kp, rep_des = get_features(sc["representative_id"], sc["rep_path"])
                mres = verifier.match_features(rep_kp, rep_des, mkp, mdes)
                if mres.status == "verified":
                    matched_cid = sc["id"]
                    if mres.transform_matrix is not None:
                        p1_s = get_proxy_shape(sc["representative_id"])
                        o1_s = orig_dims.get(sc["representative_id"], (1024, 1024))
                        p2_s = get_proxy_shape(mid)
                        o2_s = orig_dims.get(mid, (1024, 1024))
                        M_full = proxy_matrix_to_full(mres.transform_matrix, p1_s, o1_s, p2_s, o2_s)
                        cache_pair_transform(conn, sc["representative_id"], mid, M_full, verifier.model, mres.inlier_count, mres.inlier_ratio)
                    conn.execute(
                        """
                        UPDATE images
                        SET cluster_id        = ?,
                            is_keeper         = 0,
                            confidence        = 100.0,
                            link_signal       = 'sift',
                            status            = 'clustered',
                            is_low_confidence = 0,
                            updated_at        = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                        WHERE id = ?
                        """,
                        (matched_cid, mid),
                    )
                    conn.execute(
                        "UPDATE clusters SET member_count = member_count + 1 WHERE id = ?",
                        (matched_cid,),
                    )
                    break

            if matched_cid is None:
                still_split.append((mid, mpath))

        # 2. Group remaining split members amongst each other
        while still_split:
            seed_id, seed_path = still_split.pop(0)
            seed_kp, seed_des = get_features(seed_id, seed_path)
            cand_members: list[tuple[int, str]] = [(seed_id, seed_path)]

            unmatched: list[tuple[int, str]] = []
            for mid, mpath in still_split:
                mkp, mdes = get_features(mid, mpath)
                mres = verifier.match_features(seed_kp, seed_des, mkp, mdes)
                if mres.status == "verified":
                    cand_members.append((mid, mpath))
                    if mres.transform_matrix is not None:
                        p1_s = get_proxy_shape(seed_id)
                        o1_s = orig_dims.get(seed_id, (1024, 1024))
                        p2_s = get_proxy_shape(mid)
                        o2_s = orig_dims.get(mid, (1024, 1024))
                        M_full = proxy_matrix_to_full(mres.transform_matrix, p1_s, o1_s, p2_s, o2_s)
                        cache_pair_transform(conn, seed_id, mid, M_full, verifier.model, mres.inlier_count, mres.inlier_ratio)
                else:
                    unmatched.append((mid, mpath))

            still_split = unmatched

            if len(cand_members) >= 2:
                # Form new cluster
                conn.execute(
                    """
                    INSERT INTO clusters (representative_id, stage_detected, member_count)
                    VALUES (?, 3, ?)
                    """,
                    (seed_id, len(cand_members)),
                )
                new_cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                has_low = 1 if len(seed_kp) < verifier.min_keypoints else 0

                conn.execute(
                    """
                    UPDATE images
                    SET cluster_id        = ?,
                        is_keeper         = 1,
                        confidence        = 100.0,
                        link_signal       = 'sift',
                        status            = 'clustered',
                        is_low_confidence = ?,
                        updated_at        = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                    WHERE id = ?
                    """,
                    (new_cid, has_low, seed_id),
                )
                for mid, _mpath in cand_members[1:]:
                    conn.execute(
                        """
                        UPDATE images
                        SET cluster_id        = ?,
                            is_keeper         = 0,
                            confidence        = 100.0,
                            link_signal       = 'sift',
                            status            = 'clustered',
                            is_low_confidence = ?,
                            updated_at        = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                        WHERE id = ?
                        """,
                        (new_cid, has_low, mid),
                    )
                res.new_clusters_formed += 1
            else:
                conn.execute(
                    """
                    UPDATE images
                    SET cluster_id        = NULL,
                        is_keeper         = 0,
                        confidence        = NULL,
                        link_signal       = NULL,
                        status            = 'embedded',
                        is_low_confidence = 0,
                        updated_at        = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                    WHERE id = ?
                    """,
                    (seed_id,),
                )

    res.low_confidence_members = conn.execute(
        "SELECT COUNT(*) FROM images WHERE cluster_id IS NOT NULL AND is_low_confidence = 1"
    ).fetchone()[0]

    conn.commit()

    t1 = time.perf_counter()
    res.elapsed_seconds = t1 - t0
    if res.pairs_evaluated > 0:
        res.time_per_1000 = (res.elapsed_seconds / res.pairs_evaluated) * 1000.0

    return res
