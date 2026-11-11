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
import logging
import pickle
import sqlite3
import struct
import time
from typing import Any, Callable
import zlib

import cv2
import numpy as np
from PIL import Image

from sources.base import ImageSource

logger = logging.getLogger("argus.verify")


@dataclass
class MatchResult:
    """Detailed result of keypoint matching and RANSAC verification between two images."""
    status: str = "inconclusive"  # "verified" | "failed" | "inconclusive"
    verified: bool = False
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    good_matches: int = 0
    total_keypoints_1: int = 0
    total_keypoints_2: int = 0
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


_KP_MAGIC = b"SMXKPZ01"
_DS_MAGIC = b"SMXDSZ01"


def serialize_keypoints(keypoints: list[cv2.KeyPoint]) -> bytes:
    """Serialize OpenCV KeyPoints into a zlib-compressed float32 binary format."""
    if not keypoints:
        return b""
    arr = np.array(
        [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints],
        dtype=np.float32,
    )
    return _KP_MAGIC + zlib.compress(arr.tobytes(), level=1)


def deserialize_keypoints(data: bytes) -> list[cv2.KeyPoint]:
    """Deserialize binary format (compressed or legacy) back into OpenCV KeyPoints."""
    if not data:
        return []
    raw = zlib.decompress(data[len(_KP_MAGIC):]) if data.startswith(_KP_MAGIC) else data
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, 7)
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
    """Serialize numpy feature descriptors into a compact zlib-compressed binary payload."""
    if descriptors is None or len(descriptors) == 0:
        return b""
    arr = np.ascontiguousarray(descriptors)
    if arr.ndim != 2:
        return _DS_MAGIC + b"PK" + struct.pack("<II", 0, 0) + zlib.compress(pickle.dumps(arr), level=1)

    rows, cols = arr.shape
    if arr.dtype == np.uint8:
        tag = b"U8"
        payload = arr.tobytes()
    elif arr.dtype == np.float32 and np.all((arr >= 0) & (arr <= 255) & (arr == np.floor(arr))):
        # OpenCV SIFT descriptors are integer values in [0, 255] stored as float32; store losslessly as uint8.
        tag = b"U3"
        payload = arr.astype(np.uint8).tobytes()
    elif arr.dtype == np.float32:
        tag = b"F3"
        payload = arr.tobytes()
    else:
        return _DS_MAGIC + b"PK" + struct.pack("<II", rows, cols) + zlib.compress(pickle.dumps(arr), level=1)

    return _DS_MAGIC + tag + struct.pack("<II", rows, cols) + zlib.compress(payload, level=1)


def deserialize_descriptors(data: bytes) -> np.ndarray | None:
    """Deserialize bytes (compressed or legacy pickle) back into numpy feature descriptors."""
    if not data:
        return None
    if not data.startswith(_DS_MAGIC):
        return pickle.loads(data)

    header_len = len(_DS_MAGIC) + 2 + 8
    tag = data[len(_DS_MAGIC):len(_DS_MAGIC) + 2]
    rows, cols = struct.unpack("<II", data[len(_DS_MAGIC) + 2:header_len])
    raw = zlib.decompress(data[header_len:])

    if tag == b"U3":
        return np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols).astype(np.float32)
    if tag == b"U8":
        return np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols).copy()
    if tag == b"F3":
        return np.frombuffer(raw, dtype=np.float32).reshape(rows, cols).copy()
    return pickle.loads(raw)


def prune_keypoint_cache(
    conn: sqlite3.Connection,
    keep_representatives: bool = True,
    max_rep_rows: int = 500,
    vacuum_threshold_mb: float = 10.0,
) -> int:
    """Prune bloated keypoint_cache rows after verification to reclaim SQLite disk space."""
    tbl = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='keypoint_cache'"
    ).fetchone()
    if not tbl:
        return 0

    if keep_representatives:
        # Remove entries for non-representatives (singletons and duplicate members)
        cur = conn.execute(
            """
            DELETE FROM keypoint_cache
            WHERE image_id NOT IN (
                SELECT representative_id FROM clusters
                WHERE representative_id IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (max(0, int(max_rep_rows)),),
        )
    else:
        cur = conn.execute("DELETE FROM keypoint_cache")

    deleted = cur.rowcount or 0
    conn.commit()

    if deleted > 0:
        freelist_pages = conn.execute("PRAGMA freelist_count;").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size;").fetchone()[0]
        freed_mb = (freelist_pages * page_size) / (1024 * 1024)
        if freed_mb >= vacuum_threshold_mb:
            conn.execute("VACUUM;")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    return deleted


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
        # Drive-aware loading: resolve drive:// URIs through the local thumbnail cache
        # regardless of what source type was passed in (fixes the case where
        # LocalFolderSource is used but the DB contains gdrive images).
        if file_path.startswith("drive://"):
            drive_id = file_path[8:].split("/")[0]
            cache_dirs: list[Path] = []
            if source is not None and hasattr(source, "cache_dir") and getattr(source, "cache_dir", None):
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
            if (Path.home() / ".similix" / "users").is_dir():
                cache_dirs.extend(sorted((Path.home() / ".similix" / "users").glob("*/drive_cache")))
            if (Path.cwd() / ".similix" / "users").is_dir():
                cache_dirs.extend(sorted((Path.cwd() / ".similix" / "users").glob("*/drive_cache")))
            cached_file: Path | None = None
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
                    im = raw_im.copy()
            elif source is not None and hasattr(source, "get_thumbnail"):
                im = source.get_thumbnail(file_path, size=(proxy_size, proxy_size))
            else:
                # Cannot resolve drive URI — raise so the caller can handle gracefully
                raise FileNotFoundError(f"drive:// URI not resolvable from cache: {file_path}")
        elif source is not None:
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
        cv2.setRNGSeed(42)

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

        verified = (inlier_count >= self.min_inliers) and (inlier_ratio >= self.min_inlier_ratio)
        if verified:
            return MatchResult(
                status="verified",
                verified=True,
                inlier_count=inlier_count,
                inlier_ratio=round(inlier_ratio, 4),
                good_matches=len(good_matches),
                total_keypoints_1=nkp1,
                total_keypoints_2=nkp2,
                message="Verified",
                transform_matrix=M,
            )

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

        return MatchResult(
            status="failed",
            verified=False,
            inlier_count=inlier_count,
            inlier_ratio=round(inlier_ratio, 4),
            good_matches=len(good_matches),
            total_keypoints_1=nkp1,
            total_keypoints_2=nkp2,
            message=f"Failed thresholds (inliers={inlier_count}/{self.min_inliers}, ratio={inlier_ratio:.3f}/{self.min_inlier_ratio})",
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
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> Stage3Result:
    """
    Execute Stage 3 verification on all duplicate clusters in SQLite.

    For each cluster:
    1. Loads the cluster representative proxy thumbnail.
    2. Compares each non-representative cluster member against the representative.
    3. If either image has < min_keypoints, verdict is 'inconclusive'; the member keeps
       its cluster, gets is_low_confidence = 1 in the database, and is excluded from auto-delete.
    4. Members that fail geometric verification leave the cluster and become singletons (cluster_id = NULL).
    5. Disbands clusters reduced to fewer than 2 members.
    6. Caches keypoints and descriptors in SQLite table keypoint_cache and in-memory.
    """
    t0 = time.perf_counter()
    cv2.setRNGSeed(42)
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

    total_pairs = conn.execute(
        "SELECT COUNT(*) FROM images WHERE cluster_id IS NOT NULL AND is_keeper = 0"
    ).fetchone()[0]

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

        try:
            proxy = load_proxy_grayscale(file_path, proxy_size=verifier.proxy_size, source=source)
            kps, des = verifier.extract_features(proxy)
        except Exception as exc:
            logger.warning("Failed to extract keypoints for %s: %s", file_path, exc)
            kps, des = [], None

        conn.execute(
            "INSERT OR REPLACE INTO keypoint_cache (image_id, keypoints, descriptors) VALUES (?, ?, ?)",
            (image_id, serialize_keypoints(kps), serialize_descriptors(des)),
        )
        mem_cache[image_id] = (kps, des)
        return mem_cache[image_id]

    for c_row in clusters_rows:
        cluster_id = c_row["id"]
        rep_id = c_row["representative_id"]
        rep_path = c_row["rep_path"]

        rep_kp, rep_des = get_features(rep_id, rep_path)
        rep_md5_row = conn.execute("SELECT md5 FROM images WHERE id = ?", (rep_id,)).fetchone()
        rep_md5 = rep_md5_row["md5"] if rep_md5_row else None

        members = conn.execute(
            """
            SELECT id, file_path, md5
            FROM images
            WHERE cluster_id = ? AND id != ?
            ORDER BY id
            """,
            (cluster_id, rep_id),
        ).fetchall()

        members_to_split: list[tuple[int, str]] = []
        cluster_has_inconclusive = False
        verified_members: list[tuple[int, str]] = [(rep_id, rep_path)]

        for m_row in members:
            m_id = m_row["id"]
            m_path = m_row["file_path"]
            m_md5 = m_row["md5"]

            # Fast path: Byte-for-byte exact duplicate files bypass SIFT entirely
            if rep_md5 and m_md5 and rep_md5 == m_md5:
                match_res = MatchResult(
                    verified=True,
                    status="verified",
                    inlier_count=500,
                    inlier_ratio=1.0,
                    good_matches=500,
                    total_keypoints_1=500,
                    total_keypoints_2=500,
                    transform_matrix=np.eye(3, dtype=np.float32),
                    message="Exact MD5 duplicate",
                )
            else:
                m_kp, m_des = get_features(m_id, m_path)
                match_res = verifier.match_features(rep_kp, rep_des, m_kp, m_des)

            res.pairs_evaluated += 1
            if progress_callback:
                try:
                    progress_callback(res.pairs_evaluated, max(1, total_pairs), m_path)
                except Exception:
                    pass

            if match_res.status == "inconclusive":
                # Guard against impostor false-positives masquerading as inconclusive due to low keypoints
                rep_meta = conn.execute("SELECT phash, clip_vector FROM images WHERE id = ?", (rep_id,)).fetchone()
                m_meta = conn.execute("SELECT phash, clip_vector FROM images WHERE id = ?", (m_id,)).fetchone()
                is_obvious_mismatch = False
                is_confirmed_match = False
                if rep_meta and m_meta:
                    p1, p2 = rep_meta["phash"], m_meta["phash"]
                    v1, v2 = rep_meta["clip_vector"], m_meta["clip_vector"]
                    pdist = None
                    if p1 and p2 and len(p1) == len(p2):
                        try:
                            pdist = (int(p1, 16) ^ int(p2, 16)).bit_count()
                        except Exception:
                            pass
                    csim = None
                    if v1 and v2:
                        try:
                            from core.embeddings import deserialize_embedding
                            arr1 = deserialize_embedding(v1).astype(np.float32)
                            arr2 = deserialize_embedding(v2).astype(np.float32)
                            norm1 = np.linalg.norm(arr1)
                            norm2 = np.linalg.norm(arr2)
                            if norm1 > 0 and norm2 > 0:
                                csim = float(np.dot(arr1, arr2) / (norm1 * norm2))
                        except Exception:
                            pass

                    # 1. Obvious mismatch: different scenes (e.g., baseball glove vs coffee cup)
                    if pdist is not None and pdist > 12 and (csim is None or csim < 0.90):
                        is_obvious_mismatch = True
                    elif csim is not None and csim < 0.80:
                        is_obvious_mismatch = True
                    elif (pdist is not None and pdist > 8) and (csim is not None and csim < 0.88):
                        is_obvious_mismatch = True

                    # 2. Confirmed genuine duplicate on low-texture flat surfaces (insufficient keypoints)
                    if pdist is not None and pdist <= 8:
                        is_confirmed_match = True

                if is_obvious_mismatch:
                    match_res.status = "failed"
                    match_res.verified = False
                    match_res.message = "Failed: distinct images (pHash/CLIP mismatch)"
                elif is_confirmed_match:
                    match_res.status = "verified"
                    match_res.verified = True
                    match_res.message = "Verified: low-texture match confirmed by pHash"

            # Connected verification fallback for burst sequences:
            # If star comparison to cluster representative failed, check against other already-verified cluster members
            if match_res.status == "failed" and len(verified_members) > 1:
                for v_id, v_path in list(verified_members):
                    if v_id == rep_id:
                        continue
                    v_kp, v_des = get_features(v_id, v_path)
                    v_res = verifier.match_features(v_kp, v_des, m_kp, m_des)
                    if v_res.status == "verified":
                        match_res = v_res
                        if match_res.transform_matrix is not None:
                            p1_s = get_proxy_shape(v_id)
                            o1_s = orig_dims.get(v_id, (1024, 1024))
                            p2_s = get_proxy_shape(m_id)
                            o2_s = orig_dims.get(m_id, (1024, 1024))
                            M_full = proxy_matrix_to_full(match_res.transform_matrix, p1_s, o1_s, p2_s, o2_s)
                            cache_pair_transform(conn, v_id, m_id, M_full, verifier.model, match_res.inlier_count, match_res.inlier_ratio)
                        break

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
                verified_members.append((m_id, m_path))
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
            # Cluster reduced below 2 members: disband into singletons
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
        conn.commit()

    # Populate images.confidence using the fused logistic regression model
    update_cluster_duplicate_confidences(conn)

    res.low_confidence_members = conn.execute(
        "SELECT COUNT(*) FROM images WHERE cluster_id IS NOT NULL AND is_low_confidence = 1"
    ).fetchone()[0]

    conn.commit()

    t1 = time.perf_counter()
    res.elapsed_seconds = t1 - t0
    if res.pairs_evaluated > 0:
        res.time_per_1000 = (res.elapsed_seconds / res.pairs_evaluated) * 1000.0

    return res


def update_cluster_duplicate_confidences(
    conn: sqlite3.Connection,
    cluster_ids: list[int] | None = None,
    model_path: str | Path | None = None,
) -> int:
    """
    Populate images.confidence using the fused logistic regression model
    (models/duplicate_confidence.json) for all clustered images.
    - Exact MD5 matches and cluster representatives are fixed at 100.0%.
    - Non-MD5 members use pHash distance, best CLIP cosine across rotations,
      and SIFT inlier count / ratio (from pair_transforms table if present).
    """
    from core.keeper import predict_duplicate_confidence
    from core.embeddings import deserialize_embedding

    if cluster_ids is not None:
        if not cluster_ids:
            return 0
        placeholders = ",".join("?" for _ in cluster_ids)
        clusters = conn.execute(
            f"SELECT id, representative_id FROM clusters WHERE id IN ({placeholders})",
            cluster_ids,
        ).fetchall()
    else:
        clusters = conn.execute(
            "SELECT id, representative_id FROM clusters WHERE representative_id IS NOT NULL"
        ).fetchall()

    updated_count = 0
    for c in clusters:
        cid = c["id"]
        rep_id = c["representative_id"]
        if rep_id is None:
            continue

        rep = conn.execute(
            "SELECT id, md5, phash, clip_vector FROM images WHERE id = ?",
            (rep_id,),
        ).fetchone()
        if not rep:
            continue

        # Representative is anchor -> 1.0
        conn.execute(
            "UPDATE images SET confidence = 1.0 WHERE id = ?",
            (rep_id,),
        )

        members = conn.execute(
            """
            SELECT id, md5, phash, link_signal,
                   clip_vector, clip_vector_90, clip_vector_180, clip_vector_270
            FROM images
            WHERE cluster_id = ? AND id != ?
            """,
            (cid, rep_id),
        ).fetchall()

        rep_clip = deserialize_embedding(rep["clip_vector"]) if rep["clip_vector"] else None

        for m in members:
            mid = m["id"]
            is_md5 = (m["md5"] == rep["md5"]) or (m["link_signal"] == "md5")
            if is_md5:
                conf = 1.0
            else:
                # 1. pHash distance
                if rep["phash"] and m["phash"]:
                    h_dist = bin(int(rep["phash"], 16) ^ int(m["phash"], 16)).count("1")
                else:
                    h_dist = 0

                # 2. best CLIP cosine across rotations
                best_cos = 1.0
                if rep_clip is not None and m["clip_vector"]:
                    m_clip = deserialize_embedding(m["clip_vector"])
                    cosines = [float(np.dot(rep_clip, m_clip))]
                    for rot_col in ["clip_vector_90", "clip_vector_180", "clip_vector_270"]:
                        if m[rot_col]:
                            v_rot = deserialize_embedding(m[rot_col])
                            cosines.append(float(np.dot(rep_clip, v_rot)))
                    best_cos = max(cosines)

                # 3. SIFT features from pair_transforms table if present
                pt = conn.execute(
                    """
                    SELECT inlier_count, inlier_ratio FROM pair_transforms
                    WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)
                    """,
                    (rep_id, mid, mid, rep_id),
                ).fetchone()

                inliers = int(pt["inlier_count"]) if pt else 0
                ratio = float(pt["inlier_ratio"]) if pt else 0.0

                conf = predict_duplicate_confidence(
                    phash_distance=h_dist,
                    best_clip_cosine=best_cos,
                    sift_inlier_ratio=ratio,
                    inlier_count=inliers,
                    md5_match=False,
                    model_path=model_path,
                )

            conn.execute(
                "UPDATE images SET confidence = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id = ?",
                (conf, mid),
            )
            updated_count += 1

    return updated_count
