"""
core/embeddings.py — Stage 2: CLIP embeddings with Apple MPS acceleration.

Features:
- open_clip with ViT-B-32 model (permissively licensed under MIT).
- Apple MPS backend when available, fallback to CPU.
- Uses ~256-512px thumbnails via ImageSource (never full-res files).
- Float16 precision, L2-normalized vectors (512 dimensions = 1024 bytes per image).
- Small configurable batches (default 32) with explicit memory cleanup between batches.
- Rotation handling: optionally stores 90/180/270 degree rotated thumbnail embeddings
  (default disabled in config.toml to avoid compute overhead).
- Resumable: skips images that already have embeddings.
"""

from __future__ import annotations

import gc
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    import faiss  # noqa: F401
except ImportError:
    pass

import numpy as np
import open_clip
import torch
from PIL import Image

from sources.base import ImageSource
from sources.local_folder import LocalFolderSource


def get_device() -> torch.device:
    """Return Apple Silicon MPS device if available and built, else CPU."""
    dev_str = os.environ.get("ARGUS_DEVICE") or os.environ.get("TORCH_DEVICE")
    if dev_str:
        return torch.device(dev_str)
    if os.environ.get("CI"):
        return torch.device("cpu")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def is_clip_weights_cached(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
) -> bool:
    """Check if open_clip model weights are available in local cache without network access."""
    try:
        import open_clip.pretrained
        cfg = open_clip.pretrained.get_pretrained_cfg(model_name, pretrained)
        if not cfg:
            return False
        hf_hub = cfg.get("hf_hub")
        if hf_hub:
            from huggingface_hub import try_to_load_from_cache
            repo_id = hf_hub.strip("/")
            for fname in ("open_clip_model.safetensors", "open_clip_pytorch_model.bin"):
                res = try_to_load_from_cache(repo_id, fname)
                if isinstance(res, (str, Path)) and Path(res).exists():
                    return True
        url = cfg.get("url")
        if url:
            torch_hub = Path(torch.hub.get_dir()) / "checkpoints"
            fname = Path(url).name
            if (torch_hub / fname).exists():
                return True
    except Exception:
        pass
    return False


class CLIPEmbedder:
    """Wrapper around open_clip model and preprocessing pipeline."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: torch.device | None = None,
    ):
        self.device = device or get_device()
        self.model_name = model_name
        self.pretrained = pretrained

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()

    def embed_thumbnails(self, thumbnails: list[Image.Image]) -> np.ndarray:
        """
        Compute L2-normalized float16 embeddings for a batch of PIL thumbnails.

        Returns:
            np.ndarray of shape (N, 512), dtype float16, unit L2 norm.
        """
        if not thumbnails:
            return np.empty((0, 512), dtype=np.float16)

        tensors = torch.stack([self.preprocess(t) for t in thumbnails]).to(self.device)

        with torch.inference_mode():
            features = self.model.encode_image(tensors)
            # L2 normalize
            features = features / features.norm(dim=-1, keepdim=True)
            emb_np = features.cpu().to(torch.float16).numpy()

        # Free memory immediately
        del tensors, features
        if self.device.type == "mps":
            torch.mps.empty_cache()

        return emb_np


def serialize_embedding(arr: np.ndarray) -> bytes:
    """Serialize float16 embedding numpy array to raw bytes for SQLite BLOB storage."""
    if arr.dtype != np.float16:
        arr = arr.astype(np.float16)
    return arr.tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize raw bytes from SQLite BLOB into float16 numpy array."""
    return np.frombuffer(blob, dtype=np.float16)


@dataclass
class EmbeddingResult:
    images_embedded: int = 0
    images_skipped: int = 0
    device_used: str = "cpu"
    batch_size: int = 32
    rotations_enabled: bool = False


def run_embeddings(
    source_or_root: ImageSource | Path | str,
    conn: sqlite3.Connection,
    config: dict,
    progress_callback: Callable[[int, int], None] | None = None,
    embedder: CLIPEmbedder | None = None,
) -> EmbeddingResult:
    """
    Generate and store CLIP embeddings for all pending images in batches.

    Resumable: only processes images where clip_vector IS NULL.
    """
    if isinstance(source_or_root, ImageSource):
        source = source_or_root
    else:
        source = LocalFolderSource(source_or_root)

    stage2_cfg = config.get("stage2", {})
    batch_size = stage2_cfg.get("batch_size", 32)
    model_name = stage2_cfg.get("clip_model", "ViT-B-32")
    pretrained = stage2_cfg.get("clip_pretrained", "laion2b_s34b_b79k")
    enable_rotations = stage2_cfg.get("enable_rotation_embeddings", False)
    thumb_size = config.get("general", {}).get("thumbnail_size", 256)

    # Count pending images
    total_pending = conn.execute(
        """
        SELECT COUNT(*) FROM images
        WHERE clip_vector IS NULL
          AND is_raw_pair = 0
          AND (status IS NULL OR status != 'skipped')
        """
    ).fetchone()[0]

    already_done = conn.execute(
        """
        SELECT COUNT(*) FROM images
        WHERE clip_vector IS NOT NULL
        """
    ).fetchone()[0]

    if total_pending == 0:
        return EmbeddingResult(
            images_embedded=0,
            images_skipped=already_done,
            device_used=str(embedder.device if embedder else get_device()),
            batch_size=batch_size,
            rotations_enabled=enable_rotations,
        )

    # Initialize model if not passed in
    if embedder is None:
        embedder = CLIPEmbedder(
            model_name=model_name,
            pretrained=pretrained,
        )

    processed_count = 0

    while True:
        batch_rows = conn.execute(
            """
            SELECT id, file_path FROM images
            WHERE clip_vector IS NULL
              AND is_raw_pair = 0
              AND (status IS NULL OR status != 'skipped')
            ORDER BY id
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()

        if not batch_rows:
            break

        # Load thumbnails (draft mode on JPEG)
        thumbs_0: list[Image.Image] = []
        valid_ids: list[int] = []
        failed_ids: list[int] = []

        for row in batch_rows:
            img_id = row["id"]
            file_id = row["file_path"]
            try:
                t = source.get_thumbnail(file_id, size=(thumb_size, thumb_size))
                thumbs_0.append(t)
                valid_ids.append(img_id)
            except Exception:
                failed_ids.append(img_id)
                continue

        if failed_ids:
            conn.executemany(
                """
                UPDATE images
                SET status = 'skipped',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                WHERE id = ?
                """,
                [(fid,) for fid in failed_ids],
            )
            conn.commit()

        if not valid_ids:
            continue

        # Embed standard (0 degree) thumbnails
        embs_0 = embedder.embed_thumbnails(thumbs_0)

        embs_90: np.ndarray | None = None
        embs_180: np.ndarray | None = None
        embs_270: np.ndarray | None = None

        if enable_rotations:
            # 90 degrees
            thumbs_90 = [t.rotate(90, expand=True) for t in thumbs_0]
            embs_90 = embedder.embed_thumbnails(thumbs_90)
            for t in thumbs_90:
                t.close()

            # 180 degrees
            thumbs_180 = [t.rotate(180, expand=True) for t in thumbs_0]
            embs_180 = embedder.embed_thumbnails(thumbs_180)
            for t in thumbs_180:
                t.close()

            # 270 degrees
            thumbs_270 = [t.rotate(270, expand=True) for t in thumbs_0]
            embs_270 = embedder.embed_thumbnails(thumbs_270)
            for t in thumbs_270:
                t.close()

        # Close base thumbnails
        for t in thumbs_0:
            t.close()

        # Write to SQLite
        update_params = []
        for i, img_id in enumerate(valid_ids):
            blob_0 = serialize_embedding(embs_0[i])
            blob_90 = serialize_embedding(embs_90[i]) if embs_90 is not None else None
            blob_180 = serialize_embedding(embs_180[i]) if embs_180 is not None else None
            blob_270 = serialize_embedding(embs_270[i]) if embs_270 is not None else None

            update_params.append((blob_0, blob_90, blob_180, blob_270, img_id))

        conn.executemany(
            """
            UPDATE images
            SET clip_vector = ?,
                clip_vector_90 = ?,
                clip_vector_180 = ?,
                clip_vector_270 = ?,
                status = 'embedded',
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            WHERE id = ?
            """,
            update_params,
        )
        conn.commit()

        processed_count += len(valid_ids)
        if progress_callback:
            progress_callback(processed_count, total_pending)

        # Free memory between batches
        gc.collect()

    return EmbeddingResult(
        images_embedded=processed_count,
        images_skipped=already_done,
        device_used=str(embedder.device),
        batch_size=batch_size,
        rotations_enabled=enable_rotations,
    )
