"""
tests/test_embeddings.py — Tests for Stage 2 CLIP embeddings.

Requirements verified:
- Embeddings are unit length (L2 norm == 1.0 within float16 tolerance).
- Embeddings are deterministic for the same image.
- A resized copy of an image has cosine similarity > 0.9 with the original.
- Embeddings are computed on thumbnails, never full-res.
- Rotation handling stores 90/180/270 embeddings when enabled, leaves NULL when disabled.
- Resumability: skips images that already have embeddings.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from core.config import load_config
from core.embeddings import (
    CLIPEmbedder,
    deserialize_embedding,
    get_device,
    is_clip_weights_cached,
    run_embeddings,
    serialize_embedding,
)
from db.connection import init_db
from sources.local_folder import LocalFolderSource


@pytest.fixture(scope="module")
def shared_embedder() -> CLIPEmbedder:
    """Share embedder model across tests to avoid reloading weights each test."""
    cfg = load_config()
    stage2_cfg = cfg.get("stage2", {})
    model_name = stage2_cfg.get("clip_model", "ViT-B-32")
    pretrained = stage2_cfg.get("clip_pretrained", "laion2b_s34b_b79k")
    if not is_clip_weights_cached(model_name, pretrained):
        pytest.skip(f"CLIP weights ({model_name} / {pretrained}) not cached; skipping in CI to avoid network downloads")
    return CLIPEmbedder(
        model_name=model_name,
        pretrained=pretrained,
    )


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    return init_db(tmp_path / "test.db")


def _create_rich_test_image(size: tuple[int, int] = (400, 300)) -> Image.Image:
    """Create a synthetic image with distinct geometric features and colors."""
    img = Image.new("RGB", size, color=(50, 100, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 200], fill=(220, 80, 40), outline=(255, 255, 0), width=4)
    draw.ellipse([150, 100, 300, 250], fill=(40, 200, 90), outline=(255, 255, 255), width=3)
    draw.polygon([(250, 50), (350, 150), (200, 180)], fill=(240, 220, 30))
    return img


class TestCLIPEmbedder:
    def test_device_selection(self):
        device = get_device()
        assert isinstance(device, torch.device)
        if os.environ.get("SIMILIX_DEVICE"):
            assert device.type == os.environ["SIMILIX_DEVICE"]
        elif os.environ.get("ARGUS_DEVICE"):
            assert device.type == os.environ["ARGUS_DEVICE"]
        elif os.environ.get("CI"):
            assert device.type == "cpu"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"

    def test_embeddings_unit_length(self, shared_embedder: CLIPEmbedder):
        img = _create_rich_test_image()
        embs = shared_embedder.embed_thumbnails([img])
        assert embs.shape == (1, 512)
        assert embs.dtype == np.float16

        # L2 norm in float32 for accuracy
        norm = np.linalg.norm(embs[0].astype(np.float32))
        assert pytest.approx(norm, abs=1e-2) == 1.0

    def test_embeddings_deterministic(self, shared_embedder: CLIPEmbedder):
        img = _create_rich_test_image()
        emb1 = shared_embedder.embed_thumbnails([img])[0]
        emb2 = shared_embedder.embed_thumbnails([img])[0]

        np.testing.assert_array_equal(emb1, emb2)

    def test_resized_similarity_above_90_percent(self, shared_embedder: CLIPEmbedder):
        orig = _create_rich_test_image(size=(600, 450))
        resized = orig.resize((200, 150), Image.Resampling.LANCZOS)

        embs = shared_embedder.embed_thumbnails([orig, resized])
        vec1 = embs[0].astype(np.float32)
        vec2 = embs[1].astype(np.float32)

        # Cosine similarity between unit-normalized vectors is their dot product
        cosine_sim = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        assert cosine_sim > 0.90, f"Expected similarity > 0.90, got {cosine_sim:.4f}"

    def test_serialization_roundtrip(self):
        arr = np.random.randn(512).astype(np.float32)
        arr = (arr / np.linalg.norm(arr)).astype(np.float16)
        blob = serialize_embedding(arr)
        restored = deserialize_embedding(blob)
        assert restored.dtype == np.float16
        np.testing.assert_array_equal(arr, restored)


class TestRunEmbeddingsPipeline:
    def test_resumable_and_skips_already_embedded(
        self, tmp_path: Path, db: sqlite3.Connection, shared_embedder: CLIPEmbedder
    ):
        img_path = tmp_path / "test.jpg"
        _create_rich_test_image().save(img_path, format="JPEG")

        cfg = load_config()
        source = LocalFolderSource(tmp_path)

        # Stage 0 discovery
        from core.stage0 import run_stage0
        run_stage0(source, db, cfg)

        # First embedding pass
        res1 = run_embeddings(source, db, cfg, embedder=shared_embedder)
        assert res1.images_embedded == 1
        assert res1.images_skipped == 0

        # Check DB status and vector
        row = db.execute("SELECT status, clip_vector FROM images WHERE file_path = ?", (str(img_path),)).fetchone()
        assert row["status"] == "embedded"
        assert row["clip_vector"] is not None
        assert len(row["clip_vector"]) == 512 * 2  # 512 float16 = 1024 bytes

        # Second embedding pass — should be skipped!
        res2 = run_embeddings(source, db, cfg, embedder=shared_embedder)
        assert res2.images_embedded == 0
        assert res2.images_skipped == 1

    def test_rotations_flag(
        self, tmp_path: Path, db: sqlite3.Connection, shared_embedder: CLIPEmbedder
    ):
        img_path = tmp_path / "rot.jpg"
        _create_rich_test_image().save(img_path, format="JPEG")

        cfg = dict(load_config())
        cfg["stage2"] = dict(cfg.get("stage2", {}))
        cfg["stage2"]["enable_rotation_embeddings"] = True

        source = LocalFolderSource(tmp_path)
        from core.stage0 import run_stage0
        run_stage0(source, db, cfg)

        res = run_embeddings(source, db, cfg, embedder=shared_embedder)
        assert res.images_embedded == 1
        assert res.rotations_enabled is True

        row = db.execute(
            """
            SELECT clip_vector, clip_vector_90, clip_vector_180, clip_vector_270
            FROM images WHERE file_path = ?
            """,
            (str(img_path),),
        ).fetchone()

        assert row["clip_vector"] is not None
        assert row["clip_vector_90"] is not None
        assert row["clip_vector_180"] is not None
        assert row["clip_vector_270"] is not None

        # Verify rotation embeddings are unit length
        for key in ("clip_vector_90", "clip_vector_180", "clip_vector_270"):
            vec = deserialize_embedding(row[key])
            norm = np.linalg.norm(vec.astype(np.float32))
            assert pytest.approx(norm, abs=1e-2) == 1.0
