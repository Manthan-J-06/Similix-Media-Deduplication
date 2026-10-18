"""
tests/test_categories.py — Unit tests for zero-shot semantic categorization.

Requirements verified:
1. Text embeddings are unit L2 length.
2. Rotated vectors (clip_vector_90/180/270) are never used.
3. A synthetic vector close to a label's text embedding gets that label.
4. A near-tie gets 'uncertain'.
5. Categorize is idempotent (resumable skips labeled images, force produces identical results).
6. Database migration adds category columns to an old database without data loss.
7. Cluster-level category uses keeper's label, falling back to majority vote.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from core.categories import (
    CategoryConfig,
    TextEmbeddingCache,
    classify_vector,
    parse_category_config,
    run_categorization,
    search_images_by_text,
    update_cluster_categories,
)
from core.config import load_config
from core.embeddings import serialize_embedding
from db.connection import get_connection, init_db


@pytest.fixture(scope="module")
def shared_config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def shared_text_cache(shared_config: dict) -> TextEmbeddingCache:
    stage2 = shared_config.get("stage2", {})
    return TextEmbeddingCache(
        model_name=stage2.get("clip_model", "ViT-B-32"),
        pretrained=stage2.get("clip_pretrained", "laion2b_s34b_b79k"),
    )


class TestTextEmbeddings:

    def test_text_embeddings_are_unit_length(self, shared_config: dict, shared_text_cache: TextEmbeddingCache):
        """Label prompt embeddings must be strictly L2-normalized (unit length = 1.0)."""
        cat_cfg = parse_category_config(shared_config)
        assert len(cat_cfg.labels) >= 4, "Expected at least 4 category labels in config.toml"

        matrix, label_names = shared_text_cache.build_label_matrix(cat_cfg.labels, cat_cfg.templates)
        assert matrix.shape[0] == len(label_names)
        assert matrix.shape[1] == 512

        norms = np.linalg.norm(matrix, axis=1)
        for i, norm in enumerate(norms):
            assert norm == pytest.approx(1.0, abs=1e-5), f"Embedding for {label_names[i]} has norm {norm} != 1.0"

    def test_query_embedding_is_unit_length(self, shared_text_cache: TextEmbeddingCache):
        """Search query text embedding must be unit length."""
        q_vec = shared_text_cache.encode_query("a scenic mountain lake")
        assert q_vec.shape == (512,)
        assert np.linalg.norm(q_vec) == pytest.approx(1.0, abs=1e-5)


class TestClassificationLogic:

    def test_synthetic_vector_close_to_label_gets_that_label(self, shared_config: dict, shared_text_cache: TextEmbeddingCache):
        """A synthetic vector aligned with a label's text embedding gets that label with high probability."""
        cat_cfg = parse_category_config(shared_config)
        matrix, label_names = shared_text_cache.build_label_matrix(cat_cfg.labels, cat_cfg.templates)

        target_idx = label_names.index("landscape")
        target_vec = matrix[target_idx]

        # Add slight noise while maintaining direction
        noise = np.random.RandomState(42).randn(512).astype(np.float32) * 0.05
        synth_vec = target_vec + noise
        synth_vec = synth_vec / np.linalg.norm(synth_vec)

        cat, prob, top3 = classify_vector(
            synth_vec,
            matrix,
            label_names,
            temperature=cat_cfg.temperature,
            uncertain_margin=cat_cfg.uncertain_margin,
            min_top_prob=cat_cfg.min_top_prob,
        )

        assert cat == "landscape"
        assert prob > 0.70
        assert top3[0]["label"] == "landscape"
        assert top3[0]["prob"] == prob

    def test_near_tie_gets_uncertain(self, shared_config: dict, shared_text_cache: TextEmbeddingCache):
        """When top-1 and top-2 probabilities differ by less than uncertain_margin, label is 'uncertain'."""
        cat_cfg = parse_category_config(shared_config)
        matrix, label_names = shared_text_cache.build_label_matrix(cat_cfg.labels, cat_cfg.templates)

        idx1 = label_names.index("document")
        idx2 = label_names.index("screenshot")
        v1 = matrix[idx1]
        v2 = matrix[idx2]

        # Midpoint vector gives nearly identical cosine similarity to v1 and v2
        mid_vec = (v1 + v2) / np.linalg.norm(v1 + v2)

        cat, prob, top3 = classify_vector(
            mid_vec,
            matrix,
            label_names,
            temperature=cat_cfg.temperature,
            uncertain_margin=cat_cfg.uncertain_margin,
            min_top_prob=cat_cfg.min_top_prob,
        )

        assert cat == "uncertain"
        # top3 should still identify the contending labels
        top_contenders = {top3[0]["label"], top3[1]["label"]}
        assert "document" in top_contenders or "screenshot" in top_contenders
        assert abs(top3[0]["prob"] - top3[1]["prob"]) < cat_cfg.uncertain_margin

    def test_low_probability_gets_uncertain(self, shared_config: dict, shared_text_cache: TextEmbeddingCache):
        """When top-1 probability is below min_top_prob, category is 'uncertain'."""
        cat_cfg = parse_category_config(shared_config)
        matrix, label_names = shared_text_cache.build_label_matrix(cat_cfg.labels, cat_cfg.templates)

        # High temperature distributes probability evenly across all labels
        synth_vec = np.ones(512, dtype=np.float32) / np.sqrt(512)

        cat, prob, top3 = classify_vector(
            synth_vec,
            matrix,
            label_names,
            temperature=100.0,  # very high temperature spreads mass
            uncertain_margin=cat_cfg.uncertain_margin,
            min_top_prob=0.90,  # high min prob threshold
        )

        assert cat == "uncertain"


class TestDatabaseAndRotations:

    def test_rotated_vectors_are_never_used(self, tmp_path: Path, shared_config: dict, shared_text_cache: TextEmbeddingCache):
        """
        Categorization must strictly use the UPRIGHT clip_vector and never rotated ones.
        Even if rotated vectors point strongly to a different category or contain garbage,
        the assigned category must match the upright vector.
        """
        db_path = tmp_path / "test_rot.db"
        conn = init_db(db_path)

        cat_cfg = parse_category_config(shared_config)
        matrix, label_names = shared_text_cache.build_label_matrix(cat_cfg.labels, cat_cfg.templates)

        v_landscape = matrix[label_names.index("landscape")]
        v_food = matrix[label_names.index("food")]

        # Upright is landscape, rotated are food or random
        blob_upright = serialize_embedding(v_landscape)
        blob_rot90 = serialize_embedding(v_food)
        blob_rot180 = serialize_embedding(v_food)
        blob_rot270 = serialize_embedding(np.zeros(512, dtype=np.float16))

        conn.execute(
            """
            INSERT INTO images (file_path, clip_vector, clip_vector_90, clip_vector_180, clip_vector_270)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("test_image.jpg", blob_upright, blob_rot90, blob_rot180, blob_rot270),
        )
        conn.commit()

        res = run_categorization(conn, shared_config, text_cache=shared_text_cache)
        assert res.images_categorized == 1

        row = conn.execute("SELECT category_top1, category_top1_prob, category_top3 FROM images WHERE id = 1").fetchone()
        assert row["category_top1"] == "landscape", f"Expected landscape, got {row['category_top1']}"
        assert row["category_top1_prob"] > 0.70

        top3 = json.loads(row["category_top3"])
        assert top3[0]["label"] == "landscape"

    def test_categorize_is_idempotent(self, tmp_path: Path, shared_config: dict, shared_text_cache: TextEmbeddingCache):
        """Running categorization repeatedly is idempotent and resumable."""
        db_path = tmp_path / "test_idempotent.db"
        conn = init_db(db_path)

        cat_cfg = parse_category_config(shared_config)
        matrix, label_names = shared_text_cache.build_label_matrix(cat_cfg.labels, cat_cfg.templates)

        v_animal = matrix[label_names.index("animal")]
        conn.execute("INSERT INTO images (file_path, clip_vector) VALUES (?, ?)", ("cat.jpg", serialize_embedding(v_animal)))
        conn.commit()

        # Run 1: processes image
        res1 = run_categorization(conn, shared_config, force=False, text_cache=shared_text_cache)
        assert res1.images_categorized == 1
        assert res1.images_skipped == 0

        row1 = conn.execute("SELECT category_top1, category_top1_prob FROM images WHERE id = 1").fetchone()
        assert row1["category_top1"] == "animal"

        # Run 2 without force: skips already labeled image
        res2 = run_categorization(conn, shared_config, force=False, text_cache=shared_text_cache)
        assert res2.images_categorized == 0
        assert res2.images_skipped == 1

        row2 = conn.execute("SELECT category_top1, category_top1_prob FROM images WHERE id = 1").fetchone()
        assert row2["category_top1"] == row1["category_top1"]
        assert row2["category_top1_prob"] == row1["category_top1_prob"]

        # Run 3 with force: re-categorizes but produces identical values
        res3 = run_categorization(conn, shared_config, force=True, text_cache=shared_text_cache)
        assert res3.images_categorized == 1
        assert res3.images_skipped == 0

        row3 = conn.execute("SELECT category_top1, category_top1_prob FROM images WHERE id = 1").fetchone()
        assert row3["category_top1"] == row1["category_top1"]
        assert row3["category_top1_prob"] == row1["category_top1_prob"]


class TestClusterCategoryAndMigration:

    def test_cluster_category_uses_keeper_label_and_majority_fallback(self, tmp_path: Path):
        """Cluster-level category uses keeper's label if present; falls back to majority vote."""
        db_path = tmp_path / "test_cluster_cat.db"
        conn = init_db(db_path)

        # Cluster 1: Keeper is labeled 'landscape', member is 'portrait'
        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected) VALUES (1, NULL, 0)")
        conn.execute(
            "INSERT INTO images (id, file_path, cluster_id, is_keeper, category_top1) VALUES (1, 'k.jpg', 1, 1, 'landscape')"
        )
        conn.execute(
            "INSERT INTO images (id, file_path, cluster_id, is_keeper, category_top1) VALUES (2, 'm.jpg', 1, 0, 'portrait')"
        )
        conn.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")

        # Cluster 2: No keeper set, members are ['food', 'food', 'animal']
        conn.execute("INSERT INTO clusters (id, representative_id, stage_detected) VALUES (2, NULL, 1)")
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, category_top1) VALUES (3, 'f1.jpg', 2, 0, 'food')")
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, category_top1) VALUES (4, 'f2.jpg', 2, 0, 'food')")
        conn.execute("INSERT INTO images (id, file_path, cluster_id, is_keeper, category_top1) VALUES (5, 'a1.jpg', 2, 0, 'animal')")

        conn.commit()

        updated = update_cluster_categories(conn)
        assert updated == 2

        c1 = conn.execute("SELECT category FROM clusters WHERE id = 1").fetchone()
        assert c1["category"] == "landscape", f"Expected keeper's category 'landscape', got {c1['category']}"

        c2 = conn.execute("SELECT category FROM clusters WHERE id = 2").fetchone()
        assert c2["category"] == "food", f"Expected majority vote category 'food', got {c2['category']}"

    def test_migration_works_on_old_db(self, tmp_path: Path):
        """Old DB without category columns upgrades cleanly via init_db without losing data."""
        old_db_path = tmp_path / "old_version.db"
        old_conn = sqlite3.connect(str(old_db_path))

        # Create older Argus schema without category columns
        old_conn.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                file_size INTEGER,
                modified_time REAL,
                width INTEGER,
                height INTEGER,
                format TEXT,
                md5 TEXT,
                phash TEXT,
                clip_vector BLOB,
                status TEXT NOT NULL DEFAULT 'discovered',
                cluster_id INTEGER,
                is_keeper INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
        """)
        old_conn.execute("""
            CREATE TABLE clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                representative_id INTEGER,
                stage_detected INTEGER NOT NULL,
                member_count INTEGER NOT NULL DEFAULT 0,
                reference_uncertain INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
        """)
        old_conn.execute("INSERT INTO images (file_path, file_size, md5) VALUES ('old_file.jpg', 12345, 'abcde');")
        old_conn.execute("INSERT INTO clusters (representative_id, stage_detected) VALUES (1, 0);")
        old_conn.commit()
        old_conn.close()

        # Run init_db which applies safe column migrations
        migrated_conn = init_db(old_db_path)

        img_cols = {r[1] for r in migrated_conn.execute("PRAGMA table_info(images)").fetchall()}
        assert "category_top1" in img_cols
        assert "category_top1_prob" in img_cols
        assert "category_top3" in img_cols

        cluster_cols = {r[1] for r in migrated_conn.execute("PRAGMA table_info(clusters)").fetchall()}
        assert "category" in cluster_cols

        # Verify existing data is intact
        img = migrated_conn.execute("SELECT file_path, file_size, category_top1 FROM images WHERE id = 1").fetchone()
        assert img["file_path"] == "old_file.jpg"
        assert img["file_size"] == 12345
        assert img["category_top1"] is None


class TestTextSearch:

    def test_search_images_by_text(self, tmp_path: Path, shared_config: dict, shared_text_cache: TextEmbeddingCache):
        """Natural language text query ranks images by semantic similarity."""
        db_path = tmp_path / "test_search.db"
        conn = init_db(db_path)

        cat_cfg = parse_category_config(shared_config)
        matrix, label_names = shared_text_cache.build_label_matrix(cat_cfg.labels, cat_cfg.templates)

        v_landscape = matrix[label_names.index("landscape")]
        v_food = matrix[label_names.index("food")]

        conn.execute(
            "INSERT INTO images (file_path, width, height, clip_vector, category_top1, category_top1_prob) VALUES (?, ?, ?, ?, ?, ?)",
            ("mountain.jpg", 1920, 1080, serialize_embedding(v_landscape), "landscape", 0.95),
        )
        conn.execute(
            "INSERT INTO images (file_path, width, height, clip_vector, category_top1, category_top1_prob) VALUES (?, ?, ?, ?, ?, ?)",
            ("pizza.jpg", 800, 600, serialize_embedding(v_food), "food", 0.92),
        )
        conn.commit()

        results = search_images_by_text(
            "mountain valley outdoors",
            conn,
            shared_config,
            limit=2,
            text_cache=shared_text_cache,
        )

        assert len(results) == 2
        assert results[0]["file_path"] == "mountain.jpg"
        assert results[0]["score"] > results[1]["score"]


class TestCategoryEvaluationAndCLI:

    def test_generate_draft_and_evaluate(self, tmp_path: Path):
        """Verify draft CSV generation and accuracy evaluation against ground truth CSV."""
        from bench.category_eval import generate_draft_csv, evaluate_categories
        db_path = tmp_path / "test_eval.db"
        conn = init_db(db_path)

        conn.execute("INSERT INTO images (file_path, category_top1, category_top1_prob) VALUES ('img1.jpg', 'landscape', 0.95)")
        conn.execute("INSERT INTO images (file_path, category_top1, category_top1_prob) VALUES ('img2.jpg', 'portrait', 0.88)")
        conn.execute("INSERT INTO images (file_path, category_top1, category_top1_prob) VALUES ('img3.jpg', 'uncertain', 0.25)")
        conn.commit()

        # Generate draft CSV
        draft_csv = tmp_path / "draft.csv"
        count = generate_draft_csv(conn, draft_csv)
        assert count == 3
        assert draft_csv.exists()

        # Create ground truth CSV: img1 is landscape (match), img2 is food (mismatch), img3 is landscape (uncertain)
        gt_csv = tmp_path / "ground_truth.csv"
        gt_csv.write_text("file,label\nimg1.jpg,landscape\nimg2.jpg,food\nimg3.jpg,landscape\n", encoding="utf-8")

        res = evaluate_categories(gt_csv, conn)
        assert res["total_evaluated"] == 3
        assert res["correct"] == 1  # only img1 matches
        assert res["overall_accuracy"] == pytest.approx(33.33, abs=0.1)
        assert res["uncertain_count"] == 1
        assert res["uncertain_percent"] == pytest.approx(33.33, abs=0.1)
        assert res["per_label"]["landscape"]["total"] == 2
        assert res["per_label"]["landscape"]["correct"] == 1
        assert res["per_label"]["food"]["total"] == 1
        assert res["per_label"]["food"]["correct"] == 0
        assert res["confusion"]["landscape"]["landscape"] == 1
        assert res["confusion"]["landscape"]["uncertain"] == 1
        assert res["confusion"]["food"]["portrait"] == 1

    def test_cli_subcommands(self, tmp_path: Path, shared_config: dict, shared_text_cache: TextEmbeddingCache, capsys):
        """Verify argus categorize, argus search, and argus list CLI invocations."""
        from core.cli import main as cli_main
        db_path = tmp_path / "test_cli.db"
        conn = init_db(db_path)

        cat_cfg = parse_category_config(shared_config)
        matrix, label_names = shared_text_cache.build_label_matrix(cat_cfg.labels, cat_cfg.templates)
        v_doc = matrix[label_names.index("document")]
        v_land = matrix[label_names.index("landscape")]

        conn.execute("INSERT INTO images (file_path, width, height, clip_vector) VALUES ('receipt.jpg', 600, 800, ?)", (serialize_embedding(v_doc),))
        conn.execute("INSERT INTO images (file_path, width, height, clip_vector) VALUES ('sunset.jpg', 1920, 1080, ?)", (serialize_embedding(v_land),))
        conn.commit()

        # 1. argus categorize --db
        cli_main(["categorize", "--db", str(db_path)])
        captured = capsys.readouterr().out
        assert "Images categorized : 2" in captured

        # 2. argus list --category document --db
        cli_main(["list", "--category", "document", "--db", str(db_path)])
        captured = capsys.readouterr().out
        assert "receipt.jpg" in captured
        assert "sunset.jpg" not in captured

        # 3. argus search "a printed paper receipt" --db --limit 1
        cli_main(["search", "a printed paper receipt", "--db", str(db_path), "--limit", "1"])
        captured = capsys.readouterr().out
        assert "receipt.jpg" in captured

