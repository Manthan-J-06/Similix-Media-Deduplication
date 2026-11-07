import os
import shutil
import sqlite3
import pytest
from pathlib import Path
from db.connection import init_db
from core.worker import run_pipeline

BENCH_DIR = Path(__file__).parent.parent / "data" / "benchmark"
ORIG_DIR = Path(__file__).parent.parent / "data" / "originals"


@pytest.fixture
def benchmark_available() -> bool:
    return BENCH_DIR.is_dir() and any(BENCH_DIR.glob("*.jpg"))


class TestIdempotencyAndDeterminism:
    def test_rescan_twice_gives_identical_clusters(self, tmp_path, benchmark_available):
        """
        Test (a): Running the full pipeline twice on the same dataset must yield
        identical cluster counts, identical member counts, and identical member groupings.
        """
        if not benchmark_available:
            pytest.skip("Benchmark images not available in data/benchmark")

        # Create a test folder with 4 distinct images and 3 duplicates
        test_images_dir = tmp_path / "images_set"
        test_images_dir.mkdir(parents=True, exist_ok=True)

        # Base images
        base_names = ["picsum_0001.jpg", "picsum_0002.jpg", "picsum_0003.jpg", "picsum_0004.jpg"]
        for b in base_names:
            src = ORIG_DIR / b
            if src.exists():
                shutil.copy(src, test_images_dir / b)

        # Known duplicate variants
        dup_names = ["picsum_0001__exact.jpg", "picsum_0002__exact.jpg", "picsum_0001__resize_512.jpg"]
        for d in dup_names:
            src = BENCH_DIR / d
            if src.exists():
                shutil.copy(src, test_images_dir / d)

        # Run 1:
        db1 = tmp_path / "run1.db"
        job1 = tmp_path / "job1.json"
        run_pipeline(
            job_id="test_run1",
            db_path=db1,
            folder=test_images_dir,
            job_file=job1,
        )

        conn1 = init_db(db1)
        clusters_run1 = conn1.execute(
            """
            SELECT c.id, c.member_count, GROUP_CONCAT(i.file_path, '|') as paths
            FROM clusters c
            JOIN images i ON i.cluster_id = c.id
            GROUP BY c.id
            ORDER BY c.member_count DESC, paths
            """
        ).fetchall()

        # Run 2:
        db2 = tmp_path / "run2.db"
        job2 = tmp_path / "job2.json"
        run_pipeline(
            job_id="test_run2",
            db_path=db2,
            folder=test_images_dir,
            job_file=job2,
        )

        conn2 = init_db(db2)
        clusters_run2 = conn2.execute(
            """
            SELECT c.id, c.member_count, GROUP_CONCAT(i.file_path, '|') as paths
            FROM clusters c
            JOIN images i ON i.cluster_id = c.id
            GROUP BY c.id
            ORDER BY c.member_count DESC, paths
            """
        ).fetchall()

        # Assert cluster count matches exactly
        assert len(clusters_run1) == len(clusters_run2), (
            f"Non-deterministic cluster count: Run 1 found {len(clusters_run1)}, Run 2 found {len(clusters_run2)}"
        )
        assert len(clusters_run1) > 0, "Expected at least one cluster from planted duplicates"

        # Assert cluster member groupings and member counts match exactly
        for (c1_id, c1_count, c1_paths), (c2_id, c2_count, c2_paths) in zip(clusters_run1, clusters_run2):
            assert c1_count == c2_count
            p1_set = {Path(p).name for p in c1_paths.split("|")}
            p2_set = {Path(p).name for p in c2_paths.split("|")}
            assert p1_set == p2_set, f"Cluster member mismatch: {p1_set} != {p2_set}"

    def test_run_on_clean_keepers_finds_zero_clusters(self, tmp_path, benchmark_available):
        """
        Test (b): Running the deduplication pipeline on the keepers + uniques
        (clean dataset) must find exactly 0 duplicate clusters.
        """
        if not benchmark_available:
            pytest.skip("Benchmark images not available in data/benchmark")

        test_images_dir = tmp_path / "mixed_images"
        test_images_dir.mkdir(parents=True, exist_ok=True)

        base_names = ["picsum_0005.jpg", "picsum_0006.jpg", "picsum_0007.jpg"]
        for b in base_names:
            src = ORIG_DIR / b
            if src.exists():
                shutil.copy(src, test_images_dir / b)

        dup_names = ["picsum_0005__exact.jpg", "picsum_0006__resize_512.jpg"]
        for d in dup_names:
            src = BENCH_DIR / d
            if src.exists():
                shutil.copy(src, test_images_dir / d)

        # Run pipeline on mixed set
        db_mixed = tmp_path / "mixed.db"
        job_mixed = tmp_path / "job_mixed.json"
        run_pipeline(
            job_id="test_mixed",
            db_path=db_mixed,
            folder=test_images_dir,
            job_file=job_mixed,
        )

        conn_mixed = init_db(db_mixed)
        # Ensure keeper pipeline has run
        from core.keeper import run_keeper_pipeline
        run_keeper_pipeline(conn_mixed, mode="balanced")

        # Fetch keepers and uniques (i.e. images where delete_safe == 0)
        keepers_rows = conn_mixed.execute(
            """
            SELECT file_path FROM images
            WHERE (delete_safe = 0 OR delete_safe IS NULL)
              AND (status IS NULL OR status NOT IN ('trashed', 'deleted'))
            """
        ).fetchall()

        # Copy clean keepers + uniques to a new folder
        clean_dir = tmp_path / "clean_keepers_dataset"
        clean_dir.mkdir(parents=True, exist_ok=True)
        for r in keepers_rows:
            fp = Path(r[0])
            shutil.copy(fp, clean_dir / fp.name)

        # Run pipeline on the clean keepers dataset
        db_clean = tmp_path / "clean.db"
        job_clean = tmp_path / "job_clean.json"
        run_pipeline(
            job_id="test_clean",
            db_path=db_clean,
            folder=clean_dir,
            job_file=job_clean,
        )

        conn_clean = init_db(db_clean)
        clean_clusters = conn_clean.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]

        assert clean_clusters == 0, (
            f"Expected 0 duplicate clusters on clean keepers dataset, but found {clean_clusters} clusters!"
        )
