"""
bench/bench_drive_100.py — Measure and report timing per stage for a 100-image Drive folder.

Simulates a 100-image Google Drive dataset with:
- 100 images total
- Multiple exact duplicates and near duplicates
- Full pipeline run (Scan, pHash with 8-worker thread pool, CLIP Embeddings,
  Clustering, Geometric Verification, Quality & Keeper, Categorization)
- Measures wall-clock time per stage and prints detailed benchmark summary.
"""

from __future__ import annotations

import io
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from core.config import load_config
from core.worker import run_pipeline
from db.connection import init_db
from sources.google_drive import GoogleDriveSource


def generate_100_drive_images(tmp_path: Path) -> tuple[dict, dict[str, bytes]]:
    """Generate 100 synthetic images and Google Drive API file list metadata."""
    files_list = []
    thumbnails: dict[str, bytes] = {}

    # 100 images: 70 unique base images, 20 exact duplicates, 10 resized/near-duplicates
    for i in range(100):
        file_id = f"drive_file_{i:03d}"
        if i < 70:
            color = (int(i * 3.6) % 256, (i * 7) % 256, (i * 11) % 256)
            name = f"photo_{i:03d}.jpg"
            md5 = f"md5_unique_{i:03d}"
        elif i < 90:
            # Exact duplicate of photo_(i-70)
            orig_idx = i - 70
            color = (int(orig_idx * 3.6) % 256, (orig_idx * 7) % 256, (orig_idx * 11) % 256)
            name = f"copy_{i:03d}_of_{orig_idx:03d}.jpg"
            md5 = f"md5_unique_{orig_idx:03d}"
        else:
            # Near duplicate of photo_(i-90)
            orig_idx = i - 90
            color = (int(orig_idx * 3.6) % 256, (orig_idx * 7) % 256, (orig_idx * 11) % 256)
            name = f"near_{i:03d}_of_{orig_idx:03d}.jpg"
            md5 = f"md5_near_{i:03d}"

        # Create thumbnail image
        im = Image.new("RGB", (128, 128), color=color)
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), f"ID:{i}", fill=(255, 255, 255))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()
        thumbnails[file_id] = raw_bytes

        files_list.append(
            {
                "id": file_id,
                "name": name,
                "mimeType": "image/jpeg",
                "size": str(len(raw_bytes)),
                "md5Checksum": md5,
                "modifiedTime": f"2026-01-01T{i%24:02d}:00:00.000Z",
                "imageMediaMetadata": {"width": 1024, "height": 768},
                "thumbnailLink": f"https://lh3.googleusercontent.com/thumb_{file_id}=s220",
                "trashed": False,
            }
        )

    api_response = {"nextPageToken": None, "files": files_list}
    return api_response, thumbnails


def run_benchmark():
    """Execute the full 100-image Drive pipeline benchmark and print timing report."""
    print("=" * 64)
    print("  ARGUS GOOGLE DRIVE PIPELINE BENCHMARK (100 IMAGES)")
    print("=" * 64)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        db_path = tmp_dir / "bench_drive.db"
        cache_dir = tmp_dir / "drive_cache"
        job_file = tmp_dir / "bench_job.json"

        os.environ["ARGUS_DATA_DIR"] = str(tmp_dir)

        print("[1/3] Generating 100 synthetic Google Drive images & metadata...")
        t0_gen = time.perf_counter()
        api_response, thumbnails = generate_100_drive_images(tmp_dir)
        print(f"      Generated 100 images in {time.perf_counter() - t0_gen:.3f}s")

        # Mock Google Drive Service
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = api_response

        def mock_get(fileId=None, **kwargs):
            req = MagicMock()
            f_meta = next((f for f in api_response["files"] if f["id"] == fileId), None)
            req.execute.return_value = f_meta or {}
            return req

        mock_service.files().get.side_effect = mock_get

        # Mock thumbnail download responses
        def mock_urlopen(req, timeout=20):
            url_str = req.full_url if hasattr(req, "full_url") else str(req)
            # Find fileId from url
            matched_bytes = None
            for fid, b in thumbnails.items():
                if fid in url_str:
                    matched_bytes = b
                    break
            if matched_bytes is None:
                matched_bytes = next(iter(thumbnails.values()))

            resp = MagicMock()
            resp.status = 200
            resp.read.return_value = matched_bytes
            resp.__enter__.return_value = resp
            return resp

        print("[2/3] Executing pipeline across all stages (8 workers thread pool)...")
        t_pipeline_start = time.perf_counter()

        with patch("sources.google_drive.get_drive_service", return_value=mock_service), \
             patch("sources.google_drive.get_drive_credentials", return_value=MagicMock()), \
             patch("urllib.request.urlopen", side_effect=mock_urlopen):

            run_pipeline(
                job_id="bench_100_drive",
                db_path=db_path,
                source_type="drive",
                drive_folder_id="root_bench_folder",
                job_file=job_file,
            )

        t_total_pipeline = time.perf_counter() - t_pipeline_start

        # Read job file for stage timings and results
        import json
        job_data = json.loads(job_file.read_text())
        timings = job_data.get("stage_timings", {})

        print("\n" + "=" * 64)
        print("  STAGE TIMING REPORT: 100-IMAGE DRIVE FOLDER")
        print("=" * 64)
        print(f"  Stage 0 (Scan & Metadata Discovery) : {timings.get('scan', 0.0):>8.3f}s")
        print(f"  Stage 1 (pHash Thumbnail Fetching)   : {timings.get('hash', 0.0):>8.3f}s (8 workers)")
        print(f"  Stage 2 (CLIP Embeddings)            : {timings.get('embed', 0.0):>8.3f}s")
        print(f"  Stage 2 (Duplicate Clustering)       : {timings.get('cluster', 0.0):>8.3f}s")
        print(f"  Stage 3 (Geometric Verification)     : {timings.get('verify', 0.0):>8.3f}s")
        print(f"  Stage 4 (Quality & Keeper Selection) : {timings.get('quality', 0.0):>8.3f}s")
        print(f"  Stage 5 (Semantic Categorization)    : {timings.get('categorize', 0.0):>8.3f}s")
        print("  " + "-" * 58)
        print(f"  Total Wall-Clock Pipeline Time       : {t_total_pipeline:>8.3f}s")
        print("=" * 64)
        print(f"  Images Indexed   : {job_data.get('total_images')}")
        print(f"  Clusters Created : {job_data.get('clusters_created')}")
        print(f"  Final Status     : {job_data.get('status')}")
        print("=" * 64 + "\n")

        return timings, t_total_pipeline


if __name__ == "__main__":
    run_benchmark()
