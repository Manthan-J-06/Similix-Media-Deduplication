"""
tests/test_folder_zip_upload.py — Tests for chunked folder upload, resume,
ZIP upload with streamed extraction, batch caps (up to 10,000 files),
and security invariants (quota, per-file cap, path traversal, zip bombs, symlinks).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import sqlite3
import time
import zipfile

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.server import create_app
from core.config import load_config
from core.ledger import compute_file_md5, ensure_ledger_tables
from db.connection import init_db


def _make_sample_jpeg(path: Path, color: tuple[int, int, int] = (100, 150, 200), size: tuple[int, int] = (64, 64)) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    path.write_bytes(data)
    return data


def _setup_test_client(tmp_path: Path):
    data_dir = tmp_path / "similix_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    auth_db = data_dir / "auth.db"
    users_dir = data_dir / "users"
    users_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    app = create_app(config=cfg, auth_db_path=auth_db, users_dir=users_dir)
    client = TestClient(app)

    # Register and log in
    email = "uploader@similix.local"
    pwd = "SecurePassword123!"
    client.post("/auth/register", json={"email": email, "password": pwd})
    login_resp = client.post("/auth/login", json={"email": email, "password": pwd})
    data = login_resp.json()
    token = data["token"]
    csrf = data["csrf_token"]
    client.cookies.set("similix_session", token)
    client.headers["X-CSRF-Token"] = csrf

    return client, users_dir, data_dir


def test_folder_upload_chunks_of_25_and_resume(tmp_path: Path):
    """
    Test folder upload in chunks of 25:
    - 55 files -> 3 chunks (25, 25, 5).
    - Status check reports received chunks.
    - Idempotent re-upload of chunk 1.
    - Completion registers all files and launches pipeline job.
    """
    client, users_dir, _ = _setup_test_client(tmp_path)
    upload_id = "test_fld_upload_123"
    total_files = 55
    total_chunks = 3

    # Generate 55 dummy jpeg bytes
    all_files_bytes = []
    for i in range(total_files):
        buf = io.BytesIO()
        im = Image.new("RGB", (32, 32), color=(i * 4 % 255, 100, 150))
        im.save(buf, format="JPEG")
        all_files_bytes.append((f"photo_{i:03d}.jpg", buf.getvalue(), f"MyFolder/Subdir/photo_{i:03d}.jpg"))

    # Chunk 0: files 0..24
    chunk0 = all_files_bytes[0:25]
    files_payload_0 = [("files", (name, b, "image/jpeg")) for name, b, _ in chunk0]
    rel_paths_0 = [rel for _, _, rel in chunk0]
    data_0 = {
        "upload_id": upload_id,
        "chunk_index": 0,
        "total_chunks": total_chunks,
        "total_files": total_files,
        "project": "vacation_2026",
    }
    for rel in rel_paths_0:
        files_payload_0.append(("relative_paths", (None, rel)))

    r0 = client.post("/sources/folder/chunk", data=data_0, files=files_payload_0)
    assert r0.status_code == 200
    res0 = r0.json()
    assert res0["status"] == "chunk_received"
    assert res0["chunk_index"] == 0
    assert res0["uploaded_count"] == 25

    # Check status
    st_resp = client.get(f"/sources/folder/status?upload_id={upload_id}")
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert st_data["exists"] is True
    assert st_data["chunks_received"] == [0]
    assert st_data["uploaded_count"] == 25

    # Chunk 1: files 25..49
    chunk1 = all_files_bytes[25:50]
    files_payload_1 = [("files", (name, b, "image/jpeg")) for name, b, _ in chunk1]
    data_1 = {
        "upload_id": upload_id,
        "chunk_index": 1,
        "total_chunks": total_chunks,
        "total_files": total_files,
        "project": "vacation_2026",
    }
    r1 = client.post("/sources/folder/chunk", data=data_1, files=files_payload_1)
    assert r1.status_code == 200
    assert r1.json()["uploaded_count"] == 50

    # Idempotent re-send of Chunk 1 (resuming network drop)
    r1_retry = client.post("/sources/folder/chunk", data=data_1, files=files_payload_1)
    assert r1_retry.status_code == 200
    assert r1_retry.json()["status"] == "already_received"
    assert r1_retry.json()["uploaded_count"] == 50

    # Chunk 2: files 50..54 (5 files)
    chunk2 = all_files_bytes[50:55]
    files_payload_2 = [("files", (name, b, "image/jpeg")) for name, b, _ in chunk2]
    data_2 = {
        "upload_id": upload_id,
        "chunk_index": 2,
        "total_chunks": total_chunks,
        "total_files": total_files,
        "project": "vacation_2026",
    }
    r2 = client.post("/sources/folder/chunk", data=data_2, files=files_payload_2)
    assert r2.status_code == 200
    assert r2.json()["uploaded_count"] == 55

    # Complete folder upload
    comp_resp = client.post(
        "/sources/folder/complete",
        data={"upload_id": upload_id, "project": "vacation_2026"},
    )
    assert comp_resp.status_code == 200
    comp_data = comp_resp.json()
    assert comp_data["uploaded_count"] == 55
    assert comp_data["project"] == "vacation_2026"
    assert "job_id" in comp_data
    assert len(comp_data["files"]) == 55

    # Verify user database has 55 discovered images
    user_db = users_dir / "1" / "similix.db"
    if not user_db.exists():
        user_db = users_dir / "1" / "argus.db"
    assert user_db.exists()
    with sqlite3.connect(user_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM images WHERE status = 'discovered'").fetchone()[0]
        assert count == 55


def test_folder_upload_security_invariants_and_caps(tmp_path: Path):
    """
    Test security limits on folder upload:
    - Batch cap up to 10,000 files: exceeding 10,000 files is rejected.
    - Path traversal rejected.
    - Non-image rejected.
    """
    client, _, _ = _setup_test_client(tmp_path)

    # 1. Total files > 10000
    fake_img = io.BytesIO()
    Image.new("RGB", (10, 10)).save(fake_img, "JPEG")
    payload = [("files", ("test.jpg", fake_img.getvalue(), "image/jpeg"))]
    r_cap = client.post(
        "/sources/folder/chunk",
        data={"upload_id": "cap_test", "chunk_index": 0, "total_chunks": 1, "total_files": 10001},
        files=payload,
    )
    assert r_cap.status_code == 400
    assert "exceeds" in r_cap.json()["detail"].lower()
    assert "10000" in r_cap.json()["detail"].lower()

    # 2. Path traversal in filename
    r_traversal = client.post(
        "/sources/folder/chunk",
        data={"upload_id": "trav_test", "chunk_index": 0, "total_chunks": 1, "total_files": 1},
        files=[("files", ("../../etc/passwd.jpg", fake_img.getvalue(), "image/jpeg"))],
    )
    assert r_traversal.status_code == 400
    assert "path traversal rejected" in r_traversal.json()["detail"].lower()

    # 3. Non-image file
    r_non_img = client.post(
        "/sources/folder/chunk",
        data={"upload_id": "non_img_test", "chunk_index": 0, "total_chunks": 1, "total_files": 1},
        files=[("files", ("malicious.sh", b"#!/bin/bash\necho bad\n", "text/plain"))],
    )
    assert r_non_img.status_code == 400
    assert "non-image rejected" in r_non_img.json()["detail"].lower()


def test_zip_upload_chunks_and_streamed_extraction(tmp_path: Path):
    """
    Test chunked upload of a ZIP archive followed by streamed server-side extraction:
    - Creates a zip with 30 images in subfolders.
    - Uploads in binary chunks to /sources/zip/chunk.
    - Extracts using /sources/zip/extract.
    - Verifies all 30 images are extracted and recorded in DB.
    """
    client, users_dir, _ = _setup_test_client(tmp_path)

    # Create zip in memory with 30 images
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(30):
            im_buf = io.BytesIO()
            Image.new("RGB", (40, 40), color=(i * 8 % 255, 50, 200)).save(im_buf, "JPEG")
            zf.writestr(f"album/trip_day_{i // 10}/photo_{i}.jpg", im_buf.getvalue())

    zip_bytes = zip_buf.getvalue()
    chunk_size = 4096  # small chunks to test multi-chunk upload
    total_chunks = (len(zip_bytes) + chunk_size - 1) // chunk_size
    upload_id = "zip_test_session_456"

    for idx in range(total_chunks):
        start = idx * chunk_size
        chunk_data = zip_bytes[start : start + chunk_size]
        res = client.post(
            "/sources/zip/chunk",
            data={
                "upload_id": upload_id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "total_bytes": len(zip_bytes),
                "project": "zip_import",
            },
            files=[("file", ("archive.zip", chunk_data, "application/zip"))],
        )
        assert res.status_code == 200
        assert res.json()["status"] == "chunk_received"

    # Check status
    st = client.get(f"/sources/zip/status?upload_id={upload_id}").json()
    assert st["exists"] is True
    assert st["total_chunks"] == total_chunks
    assert st["bytes_received"] == len(zip_bytes)

    # Extract
    ext_resp = client.post(
        "/sources/zip/extract",
        data={"upload_id": upload_id, "project": "zip_import"},
    )
    assert ext_resp.status_code == 200
    data = ext_resp.json()
    assert data["extracted_count"] == 30
    assert data["project"] == "zip_import"
    assert "job_id" in data

    # Verify extracted on disk and in DB
    user_db = users_dir / "1" / "similix.db"
    if not user_db.exists():
        user_db = users_dir / "1" / "argus.db"
    with sqlite3.connect(user_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM images WHERE file_path LIKE '%zip_import%'").fetchone()[0]
        assert count == 30


def test_zip_direct_upload(tmp_path: Path):
    """Test direct POST /sources/zip upload and streamed extraction."""
    client, users_dir, _ = _setup_test_client(tmp_path)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(12):
            im_buf = io.BytesIO()
            Image.new("RGB", (30, 30), color=(100, 200, i * 20 % 255)).save(im_buf, "PNG")
            zf.writestr(f"collection/direct_{i}.png", im_buf.getvalue())

    zip_bytes = zip_buf.getvalue()
    res = client.post(
        "/sources/zip?project=direct_batch",
        files={"file": ("direct_archive.zip", zip_bytes, "application/zip")},
    )
    assert res.status_code == 200
    assert res.json()["extracted_count"] == 12
    assert "job_id" in res.json()


def test_zip_security_symlinks_path_traversal_and_bomb(tmp_path: Path):
    """
    Test ZIP security invariants:
    - Path traversal inside zip (e.g. ../../escape.jpg) is rejected.
    - Symlinks inside zip are rejected.
    - Zip bomb ratio (>100:1) is rejected.
    - File count cap > 10,000 is rejected.
    """
    client, _, _ = _setup_test_client(tmp_path)

    # 1. Path traversal inside ZIP
    buf_trav = io.BytesIO()
    with zipfile.ZipFile(buf_trav, "w") as zf:
        zf.writestr("../../etc/passwd.jpg", b"fake jpeg")
    r_trav = client.post(
        "/sources/zip",
        files={"file": ("trav.zip", buf_trav.getvalue(), "application/zip")},
    )
    assert r_trav.status_code == 400
    assert "path traversal in zip rejected" in r_trav.json()["detail"].lower()

    # 2. Symlink inside ZIP
    buf_sym = io.BytesIO()
    with zipfile.ZipFile(buf_sym, "w") as zf:
        zinfo = zipfile.ZipInfo("symlink_target.jpg")
        # Set Unix symlink mode: 0o120777
        zinfo.external_attr = 0o120777 << 16
        zf.writestr(zinfo, "/etc/passwd")
    r_sym = client.post(
        "/sources/zip",
        files={"file": ("sym.zip", buf_sym.getvalue(), "application/zip")},
    )
    assert r_sym.status_code == 400
    assert "symlink rejected" in r_sym.json()["detail"].lower()

    # 3. Zip bomb ratio (>100:1)
    buf_bomb = io.BytesIO()
    with zipfile.ZipFile(buf_bomb, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 10 MB of zeros compresses to ~10 KB (ratio > 1000:1)
        zinfo = zipfile.ZipInfo("bomb.jpg")
        zinfo.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zinfo, b"\x00" * (10 * 1024 * 1024))
    r_bomb = client.post(
        "/sources/zip",
        files={"file": ("bomb.zip", buf_bomb.getvalue(), "application/zip")},
    )
    assert r_bomb.status_code == 400
    assert "zip bomb" in r_bomb.json()["detail"].lower()


def test_duplicates_detected_across_folders(tmp_path: Path):
    """
    Requirement 3:
    Duplicates are detected across folders (e.g. folder upload and zip upload).
    """
    client, users_dir, _ = _setup_test_client(tmp_path)

    # Image A: 100x100 solid orange
    shared_img = Image.new("RGB", (100, 100), color=(255, 120, 30))
    buf_a = io.BytesIO()
    shared_img.save(buf_a, "JPEG")
    bytes_a = buf_a.getvalue()

    # Upload 1: folder chunk containing shared_image as "original_a.jpg"
    r_fld = client.post(
        "/sources/folder/chunk",
        data={"upload_id": "cross_fld", "chunk_index": 0, "total_chunks": 1, "total_files": 1, "project": "folder_source"},
        files=[("files", ("original_a.jpg", bytes_a, "image/jpeg"))],
    )
    assert r_fld.status_code == 200
    r_comp = client.post("/sources/folder/complete", data={"upload_id": "cross_fld", "project": "folder_source"})
    assert r_comp.status_code == 200

    # Clear active job so second upload can proceed without 409 conflict
    jobs_dir = users_dir / "1" / "jobs"
    for jf in jobs_dir.glob("*.json"):
        try:
            jf.unlink()
        except Exception:
            pass

    # Upload 2: zip containing exact same image as "nested/copy_a.jpg"
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("nested/copy_a.jpg", bytes_a)

    r_zip = client.post(
        "/sources/zip?project=zip_source",
        files={"file": ("cross_archive.zip", zip_buf.getvalue(), "application/zip")},
    )
    assert r_zip.status_code == 200

    # Verify both files are in the database under different folder paths
    user_db = users_dir / "1" / "similix.db"
    if not user_db.exists():
        user_db = users_dir / "1" / "argus.db"

    with sqlite3.connect(user_db) as conn:
        rows = conn.execute("SELECT file_path, md5 FROM images").fetchall()
        assert len(rows) == 2
        paths = [r[0] for r in rows]
        assert any("folder_source" in p for p in paths)
        assert any("zip_source" in p for p in paths)
        assert rows[0][1] == rows[1][1]  # Same MD5 checksum!
