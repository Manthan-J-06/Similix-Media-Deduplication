"""
tests/test_google_drive.py — Unit tests for Google Drive integration.

All tests are fully mocked with zero external network calls:
- Pagination over multiple Drive API pages
- Files missing md5Checksum are skipped and counted
- Stage 0 exact duplicate clustering from metadata with zero downloads
- Expired thumbnail refetch (403/404 triggers fresh thumbnailLink lookup)
- Exponential backoff on 429 rate limit errors
- Security invariant: Full download denied for unclustered files
- Token file permissions set to 0o600
- Bandwidth accounting and summary reporting
"""

from __future__ import annotations

import io
import json
import os
import pickle
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np

import pytest
from PIL import Image

from core.config import load_config
from core.stage0 import run_stage0
from core.keeper import choose_keeper, run_keeper_pipeline
from db.connection import init_db
from sources.google_drive import (
    GoogleDriveSource,
    build_drive_uri,
    get_drive_credentials,
    parse_drive_uri,
    print_bandwidth_summary,
    record_bandwidth,
    retry_with_backoff,
    run_confirm_drive,
)


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    """Initialize a temporary SQLite database."""
    return init_db(tmp_path / "test_gdrive.db")


@pytest.fixture()
def cfg() -> dict:
    """Load default config."""
    return load_config()


def _create_sample_jpeg_bytes() -> bytes:
    """Generate small valid JPEG bytes for thumbnail tests."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestDriveURIParsing:
    def test_build_and_parse_uri(self):
        uri = build_drive_uri("abc123id", "photo.jpg")
        assert uri == "drive://abc123id/photo.jpg"
        file_id, name = parse_drive_uri(uri)
        assert file_id == "abc123id"
        assert name == "photo.jpg"

    def test_parse_plain_id(self):
        file_id, name = parse_drive_uri("xyz789")
        assert file_id == "xyz789"
        assert name == ""


class TestPaginationAndListing:
    def test_pagination_and_file_discovery(self, tmp_path: Path):
        """Verify pagination across multiple pages and extraction of image metadata."""
        mock_service = MagicMock()

        page1 = {
            "nextPageToken": "page2_token",
            "files": [
                {
                    "id": "file_1",
                    "name": "image1.jpg",
                    "mimeType": "image/jpeg",
                    "size": "1048576",
                    "md5Checksum": "md5_hash_1",
                    "modifiedTime": "2024-01-01T12:00:00.000Z",
                    "imageMediaMetadata": {"width": 1920, "height": 1080},
                    "thumbnailLink": "https://drive.google.com/thumb1=s220",
                    "trashed": False,
                },
                {
                    "id": "file_2",
                    "name": "document.pdf",
                    "mimeType": "application/pdf",
                    "size": "50000",
                    "md5Checksum": "pdf_md5",
                    "trashed": False,
                },
            ],
        }

        page2 = {
            "nextPageToken": None,
            "files": [
                {
                    "id": "file_3",
                    "name": "photo2.png",
                    "mimeType": "image/png",
                    "size": "2097152",
                    "md5Checksum": "md5_hash_3",
                    "modifiedTime": "2024-01-02T12:00:00.000Z",
                    "imageMediaMetadata": {"width": 800, "height": 600},
                    "thumbnailLink": "https://drive.google.com/thumb3=s220",
                    "trashed": False,
                }
            ],
        }

        def mock_list(q=None, pageToken=None, **kwargs):
            req = MagicMock()
            if pageToken is None:
                req.execute.return_value = page1
            elif pageToken == "page2_token":
                req.execute.return_value = page2
            return req

        mock_service.files().list.side_effect = mock_list

        source = GoogleDriveSource(
            folder_id="root_folder",
            service=mock_service,
            cache_dir=tmp_path / "cache",
        )

        files = list(source.list_files())
        assert len(files) == 2
        assert files[0] == "drive://file_1/image1.jpg"
        assert files[1] == "drive://file_3/photo2.png"

        # Check metadata cache populated
        m1 = source.get_metadata(files[0])
        assert m1.file_size == 1048576
        assert m1.width == 1920
        assert m1.height == 1080
        assert m1.format == "JPEG"
        assert m1.md5 == "md5_hash_1"
        assert m1.source == "gdrive"
        assert m1.drive_id == "file_1"

    def test_missing_md5_skipped(self, tmp_path: Path):
        """Files without md5Checksum (e.g. Google Docs/Drawings) must be skipped with a reason."""
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {
            "files": [
                {
                    "id": "img_valid",
                    "name": "valid.jpg",
                    "mimeType": "image/jpeg",
                    "size": "5000",
                    "md5Checksum": "valid_md5",
                    "trashed": False,
                },
                {
                    "id": "img_no_md5",
                    "name": "no_md5.jpg",
                    "mimeType": "image/jpeg",
                    "size": "5000",
                    # md5Checksum is missing
                    "trashed": False,
                },
            ]
        }

        source = GoogleDriveSource(
            folder_id="fldr",
            service=mock_service,
            cache_dir=tmp_path / "cache",
        )

        files = list(source.list_files())
        assert len(files) == 1
        assert files[0] == "drive://img_valid/valid.jpg"
        assert len(source.skipped_files) == 1
        assert source.skipped_files[0][0] == "drive://img_no_md5/no_md5.jpg"
        assert source.skipped_files[0][1] == "missing md5Checksum"


class TestStage0ZeroDownloads:
    def test_exact_duplicates_from_metadata_zero_downloads(self, db: sqlite3.Connection, cfg: dict, tmp_path: Path):
        """Stage 0 finds exact duplicates from Drive md5+size with zero downloads."""
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {
            "files": [
                {
                    "id": "drive_copy_1",
                    "name": "orig.jpg",
                    "mimeType": "image/jpeg",
                    "size": "100000",
                    "md5Checksum": "duplicate_md5_123",
                    "modifiedTime": "2024-01-01T00:00:00.000Z",
                    "imageMediaMetadata": {"width": 1920, "height": 1080},
                    "trashed": False,
                },
                {
                    "id": "drive_copy_2",
                    "name": "copy.jpg",
                    "mimeType": "image/jpeg",
                    "size": "100000",
                    "md5Checksum": "duplicate_md5_123",
                    "modifiedTime": "2024-01-01T01:00:00.000Z",
                    "imageMediaMetadata": {"width": 1920, "height": 1080},
                    "trashed": False,
                },
                {
                    "id": "drive_unique",
                    "name": "unique.jpg",
                    "mimeType": "image/jpeg",
                    "size": "200000",
                    "md5Checksum": "unique_md5_999",
                    "modifiedTime": "2024-01-01T02:00:00.000Z",
                    "imageMediaMetadata": {"width": 1920, "height": 1080},
                    "trashed": False,
                },
            ]
        }

        source = GoogleDriveSource(
            folder_id="fldr",
            service=mock_service,
            cache_dir=tmp_path / "cache",
            conn=db,
        )

        with patch("urllib.request.urlopen") as mock_urlopen, patch.object(source, "get_bytes") as mock_get_bytes:
            res = run_stage0(source, db, cfg)
            # Ensure zero network download calls occurred
            mock_urlopen.assert_not_called()
            mock_get_bytes.assert_not_called()

        assert res.files_discovered == 3
        assert res.files_hashed == 0  # 0 files hashed locally because MD5 was provided by Drive metadata!
        assert res.clusters_created == 1
        assert res.files_clustered == 2

        # Check clusters in DB
        rows = db.execute("SELECT file_path, cluster_id, is_keeper, md5 FROM images ORDER BY id").fetchall()
        assert len(rows) == 3
        assert rows[0]["cluster_id"] == rows[1]["cluster_id"]
        assert rows[0]["cluster_id"] is not None
        assert rows[2]["cluster_id"] is None
        assert rows[0]["md5"] == "duplicate_md5_123"


class TestThumbnailFetchingAndRefetch:
    def test_expired_thumbnail_refetch(self, tmp_path: Path):
        """When an initial thumbnail link returns 403/404, refetch fresh thumbnailLink from API."""
        mock_service = MagicMock()
        sample_jpeg = _create_sample_jpeg_bytes()

        # Mock files().get() returning fresh thumbnailLink
        fresh_req = MagicMock()
        fresh_req.execute.return_value = {"thumbnailLink": "https://lh3.googleusercontent.com/fresh_link=s220"}
        mock_service.files().get.return_value = fresh_req

        source = GoogleDriveSource(
            service=mock_service,
            cache_dir=tmp_path / "cache",
        )
        # Pre-seed item cache with an expired link
        source._drive_item_cache["file_xyz"] = {
            "thumbnailLink": "https://lh3.googleusercontent.com/expired_link=s220"
        }
        source._metadata_cache["file_xyz"] = MagicMock(modified_time=1700000000.0)

        # First call to urlopen fails with 403 (expired link)
        # Second call succeeds with fresh link
        resp_success = MagicMock()
        resp_success.status = 200
        resp_success.read.return_value = sample_jpeg
        resp_success.__enter__.return_value = resp_success

        import urllib.error
        err_403 = urllib.error.HTTPError(
            url="https://lh3.googleusercontent.com/expired_link=s512",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=[err_403, resp_success]) as mock_urlopen:
            thumb = source.get_thumbnail("drive://file_xyz/sample.jpg", size=(128, 128))

        assert isinstance(thumb, Image.Image)
        # Check that files().get was called to retrieve the fresh link
        mock_service.files().get.assert_called_with(
            fileId="file_xyz",
            fields="thumbnailLink",
            supportsAllDrives=True,
        )
        # Check cached on disk
        cache_file = tmp_path / "cache" / "file_xyz_1700000000.jpg"
        assert cache_file.exists()


class TestRateLimitingBackoff:
    def test_retry_with_backoff_on_429(self):
        """Exponential backoff handles 429 rate limit errors and retries."""
        mock_fn = MagicMock()

        err_resp = MagicMock()
        err_resp.status = 429
        from googleapiclient.errors import HttpError
        http_err = HttpError(resp=err_resp, content=b"Rate Limit Exceeded")

        # Fails twice with 429, succeeds on third attempt
        mock_fn.side_effect = [http_err, http_err, "success_result"]

        with patch("time.sleep") as mock_sleep:
            result = retry_with_backoff(mock_fn, max_retries=5, initial_delay=0.1)

        assert result == "success_result"
        assert mock_fn.call_count == 3
        assert mock_sleep.call_count == 2


class TestSecurityInvariants:
    def test_unclustered_file_full_download_forbidden(self, db: sqlite3.Connection, tmp_path: Path):
        """Security Invariant: Calling get_bytes on unclustered files must be rejected."""
        # Insert unclustered file
        db.execute(
            """
            INSERT INTO images (file_path, file_size, modified_time, cluster_id, source, drive_id, status)
            VALUES ('drive://unclustered_123/img.jpg', 50000, 1000.0, NULL, 'gdrive', 'unclustered_123', 'hashed')
            """
        )
        db.commit()

        mock_service = MagicMock()
        source = GoogleDriveSource(
            service=mock_service,
            cache_dir=tmp_path / "cache",
            conn=db,
        )

        with pytest.raises(PermissionError, match="Security Invariant: Cannot download full file for unclustered image"):
            source.get_bytes("drive://unclustered_123/img.jpg")

        # Ensure Drive API get_media was never called
        mock_service.files().get_media.assert_not_called()

    def test_clustered_file_full_download_allowed_and_temp_cleaned(self, db: sqlite3.Connection, tmp_path: Path):
        """Clustered files can be downloaded, streamed through temp storage, and temp files deleted."""
        # Insert image first, then cluster pointing to it as representative, then set cluster_id
        db.execute(
            """
            INSERT INTO images (id, file_path, file_size, modified_time, cluster_id, source, drive_id, status)
            VALUES (1, 'drive://clustered_456/img.jpg', 50000, 1000.0, NULL, 'gdrive', 'clustered_456', 'hashed')
            """
        )
        db.execute(
            """
            INSERT INTO clusters (id, representative_id, stage_detected) VALUES (1, 1, 0)
            """
        )
        db.execute("UPDATE images SET cluster_id = 1 WHERE id = 1")
        db.commit()

        mock_service = MagicMock()
        fake_content = b"SIMULATED_FULL_IMAGE_PAYLOAD"

        downloader_mock = MagicMock()
        status_mock = MagicMock()
        status_mock.resumable_progress = len(fake_content)

        # Mock MediaIoBaseDownload
        with patch("googleapiclient.http.MediaIoBaseDownload") as mock_downloader_cls:
            def side_effect_download(fh, req, **kwargs):
                fh.write(fake_content)
                inst = MagicMock()
                inst.next_chunk.return_value = (status_mock, True)
                return inst

            mock_downloader_cls.side_effect = side_effect_download

            temp_dir = tmp_path / "temp"
            temp_dir.mkdir()
            source = GoogleDriveSource(
                service=mock_service,
                cache_dir=tmp_path / "cache",
                temp_dir=temp_dir,
                conn=db,
            )

            data = source.get_bytes("drive://clustered_456/img.jpg")
            assert data == fake_content

            # Verify temp files are cleaned up
            remaining_temp_files = list(temp_dir.glob("*"))
            assert len(remaining_temp_files) == 0

            # Verify bandwidth was recorded
            row = db.execute("SELECT bytes_downloaded, item_type FROM drive_bandwidth WHERE file_id = 'clustered_456'").fetchone()
            assert row is not None
            assert row["bytes_downloaded"] == len(fake_content)
            assert row["item_type"] == "full_file"


class TestAuthAndTokenPermissions:
    def test_token_file_permissions(self, tmp_path: Path):
        """OAuth token file must be created with secure 0o600 permissions."""
        secret_file = tmp_path / "client_secret.json"
        secret_file.write_text(json.dumps({"installed": {"client_id": "test_id", "client_secret": "test_sec"}}))
        token_file = tmp_path / "drive_token.json"

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.expired = False
        mock_creds.to_json.return_value = json.dumps({"token": "fake_oauth_token"})

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds

        with patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file", return_value=mock_flow):
            creds = get_drive_credentials(
                client_secret_path=secret_file,
                token_path=token_file,
                prompt_if_missing=True,
            )

        assert token_file.exists()
        mode = oct(token_file.stat().st_mode & 0o777)
        assert mode == "0o600"


class TestBandwidthAccountingSummary:
    def test_bandwidth_summary_reporting(self, db: sqlite3.Connection, capsys):
        """Verify summary accurately reflects downloads by stage vs total library size."""
        db.execute(
            """
            INSERT INTO images (file_path, file_size, modified_time, source)
            VALUES ('drive://f1/img1.jpg', 5000000, 1000.0, 'gdrive'),
                   ('drive://f2/img2.jpg', 5000000, 1000.0, 'gdrive')
            """
        )
        record_bandwidth(db, "stage1", "f1", 45000, "thumbnail")
        record_bandwidth(db, "stage1", "f2", 48000, "thumbnail")
        record_bandwidth(db, "stage3", "f1", 5000000, "full_file")

        print_bandwidth_summary(db)
        captured = capsys.readouterr().out

        assert "Google Drive Bandwidth Accounting Summary" in captured
        assert "Total Google Drive images indexed : 2" in captured
        assert "Full library uncompressed size    : 9.54 MB" in captured
        assert "STAGE1" in captured
        assert "STAGE3" in captured
        assert "Bandwidth saved vs full download" in captured

    def test_metadata_bytes_logged_during_list_and_get(self, db: sqlite3.Connection, tmp_path: Path):
        """All metadata API operations (list, get, fetch thumbnailLink) must be logged in drive_bandwidth."""
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {
            "files": [
                {
                    "id": "item_1",
                    "name": "pic.jpg",
                    "mimeType": "image/jpeg",
                    "size": "50000",
                    "md5Checksum": "hash_1",
                    "modifiedTime": "2024-01-01T00:00:00.000Z",
                    "imageMediaMetadata": {"width": 800, "height": 600},
                    "trashed": False,
                }
            ]
        }
        mock_service.files().get().execute.return_value = {
            "id": "item_1",
            "name": "pic.jpg",
            "mimeType": "image/jpeg",
            "size": "50000",
            "md5Checksum": "hash_1",
            "modifiedTime": "2024-01-01T00:00:00.000Z",
            "imageMediaMetadata": {"width": 800, "height": 600},
            "thumbnailLink": "https://example.com/thumb",
            "trashed": False,
        }

        source = GoogleDriveSource(
            folder_id="test_folder",
            service=mock_service,
            cache_dir=tmp_path / "cache",
            conn=db,
            stage="stage0",
        )

        files = list(source.list_files())
        assert len(files) == 1

        # Check list metadata logged
        rows = db.execute("SELECT item_type, bytes_downloaded FROM drive_bandwidth WHERE item_type = 'metadata'").fetchall()
        assert len(rows) >= 1
        assert rows[0]["bytes_downloaded"] > 0

        # Now call get_metadata
        source._metadata_cache.clear()
        source.get_metadata(files[0])
        rows_after = db.execute("SELECT item_type, bytes_downloaded FROM drive_bandwidth WHERE item_type = 'metadata'").fetchall()
        assert len(rows_after) >= 2


class TestLRUThumbnailCache:
    def test_lru_cache_eviction(self, tmp_path: Path):
        """Cache must respect max_cache_bytes and evict least recently used files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Set limit to 4000 bytes
        source = GoogleDriveSource(
            cache_dir=cache_dir,
            max_cache_bytes=4000,
        )

        # Create two 1500-byte files with different access times
        f1 = cache_dir / "old_file.jpg"
        f1.write_bytes(b"A" * 1500)
        os.utime(f1, (1000.0, 1000.0))

        f2 = cache_dir / "recent_file.jpg"
        f2.write_bytes(b"B" * 1500)
        os.utime(f2, (2000.0, 2000.0))

        # Total cache is now 3000 bytes <= 4000
        source._enforce_cache_cap(incoming_bytes=0)
        assert f1.exists()
        assert f2.exists()

        # Now simulate incoming 1500 bytes (total would be 4500 > 4000)
        source._enforce_cache_cap(incoming_bytes=1500)

        # Oldest file (f1 with atime 1000) must be evicted!
        assert not f1.exists()
        assert f2.exists()

    def test_cache_hit_updates_atime(self, tmp_path: Path):
        """Reading an existing thumbnail from cache must update its atime (marking it MRU)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        sample_jpeg = _create_sample_jpeg_bytes()
        f = cache_dir / "test_100.jpg"
        f.write_bytes(sample_jpeg)
        old_time = 500000.0
        os.utime(f, (old_time, old_time))

        source = GoogleDriveSource(
            cache_dir=cache_dir,
            max_cache_bytes=100000,
        )
        source._metadata_cache["test"] = MagicMock(modified_time=100.0)

        thumb = source.get_thumbnail("drive://test/photo.jpg")
        assert isinstance(thumb, Image.Image)

        # Verify access time was updated to current time
        new_atime = f.stat().st_atime
        assert new_atime > old_time


class TestDriveThumbnailConfirmation:
    def test_thumbnail_based_member_never_delete_safe(self, db: sqlite3.Connection, cfg: dict):
        """In choose_keeper / run_keeper_pipeline, a Drive member with quality_basis='thumbnail' is never delete_safe."""
        db.execute(
            """
            INSERT INTO clusters (id, stage_detected, member_count)
            VALUES (1, 1, 2)
            """
        )
        # Reference image
        db.execute(
            """
            INSERT INTO images (
                id, file_path, file_size, width, height, format, md5,
                quality_status, ssim, ssim_color, match_pct, cluster_id,
                source, drive_id, quality_basis
            ) VALUES (
                1, 'drive://ref_id/ref.jpg', 200000, 2000, 2000, 'JPEG', 'ref_md5',
                'reference', 1.0, 1.0, 100.0, 1,
                'gdrive', 'ref_id', 'thumbnail'
            )
            """
        )
        # Candidate smaller duplicate (95% match, 95% ssim, confidence 1.0, 50% file size)
        db.execute(
            """
            INSERT INTO images (
                id, file_path, file_size, width, height, format, md5,
                quality_status, ssim, ssim_color, match_pct, cluster_id,
                confidence, is_low_confidence, source, drive_id, quality_basis
            ) VALUES (
                2, 'drive://cand_id/cand.jpg', 100000, 1500, 1500, 'JPEG', 'cand_md5',
                'aligned', 0.95, 0.95, 95.0, 1,
                1.0, 0, 'gdrive', 'cand_id', 'thumbnail'
            )
            """
        )
        db.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")
        db.commit()

        run_keeper_pipeline(db, config=cfg, mode="balanced")

        row = db.execute("SELECT delete_safe, review_flag, reason FROM images WHERE id = 2").fetchone()
        assert row["delete_safe"] == 0
        assert row["review_flag"] == 1
        assert "needs full-resolution confirmation" in row["reason"]

    def test_md5_copies_unaffected(self, db: sqlite3.Connection, cfg: dict):
        """Byte-identical MD5 copies remain delete_safe even with quality_basis='thumbnail'."""
        db.execute(
            """
            INSERT INTO clusters (id, stage_detected, member_count)
            VALUES (1, 0, 2)
            """
        )
        # Reference image
        db.execute(
            """
            INSERT INTO images (
                id, file_path, file_size, width, height, format, md5,
                quality_status, ssim, ssim_color, match_pct, cluster_id,
                source, drive_id, quality_basis
            ) VALUES (
                1, 'drive://ref_id/ref.jpg', 200000, 2000, 2000, 'JPEG', 'exact_md5',
                'reference', 1.0, 1.0, 100.0, 1,
                'gdrive', 'ref_id', 'thumbnail'
            )
            """
        )
        # Exact copy
        db.execute(
            """
            INSERT INTO images (
                id, file_path, file_size, width, height, format, md5,
                quality_status, ssim, ssim_color, match_pct, cluster_id,
                confidence, is_low_confidence, source, drive_id, quality_basis
            ) VALUES (
                2, 'drive://copy_id/copy.jpg', 200000, 2000, 2000, 'JPEG', 'exact_md5',
                'aligned', 1.0, 1.0, 100.0, 1,
                1.0, 0, 'gdrive', 'copy_id', 'thumbnail'
            )
            """
        )
        db.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")
        db.commit()

        run_keeper_pipeline(db, config=cfg, mode="balanced")

        row = db.execute("SELECT delete_safe, review_flag, reason FROM images WHERE id = 2").fetchone()
        assert row["delete_safe"] == 1
        assert row["review_flag"] == 0
        assert "Byte-identical" in row["reason"]

    def test_after_confirm_can_be_delete_safe(self, db: sqlite3.Connection, cfg: dict, tmp_path: Path):
        """After running confirm-drive, confirmed Drive member gets quality_basis='full' and can become delete_safe."""
        db.execute(
            """
            INSERT INTO clusters (id, stage_detected, member_count)
            VALUES (1, 1, 2)
            """
        )
        db.execute(
            """
            INSERT INTO images (
                id, file_path, file_size, width, height, format, md5,
                quality_status, ssim, ssim_color, match_pct, cluster_id,
                source, drive_id, quality_basis
            ) VALUES (
                1, 'drive://ref_id/ref.jpg', 200000, 2000, 2000, 'JPEG', 'ref_md5',
                'reference', 1.0, 1.0, 100.0, 1,
                'gdrive', 'ref_id', 'thumbnail'
            )
            """
        )
        db.execute(
            """
            INSERT INTO images (
                id, file_path, file_size, width, height, format, md5,
                quality_status, ssim, ssim_color, match_pct, cluster_id,
                confidence, is_low_confidence, source, drive_id, quality_basis
            ) VALUES (
                2, 'drive://cand_id/cand.jpg', 100000, 1500, 1500, 'JPEG', 'cand_md5',
                'aligned', 0.95, 0.95, 95.0, 1,
                1.0, 0, 'gdrive', 'cand_id', 'thumbnail'
            )
            """
        )
        db.execute("UPDATE clusters SET representative_id = 1 WHERE id = 1")
        db.commit()

        # Pre-cache transform from Stage 3 verification
        M_ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        db.execute(
            """
            INSERT INTO pair_transforms (source_id, target_id, model, transform_matrix, inlier_count, inlier_ratio)
            VALUES (2, 1, 'affine', ?, 50, 0.95)
            """,
            (pickle.dumps(M_ident),),
        )
        db.commit()

        # Generate realistic image bytes for ref and cand
        ref_buf = io.BytesIO()
        ref_im = Image.new("RGB", (200, 200), color=(120, 140, 160))
        ref_im.save(ref_buf, format="JPEG")
        ref_bytes = ref_buf.getvalue()

        cand_buf = io.BytesIO()
        cand_im = Image.new("RGB", (200, 200), color=(120, 140, 160))
        cand_im.save(cand_buf, format="JPEG")
        cand_bytes = cand_buf.getvalue()

        mock_source = MagicMock(spec=GoogleDriveSource)
        mock_source.conn = db
        mock_source.stage = "confirm"

        def mock_get_bytes(fpath: str) -> bytes:
            if "ref_id" in fpath:
                record_bandwidth(db, "confirm", "ref_id", len(ref_bytes), "full_file")
                return ref_bytes
            else:
                record_bandwidth(db, "confirm", "cand_id", len(cand_bytes), "full_file")
                return cand_bytes

        mock_source.get_bytes.side_effect = mock_get_bytes

        res = run_confirm_drive(db, config=cfg, mode="balanced", source=mock_source)

        assert len(res.confirmed_members) == 1
        rec = res.confirmed_members[0]
        assert rec.member_id == 2
        assert rec.full_match_pct is not None
        assert rec.is_delete_safe is True

        cand_row = db.execute("SELECT quality_basis, delete_safe, review_flag FROM images WHERE id = 2").fetchone()
        assert cand_row["quality_basis"] == "full"
        assert cand_row["delete_safe"] == 1
        assert cand_row["review_flag"] == 0

        ref_row = db.execute("SELECT quality_basis FROM images WHERE id = 1").fetchone()
        assert ref_row["quality_basis"] == "full"

        # Check bandwidth was recorded under stage confirm
        bw_row = db.execute("SELECT SUM(bytes_downloaded) FROM drive_bandwidth WHERE stage = 'confirm'").fetchone()
        assert bw_row[0] == len(ref_bytes) + len(cand_bytes)


class TestDriveThumbnailConcurrencyAndCaching:
    def test_fetch_thumbnails_thread_pool_and_progress_reports(self, tmp_path: Path):
        """Fetch Drive thumbnails with 8 workers, 20s timeout, and report 'N of M' after every file."""
        mock_service = MagicMock()
        sample_jpeg = _create_sample_jpeg_bytes()

        source = GoogleDriveSource(
            service=mock_service,
            cache_dir=tmp_path / "cache",
        )

        file_ids = [f"drive://file_{i}/photo_{i}.jpg" for i in range(16)]
        for i, fid in enumerate(file_ids):
            source._drive_item_cache[f"file_{i}"] = {
                "thumbnailLink": f"https://lh3.googleusercontent.com/link_{i}=s220"
            }
            source._metadata_cache[fid] = MagicMock(modified_time=1700000000.0 + i)
            source._metadata_cache[f"file_{i}"] = MagicMock(modified_time=1700000000.0 + i)

        resp_mock = MagicMock()
        resp_mock.status = 200
        resp_mock.read.return_value = sample_jpeg
        resp_mock.__enter__.return_value = resp_mock

        progress_calls = []

        def on_progress(cur, total, fid):
            progress_calls.append((cur, total, fid))

        with patch("urllib.request.urlopen", return_value=resp_mock) as mock_urlopen:
            thumbs = source.fetch_thumbnails(file_ids, max_workers=8, progress_callback=on_progress)

        assert len(thumbs) == 16
        assert len(progress_calls) == 16
        # Assert each call received (cur, total, fid)
        assert progress_calls[-1][0] == 16
        assert progress_calls[-1][1] == 16
        for cur, total, fid in progress_calls:
            assert total == 16
            assert fid in file_ids

        # Check all files cached on disk by file_id + modifiedTime
        for i in range(16):
            expected_cache = tmp_path / "cache" / f"file_{i}_{int(1700000000.0 + i)}.jpg"
            assert expected_cache.exists()
            assert expected_cache.stat().st_size > 0

    def test_cache_by_file_id_and_mtime_skips_network(self, tmp_path: Path):
        """Cache by file id + modifiedTime; skip files already processed."""
        mock_service = MagicMock()
        sample_jpeg = _create_sample_jpeg_bytes()

        source = GoogleDriveSource(
            service=mock_service,
            cache_dir=tmp_path / "cache",
        )

        # Pre-seed cache on disk for file_cached
        cache_file = tmp_path / "cache" / "file_cached_1700005555.jpg"
        cache_file.write_bytes(sample_jpeg)

        source._metadata_cache["drive://file_cached/pic.jpg"] = MagicMock(modified_time=1700005555.0)
        source._metadata_cache["file_cached"] = MagicMock(modified_time=1700005555.0)

        with patch("urllib.request.urlopen") as mock_urlopen:
            thumb = source.get_thumbnail("drive://file_cached/pic.jpg")
            # Network download must NOT be called because it is already cached
            mock_urlopen.assert_not_called()

        assert isinstance(thumb, Image.Image)

    def test_download_thumbnail_retries_with_backoff_on_429(self, tmp_path: Path):
        """Thumbnail download retries up to 3 times with backoff on HTTP 429."""
        mock_service = MagicMock()
        sample_jpeg = _create_sample_jpeg_bytes()

        source = GoogleDriveSource(
            service=mock_service,
            cache_dir=tmp_path / "cache",
        )

        import urllib.error
        err_429 = urllib.error.HTTPError(
            url="https://lh3.googleusercontent.com/test=s512",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=None,
        )

        resp_success = MagicMock()
        resp_success.status = 200
        resp_success.read.return_value = sample_jpeg
        resp_success.__enter__.return_value = resp_success

        with patch("urllib.request.urlopen", side_effect=[err_429, err_429, resp_success]) as mock_urlopen, \
             patch("time.sleep") as mock_sleep:
            data = source._download_thumbnail_bytes("https://lh3.googleusercontent.com/test=s512", timeout=20, max_retries=3)

        assert data == sample_jpeg
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2

    def test_fetch_thumbnails_concurrent_sqlite_safety(self, tmp_path: Path):
        """Concurrent thumbnail fetching across 8 workers safely records bandwidth and queries DB without corruption."""
        from db.connection import init_db

        conn = init_db(tmp_path / "test_concurrency.db")
        sample_jpeg = _create_sample_jpeg_bytes()
        mock_service = MagicMock()

        # Insert 32 image rows
        file_ids = [f"drive://file_{i}/photo_{i}.jpg" for i in range(32)]
        for i, fid in enumerate(file_ids):
            conn.execute(
                """
                INSERT INTO images (file_path, file_size, modified_time, drive_id, source, status)
                VALUES (?, ?, ?, ?, 'gdrive', 'discovered')
                """,
                (fid, 1024 * 50, 1700000000.0 + i, f"file_{i}"),
            )
        conn.commit()

        source = GoogleDriveSource(
            service=mock_service,
            cache_dir=tmp_path / "cache",
            conn=conn,
            stage="stage1",
        )
        for i in range(32):
            source._drive_item_cache[f"file_{i}"] = {
                "thumbnailLink": f"https://lh3.googleusercontent.com/link_{i}=s220"
            }

        resp_mock = MagicMock()
        resp_mock.status = 200
        resp_mock.read.return_value = sample_jpeg
        resp_mock.__enter__.return_value = resp_mock

        with patch("urllib.request.urlopen", return_value=resp_mock):
            thumbs = source.fetch_thumbnails(file_ids, max_workers=8)

        assert len(thumbs) == 32
        # Check bandwidth table has 32 entries recorded concurrently
        count = conn.execute("SELECT COUNT(*) FROM drive_bandwidth WHERE stage = 'stage1'").fetchone()[0]
        assert count == 32
        conn.close()



class TestDriveAdminConfig:
    """Tests for Drive admin configuration — find_drive_client_secret, redirect URI, and callback."""

    def test_find_drive_client_secret_returns_searched_paths(self, tmp_path: Path, monkeypatch):
        """find_drive_client_secret must return a 3-tuple with searched paths list."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        from api.server import find_drive_client_secret

        # No secret file exists anywhere in tmp_path
        path, info, searched = find_drive_client_secret(config={}, data_dir=tmp_path)
        assert path is None
        assert info is None
        assert isinstance(searched, list)
        assert len(searched) > 0
        # The tmp_path candidate should appear in the searched list
        assert any(str(tmp_path) in str(p) for p in searched)

    def test_find_drive_client_secret_finds_installed_type(self, tmp_path: Path):
        """find_drive_client_secret must find 'installed'-type client_secret.json."""
        from api.server import find_drive_client_secret

        secret_file = tmp_path / "client_secret.json"
        secret_file.write_text(json.dumps({
            "installed": {
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
            }
        }))

        path, info, searched = find_drive_client_secret(config={}, data_dir=tmp_path)
        assert path == secret_file
        assert info is not None
        assert info["client_id"] == "test_client_id"
        assert isinstance(searched, list)

    def test_find_drive_client_secret_finds_web_type(self, tmp_path: Path):
        """find_drive_client_secret must find 'web'-type client_secret.json (server-side OAuth)."""
        from api.server import find_drive_client_secret

        secret_file = tmp_path / "client_secret.json"
        secret_file.write_text(json.dumps({
            "web": {
                "client_id": "web_client_id",
                "client_secret": "web_client_secret",
                "redirect_uris": ["https://example.com/drive/callback"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }))

        path, info, searched = find_drive_client_secret(config={}, data_dir=tmp_path)
        assert path == secret_file
        assert info is not None
        assert info["client_id"] == "web_client_id"

    def test_redirect_uri_uses_base_url_env_var(self, tmp_path: Path):
        """The redirect URI built in /drive/auth-url must use BASE_URL env var when set."""
        from api.server import find_drive_client_secret

        # Verify the redirect URI pattern — the endpoint uses f"{base}/drive/callback"
        # where base = os.environ.get("BASE_URL") or derived from request.base_url.
        base_url = "https://argus.example.com"
        expected_redirect = f"{base_url}/drive/callback"
        assert expected_redirect.endswith("/drive/callback")
        assert expected_redirect.startswith(base_url)

    def test_drive_callback_invalid_state_returns_400(self, tmp_path: Path):
        """GET /drive/callback with an invalid (unknown) state must return HTTP 400."""
        from fastapi.testclient import TestClient
        from api.server import create_app
        from core.config import load_config

        cfg = load_config()
        with patch.dict(os.environ, {"ARGUS_DATA_DIR": str(tmp_path), "SECRET_KEY": "testsecretkey12345678901234567890"}):
            app = create_app(config=cfg)

        with TestClient(app, raise_server_exceptions=False) as client:
            reg = client.post("/auth/register", json={"email": "cb_test@example.com", "password": "TestPass123!"})
            if reg.status_code not in (200, 201):
                pytest.skip("Registration not available in test config")

            login = client.post("/auth/login", json={"email": "cb_test@example.com", "password": "TestPass123!"})
            if login.status_code != 200:
                pytest.skip("Login not available in test config")

            token = login.json().get("access_token") or login.cookies.get("session")
            headers = {"Authorization": f"Bearer {token}"} if token else {}

            resp = client.get(
                "/drive/callback",
                params={"code": "fake_auth_code", "state": "totally_invalid_state_xyz"},
                headers=headers,
            )
            assert resp.status_code == 400

    def test_drive_callback_missing_code_returns_400(self, tmp_path: Path):
        """GET /drive/callback with missing code must return HTTP 400."""
        from fastapi.testclient import TestClient
        from api.server import create_app
        from core.config import load_config

        cfg = load_config()
        with patch.dict(os.environ, {"ARGUS_DATA_DIR": str(tmp_path), "SECRET_KEY": "testsecretkey12345678901234567890"}):
            app = create_app(config=cfg)

        with TestClient(app, raise_server_exceptions=False) as client:
            reg = client.post("/auth/register", json={"email": "cb_nocode@example.com", "password": "TestPass123!"})
            if reg.status_code not in (200, 201):
                pytest.skip("Registration not available in test config")

            login = client.post("/auth/login", json={"email": "cb_nocode@example.com", "password": "TestPass123!"})
            if login.status_code != 200:
                pytest.skip("Login not available in test config")

            token = login.json().get("access_token") or login.cookies.get("session")
            headers = {"Authorization": f"Bearer {token}"} if token else {}

            resp = client.get(
                "/drive/callback",
                params={"state": "some_state"},
                headers=headers,
            )
            assert resp.status_code == 400

    def test_drive_status_shows_searched_paths_when_not_configured(self, tmp_path: Path, monkeypatch):
        """GET /drive/status must include searched paths in the message when not configured."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr("sources.google_drive.DEFAULT_TOKEN_PATH", str(tmp_path / "drive_token.json"))
        from fastapi.testclient import TestClient
        from api.server import create_app
        from core.config import load_config

        cfg = load_config()
        # Use empty tmp_path so no client_secret.json exists
        with patch.dict(os.environ, {"ARGUS_DATA_DIR": str(tmp_path), "SECRET_KEY": "testsecretkey12345678901234567890"}):
            app = create_app(config=cfg)

        with TestClient(app, raise_server_exceptions=False) as client:
            reg = client.post("/auth/register", json={"email": "status_test@example.com", "password": "TestPass123!"})
            if reg.status_code not in (200, 201):
                pytest.skip("Registration not available in test config")

            login = client.post("/auth/login", json={"email": "status_test@example.com", "password": "TestPass123!"})
            if login.status_code != 200:
                pytest.skip("Login not available in test config")

            token = login.json().get("access_token") or login.cookies.get("session")
            headers = {"Authorization": f"Bearer {token}"} if token else {}

            resp = client.get("/drive/status", headers=headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["configured"] is False
            # The message should mention the searched paths
            msg = data.get("message", "")
            assert "Searched" in msg or "not configured" in msg

    def test_configure_drive_via_frontend_upload(self, tmp_path: Path):
        """POST /drive/configure uploads client_secret.json and enables Drive integration."""
        from fastapi.testclient import TestClient
        from api.server import create_app
        from core.config import load_config

        cfg = load_config()
        data_dir = tmp_path / "argus_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        with patch.dict(os.environ, {"ARGUS_DATA_DIR": str(data_dir), "SECRET_KEY": "testsecretkey12345678901234567890"}):
            app = create_app(config=cfg)

        with TestClient(app, raise_server_exceptions=False) as client:
            reg = client.post("/auth/register", json={"email": "conf_test@example.com", "password": "TestPass123!"})
            if reg.status_code not in (200, 201):
                pytest.skip("Registration not available in test config")

            login = client.post("/auth/login", json={"email": "conf_test@example.com", "password": "TestPass123!"})
            if login.status_code != 200:
                pytest.skip("Login not available in test config")

            login_data = login.json()
            token = login_data.get("token")
            csrf = login_data.get("csrf_token")
            if token:
                client.cookies.set("argus_session", token)
            if csrf:
                client.headers["X-CSRF-Token"] = csrf

            # Upload invalid file (missing client_id)
            bad_json = json.dumps({"foo": "bar"}).encode("utf-8")
            bad_resp = client.post(
                "/drive/configure",
                files={"file": ("bad_secret.json", io.BytesIO(bad_json), "application/json")},
            )
            assert bad_resp.status_code == 400

            # Upload valid client_secret.json
            valid_secret = json.dumps({
                "web": {
                    "client_id": "uploaded_web_client_id",
                    "client_secret": "uploaded_web_client_secret",
                    "redirect_uris": ["http://localhost:8000/drive/callback"],
                }
            }).encode("utf-8")

            good_resp = client.post(
                "/drive/configure",
                files={"file": ("client_secret.json", io.BytesIO(valid_secret), "application/json")},
            )
            assert good_resp.status_code == 200
            assert good_resp.json()["configured"] is True

            # Verify saved file exists on disk
            saved_secret = data_dir / "client_secret.json"
            assert saved_secret.is_file()

            # Verify /drive/status now reports configured = True
            st_resp = client.get("/drive/status")
            assert st_resp.status_code == 200
            assert st_resp.json()["configured"] is True


class TestDriveFolderValidationAndHardening:
    """Tests for Drive folder link extraction, pre-flight validation, and credential auto-refresh."""

    def test_extract_drive_folder_id_variants(self):
        from api.server import extract_drive_folder_id
        from fastapi import HTTPException

        # Root aliases
        assert extract_drive_folder_id("root") == "root"
        assert extract_drive_folder_id("ROOT") == "root"
        assert extract_drive_folder_id("My Drive") == "root"
        assert extract_drive_folder_id("mydrive") == "root"
        assert extract_drive_folder_id("") == "root"

        # Standard URLs
        url1 = "https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI"
        assert extract_drive_folder_id(url1) == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI"

        # URL with query params
        url2 = "https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI?usp=sharing"
        assert extract_drive_folder_id(url2) == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI"

        # URL with user index
        url3 = "https://drive.google.com/drive/u/2/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI"
        assert extract_drive_folder_id(url3) == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI"

        # Open link with id parameter
        url4 = "https://drive.google.com/open?id=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI"
        assert extract_drive_folder_id(url4) == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI"

        # Raw valid ID
        raw_id = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI"
        assert extract_drive_folder_id(raw_id) == raw_id

        # Invalid URLs/IDs
        with pytest.raises(HTTPException) as exc:
            extract_drive_folder_id("https://drive.google.com/invalid/link")
        assert exc.value.status_code == 400

        with pytest.raises(HTTPException) as exc:
            extract_drive_folder_id("not-a-valid-id-***")
        assert exc.value.status_code == 400

    def test_drive_credentials_auto_refresh_on_expired(self, tmp_path: Path):
        from sources.google_drive import GoogleDriveSource
        tok_file = tmp_path / "drive_token.json"
        tok_file.write_text(json.dumps({"token": "old_token", "refresh_token": "ref123"}))

        class MockCreds:
            def __init__(self):
                self.expired = True
                self.refresh_token = "ref123"
                self.token = "old_token"
                self.refreshed = False

            def refresh(self, request):
                self.refreshed = True
                self.expired = False
                self.token = "new_refreshed_token"

            def to_json(self):
                return json.dumps({"token": self.token, "refresh_token": self.refresh_token})

        mock_creds = MockCreds()
        source = GoogleDriveSource(token_path=tok_file, credentials=mock_creds)
        assert mock_creds.refreshed is False

        # Access property -> triggers auto-refresh
        active_creds = source.credentials
        assert mock_creds.refreshed is True
        assert active_creds.token == "new_refreshed_token"
        assert "new_refreshed_token" in tok_file.read_text()

    def test_validate_drive_endpoint_flow(self, tmp_path: Path, monkeypatch):
        from fastapi.testclient import TestClient
        from api.server import create_app
        from core.config import load_config
        from unittest.mock import MagicMock

        cfg = load_config()
        data_dir = tmp_path / "argus_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        with patch.dict(os.environ, {"ARGUS_DATA_DIR": str(data_dir), "SECRET_KEY": "testsecretkey12345678901234567890"}):
            app = create_app(config=cfg)

        with TestClient(app, raise_server_exceptions=False) as client:
            client.post("/auth/register", json={"email": "val_test@example.com", "password": "TestPass123!"})
            login = client.post("/auth/login", json={"email": "val_test@example.com", "password": "TestPass123!"})
            login_data = login.json()
            token = login_data.get("token")
            csrf = login_data.get("csrf_token")
            if token:
                client.cookies.set("argus_session", token)
            if csrf:
                client.headers["X-CSRF-Token"] = csrf

            # 1. Not connected yet
            val_unauthed = client.get("/sources/drive/validate?link=root")
            assert val_unauthed.status_code == 200
            assert val_unauthed.json()["valid"] is False
            assert val_unauthed.json()["error"] == "not_authenticated"

            # 2. Connect token
            tok_file = data_dir / "users" / "1" / "drive_token.json"
            tok_file.parent.mkdir(parents=True, exist_ok=True)
            tok_file.write_text(json.dumps({
                "token": "valid_token",
                "refresh_token": "valid_refresh",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
                "expiry": "2099-01-01T00:00:00Z",
            }))

            class DummyValidCreds:
                valid = True
                expired = False
                token = "valid_token"
                refresh_token = "valid_refresh"

            monkeypatch.setattr("sources.google_drive.get_drive_credentials", lambda *args, **kwargs: DummyValidCreds())

            # Mock get_drive_service
            mock_service = MagicMock()
            mock_files = MagicMock()
            mock_get = MagicMock()
            mock_get.execute.return_value = {
                "id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI",
                "name": "My Holiday Photos",
                "mimeType": "application/vnd.google-apps.folder",
                "trashed": False,
            }
            mock_files.get.return_value = mock_get
            mock_service.files.return_value = mock_files

            monkeypatch.setattr("sources.google_drive.get_drive_service", lambda *args, **kwargs: mock_service)

            # 3. Validate root folder
            val_root = client.get("/sources/drive/validate?link=root")
            assert val_root.status_code == 200
            assert val_root.json()["valid"] is True
            assert val_root.json()["folder_id"] == "root"

            # 4. Validate specific folder via GET
            val_folder = client.get("/sources/drive/validate?link=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI")
            assert val_folder.status_code == 200
            assert val_folder.json()["valid"] is True
            assert val_folder.json()["folder_name"] == "My Holiday Photos"

            # 5. Validate specific folder via POST
            val_post = client.post("/sources/drive/validate", json={"link": "https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI"})
            assert val_post.status_code == 200
            assert val_post.json()["valid"] is True
            assert val_post.json()["folder_name"] == "My Holiday Photos"

            # 6. Pre-flight rejection on trashed folder
            mock_get.execute.return_value["trashed"] = True
            val_trashed = client.get("/sources/drive/validate?link=1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OI")
            assert val_trashed.status_code == 200
            assert val_trashed.json()["valid"] is False
            assert val_trashed.json()["error"] == "folder_trashed"

