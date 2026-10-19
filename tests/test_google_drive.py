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
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from PIL import Image

from core.config import load_config
from core.stage0 import run_stage0
from db.connection import init_db
from sources.google_drive import (
    GoogleDriveSource,
    build_drive_uri,
    get_drive_credentials,
    parse_drive_uri,
    print_bandwidth_summary,
    record_bandwidth,
    retry_with_backoff,
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

