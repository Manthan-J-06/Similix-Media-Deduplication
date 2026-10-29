"""
End-to-End browser test for Argus using Playwright (Chromium headless).
Verifies:
1. Registration & Login
2. Uploading 12 images (4 are copies)
3. Job banner progress to completion WITHOUT page reload
4. Duplicate clusters appearance within 60 seconds
5. Mode switching across conservative, balanced, aggressive
6. Opening cluster and inspecting copy cards
7. Category dropdown population / explanatory message
8. Account deletion
"""

import os
import re
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

from PIL import Image
import pytest

from core.embeddings import is_clip_weights_cached


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _create_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (240, 240)) -> None:
    import numpy as np
    from PIL import ImageDraw

    # Generate repeatable pseudo-random pattern based on color to ensure ample SIFT keypoints
    seed = (color[0] * 10000 + color[1] * 100 + color[2]) % 1000000
    rng = np.random.RandomState(seed)
    noise = (rng.rand(size[1], size[0], 3) * 120).astype(np.uint8)
    base = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    combined = np.clip(base // 2 + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(combined, mode="RGB")
    draw = ImageDraw.Draw(im)
    draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline=(255 - color[0], 255 - color[1], 255 - color[2]), width=4)
    draw.line([0, 0, size[0], size[1]], fill=(255 - color[0], 255 - color[1], 255 - color[2]), width=3)
    im.save(str(path), format="JPEG", quality=95)


@pytest.fixture(scope="function")
def live_server(tmp_path):
    """Start real Argus server in a temporary data directory."""
    port = _get_free_port()
    data_dir = tmp_path / "argus_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ARGUS_DATA_DIR"] = str(data_dir)
    env["HOST"] = "127.0.0.1"
    env["PORT"] = str(port)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "core.cli",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    log_path = tmp_path / "server.log"
    log_file = open(log_path, "w", encoding="utf-8")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20.0
    ready = False

    while time.time() < deadline:
        if proc.poll() is not None:
            log_file.flush()
            server_output = log_path.read_text(encoding="utf-8")
            raise RuntimeError(
                f"Argus server exited prematurely (code {proc.returncode}):\n{server_output}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/login", timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.3)

    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        log_file.flush()
        server_output = log_path.read_text(encoding="utf-8")
        raise TimeoutError(f"Argus server failed to become ready on {base_url} within 20s:\n{server_output}")

    try:
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        log_file.close()


@pytest.mark.skipif(not is_clip_weights_cached(), reason="CLIP weights not cached; skipping e2e browser test")
def test_full_web_flow(live_server, tmp_path):
    from playwright.sync_api import sync_playwright

    base_url = live_server
    test_email = f"e2e_user_{uuid.uuid4().hex[:8]}@example.com"
    test_password = "SecurePassword123!"

    # 1. Prepare 12 images (4 are copies)
    img_dir = tmp_path / "test_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Cluster A: 3 copies (orig + 2 copies)
    p_a0 = img_dir / "img_a0.jpg"
    _create_image(p_a0, color=(220, 50, 50), size=(180, 180))
    p_a1 = img_dir / "img_a1.jpg"
    shutil.copyfile(p_a0, p_a1)
    p_a2 = img_dir / "img_a2.jpg"
    shutil.copyfile(p_a0, p_a2)

    # Cluster B: 3 copies (orig + 2 copies)
    p_b0 = img_dir / "img_b0.jpg"
    _create_image(p_b0, color=(50, 50, 220), size=(200, 200))
    p_b1 = img_dir / "img_b1.jpg"
    shutil.copyfile(p_b0, p_b1)
    p_b2 = img_dir / "img_b2.jpg"
    shutil.copyfile(p_b0, p_b2)

    # 6 distinct standalone images
    other_images = []
    colors = [
        (40, 200, 40),
        (220, 220, 40),
        (200, 40, 200),
        (220, 140, 40),
        (40, 220, 220),
        (100, 100, 100),
    ]
    for idx, col in enumerate(colors):
        p_other = img_dir / f"img_other_{idx}.jpg"
        _create_image(p_other, color=col, size=(160 + idx * 10, 160 + idx * 10))
        other_images.append(p_other)

    all_12_images = [p_a0, p_a1, p_a2, p_b0, p_b1, p_b2] + other_images
    assert len(all_12_images) == 12

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # 2. Register new account
        page.goto(f"{base_url}/login")
        page.wait_for_selector("#tab-register", timeout=10000)
        page.click("#tab-register")

        page.fill("#email", test_email)
        page.fill("#password", test_password)
        page.click("#submit-btn")

        # Wait for redirect to main app "/"
        try:
            page.wait_for_url(f"{base_url}/", timeout=15000)
        except Exception:
            # Fallback in case auto-login didn't redirect automatically
            if page.is_visible("#tab-login.active"):
                page.fill("#email", test_email)
                page.fill("#password", test_password)
                page.click("#submit-btn")
                page.wait_for_url(f"{base_url}/", timeout=15000)

        page.wait_for_selector("#user-email-display", timeout=10000)
        email_text = page.text_content("#user-email-display") or ""
        assert test_email in email_text

        # 3. Upload 12 images
        page.click("#btn-open-sources")
        page.wait_for_selector("#sources-modal", state="visible", timeout=5000)
        page.set_input_files("#file-input", [str(img) for img in all_12_images])
        page.wait_for_selector("#btn-start-upload", state="visible", timeout=5000)
        page.click("#btn-start-upload")

        # 4. Watch banner to completion WITHOUT reloading
        page.wait_for_selector("#job-progress-banner", state="visible", timeout=10000)
        page.wait_for_function(
            "() => document.getElementById('job-percent-text')?.textContent === '100%' || "
            "(document.getElementById('job-status-text')?.textContent || '').includes('Pipeline complete') || "
            "(document.getElementById('job-status-text')?.textContent || '').includes('Pipeline completed')",
            timeout=60000,
        )

        # Assert clusters appear within 60s
        page.wait_for_selector("#cluster-list .cluster-item", timeout=60000)
        clusters = page.query_selector_all("#cluster-list .cluster-item")
        assert len(clusters) >= 1, "Expected duplicate clusters to appear"

        # 5. Switch all three modes
        for mode in ["balanced", "aggressive", "conservative"]:
            btn_id = f"#mode-{mode}"
            page.click(btn_id)
            page.wait_for_selector(f"{btn_id}.active", timeout=5000)
            page.wait_for_selector("#mode-spinner", state="hidden", timeout=5000)
            page.wait_for_selector("#cluster-list .cluster-item", timeout=5000)
            cur_clusters = page.query_selector_all("#cluster-list .cluster-item")
            assert len(cur_clusters) >= 1

        # 6. Open a cluster
        first_cluster = page.query_selector("#cluster-list .cluster-item")
        assert first_cluster is not None
        first_cluster.click()

        page.wait_for_selector("#detail-content", state="visible", timeout=5000)
        title = page.text_content("#detail-cluster-title") or ""
        assert "Cluster #" in title

        members = page.query_selector_all("#member-grid .member-card")
        assert len(members) >= 2, f"Expected cluster to contain >=2 copies, got {len(members)}"

        # 7. Assert categories dropdown is populated or shows explanatory message
        page.wait_for_selector("#category-filter", timeout=5000)
        cat_options = page.eval_on_selector_all("#category-filter option", "opts => opts.map(o => o.textContent)")
        assert len(cat_options) >= 1
        valid_cat_state = any(
            "All Categories" in opt or "Categories are computed after processing" in opt
            for opt in cat_options
        )
        assert valid_cat_state, f"Unexpected category dropdown options: {cat_options}"

        # 8. Delete the account
        page.click("#btn-account-menu")
        page.wait_for_selector("#btn-open-delete-account", state="visible", timeout=5000)
        page.click("#btn-open-delete-account")

        page.wait_for_selector("#delete-account-modal", state="visible", timeout=5000)
        page.fill("#delete-account-password", test_password)
        page.check("#delete-account-confirm-checkbox")

        page.wait_for_selector("#btn-confirm-delete-account:not([disabled])", timeout=5000)
        page.click("#btn-confirm-delete-account")

        page.wait_for_url(re.compile(r".*/login.*"), timeout=15000)

        context.close()
        browser.close()


@pytest.mark.skipif(not is_clip_weights_cached(), reason="CLIP weights not cached; skipping e2e browser test")
def test_move_to_trash_and_undo(live_server, tmp_path):
    from playwright.sync_api import sync_playwright

    base_url = live_server
    test_email = f"trash_user_{uuid.uuid4().hex[:8]}@example.com"
    test_password = "SecurePassword123!"

    # 1. Prepare images: original + identical copy + distinct background image
    img_dir = tmp_path / "trash_test_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    p_orig = img_dir / "item_orig.jpg"
    _create_image(p_orig, color=(220, 40, 40), size=(180, 180))

    p_copy = img_dir / "item_copy.jpg"
    shutil.copyfile(p_orig, p_copy)

    p_other = img_dir / "item_other.jpg"
    _create_image(p_other, color=(40, 40, 220), size=(180, 180))

    test_images = [p_orig, p_copy, p_other]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Fail test if any alert()/confirm()/prompt() popup appears
        dialog_events = []

        def handle_dialog(dialog):
            dialog_events.append(f"{dialog.type}: {dialog.message}")
            dialog.dismiss()

        page.on("dialog", handle_dialog)

        # 2. Register & Login
        page.goto(f"{base_url}/login")
        page.wait_for_selector("#tab-register", timeout=10000)
        page.click("#tab-register")

        page.fill("#email", test_email)
        page.fill("#password", test_password)
        page.click("#submit-btn")

        try:
            page.wait_for_url(f"{base_url}/", timeout=15000)
        except Exception:
            if page.is_visible("#tab-login.active"):
                page.fill("#email", test_email)
                page.fill("#password", test_password)
                page.click("#submit-btn")
                page.wait_for_url(f"{base_url}/", timeout=15000)

        page.wait_for_selector("#user-email-display", timeout=10000)

        # 3. Upload images
        page.click("#btn-open-sources")
        page.wait_for_selector("#sources-modal", state="visible", timeout=5000)
        page.set_input_files("#file-input", [str(img) for img in test_images])
        page.wait_for_selector("#btn-start-upload", state="visible", timeout=5000)
        page.click("#btn-start-upload")

        # 4. Wait for processing to complete
        page.wait_for_selector("#job-progress-banner", state="visible", timeout=10000)
        page.wait_for_function(
            "() => document.getElementById('job-percent-text')?.textContent === '100%' || "
            "(document.getElementById('job-status-text')?.textContent || '').includes('Pipeline complete') || "
            "(document.getElementById('job-status-text')?.textContent || '').includes('Pipeline completed')",
            timeout=60000,
        )

        # 5. Open the duplicate cluster
        page.wait_for_selector("#cluster-list .cluster-item", timeout=60000)
        page.click("#cluster-list .cluster-item")

        page.wait_for_selector("#detail-content", state="visible", timeout=5000)
        page.wait_for_selector("#member-grid .member-card", timeout=5000)
        member_cards = page.query_selector_all("#member-grid .member-card")
        assert len(member_cards) >= 2, f"Expected at least 2 member cards, found {len(member_cards)}"

        # Find the safe copy card (non-keeper member)
        copy_card = page.query_selector(".member-card:not(.is-keeper)")
        assert copy_card is not None, "Expected a non-keeper copy card"

        copy_filename_el = copy_card.query_selector(".card-filename")
        assert copy_filename_el is not None
        copy_disk_path_str = copy_filename_el.get_attribute("title")
        assert copy_disk_path_str, "Expected title attribute with disk file path"
        copy_disk_path = Path(copy_disk_path_str)
        assert copy_disk_path.exists(), f"File should exist on disk before deletion: {copy_disk_path}"

        # 6. Click 'Select all safe copies'
        page.click("#btn-select-safe")
        page.wait_for_timeout(300)

        # 7. Click 'Move to trash'
        page.click("#btn-plan-deletion")
        page.wait_for_selector("#confirm-modal", state="visible", timeout=5000)
        page.wait_for_selector("#btn-confirm-apply:not([disabled])", timeout=5000)

        # 8. Click 'Confirm & Move'
        page.click("#btn-confirm-apply")
        page.wait_for_selector("#apply-success-box", state="visible", timeout=10000)

        # Wait up to 5s for disk file to be moved
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not copy_disk_path.exists():
                break
            time.sleep(0.2)

        # 9. Assert file disappears from disk
        assert not copy_disk_path.exists(), f"File should have been moved from {copy_disk_path}"

        # Assert file disappears from active cluster member cards
        page.wait_for_timeout(500)
        remaining_cards = page.query_selector_all("#member-grid .member-card")
        remaining_paths = [
            c.query_selector(".card-filename").get_attribute("title")
            for c in remaining_cards
            if c.query_selector(".card-filename")
        ]
        assert str(copy_disk_path) not in remaining_paths, f"Deleted file {copy_disk_path} still in cluster view"

        # 10. Click 'Undo This Deletion'
        page.wait_for_selector("#btn-undo-apply", state="visible", timeout=5000)
        page.click("#btn-undo-apply")

        # 11. Assert file returns to disk
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if copy_disk_path.exists():
                break
            time.sleep(0.3)

        assert copy_disk_path.exists(), f"File should be restored on disk after undo: {copy_disk_path}"

        # Assert file returns to cluster in UI
        page.wait_for_selector("#cluster-list .cluster-item", timeout=10000)
        page.click("#cluster-list .cluster-item")
        page.wait_for_selector("#member-grid .member-card", timeout=5000)

        restored_cards = page.query_selector_all("#member-grid .member-card")
        restored_paths = [
            c.query_selector(".card-filename").get_attribute("title")
            for c in restored_cards
            if c.query_selector(".card-filename")
        ]
        assert str(copy_disk_path) in restored_paths, f"Restored file {copy_disk_path} not found in cluster view"

        # Assert no alert() dialogs were triggered
        assert len(dialog_events) == 0, f"Expected zero alert/dialog popups, got: {dialog_events}"

        context.close()
        browser.close()


def test_logo_rendering(live_server):
    from playwright.sync_api import sync_playwright

    base_url = live_server

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for width in [1280, 400]:
            for theme in ["light", "dark"]:
                context = browser.new_context(
                    viewport={"width": width, "height": 800},
                    color_scheme=theme,
                )
                page = context.new_page()
                page.goto(f"{base_url}/login")
                page.wait_for_selector(".brand", timeout=10000)

                # Check eye SVG
                eye = page.locator(".brand svg.argus-eye")
                assert eye.is_visible()
                assert eye.get_attribute("viewBox") == "0 0 40 40"
                path_stroke = page.locator(".brand svg.argus-eye path").get_attribute("stroke")
                assert path_stroke == "#2DD4BF"

                # Check wordmark
                wordmark = page.locator(".brand .argus-wordmark")
                assert wordmark.is_visible()
                assert wordmark.inner_text().strip() == "ARGUS"

                # Check styles
                font_weight = wordmark.evaluate("el => window.getComputedStyle(el).fontWeight")
                assert font_weight in ("700", "bold", "800", "900")

                box = wordmark.bounding_box()
                assert box is not None
                assert box["width"] > 0
                assert box["height"] > 0
                # Ensure wordmark is within the viewport width
                assert box["x"] + box["width"] <= width

                context.close()

        browser.close()

