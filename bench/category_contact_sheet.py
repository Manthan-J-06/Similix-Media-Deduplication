"""
bench/category_contact_sheet.py — Generate visual contact sheet for categorized images.

Features:
- Queries images categorized in SQLite database.
- Renders up to 12 thumbnails per predicted category (including 'uncertain').
- Displays filename, top-1 predicted category, probability, and top-3 breakdown.
- Generates a standalone, self-contained HTML report with embedded base64 thumbnails
  and responsive CSS layout.

Usage:
    python -m bench.category_contact_sheet --db data/category_eval/eval.db --output contact_sheet.html
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

from PIL import Image

from db.connection import init_db


def create_thumbnail_b64(image_path: str | Path, max_size: int = 240) -> str:
    """Generate a base64-encoded JPEG thumbnail for an image file."""
    p = Path(image_path)
    if not p.is_file():
        return ""
    try:
        with Image.open(p) as img:
            img = img.convert("RGB")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def generate_contact_sheet_html(
    conn: sqlite3.Connection,
    output_path: str | Path,
    per_category: int = 12,
    thumbnail_size: int = 240,
) -> Path:
    """
    Generate an HTML contact sheet showing up to `per_category` thumbnails
    for each predicted category in the database.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch all categorized images grouped by category_top1
    rows = conn.execute(
        """
        SELECT id, file_path, category_top1, category_top1_prob, category_top3
        FROM images
        WHERE category_top1 IS NOT NULL
        ORDER BY category_top1 ASC, category_top1_prob DESC, id ASC
        """
    ).fetchall()

    if not rows:
        raise ValueError("No categorized images found in database.")

    # Group rows by category_top1
    categories: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        cat = r["category_top1"] or "uncertain"
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(dict(r))

    # Build HTML content
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Argus — Category Contact Sheet</title>
<style>
  :root {
    --bg-primary: #121418;
    --bg-secondary: #1a1e24;
    --bg-card: #222730;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent: #58a6ff;
    --accent-green: #3fb950;
    --accent-yellow: #d29922;
    --accent-red: #f85149;
    --border: #30363d;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    padding: 24px;
    line-height: 1.5;
  }
  header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 24px;
  }
  h1 { font-size: 24px; font-weight: 600; color: var(--text-primary); }
  .subtitle { color: var(--text-secondary); font-size: 14px; margin-top: 4px; }
  .nav-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 16px 0;
  }
  .nav-pill {
    padding: 6px 12px;
    border-radius: 20px;
    background: var(--bg-card);
    color: var(--text-primary);
    text-decoration: none;
    font-size: 13px;
    border: 1px solid var(--border);
    transition: background 0.15s, border-color 0.15s;
  }
  .nav-pill:hover {
    background: var(--border);
    border-color: var(--accent);
  }
  .category-section {
    margin-bottom: 40px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
  }
  .category-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
  }
  .category-title {
    font-size: 18px;
    font-weight: 600;
    text-transform: capitalize;
  }
  .category-count {
    font-size: 13px;
    color: var(--text-secondary);
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
  }
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: transform 0.15s, border-color 0.15s;
  }
  .card:hover {
    transform: translateY(-2px);
    border-color: var(--accent);
  }
  .img-wrap {
    width: 100%;
    height: 180px;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .img-wrap img {
    max-width: 100%;
    max-height: 100%;
    object-fit: cover;
  }
  .card-body {
    padding: 10px;
    font-size: 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .filename {
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    title: attr(data-full);
  }
  .prob-badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    align-self: flex-start;
  }
  .prob-high { background: rgba(63, 185, 80, 0.2); color: var(--accent-green); }
  .prob-med { background: rgba(210, 153, 34, 0.2); color: var(--accent-yellow); }
  .prob-low { background: rgba(248, 81, 73, 0.2); color: var(--accent-red); }
  .top3-list {
    margin-top: 4px;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.3;
  }
</style>
</head>
<body>
<header>
  <h1>Argus — Category Contact Sheet</h1>
  <div class=\"subtitle\">Visual verification of predicted categories and confidence scores</div>
  <div class=\"nav-pills\">
""")

    sorted_categories = sorted(categories.keys())
    # Place 'uncertain' at the end if present
    if "uncertain" in sorted_categories:
        sorted_categories.remove("uncertain")
        sorted_categories.append("uncertain")

    for cat in sorted_categories:
        count = len(categories[cat])
        html_parts.append(f'    <a class="nav-pill" href="#{cat}">{cat} ({count})</a>\n')

    html_parts.append("""  </div>
</header>
<main>
""")

    for cat in sorted_categories:
        items = categories[cat][:per_category]
        total_in_cat = len(categories[cat])
        shown = len(items)

        html_parts.append(f"""
  <section class="category-section" id="{cat}">
    <div class="category-header">
      <div class="category-title">{cat}</div>
      <div class="category-count">Showing {shown} of {total_in_cat} images</div>
    </div>
    <div class="grid">
""")

        for item in items:
            p = Path(item["file_path"])
            fname = p.name
            prob = float(item["category_top1_prob"] or 0.0)
            b64_img = create_thumbnail_b64(p, max_size=thumbnail_size)

            if prob >= 0.70:
                badge_cls = "prob-high"
            elif prob >= 0.40:
                badge_cls = "prob-med"
            else:
                badge_cls = "prob-low"

            top3_json = item["category_top3"]
            top3_str = ""
            if top3_json:
                try:
                    top3_data = json.loads(top3_json) if isinstance(top3_json, str) else top3_json
                    parts = [f"{entry['label']} {entry['prob']*100:.0f}%" for entry in top3_data[:3]]
                    top3_str = " &bull; ".join(parts)
                except Exception:
                    top3_str = ""

            img_tag = f'<img src="data:image/jpeg;base64,{b64_img}" alt="{fname}" loading="lazy" />' if b64_img else '<div style="color:#666;font-size:11px;">No preview</div>'

            html_parts.append(f"""      <div class="card">
        <div class="img-wrap">{img_tag}</div>
        <div class="card-body">
          <div class="filename" title="{fname}">{fname}</div>
          <div><span class="prob-badge {badge_cls}">{cat} &bull; category confidence: {prob*100:.1f}%</span></div>
          {f'<div class="top3-list">{top3_str}</div>' if top3_str else ''}
        </div>
      </div>
""")

        html_parts.append("""    </div>
  </section>
""")

    html_parts.append("""</main>
</body>
</html>
""")

    output_path.write_text("".join(html_parts), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate category contact sheet HTML")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--output", default="category_contact_sheet.html", help="Path to output HTML")
    parser.add_argument("--per-category", type=int, default=12, help="Number of thumbnails per category (default: 12)")
    parser.add_argument("--thumb-size", type=int, default=240, help="Max thumbnail edge length (px)")

    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        sys.stderr.write(f"Database not found: {db_path}\n")
        sys.exit(1)

    conn = init_db(db_path)
    try:
        out_file = generate_contact_sheet_html(
            conn,
            output_path=args.output,
            per_category=args.per_category,
            thumbnail_size=args.thumb_size,
        )
        print(f"Contact sheet successfully written to: {out_file.resolve()}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
