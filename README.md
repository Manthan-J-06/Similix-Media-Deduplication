# Argus — Image Deduplication System

Local-first, free, open-source tool to detect exact and near-duplicate images,
cluster them, score confidence, recommend a keeper, and safely delete the rest.

> See [PROJECT.md](PROJECT.md) for the full specification.

---

## Requirements

- Python **3.11 or later**
- macOS (Apple Silicon M-series recommended; also works on Intel)

---

## Setup

### 1. Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install runtime dependencies

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

> **Stage 0–1 only** — this installs `Pillow`, `pillow-heif`, `imagehash`, `pytest`, and `pytest-cov`.
> Stage 2+ dependencies (`open-clip-torch`, `faiss-cpu`, `scikit-image`) will be added when those stages are implemented.

### 3. Verify the installation

```bash
python -c "import PIL; import pillow_heif; import imagehash; print('OK')"
```

Expected output: `OK`

### 4. Run the test suite

```bash
pytest
```

---

## Configuration

All thresholds and weights live in [`config.toml`](config.toml). Edit that file
to tune the pipeline — **never** hardcode values in source files.

Key settings:

| Key | Default | Meaning |
|-----|---------|---------|
| `general.thumbnail_size` | `256` | Thumbnail edge length (px) |
| `general.batch_size` | `500` | Images processed per batch |
| `stage1.hamming_threshold` | `10` | Max Hamming distance for pHash near-match |
| `stage2.cosine_threshold` | `0.92` | Min CLIP cosine similarity (Stage 2) |
| `stage3.ssim_threshold` | `0.85` | Min SSIM for visual equivalence (Stage 3) |

---

## Project Layout

```
argus/
├── core/        # Pipeline stages, scoring, clustering
├── sources/     # ImageSource implementations (local folder, …)
├── db/          # SQLite schema, migrations, repository classes
├── agent/       # AI orchestration layer
├── api/         # FastAPI routes
├── ui/          # Web frontend (planned)
├── bench/       # Benchmarks
├── tests/       # Unit & integration tests
├── config.toml  # ← all thresholds here
└── PROJECT.md   # Full specification
```

---

## Deletion Safety

Deletion is **always reversible**:
- Files are moved to macOS Trash, never permanently deleted.
- Every action is recorded in the SQLite `actions` ledger.
- Use `--dry-run` to preview any operation before executing it.

---

## Roadmap

| Stage | Description | Status |
|-------|-------------|--------|
| 0 | File scan + MD5 exact-match | 🔲 Planned |
| 1 | Perceptual hash (pHash) near-match | 🔲 Planned |
| 2 | CLIP embedding + FAISS ANN | 🔲 Planned |
| 3 | SSIM / quality scoring | 🔲 Planned |
| 4 | Full-res keeper selection | 🔲 Planned |
| — | FastAPI + Web UI | 🔲 Planned |

---

## License

MIT
