"""
db/schema.py — SQL DDL for the Argus database.

Stores image metadata, hashes, pipeline statuses, duplicate clusters,
and the deletion ledger.
"""

# ---------------------------------------------------------------------------
# Table: images
#   status lifecycle: 'discovered' -> 'hashed' -> 'clustered'
# ---------------------------------------------------------------------------
CREATE_IMAGES = """
CREATE TABLE IF NOT EXISTS images (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path     TEXT    NOT NULL UNIQUE,
    file_size     INTEGER,                       -- bytes
    modified_time REAL,                          -- last modified POSIX timestamp
    width         INTEGER,                       -- image width in px
    height        INTEGER,                       -- image height in px
    format        TEXT,                          -- image format (JPEG, PNG, etc.)
    md5           TEXT,                          -- hex digest (Stage 0)
    phash         TEXT,                          -- perceptual hash (Stage 1)
    clip_vector       BLOB,                          -- CLIP embedding (Stage 2)
    clip_vector_90    BLOB,                          -- CLIP embedding 90 deg rotated
    clip_vector_180   BLOB,                          -- CLIP embedding 180 deg rotated
    clip_vector_270   BLOB,                          -- CLIP embedding 270 deg rotated
    status        TEXT    NOT NULL DEFAULT 'discovered', -- 'discovered' | 'hashed' | 'embedded' | 'clustered'
    cluster_id    INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    is_keeper     INTEGER NOT NULL DEFAULT 0,    -- legacy anchor column / applied keeper
    is_cluster_anchor INTEGER NOT NULL DEFAULT 0, -- 1 = seed/anchor at clustering time (not final keeper)
    is_low_confidence INTEGER NOT NULL DEFAULT 0, -- 1 = inconclusive keypoints, exclude from auto-delete
    confidence    REAL,                          -- duplicate-confidence 0-100
    link_signal   TEXT,                          -- 'md5' | 'phash' | 'clip'
    is_raw_pair   INTEGER NOT NULL DEFAULT 0,    -- 1 = RAW/JPEG sibling, skip dedup
    ssim          REAL,                          -- SSIM score vs cluster reference (grayscale)
    ssim_color    REAL,                          -- SSIM score vs cluster reference (color)
    sharpness_ratio REAL,                        -- Laplacian variance ratio (member / reference)
    pixel_ratio   REAL,                          -- pixel count ratio (member / reference)
    bytes_ratio   REAL,                          -- file size ratio (member / reference)
    jpeg_quality_est INTEGER,                    -- estimated JPEG quality (1-100) from quantization tables
    source_likelihood REAL,                      -- mean SSIM against same-resolution lossy peers
    quality_status TEXT,                         -- 'reference' | 'aligned' | 'unalignable'
    is_recommended_keeper INTEGER NOT NULL DEFAULT 0, -- recommended keeper flag (glow flag)
    delete_safe   INTEGER NOT NULL DEFAULT 0,    -- 1 = eligible for deletion
    review_flag   INTEGER NOT NULL DEFAULT 0,    -- 1 = requires manual review
    quality_retained_pct REAL,                   -- % quality retained vs reference
    bytes_saved_if_deleted INTEGER DEFAULT 0,    -- bytes reclaimed if this member is deleted
    reason        TEXT,                          -- explanation for keeper / delete / review verdict
    match_pct     REAL,                          -- color SSIM match % vs reference (100.0 for ref/exact)
    resolution_pct REAL,                         -- pixel count % vs reference (100.0 for ref/exact)
    category_top1 TEXT,                          -- top-1 predicted category or 'uncertain'
    category_top1_prob REAL,                     -- probability of top-1 category (0.0 - 1.0)
    category_top3 TEXT,                          -- JSON array of [{"label": ..., "prob": ...}, ...]
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

# ---------------------------------------------------------------------------
# Table: clusters
#   representative_id points to the recommended keeper candidate.
# ---------------------------------------------------------------------------
CREATE_CLUSTERS = """
CREATE TABLE IF NOT EXISTS clusters (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    representative_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
    stage_detected    INTEGER NOT NULL,           -- pipeline stage that found this cluster (0, 1, 2...)
    member_count      INTEGER NOT NULL DEFAULT 0,
    reference_uncertain INTEGER NOT NULL DEFAULT 0, -- 1 = reference uncertain (SSIM tie <0.002 or EXIF/mtime conflict)
    category          TEXT,                          -- cluster-level category (from keeper or majority vote)
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

# ---------------------------------------------------------------------------
# Table: pair_transforms (cached geometric alignment matrices)
# ---------------------------------------------------------------------------
CREATE_PAIR_TRANSFORMS = """
CREATE TABLE IF NOT EXISTS pair_transforms (
    source_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    target_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    transform_matrix BLOB NOT NULL,
    model            TEXT NOT NULL DEFAULT 'affine',
    inlier_count     INTEGER NOT NULL DEFAULT 0,
    inlier_ratio     REAL NOT NULL DEFAULT 0.0,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    PRIMARY KEY (source_id, target_id)
);
"""

# ---------------------------------------------------------------------------
# Table: actions (deletion ledger)
# ---------------------------------------------------------------------------
CREATE_ACTIONS = """
CREATE TABLE IF NOT EXISTS actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    INTEGER REFERENCES images(id) ON DELETE SET NULL,
    operation   TEXT NOT NULL,                   -- 'trash' | 'restore' | 'skip'
    source_path TEXT NOT NULL,
    dest_path   TEXT,                            -- trash location after move
    status      TEXT NOT NULL DEFAULT 'pending', -- 'done' | 'failed'
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

# ---------------------------------------------------------------------------
# Tables: ledger_batches & deletion_journal (core/ledger.py)
# ---------------------------------------------------------------------------
CREATE_LEDGER_BATCHES = """
CREATE TABLE IF NOT EXISTS ledger_batches (
    batch_id     TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    executed_at  TEXT,
    mode         TEXT NOT NULL DEFAULT 'balanced',
    status       TEXT NOT NULL DEFAULT 'planned', -- 'planned' | 'executed' | 'undone' | 'purged'
    file_count   INTEGER NOT NULL DEFAULT 0,
    bytes_freed  INTEGER NOT NULL DEFAULT 0,
    trash_dir    TEXT NOT NULL
);
"""

CREATE_DELETION_JOURNAL = """
CREATE TABLE IF NOT EXISTS deletion_journal (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id      TEXT NOT NULL REFERENCES ledger_batches(batch_id) ON DELETE CASCADE,
    timestamp     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    original_path TEXT NOT NULL,
    trash_path    TEXT,
    md5           TEXT,
    size          INTEGER,
    cluster_id    INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    keeper_id     INTEGER REFERENCES images(id) ON DELETE SET NULL,
    mode          TEXT,
    reason        TEXT,
    status        TEXT NOT NULL DEFAULT 'planned', -- 'planned' | 'moved' | 'restored' | 'failed' | 'refused' | 'purged'
    is_manual     INTEGER NOT NULL DEFAULT 0,
    image_id      INTEGER REFERENCES images(id) ON DELETE SET NULL
);
"""

# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------
CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_images_path      ON images(file_path);
CREATE INDEX IF NOT EXISTS idx_images_mtime     ON images(file_path, modified_time, file_size);
CREATE INDEX IF NOT EXISTS idx_images_md5       ON images(md5);
CREATE INDEX IF NOT EXISTS idx_images_phash     ON images(phash);
CREATE INDEX IF NOT EXISTS idx_images_status    ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_cluster   ON images(cluster_id);
CREATE INDEX IF NOT EXISTS idx_actions_image    ON actions(image_id);
CREATE INDEX IF NOT EXISTS idx_pair_transforms_target ON pair_transforms(target_id);
CREATE INDEX IF NOT EXISTS idx_journal_batch    ON deletion_journal(batch_id);
CREATE INDEX IF NOT EXISTS idx_journal_status   ON deletion_journal(status);
CREATE INDEX IF NOT EXISTS idx_journal_path     ON deletion_journal(original_path);
"""

ALL_DDL = "\n".join([
    CREATE_IMAGES,
    CREATE_CLUSTERS,
    CREATE_PAIR_TRANSFORMS,
    CREATE_ACTIONS,
    CREATE_LEDGER_BATCHES,
    CREATE_DELETION_JOURNAL,
    CREATE_INDEXES,
])
