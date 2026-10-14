"""
db/schema.py — SQL DDL for the Argus database.

All three tables are defined here. Columns for later pipeline stages
are included now so the schema never needs a destructive migration.
New columns added in future stages should be done via ALTER TABLE
scripts in db/migrations.py.
"""

# ---------------------------------------------------------------------------
# Table: images
#   One row per discovered file. status tracks pipeline progress.
#   cluster_id is NULL until the image is assigned to a duplicate cluster.
# ---------------------------------------------------------------------------
CREATE_IMAGES = """
CREATE TABLE IF NOT EXISTS images (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path     TEXT    NOT NULL UNIQUE,
    file_size     INTEGER,                       -- bytes
    md5           TEXT,                          -- hex digest (Stage 0)
    phash         TEXT,                          -- perceptual hash  (Stage 1)
    clip_vector   BLOB,                          -- CLIP embedding   (Stage 2)
    status        TEXT    NOT NULL DEFAULT 'pending',
    cluster_id    INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    is_keeper     INTEGER NOT NULL DEFAULT 0,    -- 1 = recommended keeper
    confidence    REAL,                          -- duplicate-confidence 0-100
    is_raw_pair   INTEGER NOT NULL DEFAULT 0,    -- 1 = RAW/JPEG sibling, skip dedup
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

# ---------------------------------------------------------------------------
# Table: clusters
#   One row per duplicate group. representative_id points to the current
#   best-known keeper candidate (updated as later stages run).
# ---------------------------------------------------------------------------
CREATE_CLUSTERS = """
CREATE TABLE IF NOT EXISTS clusters (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    representative_id INTEGER REFERENCES images(id) ON DELETE SET NULL,
    stage_detected    INTEGER NOT NULL,           -- pipeline stage that found this cluster
    member_count      INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

# ---------------------------------------------------------------------------
# Table: actions  (deletion ledger — every operation is recorded here)
#   Enables full undo of any batch-delete.
# ---------------------------------------------------------------------------
CREATE_ACTIONS = """
CREATE TABLE IF NOT EXISTS actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    INTEGER REFERENCES images(id) ON DELETE SET NULL,
    operation   TEXT NOT NULL,   -- 'trash' | 'restore' | 'skip'
    source_path TEXT NOT NULL,
    dest_path   TEXT,            -- trash location after move
    status      TEXT NOT NULL DEFAULT 'pending',  -- 'done' | 'failed'
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------
CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_images_md5       ON images(md5);
CREATE INDEX IF NOT EXISTS idx_images_status    ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_cluster   ON images(cluster_id);
CREATE INDEX IF NOT EXISTS idx_images_phash     ON images(phash);
CREATE INDEX IF NOT EXISTS idx_actions_image    ON actions(image_id);
"""

# Convenience: run all DDL in a single call
ALL_DDL = "\n".join([
    CREATE_IMAGES,
    CREATE_CLUSTERS,
    CREATE_ACTIONS,
    CREATE_INDEXES,
])
