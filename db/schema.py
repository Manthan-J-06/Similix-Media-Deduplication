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
    is_keeper     INTEGER NOT NULL DEFAULT 0,    -- 1 = recommended keeper
    confidence    REAL,                          -- duplicate-confidence 0-100
    is_raw_pair   INTEGER NOT NULL DEFAULT 0,    -- 1 = RAW/JPEG sibling, skip dedup
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
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
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
"""

ALL_DDL = "\n".join([
    CREATE_IMAGES,
    CREATE_CLUSTERS,
    CREATE_ACTIONS,
    CREATE_INDEXES,
])
