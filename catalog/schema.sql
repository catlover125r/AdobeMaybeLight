-- AdobeMaybeLight catalog schema (Task 3)
-- SQLite. Non-destructive: edits are parametric JSON recipes, never pixels.
-- Design tenets:
--   * The catalog references files; it never owns originals.
--   * A photo's current look = ordered stack of recipe versions (history).
--   * Virtual copies are first-class: many "develops" per one negative.
--   * Heavy/derived data (thumbnails, embeddings) live in side tables/blobs.

PRAGMA journal_mode = WAL;       -- concurrent readers + one writer
PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;          -- bump on every migration

-- ─────────────────────────────────────────────────────────────────────────
-- Files on disk (the immutable negative)
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE folder (
    id          INTEGER PRIMARY KEY,
    parent_id   INTEGER REFERENCES folder(id) ON DELETE CASCADE,
    path        TEXT NOT NULL UNIQUE,        -- absolute, normalized
    volume_uuid TEXT,                         -- for offline/relinking
    created_at  INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE photo (
    id            INTEGER PRIMARY KEY,
    folder_id     INTEGER NOT NULL REFERENCES folder(id) ON DELETE CASCADE,
    filename      TEXT NOT NULL,
    content_hash  BLOB,                       -- xxh3 of raw bytes; relink/dedupe
    file_size     INTEGER,
    mtime         INTEGER,
    -- decoded intrinsics (cached from LibRaw/Exiv2 at import)
    width         INTEGER,
    height        INTEGER,
    orientation   INTEGER DEFAULT 1,          -- EXIF 1..8
    capture_time  INTEGER,                    -- unix seconds, from EXIF
    camera_make   TEXT,
    camera_model  TEXT,
    lens_model    TEXT,
    iso           INTEGER,
    shutter       REAL,
    aperture      REAL,
    focal_length  REAL,
    gps_lat       REAL,
    gps_lon       REAL,
    rating        INTEGER NOT NULL DEFAULT 0 CHECK (rating BETWEEN 0 AND 5),
    flag          INTEGER NOT NULL DEFAULT 0 CHECK (flag IN (-1,0,1)), -- reject/none/pick
    color_label   INTEGER NOT NULL DEFAULT 0, -- 0=none,1..5 = red/yellow/green/blue/purple
    imported_at   INTEGER NOT NULL DEFAULT (unixepoch()),
    UNIQUE (folder_id, filename)
);
CREATE INDEX ix_photo_capture ON photo(capture_time);
CREATE INDEX ix_photo_hash    ON photo(content_hash);
CREATE INDEX ix_photo_model   ON photo(camera_model);

-- ─────────────────────────────────────────────────────────────────────────
-- Develops: virtual copies + non-destructive history
-- One photo has >=1 develop ("Copy 1" is the default master).
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE develop (
    id          INTEGER PRIMARY KEY,
    photo_id    INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    name        TEXT NOT NULL DEFAULT 'Master',  -- "Master", "B&W", "Copy 2"...
    is_master   INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL DEFAULT (unixepoch())
);
CREATE INDEX ix_develop_photo ON develop(photo_id);

-- Every edit produces a new immutable recipe row (history). The develop's
-- current look is the row with the highest seq (or a pinned snapshot).
CREATE TABLE recipe_version (
    id          INTEGER PRIMARY KEY,
    develop_id  INTEGER NOT NULL REFERENCES develop(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,            -- monotonically increasing per develop
    label       TEXT,                         -- "Import", "Crop", "AI sky mask"...
    recipe      TEXT NOT NULL,                -- JSON; see docs/recipe-format.md
    is_snapshot INTEGER NOT NULL DEFAULT 0,   -- user-pinned named state
    created_at  INTEGER NOT NULL DEFAULT (unixepoch()),
    UNIQUE (develop_id, seq)
);
CREATE INDEX ix_recipe_develop ON recipe_version(develop_id, seq);

-- Fast pointer to the active version (avoids MAX(seq) scans in the loupe).
CREATE TABLE develop_head (
    develop_id  INTEGER PRIMARY KEY REFERENCES develop(id) ON DELETE CASCADE,
    version_id  INTEGER NOT NULL REFERENCES recipe_version(id) ON DELETE CASCADE
);

-- Reusable develop presets (a recipe fragment applied to many photos).
CREATE TABLE preset (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    group_name TEXT,
    recipe    TEXT NOT NULL                   -- partial recipe JSON
);

-- ─────────────────────────────────────────────────────────────────────────
-- Keywords / collections / stacks
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE keyword (
    id        INTEGER PRIMARY KEY,
    parent_id INTEGER REFERENCES keyword(id) ON DELETE CASCADE,  -- hierarchy
    name      TEXT NOT NULL,
    synonyms  TEXT,                            -- JSON array
    UNIQUE (parent_id, name)
);
CREATE TABLE photo_keyword (
    photo_id   INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    keyword_id INTEGER NOT NULL REFERENCES keyword(id) ON DELETE CASCADE,
    source     TEXT NOT NULL DEFAULT 'user',   -- 'user' | 'ai'
    confidence REAL,                            -- for ai
    PRIMARY KEY (photo_id, keyword_id)
);

CREATE TABLE collection (
    id        INTEGER PRIMARY KEY,
    parent_id INTEGER REFERENCES collection(id) ON DELETE CASCADE,
    name      TEXT NOT NULL,
    kind      TEXT NOT NULL DEFAULT 'static',  -- 'static' | 'smart'
    rule      TEXT                              -- JSON query for smart collections
);
CREATE TABLE collection_photo (
    collection_id INTEGER NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
    photo_id      INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    position      INTEGER,                      -- manual ordering
    PRIMARY KEY (collection_id, photo_id)
);

CREATE TABLE stack (
    id        INTEGER PRIMARY KEY,
    folder_id INTEGER REFERENCES folder(id) ON DELETE CASCADE
);
CREATE TABLE stack_member (
    stack_id  INTEGER NOT NULL REFERENCES stack(id) ON DELETE CASCADE,
    photo_id  INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    position  INTEGER NOT NULL,
    is_top    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (stack_id, photo_id)
);

-- ─────────────────────────────────────────────────────────────────────────
-- People (InsightFace) and AI artifacts
-- ─────────────────────────────────────────────────────────────────────────
CREATE TABLE person (
    id        INTEGER PRIMARY KEY,
    name      TEXT,
    cluster   INTEGER                          -- unconfirmed cluster id
);
CREATE TABLE face (
    id          INTEGER PRIMARY KEY,
    photo_id    INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    person_id   INTEGER REFERENCES person(id) ON DELETE SET NULL,
    bbox        TEXT NOT NULL,                  -- JSON [x,y,w,h] normalized
    embedding   BLOB NOT NULL,                  -- 512-d ArcFace float32
    confirmed   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX ix_face_photo  ON face(photo_id);
CREATE INDEX ix_face_person ON face(person_id);

-- CLIP/RAM++ embedding for natural-language search (cosine in app or sqlite-vec).
CREATE TABLE photo_embedding (
    photo_id  INTEGER PRIMARY KEY REFERENCES photo(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,                    -- e.g. 'clip-vit-b32'
    vector    BLOB NOT NULL                     -- float16/32 packed
);

-- Cached thumbnails/proxies (blob or path to sidecar cache dir).
CREATE TABLE thumbnail (
    photo_id  INTEGER NOT NULL REFERENCES photo(id) ON DELETE CASCADE,
    size      INTEGER NOT NULL,                 -- longest edge px (e.g. 256,1024)
    format    TEXT NOT NULL DEFAULT 'jpeg',
    data      BLOB,
    PRIMARY KEY (photo_id, size)
);

-- ─────────────────────────────────────────────────────────────────────────
-- Full-text search over filename + keywords + caption
-- ─────────────────────────────────────────────────────────────────────────
CREATE VIRTUAL TABLE photo_fts USING fts5(
    filename, keywords, caption,
    content=''                                  -- external-content/contentless
);
