//! SQLite catalog: open/create the DB, import folders of RAWs, and read/write
//! parametric recipes. Non-destructive — only references files + stores recipes.

use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, OptionalExtension};

pub const RAW_EXTS: &[&str] = &[
    "arw", "cr2", "cr3", "nef", "dng", "raf", "rw2", "orf", "pef", "srw", "raw",
];

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error(transparent)]
    Db(#[from] rusqlite::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Raw(#[from] raw_decode::RawError),
}

pub struct Catalog {
    conn: Connection,
}

/// One catalog photo, denormalized for the library grid.
#[derive(Debug, Clone)]
pub struct PhotoRow {
    pub id: i64,
    pub path: PathBuf,
    pub filename: String,
    pub camera: Option<String>,
    pub iso: Option<i64>,
    pub aperture: Option<f64>,
    pub width: i64,
    pub height: i64,
    pub rating: i64,      // 0..5 stars
    pub flag: i64,        // -1 reject, 0 none, +1 pick
    pub color_label: i64, // 0 none, 1..5 = red/yellow/green/blue/purple
}

#[derive(Debug, Default)]
pub struct ImportStats {
    pub scanned: usize,
    pub imported: usize,
    pub skipped: usize,
    pub failed: usize,
}

impl Catalog {
    /// Open (creating + initializing if empty) a catalog DB.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, CatalogError> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let initialized: bool = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='photo'",
                [],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        if !initialized {
            conn.execute_batch(include_str!("../../../catalog/schema.sql"))?;
        }
        Ok(Self { conn })
    }

    /// Recursively import every RAW under `dir`. Idempotent: re-importing an
    /// already-cataloged file is a skip, not a duplicate.
    pub fn import_folder<P: AsRef<Path>>(&mut self, dir: P) -> Result<ImportStats, CatalogError> {
        let mut stats = ImportStats::default();
        let tx = self.conn.transaction()?;
        for path in walk_raw(dir.as_ref()) {
            stats.scanned += 1;
            match import_one(&tx, &path) {
                Ok(true) => stats.imported += 1,
                Ok(false) => stats.skipped += 1,
                Err(_) => stats.failed += 1,
            }
        }
        tx.commit()?;
        Ok(stats)
    }

    pub fn photo_count(&self) -> Result<i64, CatalogError> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM photo", [], |r| r.get(0))?)
    }

    /// The active recipe for a photo's master develop (defaults if none).
    pub fn master_recipe(&self, photo_id: i64) -> Result<recipe::Recipe, CatalogError> {
        let json: Option<String> = self
            .conn
            .query_row(
                "SELECT rv.recipe FROM develop d
                 JOIN develop_head h ON h.develop_id = d.id
                 JOIN recipe_version rv ON rv.id = h.version_id
                 WHERE d.photo_id = ?1 AND d.is_master = 1",
                params![photo_id],
                |r| r.get(0),
            )
            .optional()?;
        Ok(match json {
            Some(j) => recipe::Recipe::from_json(&j)?,
            None => recipe::Recipe::default(),
        })
    }

    /// Append a new recipe version to a photo's master develop and move HEAD.
    pub fn save_master_recipe(
        &mut self,
        photo_id: i64,
        recipe: &recipe::Recipe,
        label: &str,
    ) -> Result<(), CatalogError> {
        let tx = self.conn.transaction()?;
        let develop_id: i64 = tx.query_row(
            "SELECT id FROM develop WHERE photo_id=?1 AND is_master=1",
            params![photo_id],
            |r| r.get(0),
        )?;
        let next_seq: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq)+1, 0) FROM recipe_version WHERE develop_id=?1",
            params![develop_id],
            |r| r.get(0),
        )?;
        tx.execute(
            "INSERT INTO recipe_version (develop_id, seq, label, recipe) VALUES (?1,?2,?3,?4)",
            params![develop_id, next_seq, label, recipe.to_json()],
        )?;
        let version_id = tx.last_insert_rowid();
        tx.execute(
            "INSERT INTO develop_head (develop_id, version_id) VALUES (?1,?2)
             ON CONFLICT(develop_id) DO UPDATE SET version_id=excluded.version_id",
            params![develop_id, version_id],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// All photos in the catalog, newest import first, with enough metadata to
    /// render a library grid and reopen the file for develop.
    pub fn list_photos(&self) -> Result<Vec<PhotoRow>, CatalogError> {
        let mut stmt = self.conn.prepare(
            "SELECT p.id, f.path || '/' || p.filename, p.filename,
                    p.camera_model, p.iso, p.aperture, p.width, p.height,
                    p.rating, p.flag, p.color_label
             FROM photo p JOIN folder f ON f.id = p.folder_id
             ORDER BY p.id DESC",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok(PhotoRow {
                    id: r.get(0)?,
                    path: PathBuf::from(r.get::<_, String>(1)?),
                    filename: r.get(2)?,
                    camera: r.get(3)?,
                    iso: r.get(4)?,
                    aperture: r.get(5)?,
                    width: r.get(6)?,
                    height: r.get(7)?,
                    rating: r.get(8)?,
                    flag: r.get(9)?,
                    color_label: r.get(10)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Set a photo's star rating (clamped to 0..5).
    pub fn set_rating(&self, photo_id: i64, rating: i64) -> Result<(), CatalogError> {
        self.conn.execute(
            "UPDATE photo SET rating=?2 WHERE id=?1",
            params![photo_id, rating.clamp(0, 5)],
        )?;
        Ok(())
    }

    /// Set a photo's pick/reject flag (-1 reject, 0 none, +1 pick).
    pub fn set_flag(&self, photo_id: i64, flag: i64) -> Result<(), CatalogError> {
        self.conn.execute(
            "UPDATE photo SET flag=?2 WHERE id=?1",
            params![photo_id, flag.clamp(-1, 1)],
        )?;
        Ok(())
    }

    /// Set a photo's color label (0 none, 1..5 = red/yellow/green/blue/purple).
    pub fn set_color_label(&self, photo_id: i64, label: i64) -> Result<(), CatalogError> {
        self.conn.execute(
            "UPDATE photo SET color_label=?2 WHERE id=?1",
            params![photo_id, label.clamp(0, 5)],
        )?;
        Ok(())
    }

    /// Save (or replace by name) a develop preset.
    pub fn save_preset(&self, name: &str, recipe: &recipe::Recipe) -> Result<(), CatalogError> {
        self.conn.execute("DELETE FROM preset WHERE name=?1", params![name])?;
        self.conn.execute(
            "INSERT INTO preset (name, recipe) VALUES (?1, ?2)",
            params![name, recipe.to_json()],
        )?;
        Ok(())
    }

    /// All presets as (id, name), alphabetical.
    pub fn list_presets(&self) -> Result<Vec<(i64, String)>, CatalogError> {
        let mut stmt = self.conn.prepare("SELECT id, name FROM preset ORDER BY name")?;
        let rows = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The recipe stored in a preset.
    pub fn preset_recipe(&self, preset_id: i64) -> Result<recipe::Recipe, CatalogError> {
        let json: String =
            self.conn.query_row("SELECT recipe FROM preset WHERE id=?1", params![preset_id], |r| r.get(0))?;
        Ok(recipe::Recipe::from_json(&json)?)
    }

    /// First photo id (handy for the spike CLI: "develop newest import").
    pub fn first_photo(&self) -> Result<Option<(i64, PathBuf)>, CatalogError> {
        Ok(self
            .conn
            .query_row(
                "SELECT p.id, f.path || '/' || p.filename
                 FROM photo p JOIN folder f ON f.id = p.folder_id
                 ORDER BY p.id LIMIT 1",
                [],
                |r| Ok((r.get::<_, i64>(0)?, PathBuf::from(r.get::<_, String>(1)?))),
            )
            .optional()?)
    }
}

fn is_raw(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| RAW_EXTS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Returns Ok(true) if a new photo row was created, Ok(false) if it already
/// existed (skip).
fn import_one(tx: &Connection, path: &Path) -> Result<bool, CatalogError> {
    let parent = path.parent().unwrap_or(Path::new("/"));
    let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
    let folder_path = parent.to_string_lossy().to_string();

    // Folder (upsert).
    tx.execute(
        "INSERT INTO folder (path) VALUES (?1) ON CONFLICT(path) DO NOTHING",
        params![folder_path],
    )?;
    let folder_id: i64 =
        tx.query_row("SELECT id FROM folder WHERE path=?1", params![folder_path], |r| r.get(0))?;

    // Already imported?
    let existing: Option<i64> = tx
        .query_row(
            "SELECT id FROM photo WHERE folder_id=?1 AND filename=?2",
            params![folder_id, filename],
            |r| r.get(0),
        )
        .optional()?;
    if existing.is_some() {
        return Ok(false);
    }

    let meta = raw_decode::probe(path)?;
    let fs_meta = std::fs::metadata(path).ok();
    let file_size = fs_meta.as_ref().map(|m| m.len() as i64);
    let mtime = fs_meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64);

    tx.execute(
        "INSERT INTO photo
           (folder_id, filename, file_size, mtime, width, height, orientation,
            capture_time, camera_make, camera_model, lens_model, iso, shutter,
            aperture, focal_length)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
        params![
            folder_id,
            filename,
            file_size,
            mtime,
            meta.width,
            meta.height,
            meta.flip,
            (meta.timestamp != 0).then_some(meta.timestamp),
            meta.make,
            meta.model,
            meta.lens,
            (meta.iso > 0.0).then_some(meta.iso as i64),
            (meta.shutter > 0.0).then_some(meta.shutter),
            (meta.aperture > 0.0).then_some(meta.aperture),
            (meta.focal > 0.0).then_some(meta.focal),
        ],
    )?;
    let photo_id = tx.last_insert_rowid();

    // Master develop + default recipe (seq 0) + HEAD.
    tx.execute(
        "INSERT INTO develop (photo_id, name, is_master) VALUES (?1, 'Master', 1)",
        params![photo_id],
    )?;
    let develop_id = tx.last_insert_rowid();
    tx.execute(
        "INSERT INTO recipe_version (develop_id, seq, label, recipe) VALUES (?1, 0, 'Import', ?2)",
        params![develop_id, recipe::Recipe::default().to_json()],
    )?;
    let version_id = tx.last_insert_rowid();
    tx.execute(
        "INSERT INTO develop_head (develop_id, version_id) VALUES (?1, ?2)",
        params![develop_id, version_id],
    )?;
    Ok(true)
}

/// Collect RAW file paths under a directory (recursive).
fn walk_raw(dir: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|e| e.into_path())
        .filter(|p| p.is_file() && is_raw(p))
        .collect()
}
