//! Library-metadata round-trip: rating / flag / color label persist and read
//! back via the public API. Inserts a photo row directly (no RAW file needed).

use rusqlite::{params, Connection};

fn seed_photo(db: &std::path::Path) -> i64 {
    let conn = Connection::open(db).unwrap();
    conn.execute("INSERT INTO folder (path) VALUES ('/tmp/photos')", []).unwrap();
    let folder_id = conn.last_insert_rowid();
    conn.execute(
        "INSERT INTO photo (folder_id, filename, width, height) VALUES (?1, 'a.arw', 6000, 4000)",
        params![folder_id],
    )
    .unwrap();
    conn.last_insert_rowid()
}

#[test]
fn rating_flag_label_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("cat.db");

    // Create + initialize the schema, then seed one photo.
    let cat = catalog::Catalog::open(&db).unwrap();
    let id = seed_photo(&db);

    cat.set_rating(id, 4).unwrap();
    cat.set_flag(id, 1).unwrap();
    cat.set_color_label(id, 3).unwrap();

    let row = cat.list_photos().unwrap().into_iter().find(|p| p.id == id).unwrap();
    assert_eq!(row.rating, 4);
    assert_eq!(row.flag, 1);
    assert_eq!(row.color_label, 3);

    // Clamping: out-of-range values are coerced into the valid domain.
    cat.set_rating(id, 99).unwrap();
    cat.set_flag(id, -7).unwrap();
    cat.set_color_label(id, 42).unwrap();
    let row = cat.list_photos().unwrap().into_iter().find(|p| p.id == id).unwrap();
    assert_eq!(row.rating, 5);
    assert_eq!(row.flag, -1);
    assert_eq!(row.color_label, 5);
}

/// Seed a photo plus a master develop with an initial seq-0 'Import' version.
fn seed_developed_photo(db: &std::path::Path) -> i64 {
    let conn = Connection::open(db).unwrap();
    conn.execute("INSERT INTO folder (path) VALUES ('/tmp/p')", []).unwrap();
    let folder = conn.last_insert_rowid();
    conn.execute(
        "INSERT INTO photo (folder_id, filename, width, height) VALUES (?1,'h.arw',100,100)",
        params![folder],
    )
    .unwrap();
    let photo = conn.last_insert_rowid();
    conn.execute(
        "INSERT INTO develop (photo_id, name, is_master) VALUES (?1,'Master',1)",
        params![photo],
    )
    .unwrap();
    let develop = conn.last_insert_rowid();
    conn.execute(
        "INSERT INTO recipe_version (develop_id, seq, label, recipe) VALUES (?1,0,'Import','{}')",
        params![develop],
    )
    .unwrap();
    let version = conn.last_insert_rowid();
    conn.execute(
        "INSERT INTO develop_head (develop_id, version_id) VALUES (?1,?2)",
        params![develop, version],
    )
    .unwrap();
    photo
}

#[test]
fn history_undo_redo() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("cat.db");
    let mut cat = catalog::Catalog::open(&db).unwrap();
    let id = seed_developed_photo(&db);

    let mut r1 = recipe::Recipe::default();
    r1.globals.tone.exposure_ev = 1.0;
    cat.save_master_recipe(id, &r1, "Exposure").unwrap();
    let mut r2 = recipe::Recipe::default();
    r2.globals.tone.exposure_ev = 2.0;
    cat.save_master_recipe(id, &r2, "More exposure").unwrap();

    // Three states: Import(0), Exposure(1, =1.0), More exposure(2, =2.0).
    assert_eq!(cat.history(id).unwrap().len(), 3);
    assert_eq!(cat.master_recipe(id).unwrap().globals.tone.exposure_ev, 2.0);

    assert!(cat.undo(id).unwrap());
    assert_eq!(cat.master_recipe(id).unwrap().globals.tone.exposure_ev, 1.0);
    assert!(cat.undo(id).unwrap());
    assert_eq!(cat.master_recipe(id).unwrap().globals.tone.exposure_ev, 0.0); // Import
    assert!(!cat.undo(id).unwrap()); // already oldest

    assert!(cat.redo(id).unwrap());
    assert_eq!(cat.master_recipe(id).unwrap().globals.tone.exposure_ev, 1.0);
}

#[test]
fn presets_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("cat.db");
    let cat = catalog::Catalog::open(&db).unwrap();

    let mut r = recipe::Recipe::default();
    r.globals.tone.exposure_ev = 1.5;
    r.globals.effects.grain_amount = 30.0;
    cat.save_preset("Punchy", &r).unwrap();

    // Re-saving under the same name replaces, not duplicates.
    cat.save_preset("Punchy", &r).unwrap();
    let presets = cat.list_presets().unwrap();
    assert_eq!(presets.len(), 1);
    assert_eq!(presets[0].1, "Punchy");

    let back = cat.preset_recipe(presets[0].0).unwrap();
    assert_eq!(back.globals.tone.exposure_ev, 1.5);
    assert_eq!(back.globals.effects.grain_amount, 30.0);
}
