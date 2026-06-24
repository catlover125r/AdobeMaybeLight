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
