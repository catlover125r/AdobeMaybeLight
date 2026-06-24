//! AdobeMaybeLight entry point.
//!
//!   aml                                 # open the Library (default catalog)
//!   aml --catalog db                    # open a specific catalog
//!   aml import <DIR> [--catalog db]     # catalog a folder of RAWs (headless)
//!   aml <RAW_FILE> [--recipe r.json]    # open one file straight in Develop
//!   aml <RAW_FILE> --export out.png [--recipe r.json]   # headless export
//!   aml --selftest out.png              # GPU pipeline self-test

mod gui;

use gpu::{DevelopParams, GpuContext, Scene};

fn main() {
    // Ignore the process-serial-number arg Finder passes to GUI apps.
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("-psn_"))
        .collect();

    // --selftest: synthetic linear gradient -> develop -> PNG. Proves the GPU
    // path (upload -> WGSL -> readback -> encode) without a RAW file.
    if args.first().map(String::as_str) == Some("--selftest") {
        let out = args.get(1).cloned().unwrap_or_else(|| "selftest.png".into());
        let (w, h) = (512u32, 256u32);
        let mut samples = vec![0u16; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                samples[i] = ((x as f32 / w as f32) * 65535.0) as u16; // R ramp
                samples[i + 1] = ((y as f32 / h as f32) * 65535.0) as u16; // G ramp
                samples[i + 2] = 16384; // constant B (linear)
            }
        }
        let ctx = pollster::block_on(GpuContext::new(None));
        let scene = Scene::from_linear_rgb16(&ctx, w, h, &samples);
        gpu::export_png(&ctx, &scene, DevelopParams::default(), std::path::Path::new(&out))
            .expect("export failed");
        println!("wrote {out}");
        return;
    }

    // `import <DIR>`: catalog a folder of RAWs into the SQLite DB.
    if args.first().map(String::as_str) == Some("import") {
        let mut dir: Option<String> = None;
        let mut db: Option<String> = None;
        let mut it = args[1..].iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--catalog" => db = it.next().cloned(),
                other if !other.starts_with('-') => dir = Some(other.to_string()),
                _ => {}
            }
        }
        let dir = dir.unwrap_or_else(|| {
            eprintln!("usage: aml import <DIR> [--catalog db]");
            std::process::exit(2);
        });
        let db = db.unwrap_or_else(default_catalog_path);
        let mut cat = catalog::Catalog::open(&db).expect("open catalog");
        println!("importing {dir} -> {db} ...");
        let s = cat.import_folder(&dir).expect("import failed");
        println!(
            "scanned {} · imported {} · skipped {} · failed {} · total photos {}",
            s.scanned, s.imported, s.skipped, s.failed,
            cat.photo_count().unwrap_or(0)
        );
        return;
    }

    // Parse the remaining flags: positional RAW path, --export, --recipe, --catalog.
    let mut export: Option<String> = None;
    let mut recipe_file: Option<String> = None;
    let mut catalog_db: Option<String> = None;
    let mut positional: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--export" => export = it.next().cloned(),
            "--recipe" => recipe_file = it.next().cloned(),
            "--catalog" => catalog_db = it.next().cloned(),
            other if !other.starts_with('-') => positional = Some(other.to_string()),
            _ => {}
        }
    }

    let recipe = match &recipe_file {
        Some(p) => {
            let txt = std::fs::read_to_string(p).expect("read recipe");
            recipe::Recipe::from_json(&txt).expect("parse recipe")
        }
        None => recipe::Recipe::default(),
    };

    // Headless export: positional RAW + --export, no window.
    if let (Some(raw_path), Some(out)) = (&positional, &export) {
        let raw = match raw_decode::decode(raw_path) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("decode failed: {e}");
                std::process::exit(1);
            }
        };
        let ctx = pollster::block_on(GpuContext::new(None));
        let scene = Scene::from_raw(&ctx, &raw);
        gpu::export_png(&ctx, &scene, DevelopParams::from(&recipe), std::path::Path::new(out))
            .expect("export failed");
        println!("wrote {out}");
        return;
    }

    // A single RAW on the command line -> open it directly in Develop.
    if let Some(raw_path) = positional {
        gui::run(None, Some((std::path::PathBuf::from(raw_path), recipe)));
        return;
    }

    // No file -> open the Library on the chosen (or default) catalog.
    let db = catalog_db.unwrap_or_else(default_catalog_path);
    match catalog::Catalog::open(&db) {
        Ok(cat) => gui::run(Some(cat), None),
        Err(e) => {
            rfd::MessageDialog::new()
                .set_title("Couldn't open catalog")
                .set_description(format!("{db}\n\n{e}"))
                .set_level(rfd::MessageLevel::Error)
                .show();
            std::process::exit(1);
        }
    }
}

/// Default catalog DB location (~/Pictures/AdobeMaybeLight.db, else temp).
fn default_catalog_path() -> String {
    std::env::var_os("HOME")
        .map(|h| std::path::Path::new(&h).join("Pictures"))
        .filter(|p| p.is_dir())
        .unwrap_or_else(std::env::temp_dir)
        .join("AdobeMaybeLight.db")
        .to_string_lossy()
        .into_owned()
}
