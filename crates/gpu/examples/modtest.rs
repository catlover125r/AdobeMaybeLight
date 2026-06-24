//! Functional checks for the develop modules added on top of the Phase-0 slice:
//! HSL band desaturation, post-crop vignette, and grain. Renders flat synthetic
//! scenes, reads the PNGs back, and asserts the expected pixel behavior.
//!
//!   cargo run -p gpu --example modtest

use gpu::{export_png, DevelopParams, GpuContext, Scene};

fn flat(w: u32, h: u32, r: u16, g: u16, b: u16) -> Vec<u16> {
    let mut v = vec![0u16; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
        v[i * 3] = r;
        v[i * 3 + 1] = g;
        v[i * 3 + 2] = b;
    }
    v
}

fn render(ctx: &GpuContext, w: u32, h: u32, src: &[u16], p: DevelopParams) -> Vec<u8> {
    let scene = Scene::from_linear_rgb16(ctx, w, h, src);
    let dir = std::env::temp_dir().join("aml-modtest.png");
    export_png(ctx, &scene, p, &dir).expect("export");
    image::open(&dir).expect("reload").to_rgba8().into_raw()
}

fn px(buf: &[u8], w: u32, x: u32, y: u32) -> [u8; 3] {
    let i = ((y * w + x) * 4) as usize;
    [buf[i], buf[i + 1], buf[i + 2]]
}

fn luma(p: [u8; 3]) -> u32 {
    p[0] as u32 + p[1] as u32 + p[2] as u32
}

fn main() {
    let ctx = pollster::block_on(GpuContext::new(None));
    let (w, h) = (64u32, 64u32);

    // --- Vignette: amount -100 must darken corners but not the center. ---
    let gray = flat(w, h, 16384, 16384, 16384); // linear ~0.25
    let mut vig = DevelopParams::default();
    vig.vignette = [-100.0, 40.0, 60.0, 0.0]; // full darken, falloff starts at 40% radius
    let base = render(&ctx, w, h, &gray, DevelopParams::default());
    let vigb = render(&ctx, w, h, &gray, vig);

    let c_base = luma(px(&base, w, w / 2, h / 2));
    let c_vig = luma(px(&vigb, w, w / 2, h / 2));
    let corner_base = luma(px(&base, w, 0, 0));
    let corner_vig = luma(px(&vigb, w, 0, 0));
    assert!(c_vig >= c_base.saturating_sub(2), "vignette must not darken the center");
    assert!(corner_vig + 30 < corner_base, "vignette must darken the corner ({corner_vig} vs {corner_base})");
    println!("vignette OK: center {c_base}->{c_vig}, corner {corner_base}->{corner_vig}");

    // --- HSL: red-band saturation -100 should desaturate a pure-red scene. ---
    let red = flat(w, h, 40000, 4000, 4000);
    let base_red = render(&ctx, w, h, &red, DevelopParams::default());
    let mut hsl = DevelopParams::default();
    hsl.hsl_sat[0][0] = -100.0; // band 0 = Red
    let hsl_red = render(&ctx, w, h, &red, hsl);

    let p0 = px(&base_red, w, w / 2, h / 2);
    let p1 = px(&hsl_red, w, w / 2, h / 2);
    let spread = |p: [u8; 3]| *p.iter().max().unwrap() as i32 - *p.iter().min().unwrap() as i32;
    assert!(spread(p1) + 20 < spread(p0), "HSL red sat -100 must reduce channel spread ({:?} -> {:?})", p0, p1);
    println!("HSL OK: red spread {} -> {}", spread(p0), spread(p1));

    // --- Grain: amount must introduce pixel-to-pixel variation on a flat field. ---
    let mut grain = DevelopParams::default();
    grain.grain = [100.0, 50.0, 0.0, 0.0];
    let grainb = render(&ctx, w, h, &gray, grain);
    let mut distinct = std::collections::HashSet::new();
    for x in 0..w {
        distinct.insert(px(&grainb, w, x, 0)[0]);
    }
    assert!(distinct.len() > 3, "grain must vary across a flat row (got {} levels)", distinct.len());
    println!("grain OK: {} distinct levels across a flat row", distinct.len());

    // --- Tone curve: shadows +100 lifts a dark patch, highlights -100 lowers
    //     a bright patch; the opposite region stays put. ---
    let dark = flat(w, h, 3000, 3000, 3000); // ~0.046 linear
    let bright = flat(w, h, 52000, 52000, 52000); // ~0.79 linear
    let dark_base = luma(px(&render(&ctx, w, h, &dark, DevelopParams::default()), w, 0, 0));
    let bright_base = luma(px(&render(&ctx, w, h, &bright, DevelopParams::default()), w, 0, 0));

    let mut lift = DevelopParams::default();
    lift.curve = [100.0, 0.0, 0.0, 0.0]; // shadows up
    let dark_lift = luma(px(&render(&ctx, w, h, &dark, lift), w, 0, 0));
    let bright_under_lift = luma(px(&render(&ctx, w, h, &bright, lift), w, 0, 0));

    let mut drop = DevelopParams::default();
    drop.curve = [0.0, 0.0, 0.0, -100.0]; // highlights down
    let bright_drop = luma(px(&render(&ctx, w, h, &bright, drop), w, 0, 0));

    assert!(dark_lift > dark_base + 10, "curve shadows+ must lift darks ({dark_base}->{dark_lift})");
    assert!((bright_under_lift as i32 - bright_base as i32).abs() < 8, "shadows+ must barely touch highlights");
    assert!(bright_drop + 10 < bright_base, "curve highlights- must lower brights ({bright_base}->{bright_drop})");
    println!("tone curve OK: dark {dark_base}->{dark_lift}, bright {bright_base}->{bright_drop}");

    // --- Crop: output dimensions follow the crop rectangle. ---
    let (cw, ch) = gpu::crop_output_dims(&DevelopParams::default(), w, h);
    assert_eq!((cw, ch), (w, h), "full-frame crop must keep source dims");
    let mut half = DevelopParams::default();
    half.crop = [0.25, 0.25, 0.5, 0.5];
    let (cw2, ch2) = gpu::crop_output_dims(&half, w, h);
    assert_eq!((cw2, ch2), (32, 32), "half crop of 64px must be 32px");

    // Straighten: on a diagonal-split scene, rotating the cropped view must
    // change the pixels. (finalize() preserves geom[0]=angle; it only fills the
    // aspect/active fields.)
    let mut diag = vec![0u16; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let val: u16 = if x > y { 60000 } else { 1000 };
            diag[i] = val;
            diag[i + 1] = val;
            diag[i + 2] = val;
        }
    }
    let mut inset = DevelopParams::default();
    inset.crop = [0.2, 0.2, 0.6, 0.6];
    let straight_px = render(&ctx, w, h, &diag, inset);
    let mut angled = inset;
    angled.geom[0] = 0.4; // ~23°
    let rotated_px = render(&ctx, w, h, &diag, angled);
    let diff: u32 = straight_px
        .iter()
        .zip(rotated_px.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
        .sum();
    assert!(diff > 1000, "straighten must change the cropped pixels (diff {diff})");
    println!("crop OK: full {cw}x{ch}, half {cw2}x{ch2}; straighten pixel diff {diff}");

    // --- Export formats: JPEG + TIFF round-trip to the right dimensions. ---
    let scene = Scene::from_linear_rgb16(&ctx, w, h, &gray);
    for (name, q) in [("aml-export.jpg", 90u8), ("aml-export.tif", 0u8)] {
        let path = std::env::temp_dir().join(name);
        gpu::export_image(&ctx, &scene, DevelopParams::default(), &path, q).expect("export");
        let img = image::open(&path).expect("reload export");
        assert_eq!((img.width(), img.height()), (w, h), "{name} wrong dims");
        println!("export OK: {name} {}x{}", img.width(), img.height());
    }

    println!("\nall develop-module checks passed");
}
