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

    println!("\nall develop-module checks passed");
}
