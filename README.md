# AdobeMaybeLight

Open-source, catalog-based, non-destructive RAW photo editor (Lightroom-class),
GPLv3, 100% local AI. Engine in Rust, one `wgpu` shader codebase, ONNX Runtime
for all ML.

This repo contains the **Phase-0 spike + a working Phase-1 (MVP)** with a real
desktop GUI — proven end-to-end on Apple Silicon (Metal + CoreML):

- `crates/raw-decode` — LibRaw → linear 16-bit RGB + metadata probe, via a
  stable C-ABI shim.
- `crates/recipe` — serde parametric edit-recipe (sparse, identity-default).
- `crates/catalog` — SQLite catalog: import folders of RAWs (with EXIF),
  read/write recipes, idempotent.
- `crates/gpu` — wgpu develop pipeline (WB, exposure, contrast,
  highlights/shadows/whites/blacks, vibrance/saturation, dehaze, 8-band HSL,
  post-crop vignette, grain), shared by the live preview and headless export.
  Preview == export. `cargo run -p gpu --example modtest` checks the modules.
- `crates/app` — **egui desktop app**: a Library grid (background-decoded
  thumbnails) and a Develop view with live sliders driving the GPU pipeline,
  plus the `import` / `--recipe` / `--export` / `--selftest` headless CLI.
- `crates/ai-smoke` — ONNX Runtime + CoreML EP smoke test for SAM2.
- `catalog/schema.sql` — full SQLite catalog schema (Task 3).
- `docs/` — stack decisions, recipe format, develop pipeline, model packaging.

## What's proven

| Capability | Status |
|---|---|
| LibRaw decode (real files) | ✅ Sony ARW 6024×4024, Canon CR2, Adobe DNG → correct color + orientation |
| Catalog import + EXIF | ✅ real metadata (camera/ISO/aperture), idempotent re-import |
| Recipe → develop → export | ✅ saturation −100 ⇒ exact grayscale; contrast/exposure verified |
| Develop globals: WB/exposure/tone/vibrance/sat | ✅ live sliders, linear pipeline |
| Develop globals: 8-band HSL, dehaze, vignette, grain | ✅ `gpu --example modtest` (band desat, corner darken, grain variation) |
| wgpu upload → WGSL → readback → PNG | ✅ `--selftest`, math verified (lin 0.25→sRGB 137) |
| Preview == export (one shader) | ✅ shared `make_pipeline` |
| Desktop GUI (Library + Develop) | ✅ egui/wgpu: thumbnail grid, live sliders, save/export |
| ONNX Runtime local + CoreML EP | ✅ `ai-smoke` loads ORT 1.26 dynamically |
| SQLite schema | ✅ loads clean, FTS5 + WAL |

## Prerequisites (macOS / Homebrew)

```bash
brew install libraw onnxruntime little-cms2 exiv2
# Rust 1.80+ (this spike built on 1.96)
```

`build.rs` finds LibRaw via `brew --prefix libraw` (override with `LIBRAW_DIR`).

## Build & run

```bash
cargo build

# Launch the app (Library on the default catalog ~/Pictures/AdobeMaybeLight.db)
cargo run -p app
cargo run -p app -- --catalog ~/Pictures/AML.db   # a specific catalog

# Open one RAW straight in the Develop view (no catalog needed)
cargo run -p app -- /path/to/photo.cr2
cargo run -p app -- /path/to/photo.cr2 --recipe look.json   # with a starting look

# Headless: GPU pipeline self-test (no RAW), folder import, and export
cargo run -p app -- --selftest out.png
cargo run -p app -- import /path/to/photos --catalog ~/Pictures/AML.db
cargo run -p app -- /path/to/photo.cr2 --recipe look.json --export dev.png

# ONNX Runtime / SAM2 smoke test
export ORT_DYLIB_PATH="$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"
cargo run -p ai-smoke                                    # init + EP report
cargo run -p ai-smoke -- models/sam2_image_encoder.onnx # + run inference

# Build the double-clickable macOS app (local use; links Homebrew libraw)
cargo build --release -p app && ./packaging/bundle.sh
```

### Using the app
- **Library** — click *Import folder…* to catalog a folder of RAWs; thumbnails
  decode in the background. Click a thumbnail to open it in Develop.
- **Develop** — drag the sliders (white balance, tone, presence) for a live GPU
  preview. *Save* writes a new recipe version to the catalog (non-destructive,
  versioned); *Export…* renders a full-resolution PNG via the same shader.

A recipe is sparse JSON (omitted fields = identity); see `docs/recipe-format.md`.
Minimal example:

```json
{ "globals": { "tone": { "exposure_ev": 1.0, "contrast": 50 },
               "presence": { "vibrance": 40 } } }
```

## License

GPLv3-or-later. See `docs/stack-decisions.md` for the dependency/model license
audit and the Adobe-DNG-SDK substitution (we do **not** link it).
