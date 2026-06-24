# AdobeMaybeLight

Open-source, catalog-based, non-destructive RAW photo editor (Lightroom-class),
GPLv3, 100% local AI. Engine in Rust, one `wgpu` shader codebase, ONNX Runtime
for all ML.

This repo contains the **Phase-0 spike + a working slice of Phase-1 (MVP)** —
proven end-to-end on Apple Silicon (Metal + CoreML):

- `crates/raw-decode` — LibRaw → linear 16-bit RGB + metadata probe, via a
  stable C-ABI shim.
- `crates/recipe` — serde parametric edit-recipe (sparse, identity-default).
- `crates/catalog` — SQLite catalog: import folders of RAWs (with EXIF),
  read/write recipes, idempotent.
- `crates/gpu` — wgpu develop pipeline (WB, exposure, contrast,
  highlights/shadows/whites/blacks, vibrance/saturation), shared by preview and
  headless export. Preview == export.
- `crates/app` — `winit` loupe window + `import` / `--recipe` / `--export` /
  `--selftest`.
- `crates/ai-smoke` — ONNX Runtime + CoreML EP smoke test for SAM2.
- `catalog/schema.sql` — full SQLite catalog schema (Task 3).
- `docs/` — stack decisions, recipe format, develop pipeline, model packaging.

## What's proven

| Capability | Status |
|---|---|
| LibRaw decode (real files) | ✅ Sony ARW 6024×4024, Canon CR2, Adobe DNG → correct color + orientation |
| Catalog import + EXIF | ✅ real metadata (camera/ISO/aperture), idempotent re-import |
| Recipe → develop → export | ✅ saturation −100 ⇒ exact grayscale; contrast/exposure verified |
| wgpu upload → WGSL → readback → PNG | ✅ `--selftest`, math verified (lin 0.25→sRGB 137) |
| Preview == export (one shader) | ✅ shared `make_pipeline` |
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

# GPU pipeline self-test (no RAW needed): writes a developed gradient PNG
cargo run -p app -- --selftest out.png

# Catalog: import a folder of RAWs (EXIF + default recipe per photo)
cargo run -p app -- import /path/to/photos --catalog ~/Pictures/AML.db

# Develop a real RAW file
cargo run -p app -- /path/to/photo.cr2                       # interactive loupe
cargo run -p app -- /path/to/photo.cr2 --recipe look.json    # apply a recipe
cargo run -p app -- /path/to/photo.cr2 --recipe look.json --export dev.png

# ONNX Runtime / SAM2 smoke test
export ORT_DYLIB_PATH="$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"
cargo run -p ai-smoke                                    # init + EP report
cargo run -p ai-smoke -- models/sam2_image_encoder.onnx # + run inference

# Build the double-clickable macOS app (local use; links Homebrew libraw)
cargo build --release -p app && ./packaging/bundle.sh
```

A recipe is sparse JSON (omitted fields = identity); see `docs/recipe-format.md`.
Minimal example:

```json
{ "globals": { "tone": { "exposure_ev": 1.0, "contrast": 50 },
               "presence": { "vibrance": 40 } } }
```

### Loupe keys
`↑ / ↓` exposure ±0.25 stop · `[ / ]` warm/cool white balance ·
`S` save current look to the Desktop · `Esc` quit.

## License

GPLv3-or-later. See `docs/stack-decisions.md` for the dependency/model license
audit and the Adobe-DNG-SDK substitution (we do **not** link it).
