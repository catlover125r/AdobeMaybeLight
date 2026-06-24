# Model Packaging Plan (Task 5)

Goal: every model ships as **ONNX**, **quantized**, loaded through one ORT
abstraction with the right **execution provider per platform** — all local, $0
recurring cost. (ORT 1.26 + CoreML EP load is **proven** in `ai-smoke`.)

## Pipeline: PyTorch → ONNX → optimized → quantized → packaged

```bash
# 1. Export (per model; example: SAM2 image encoder)
python -m torch.onnx.export ...           # or model's own export script
# prefer opset 17+, dynamic axes only where needed (batch, H, W)

# 2. Shape-infer + graph optimize
python -m onnxruntime.tools.symbolic_shape_infer \
    --input sam2_enc.onnx --output sam2_enc.shapes.onnx
onnxruntime_perf_test ...                  # sanity/latency baseline

# 3. Quantize
#    - CNN/segmentation (LaMa, BiRefNet, Real-ESRGAN, BiSeNet): INT8 static
#      (calibrate on ~200 representative images)
python -m onnxruntime.quantization.preprocess --input m.onnx --output m.pre.onnx
python quantize_static.py m.pre.onnx m.int8.onnx --calib ./calib/
#    - Transformers/ViT (SAM2 encoder, CLIP, ViTMatte): FP16 (INT8 often hurts
#      accuracy on attention); ORT runs FP16 well on all EPs.
python to_fp16.py sam2_enc.onnx sam2_enc.fp16.onnx

# 4. Optional: convert to ORT format for faster load + smaller footprint
python -m onnxruntime.tools.convert_onnx_models_to_ort m.int8.onnx
```

Per-model quantization decision:

| Model | Export | Quant | EP notes |
|---|---|---|---|
| SAM2 image encoder | opset17, static H/W=1024 | **FP16** | heavy; CoreML/CUDA |
| SAM2 prompt decoder | dynamic points | FP16 | tiny; runs anywhere/CPU |
| MobileSAM/EfficientSAM | static | INT8 | fast fallback path |
| BiRefNet (sky/bg) | static 1024 | INT8 static | CNN-friendly |
| ViTMatte / MODNet | dynamic | FP16 | edge quality sensitive |
| LaMa (removal) | dynamic | INT8 static | calibrate on inpaint set |
| Real-ESRGAN x2/x4 | tiled | INT8 static | tile to bound memory |
| RetinaFace + ArcFace | static | INT8 | ArcFace embeddings stable |
| BiSeNet face-parse | static 512 | INT8 | |
| CLIP / RAM++ | opset17 | FP16 | text+image encoders |

## Execution provider per platform

ORT auto-falls-back per node, so we register the accelerator first, CPU last:

| Platform | Primary EP | Fallback |
|---|---|---|
| macOS (Apple Silicon) | **CoreML** (`MLProgram`, ANE/GPU) | CPU |
| Windows | **DirectML** (any vendor GPU) | CPU |
| Linux/Windows + NVIDIA | **CUDA** (+ optional TensorRT) | CPU |
| Linux (no NVIDIA) | CPU (XNNPACK) | — |

One Rust abstraction picks the list at runtime:

```rust
// crates/ai-smoke shows the CoreML registration; production generalizes to:
let mut b = Session::builder()?;
#[cfg(target_os = "macos")]
{ b = b.with_execution_providers([CoreMLExecutionProvider::default().build()])?; }
#[cfg(target_os = "windows")]
{ b = b.with_execution_providers([DirectMLExecutionProvider::default().build()])?; }
#[cfg(all(unix, not(target_os = "macos")))]
{ b = b.with_execution_providers([
      CUDAExecutionProvider::default().build(),   // ignored if absent
  ])?; }
let session = b.commit_from_file(path)?;          // CPU fallback automatic
```

## Distribution & loading

- **ONNX Runtime lib**: ship the platform dylib next to the app; load via
  `load-dynamic` + `ORT_DYLIB_PATH` (proven here against Homebrew's 1.26).
  Bundle EP-specific providers (CoreML built-in; DirectML/CUDA as side libs).
- **Weights are not bundled in the binary** (keeps GPLv3 binary clean and small).
  Ship only **Apache/MIT/BSD** weights in a signed `models/` pack; everything
  with a restrictive/non-commercial license (InsightFace variants, SD/SDXL) is an
  **in-app opt-in download** that shows its license first. See
  `stack-decisions.md` for the per-model license audit.
- **Integrity**: each model has a manifest entry `{name, sha256, license, ep}`;
  verify on load, lazy-load on first feature use, LRU-evict from VRAM.
- **Versioning**: `models/manifest.json` is independent of the app version so
  models update without a full release.

## Memory & latency discipline (all local, no cloud)

- Tile Real-ESRGAN / LaMa to cap VRAM; stream tiles.
- Cache SAM2 image embedding per photo (encoder is the cost; the decoder is
  cheap, so multiple clicks reuse one embedding).
- Warm CPU fallback in a background thread so first-use isn't a stall.
