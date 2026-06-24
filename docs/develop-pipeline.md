# Develop GPU Pipeline Architecture (Task 4)

## Working color space

The pipeline runs in **linear, scene-referred RGB** on a **wide working
primary** — specifically **linear Rec.2020 (or ACEScg)**, 16-bit float on the
GPU (`Rgba16Float`). Rationale:

- Linear light = physically correct exposure, blends, and merges.
- Wide primaries prevent gamut clipping during saturation/color-grade before the
  final clamp.
- We convert **once in** (camera native → working) and **once out** (working →
  output/display) via lcms2-derived matrices baked into the shader.

> The Phase-0 spike currently decodes to *linear sRGB-primary* for simplicity
> and applies the sRGB OETF at the end (verified: linear 0.25 → sRGB 137). The
> production pipeline swaps the working primary to Rec.2020/ACEScg and moves the
> output transform behind a selectable display/output profile.

## Fixed module order (globals)

Globals always execute in this order regardless of recipe key order. Each is a
wgpu pass (or fused group of passes); most are fragment passes, a few
(denoise, sharpen, dehaze) are compute.

```
 0. Decode               LibRaw → linear RGB16 (CPU)            [done in spike]
 1. Input transform      camera primaries → working (matrix)
 2. White balance        channel multipliers (pre-demosaic-ideal, post here)
 3. Lens corrections     distortion / TCA / vignette (Lensfun)   [geometry]
 4. Crop / rotate / upright / perspective                        [geometry]
 5. Exposure                                                     [linear gain]
 6. Highlight reconstruction / tone (highlights/shadows/whites/blacks)
 7. Tone curve (RGB then per-channel)
 8. Dehaze                                                       [compute]
 9. Texture / Clarity    (local-contrast, multi-scale)           [compute]
10. HSL                                                          [hue-band]
11. Color grading wheels (shadow/mid/high lift-gamma-gain)
12. Vibrance / Saturation
13. Noise reduction      (profiled first; ML denoise optional)   [compute]
14. Sharpening           (unsharp / deconvolution)               [compute]
15. Effects: vignette, grain
16. Local adjustments    (mask stack — see below)
17. Spot heal/clone/red-eye  (incl. LaMa fill)                   [compute/ONNX]
18. Output transform     working → output profile + OETF; dither; export
```

Geometry passes (3–4) and spot/AI passes (16–17) are the expensive,
cache-worthy boundaries. We cache the **post-geometry linear buffer** and the
**pre-output buffer** so slider tweaks in 5–15 only re-run cheap passes.

## How masks composite

Each entry in `recipe.masks[]` produces a single-channel **alpha** in working
resolution, then its `adjust` block is applied **only where alpha > 0**, blended
back over the running image:

```
for mask in masks (UI top→bottom):
    base_alpha   = rasterize(mask.source)          // brush/linear/radial/AI
    for step in mask.combine:                       // boolean stack
        a2 = rasterize(step.source)
        base_alpha = combine(base_alpha, a2, step.op)   // add/subtract/intersect
    if mask.intersect_with: base_alpha *= other_mask_final_alpha
    if mask.matte: base_alpha = refine(base_alpha, matte_model)  // ViTMatte/feather
    if mask.invert: base_alpha = 1 - base_alpha
    adjusted = apply_local_ops(running_image, mask.adjust)        // same op kernels
    running_image = mix(running_image, adjusted, base_alpha)
```

Boolean ops on alpha:
- `add`       → `max(a, b)`  (union)
- `subtract`  → `a * (1 - b)`
- `intersect` → `a * b`

Key properties:
- **Local adjustments reuse the exact global op kernels** (exposure, WB, clarity,
  …) — one WGSL library, called globally with alpha=1 or locally with a mask.
- **AI masks are just alpha producers.** SAM2/BiRefNet/BiSeNet run via ONNX
  Runtime on CoreML/DirectML/CUDA, output a low-res mask, we upsample + optional
  matte refine on GPU, then it enters the same boolean/composite path.
- Mask alphas are **cached by content hash** so re-renders skip inference.

## Pass scheduling

- One `wgpu::CommandEncoder` per frame; passes write ping-pong `Rgba16Float`
  targets. Fuse adjacent fragment passes where bind groups allow.
- Interactive preview renders at **proxy resolution** (fit-to-view); export and
  1:1 loupe render full-res through the identical shaders (preview == export,
  proven in the spike's shared `make_pipeline` + `export_png`).
- Heavy ML/compute (denoise, dehaze, LaMa) gated behind dirty-flags + caches so
  dragging a tone slider never re-triggers them.
