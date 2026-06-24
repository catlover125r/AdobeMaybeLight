# Parametric Edit-Recipe Format (Task 3)

A recipe is the **complete, declarative description of a develop** — no pixels.
It is stored as JSON in `recipe_version.recipe` and is the single source of truth
the GPU pipeline consumes. Rules:

- **Versioned.** `schema` gates migrations.
- **Order-stable.** Global modules render in a fixed pipeline order (see
  `develop-pipeline.md`), *not* recipe key order. Local adjustments render in
  their array order (top-to-bottom in the UI).
- **Sparse.** Omitted modules = identity. A preset is just a partial recipe.
- **Mask geometry is parametric where possible** (linear/radial = a few floats;
  brush = compressed stroke list; AI masks = model + prompt + a cached
  low-res alpha hash so we can recompute losslessly and cache the heavy alpha
  out-of-band in `thumbnail`-style blobs, never inline in JSON).

## Top-level shape

```jsonc
{
  "schema": 1,
  "crop":      { /* geometry */ },
  "globals":   { /* WB, tone, color, detail, effects */ },
  "lens":      { /* corrections */ },
  "masks":     [ /* ordered local adjustments */ ],
  "spots":     [ /* heal/clone/red-eye */ ]
}
```

## Worked example

```json
{
  "schema": 1,
  "crop": {
    "angle_deg": -1.5,
    "rect": [0.05, 0.02, 0.92, 0.95],
    "aspect": "3:2",
    "perspective": { "upright": "auto", "vertical": 0.0, "horizontal": 0.0 }
  },
  "globals": {
    "white_balance": { "temp_k": 5200, "tint": 8, "as_shot": false },
    "tone": {
      "exposure_ev": 0.35, "contrast": 12,
      "highlights": -40, "shadows": 30, "whites": 8, "blacks": -6
    },
    "presence": { "texture": 15, "clarity": 10, "dehaze": 5,
                  "vibrance": 18, "saturation": -4 },
    "tone_curve": {
      "rgb":  [[0,0],[64,58],[192,200],[255,255]],
      "r": [], "g": [], "b": []
    },
    "hsl": {
      "red":    { "h": 0,  "s": 0,   "l": 0 },
      "orange": { "h": -3, "s": 10,  "l": 5 },
      "blue":   { "h": 0,  "s": 20,  "l": -10 }
    },
    "color_grade": {
      "shadows":    { "h": 215, "s": 12, "l": 0 },
      "midtones":   { "h": 40,  "s": 6,  "l": 0 },
      "highlights": { "h": 45,  "s": 8,  "l": 2 },
      "blending": 50, "balance": 0
    },
    "detail": {
      "sharpen": { "amount": 40, "radius": 1.0, "detail": 25, "masking": 30 },
      "noise":   { "luminance": 15, "color": 25, "method": "profiled" }
    },
    "effects": {
      "vignette": { "amount": -18, "midpoint": 50, "roundness": 0, "feather": 60 },
      "grain":    { "amount": 12, "size": 25, "roughness": 50 }
    }
  },
  "lens": {
    "profile": "lensfun:Canon:RF 50mm F1.8 STM",
    "distortion": true, "vignette": true, "chromatic_aberration": true,
    "manual": { "distortion": 0.0, "defringe_purple": 0, "defringe_green": 0 }
  },
  "masks": [
    {
      "id": "m1", "name": "Sky", "enabled": true, "invert": false,
      "source": { "type": "ai-segment", "model": "birefnet", "target": "sky" },
      "combine": [
        { "op": "subtract",
          "source": { "type": "linear",
                      "p0": [0.0, 0.55], "p1": [0.0, 0.75] } }
      ],
      "adjust": { "exposure_ev": -0.5, "dehaze": 20,
                  "white_balance": { "temp_k": 6200 } }
    },
    {
      "id": "m2", "name": "Subject", "enabled": true,
      "source": { "type": "ai-click", "model": "sam2",
                  "points": [{ "xy": [0.5, 0.6], "label": 1 }],
                  "alpha_cache": "blake3:9f1c…" },
      "matte": { "model": "vitmatte", "feather_px": 2 },
      "adjust": { "exposure_ev": 0.3, "clarity": 12, "texture": 10 }
    },
    {
      "id": "m3", "name": "Eyes", "enabled": true,
      "source": { "type": "face-part", "model": "bisenet", "part": "eyes" },
      "intersect_with": "m2",
      "adjust": { "clarity": 20, "saturation": 8 }
    }
  ],
  "spots": [
    { "type": "heal",  "method": "lama",
      "target": [0.31, 0.22, 0.03], "feather": 0.5 },
    { "type": "clone", "target": [0.7, 0.4], "source": [0.6, 0.4], "radius": 0.04 },
    { "type": "redeye", "center": [0.48, 0.33], "radius": 0.01 }
  ]
}
```

## Notes on each block

- **crop.rect** = normalized `[x, y, w, h]` in *post-rotation, post-upright*
  coordinates so geometry is resolution-independent.
- **white_balance** stores Kelvin/tint (UI) **and** can carry `as_shot` so a
  reset is exact; the engine converts to channel multipliers at runtime.
- **tone_curve** points are 0..255 control points; engine fits a monotonic
  spline. Per-channel R/G/B curves are optional.
- **masks[].combine** is the boolean stack: each entry `{op, source}` with
  `op ∈ {add, subtract, intersect}`, applied left-to-right onto the base
  `source`. `intersect_with` is sugar for "intersect with another mask's final
  alpha" (lets m3 = eyes ∩ subject).
- **AI mask sources never inline the alpha.** They store the prompt + a content
  hash (`alpha_cache`) keyed to the model+image; the actual alpha lives in the
  mask cache (recomputable, so the recipe stays small and portable).
- **spots[].target/source** are normalized points/rects; heal `method` selects
  classic patch vs `lama` (default per project brief).

## Why this shape

- Deterministic re-render from JSON alone (+ cached mattes) → cheap sync, cheap
  history, cheap presets (just a partial recipe).
- Decouples UI units (Kelvin, 0..100 sliders) from engine units (multipliers,
  normalized) at one boundary.
- Adding a module = adding a key; old recipes still render (sparse = identity).
