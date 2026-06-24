# Stack & Model Decisions (Task 1)

Verdict: the proposed stack is sound. Below are confirmations, three
substitutions I'd make, and the licensing/feasibility risks — with the GPLv3
(Path A) implications called out explicitly.

## Confirmed as-is

| Area | Choice | Why keep it |
|---|---|---|
| Engine | Rust + C/C++ for direct ports | Memory-safe orchestration; port DSP where it already exists |
| GPU | wgpu (Vulkan/Metal/DX12/GL) | One shader codebase; **proven in this spike on Metal** |
| RAW decode | LibRaw | De-facto standard; **proven linking + runtime here** (0.22.1) |
| Color mgmt | lcms2 | MIT; the standard CMM |
| Catalog | SQLite + JSON recipes | Single-file, transactional, zero-cost |
| Inference | ONNX Runtime, 100% local | **Proven loading here** (1.26.0, CoreML EP) |
| Masking | SAM 2 + MobileSAM/EfficientSAM | Best click-to-mask; small variants for speed |
| Sky/BG | BiRefNet | SOTA dichotomous segmentation |
| Removal | LaMa default | Fast, no diffusion weights to ship |
| People | InsightFace (RetinaFace+ArcFace) | Best open face stack |

## Substitutions I'd make

1. **Drop the Adobe DNG SDK as a hard dependency.** It is *not* GPLv3-compatible
   (custom Adobe license + patent terms) and **cannot be linked into a GPLv3
   binary**. Use **LibRaw for DNG read** (it already reads DNG) and write DNG
   via a **TIFF/EP writer over `libtiff`** for the cases we need (linear DNG,
   proxy). Full mosaic-DNG *write* is a Phase-4+ nice-to-have, not MVP. This
   removes the single biggest license landmine.

2. **Metadata: prefer the Exiv2 library; treat ExifTool as an optional external
   tool, not a linked dep.** Exiv2 is GPLv2-or-later (GPLv3-compatible). ExifTool
   is Perl (Artistic/GPL) — shell out to it only as an optional helper, never
   link. For MVP, Exiv2 alone covers read/write.

3. **UI shell: keep Tauri but be ready to fall back to winit+egui for the
   catalog chrome.** Tauri (MIT/Apache) is fine and GPL-compatible. The risk is
   wiring a zero-copy wgpu canvas *inside* a Tauri webview across all 3 OSes; the
   spike here uses **winit + raw wgpu**, which is the lower-risk path and what I'd
   ship the Develop view on. Use Tauri for Library/Map/Print chrome if you want
   web tech there; render the image with native wgpu either way.

## GPLv3 (Path A) implications — read this

- **Whole-program copyleft.** Linking GPLv3 code (darktable, RawTherapee
  modules, Hugin/enblend, potentially Lensfun consumers) makes the **entire
  distributed binary GPLv3**. Every linked dependency must be GPLv3-compatible.
- **Compatible:** lcms2 (MIT), LibRaw (LGPL2.1/CDDL dual — OK), Exiv2 (GPLv2+),
  libtiff (libtiff/BSD), wgpu/winit/Rust crates (MIT/Apache-2.0 — Apache-2.0 is
  GPLv3-compatible, *not* GPLv2), FFmpeg (LGPL/GPL build — choose LGPL config to
  keep flexibility), libgphoto2 (LGPL), MapLibre (BSD), SQLite (public domain).
- **Incompatible / keep out of the linked binary:** Adobe DNG SDK, anything
  Apache-2.0-only that you'd link into a *GPLv2* context (not our case since we
  target GPLv3), and any "source-available" model licenses (see below).
- **Patent grant:** GPLv3 gives users a patent license from contributors. Fine
  for us; just means corporate contributors must be aware.
- **Model weights are data, not linked code**, so they don't force GPL on the
  app — *but their own licenses still bind distribution.* Audit each:
  - SAM 2 — Apache-2.0 ✅ ship weights
  - MobileSAM/EfficientSAM — Apache/MIT ✅
  - BiRefNet — MIT ✅
  - LaMa — Apache-2.0 ✅
  - Real-ESRGAN — BSD-3 ✅ (note: some training data terms; weights are BSD)
  - InsightFace models — **non-commercial** on several releases ⚠️ ship only the
    permissively-licensed variants or make People an opt-in download.
  - RAM++ / CLIP — Apache/MIT ✅ (OpenAI CLIP weights MIT)
  - SD/SDXL inpaint — CreativeML OpenRAIL ⚠️ **opt-in plugin only**, never bundle.
  - ViTMatte/MODNet — check per-repo; MODNet is Apache, some matting weights are
    research-only ⚠️.

  Decision: **bundle only Apache/MIT/BSD weights**; everything else is an
  explicit in-app opt-in download with its license shown.

## Feasibility risks & mitigations

- **wgpu inside Tauri webview** → mitigated by shipping Develop on native wgpu
  (done in spike).
- **CoreML EP coverage** — some ops fall back to CPU. Acceptable; ORT does this
  per-node automatically (verified the EP registers here).
- **darktable/RawTherapee are C/C++** — port modules behind a stable C ABI shim
  (the pattern this spike uses for LibRaw), don't FFI their internal structs.
- **Color science is on us** (we don't copy Adobe). Plan ArgyllCMS ColorChecker
  calibration early so users have a credible neutral starting point.

## Recurring-cost audit: **$0**

No cloud GPU, no per-seat SDK fees, no map tiles bill (self-host or OSM with
attribution). Only optional cost is code-signing/notarization certs per OS.
