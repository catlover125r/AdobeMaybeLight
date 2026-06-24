//! Parametric edit-recipe (the JSON in `recipe_version.recipe`).
//!
//! Sparse by design: every field has a default == identity, so a preset or an
//! old recipe still deserializes and renders. See `docs/recipe-format.md`.
//! This Phase-1 subset models the globals the develop shader consumes today;
//! crop/lens/masks/spots are declared in the doc and added as the engine grows.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Recipe {
    pub schema: u32,
    pub globals: Globals,
}

impl Default for Recipe {
    fn default() -> Self {
        Self { schema: 1, globals: Globals::default() }
    }
}

impl Recipe {
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("recipe serializes")
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Globals {
    pub white_balance: WhiteBalance,
    pub tone: Tone,
    pub presence: Presence,
    pub hsl: Hsl,
    pub effects: Effects,
    pub tone_curve: ToneCurve,
    pub crop: Crop,
}

/// Crop rectangle (normalized to the source frame) + straighten angle. Default
/// is the full frame, unrotated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Crop {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
    pub angle_deg: f32,
}

impl Default for Crop {
    fn default() -> Self {
        Self { left: 0.0, top: 0.0, width: 1.0, height: 1.0, angle_deg: 0.0 }
    }
}

/// Parametric tone curve (the four region sliders). All [-100,100], 0 = identity.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToneCurve {
    pub shadows: f32,
    pub darks: f32,
    pub lights: f32,
    pub highlights: f32,
}

/// Per-band HSL/color mixer. 8 bands, in order:
/// Red, Orange, Yellow, Green, Aqua, Blue, Purple, Magenta. All [-100,100],
/// 0 = identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Hsl {
    pub hue: [f32; 8],
    pub saturation: [f32; 8],
    pub luminance: [f32; 8],
}

impl Default for Hsl {
    fn default() -> Self {
        Self { hue: [0.0; 8], saturation: [0.0; 8], luminance: [0.0; 8] }
    }
}

/// Post-crop vignette + grain. All identity-default (0).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Effects {
    pub vignette_amount: f32,   // -100 darken .. +100 lighten corners
    pub vignette_midpoint: f32, // 0..100 radius where falloff starts
    pub vignette_feather: f32,  // 0..100 falloff width
    pub grain_amount: f32,      // 0..100
    pub grain_size: f32,        // 0..100
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WhiteBalance {
    pub temp_k: f32, // UI Kelvin; 5500 = neutral relative to camera-as-shot
    pub tint: f32,   // -100 green .. +100 magenta
    pub as_shot: bool,
}

impl Default for WhiteBalance {
    fn default() -> Self {
        Self { temp_k: 5500.0, tint: 0.0, as_shot: true }
    }
}

/// All sliders are 0-centered, range roughly [-100, 100], except exposure (EV).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Tone {
    pub exposure_ev: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Presence {
    pub texture: f32,
    pub clarity: f32,
    pub dehaze: f32,
    pub vibrance: f32,
    pub saturation: f32,
}
