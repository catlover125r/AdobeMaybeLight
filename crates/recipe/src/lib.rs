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
