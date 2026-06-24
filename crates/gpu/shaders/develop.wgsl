// Phase-0 develop shader: a fullscreen triangle that samples the linear RAW
// texture and applies a tiny slice of the develop pipeline (exposure + white
// balance), then encodes sRGB for display/export.
//
// The pipeline operates in LINEAR light. Only the final stage applies the sRGB
// OETF. This is the same math used for on-screen preview and PNG export, so
// what you see is what you get.

struct Develop {
    exposure: f32,    // stops; gain = 2^exposure
    contrast: f32,    // [-100,100]
    highlights: f32,  // [-100,100]
    shadows: f32,     // [-100,100]
    whites: f32,      // [-100,100]
    blacks: f32,      // [-100,100]
    vibrance: f32,    // [-100,100]
    saturation: f32,  // [-100,100]
    wb_r: f32,
    wb_g: f32,
    wb_b: f32,
    _pad: f32,
};

const LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

@group(0) @binding(0) var img_tex: texture_2d<f32>;
@group(0) @binding(1) var img_smp: sampler;
@group(0) @binding(2) var<uniform> dev: Develop;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Oversized triangle covering the viewport.
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    var uv = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 2.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(p[vi], 0.0, 1.0);
    o.uv = uv[vi];
    return o;
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var rgb = textureSample(img_tex, img_smp, in.uv).rgb;

    // 1. White balance + exposure, in linear light.
    rgb *= vec3<f32>(dev.wb_r, dev.wb_g, dev.wb_b);
    rgb *= exp2(dev.exposure);
    rgb = max(rgb, vec3<f32>(0.0));

    // 2. Tonal sliders. NOTE: spike-grade approximations — the production
    //    pipeline applies these in proper log/perceptual domains with a real
    //    tone curve (see docs/develop-pipeline.md). They are monotonic and
    //    behave sensibly for verification.
    var luma = dot(rgb, LUMA);
    let sh_mask = clamp(1.0 - luma * 2.0, 0.0, 1.0); // strongest in shadows
    let hi_mask = clamp(luma * 2.0 - 1.0, 0.0, 1.0); // strongest in highlights
    rgb += rgb * (dev.shadows / 100.0) * sh_mask;
    rgb += rgb * (dev.highlights / 100.0) * hi_mask;
    rgb *= 1.0 + (dev.whites / 100.0) * 0.2 * clamp(luma, 0.0, 1.0);
    rgb += vec3<f32>((dev.blacks / 100.0) * 0.1 * (1.0 - clamp(luma, 0.0, 1.0)));
    rgb = max(rgb, vec3<f32>(0.0));

    // 3. Contrast around an 18% pivot.
    let k = 1.0 + dev.contrast / 100.0;
    rgb = max((rgb - 0.18) * k + 0.18, vec3<f32>(0.0));

    // 4. Saturation then vibrance (vibrance scales less-saturated pixels more).
    luma = dot(rgb, LUMA);
    rgb = mix(vec3<f32>(luma), rgb, 1.0 + dev.saturation / 100.0);
    let mx = max(rgb.r, max(rgb.g, rgb.b));
    let mn = min(rgb.r, min(rgb.g, rgb.b));
    let cur_sat = select(0.0, (mx - mn) / mx, mx > 0.0);
    let l2 = dot(rgb, LUMA);
    rgb = mix(vec3<f32>(l2), rgb, 1.0 + (dev.vibrance / 100.0) * (1.0 - cur_sat));

    // 5. Output transform: linear -> sRGB.
    let out = clamp(linear_to_srgb(max(rgb, vec3<f32>(0.0))), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(out, 1.0);
}
