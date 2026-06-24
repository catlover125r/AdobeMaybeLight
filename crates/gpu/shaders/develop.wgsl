// Develop shader: a fullscreen triangle that samples the linear RAW texture and
// applies the global develop pipeline, then encodes sRGB for display/export.
//
// The pipeline operates in LINEAR light. Only the final stage applies the sRGB
// OETF. This is the same math used for on-screen preview and PNG export, so
// what you see is what you get.
//
// NOTE: tonal/HSL/dehaze math here is spike-grade — monotonic, sensible, and
// verifiable, but not the production darktable/RawTherapee DSP (see
// docs/develop-pipeline.md). Every new module is guarded so that an identity
// setting (all zero) leaves the existing output byte-for-byte unchanged.

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
    dehaze: f32,      // [-100,100]
    // 8-band HSL, packed two vec4s per channel (Red,Orange,Yellow,Green,Aqua,
    // Blue,Purple,Magenta). Each value [-100,100].
    hsl_hue: array<vec4<f32>, 2>,
    hsl_sat: array<vec4<f32>, 2>,
    hsl_lum: array<vec4<f32>, 2>,
    vignette: vec4<f32>, // amount[-100,100], midpoint[0,100], feather[0,100], _
    grain: vec4<f32>,    // amount[0,100], size[0,100], _, _
    curve: vec4<f32>,    // parametric tone curve regions: shadows,darks,lights,highlights [-100,100]
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

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let d = mx - mn;
    var h = 0.0;
    if (d > 0.0) {
        if (mx == c.r) {
            h = (c.g - c.b) / d;
        } else if (mx == c.g) {
            h = 2.0 + (c.b - c.r) / d;
        } else {
            h = 4.0 + (c.r - c.g) / d;
        }
        h = fract(h / 6.0);
    }
    let s = select(0.0, d / mx, mx > 0.0);
    return vec3<f32>(h, s, mx);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let h = fract(c.x) * 6.0;
    let i = floor(h);
    let f = h - i;
    let p = c.z * (1.0 - c.y);
    let q = c.z * (1.0 - c.y * f);
    let t = c.z * (1.0 - c.y * (1.0 - f));
    let m = i - 6.0 * floor(i / 6.0);
    if (m < 0.5) { return vec3<f32>(c.z, t, p); }
    if (m < 1.5) { return vec3<f32>(q, c.z, p); }
    if (m < 2.5) { return vec3<f32>(p, c.z, t); }
    if (m < 3.5) { return vec3<f32>(p, q, c.z); }
    if (m < 4.5) { return vec3<f32>(t, p, c.z); }
    return vec3<f32>(c.z, p, q);
}

// 8-band HSL. Each pixel's hue is matched against band centers with a 60°
// triangular falloff; hue is rotated, saturation/value scaled by the summed
// band weights.
fn apply_hsl(rgb_in: vec3<f32>) -> vec3<f32> {
    let hsv = rgb2hsv(rgb_in);
    var h = hsv.x;
    var s = hsv.y;
    var v = hsv.z;
    if (s <= 0.0) { return rgb_in; }

    var centers = array<f32, 8>(0.0, 0.0833, 0.1667, 0.3333, 0.5, 0.6667, 0.7917, 0.8333);
    var hue = array<f32, 8>(
        dev.hsl_hue[0].x, dev.hsl_hue[0].y, dev.hsl_hue[0].z, dev.hsl_hue[0].w,
        dev.hsl_hue[1].x, dev.hsl_hue[1].y, dev.hsl_hue[1].z, dev.hsl_hue[1].w);
    var sat = array<f32, 8>(
        dev.hsl_sat[0].x, dev.hsl_sat[0].y, dev.hsl_sat[0].z, dev.hsl_sat[0].w,
        dev.hsl_sat[1].x, dev.hsl_sat[1].y, dev.hsl_sat[1].z, dev.hsl_sat[1].w);
    var lum = array<f32, 8>(
        dev.hsl_lum[0].x, dev.hsl_lum[0].y, dev.hsl_lum[0].z, dev.hsl_lum[0].w,
        dev.hsl_lum[1].x, dev.hsl_lum[1].y, dev.hsl_lum[1].z, dev.hsl_lum[1].w);

    var h_adj = 0.0;
    var s_adj = 0.0;
    var l_adj = 0.0;
    for (var i = 0; i < 8; i = i + 1) {
        var d = abs(h - centers[i]);
        d = min(d, 1.0 - d); // hue wraps
        let w = max(0.0, 1.0 - d / 0.1667);
        h_adj = h_adj + w * hue[i];
        s_adj = s_adj + w * sat[i];
        l_adj = l_adj + w * lum[i];
    }
    h = fract(h + (h_adj / 100.0) * 0.0833); // up to ~±30°
    s = clamp(s * (1.0 + s_adj / 100.0), 0.0, 1.0);
    v = max(v * (1.0 + l_adj / 100.0), 0.0);
    return hsv2rgb(vec3<f32>(h, s, v));
}

// Parametric tone curve: four region sliders (shadows, darks, lights,
// highlights) drive smooth Gaussian bumps centered across the tonal range.
// Values outside [0,1] (HDR) get ~zero weight and pass through.
fn tone_curve(x: f32) -> f32 {
    let centers = vec4<f32>(0.125, 0.375, 0.625, 0.875);
    let d = (vec4<f32>(x) - centers) / 0.18;
    let bump = exp(-0.5 * d * d);
    let amt = (dev.curve / 100.0) * 0.25;
    return max(x + dot(amt, bump), 0.0);
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var rgb = textureSample(img_tex, img_smp, in.uv).rgb;

    // 1. White balance + exposure, in linear light.
    rgb *= vec3<f32>(dev.wb_r, dev.wb_g, dev.wb_b);
    rgb *= exp2(dev.exposure);
    rgb = max(rgb, vec3<f32>(0.0));

    // 2. Tonal sliders (luma-masked, monotonic approximations).
    var luma = dot(rgb, LUMA);
    let sh_mask = clamp(1.0 - luma * 2.0, 0.0, 1.0);
    let hi_mask = clamp(luma * 2.0 - 1.0, 0.0, 1.0);
    rgb += rgb * (dev.shadows / 100.0) * sh_mask;
    rgb += rgb * (dev.highlights / 100.0) * hi_mask;
    rgb *= 1.0 + (dev.whites / 100.0) * 0.2 * clamp(luma, 0.0, 1.0);
    rgb += vec3<f32>((dev.blacks / 100.0) * 0.1 * (1.0 - clamp(luma, 0.0, 1.0)));
    rgb = max(rgb, vec3<f32>(0.0));

    // 3. Contrast around an 18% pivot.
    let k = 1.0 + dev.contrast / 100.0;
    rgb = max((rgb - 0.18) * k + 0.18, vec3<f32>(0.0));

    // 3b. Parametric tone curve (guarded: identity when all regions are zero).
    if (dot(abs(dev.curve), vec4<f32>(1.0)) > 0.0) {
        rgb = vec3<f32>(tone_curve(rgb.r), tone_curve(rgb.g), tone_curve(rgb.b));
    }

    // 4. Saturation then vibrance (vibrance scales less-saturated pixels more).
    luma = dot(rgb, LUMA);
    rgb = mix(vec3<f32>(luma), rgb, 1.0 + dev.saturation / 100.0);
    let mx = max(rgb.r, max(rgb.g, rgb.b));
    let mn = min(rgb.r, min(rgb.g, rgb.b));
    let cur_sat = select(0.0, (mx - mn) / mx, mx > 0.0);
    let l2 = dot(rgb, LUMA);
    rgb = mix(vec3<f32>(l2), rgb, 1.0 + (dev.vibrance / 100.0) * (1.0 - cur_sat));
    rgb = max(rgb, vec3<f32>(0.0));

    // 5. 8-band HSL (guarded: skipped exactly when all bands are zero).
    let hsl_sum =
        dot(abs(dev.hsl_hue[0]), vec4<f32>(1.0)) + dot(abs(dev.hsl_hue[1]), vec4<f32>(1.0)) +
        dot(abs(dev.hsl_sat[0]), vec4<f32>(1.0)) + dot(abs(dev.hsl_sat[1]), vec4<f32>(1.0)) +
        dot(abs(dev.hsl_lum[0]), vec4<f32>(1.0)) + dot(abs(dev.hsl_lum[1]), vec4<f32>(1.0));
    if (hsl_sum > 0.0) {
        rgb = apply_hsl(rgb);
    }

    // 6. Dehaze (approximate): pull a black point and boost contrast +
    //    saturation, weighted by the slider. Identity at 0.
    if (dev.dehaze != 0.0) {
        let dz = dev.dehaze / 100.0;
        rgb = max((rgb - 0.03 * dz) * (1.0 + 0.4 * dz), vec3<f32>(0.0));
        let dl = dot(rgb, LUMA);
        rgb = mix(vec3<f32>(dl), rgb, 1.0 + 0.3 * dz);
        rgb = max(rgb, vec3<f32>(0.0));
    }

    // 7. Post-crop vignette (linear multiply, identity at amount 0).
    let v_amt = dev.vignette.x / 100.0;
    if (v_amt != 0.0) {
        let mid = dev.vignette.y / 100.0;
        let feather = max(dev.vignette.z / 100.0, 0.001);
        let c = in.uv - vec2<f32>(0.5);
        let r = length(c) * 1.41421356; // 0 center .. 1 corner
        let t = clamp((r - mid) / feather, 0.0, 1.0);
        rgb *= 1.0 + v_amt * t; // amount<0 darkens corners
        rgb = max(rgb, vec3<f32>(0.0));
    }

    // 8. Output transform: linear -> sRGB.
    var out = clamp(linear_to_srgb(max(rgb, vec3<f32>(0.0))), vec3<f32>(0.0), vec3<f32>(1.0));

    // 9. Grain in display space (guarded: identity at amount 0).
    let g_amt = dev.grain.x / 100.0;
    if (g_amt > 0.0) {
        let dims = vec2<f32>(textureDimensions(img_tex));
        let gsize = mix(1.0, 5.0, dev.grain.y / 100.0);
        let n = hash21(floor(in.uv * dims / gsize));
        out = clamp(out + (n - 0.5) * g_amt * 0.25, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    return vec4<f32>(out, 1.0);
}
