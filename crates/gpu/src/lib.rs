//! Phase-0 wgpu develop pipeline: upload linear RAW -> apply exposure/WB ->
//! present or export. One WGSL codebase drives both the on-screen preview and
//! the headless PNG/TIFF export, so preview == export.

use half::f16;
use wgpu::util::DeviceExt;

/// GPU uniform for the global develop pass. Plain scalars + vec4-packed arrays
/// laid out to match the WGSL `Develop` struct exactly (176 bytes; every block
/// starts on a 16-byte boundary so std140 and `#[repr(C)]` agree).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DevelopParams {
    pub exposure: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub vibrance: f32,
    pub saturation: f32,
    pub wb_r: f32,
    pub wb_g: f32,
    pub wb_b: f32,
    pub dehaze: f32,
    // 8-band HSL, two vec4s per channel (R,O,Y,G,Aqua,B,Purple,Magenta).
    pub hsl_hue: [[f32; 4]; 2],
    pub hsl_sat: [[f32; 4]; 2],
    pub hsl_lum: [[f32; 4]; 2],
    pub vignette: [f32; 4], // amount, midpoint, feather, _
    pub grain: [f32; 4],    // amount, size, _, _
    pub curve: [f32; 4],    // parametric tone curve: shadows, darks, lights, highlights
    pub crop: [f32; 4],     // normalized left, top, width, height
    pub geom: [f32; 4],     // angle_rad, src_aspect (filled at render), crop_active, _
}

impl Default for DevelopParams {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            contrast: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            vibrance: 0.0,
            saturation: 0.0,
            wb_r: 1.0,
            wb_g: 1.0,
            wb_b: 1.0,
            dehaze: 0.0,
            hsl_hue: [[0.0; 4]; 2],
            hsl_sat: [[0.0; 4]; 2],
            hsl_lum: [[0.0; 4]; 2],
            vignette: [0.0; 4],
            grain: [0.0; 4],
            curve: [0.0; 4],
            crop: [0.0, 0.0, 1.0, 1.0],
            geom: [0.0; 4],
        }
    }
}

/// Pack an 8-element band array into the two-vec4 layout the shader expects.
fn pack8(v: &[f32; 8]) -> [[f32; 4]; 2] {
    [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]]]
}

impl From<&recipe::Recipe> for DevelopParams {
    fn from(r: &recipe::Recipe) -> Self {
        let g = &r.globals;
        // Spike-grade Kelvin/tint -> channel multipliers. The production path
        // derives these from the camera profile + chromatic-adaptation matrix.
        let (mut wb_r, mut wb_g, mut wb_b) = (1.0_f32, 1.0_f32, 1.0_f32);
        if !g.white_balance.as_shot {
            let t = (g.white_balance.temp_k - 5500.0) / 5500.0; // warm = +
            wb_r = (1.0 + 0.4 * t).max(0.05);
            wb_b = (1.0 - 0.4 * t).max(0.05);
            wb_g = (1.0 - 0.2 * g.white_balance.tint / 100.0).max(0.05);
        }
        let e = &g.effects;
        Self {
            exposure: g.tone.exposure_ev,
            contrast: g.tone.contrast,
            highlights: g.tone.highlights,
            shadows: g.tone.shadows,
            whites: g.tone.whites,
            blacks: g.tone.blacks,
            vibrance: g.presence.vibrance,
            saturation: g.presence.saturation,
            wb_r,
            wb_g,
            wb_b,
            dehaze: g.presence.dehaze,
            hsl_hue: pack8(&g.hsl.hue),
            hsl_sat: pack8(&g.hsl.saturation),
            hsl_lum: pack8(&g.hsl.luminance),
            vignette: [e.vignette_amount, e.vignette_midpoint, e.vignette_feather, 0.0],
            grain: [e.grain_amount, e.grain_size, 0.0, 0.0],
            curve: {
                let c = &g.tone_curve;
                [c.shadows, c.darks, c.lights, c.highlights]
            },
            crop: {
                let c = &g.crop;
                [c.left, c.top, c.width, c.height]
            },
            geom: [g.crop.angle_deg.to_radians(), 0.0, 0.0, 0.0],
        }
    }
}

/// The output pixel dimensions after applying `params`' crop to a source of
/// `sw`x`sh`. Straighten doesn't change the crop's pixel size.
pub fn crop_output_dims(params: &DevelopParams, sw: u32, sh: u32) -> (u32, u32) {
    let cw = params.crop[2].clamp(0.01, 1.0);
    let ch = params.crop[3].clamp(0.01, 1.0);
    (((sw as f32 * cw).round() as u32).max(1), ((sh as f32 * ch).round() as u32).max(1))
}

/// Fill the render-time geometry fields (source aspect; whether crop/straighten
/// is active) so the shader can short-circuit to an exact passthrough when the
/// frame is uncropped and unrotated.
fn finalize(mut p: DevelopParams, sw: u32, sh: u32) -> DevelopParams {
    let full = p.crop[0] == 0.0 && p.crop[1] == 0.0 && p.crop[2] >= 1.0 && p.crop[3] >= 1.0;
    let active = !full || p.geom[0] != 0.0;
    p.geom[1] = sw as f32 / sh.max(1) as f32;
    p.geom[2] = if active { 1.0 } else { 0.0 };
    p
}

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub async fn new(compatible_surface: Option<&wgpu::Surface<'_>>) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface,
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("aml-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("device request failed");
        Self { instance, adapter, device, queue }
    }
}

/// GPU-resident image + its develop uniforms and bind group.
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
}

impl Scene {
    pub fn from_raw(ctx: &GpuContext, img: &raw_decode::RawImage) -> Self {
        Self::from_linear_rgb16(ctx, img.width, img.height, img.samples())
    }

    /// Build a scene from interleaved linear 16-bit RGB samples
    /// (length = width*height*3). Shared by RAW decode and the self-test.
    pub fn from_linear_rgb16(ctx: &GpuContext, w: u32, h: u32, src: &[u16]) -> Self {
        assert_eq!(src.len(), (w * h * 3) as usize, "sample count mismatch");

        // Expand linear RGB16-uint -> RGBA16-float in [0,1]; alpha = 1.
        let mut rgba = vec![f16::ZERO; (w * h * 4) as usize];
        let one = f16::from_f32(1.0);
        for i in 0..(w * h) as usize {
            let r = src[i * 3] as f32 / 65535.0;
            let g = src[i * 3 + 1] as f32 / 65535.0;
            let b = src[i * 3 + 2] as f32 / 65535.0;
            rgba[i * 4] = f16::from_f32(r);
            rgba[i * 4 + 1] = f16::from_f32(g);
            rgba[i * 4 + 2] = f16::from_f32(b);
            rgba[i * 4 + 3] = one;
        }

        let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raw-linear"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(w * 8), // 4 channels * 2 bytes
                rows_per_image: Some(h),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("develop-uniform"),
            contents: bytemuck::bytes_of(&DevelopParams::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout =
            ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("develop-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("develop-bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
            ],
        });

        Self { width: w, height: h, bind_group_layout, bind_group, uniform }
    }

    pub fn set_params(&self, queue: &wgpu::Queue, p: DevelopParams) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&p));
    }
}

/// Build the develop render pipeline for a given color-attachment format.
pub fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    target_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("develop-wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/develop.wgsl").into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("develop-pl"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("develop-pipeline"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(target_format.into())],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// A reusable Rgba8Unorm render target that the GUI develop view draws into and
/// then hands to egui as a native texture. Sized independently of the source
/// image (the develop shader samples the full-res linear texture), so the
/// preview can be capped for responsiveness.
pub struct PreviewTarget {
    pub texture: wgpu::Texture,
    /// Linear (Rgba8Unorm) view used as the render attachment. The develop
    /// shader writes already-sRGB-encoded bytes here (no hardware re-encode).
    pub attach_view: wgpu::TextureView,
    /// sRGB reinterpreting view handed to egui. Sampling it converts the stored
    /// sRGB bytes back to linear so egui's sRGB framebuffer displays them
    /// correctly (avoids the classic double-gamma washout).
    pub sample_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl PreviewTarget {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preview-target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });
        let attach_view = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            ..Default::default()
        });
        let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            ..Default::default()
        });
        Self { texture, attach_view, sample_view, width, height }
    }
}

/// Render `scene` with `params` into a preview target using a prebuilt pipeline
/// (Rgba8Unorm). Used by the interactive develop view; cheap to call per edit.
pub fn render_to_target(
    ctx: &GpuContext,
    pipeline: &wgpu::RenderPipeline,
    scene: &Scene,
    params: DevelopParams,
    target: &PreviewTarget,
) {
    scene.set_params(&ctx.queue, finalize(params, scene.width, scene.height));
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("preview-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.attach_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    ctx.queue.submit([enc.finish()]);
}

/// Headless render of `scene` with `params` to a tightly-packed 8-bit sRGB
/// RGBA buffer. Shared by every file-export format.
pub fn render_rgba8(
    ctx: &GpuContext,
    scene: &Scene,
    params: DevelopParams,
) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error>> {
    scene.set_params(&ctx.queue, finalize(params, scene.width, scene.height));
    // Output dimensions follow the crop, not the source.
    let (w, h) = crop_output_dims(&params, scene.width, scene.height);

    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("export-target"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let pipeline = make_pipeline(&ctx.device, &scene.bind_group_layout, wgpu::TextureFormat::Rgba8Unorm);

    // Readback buffer with 256-byte-aligned rows.
    let unpadded = w * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = ((unpadded + align - 1) / align) * align;
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("export-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &scene.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &readback,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv()??;

    let mapped = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((unpadded * h) as usize);
    for row in 0..h {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    readback.unmap();
    Ok((w, h, pixels))
}

/// Export `scene`+`params` to a file, choosing the encoder by extension:
/// `.jpg`/`.jpeg` (8-bit, quality `jpeg_quality`), `.tif`/`.tiff`, else PNG.
pub fn export_image(
    ctx: &GpuContext,
    scene: &Scene,
    params: DevelopParams,
    path: &std::path::Path,
    jpeg_quality: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h, pixels) = render_rgba8(ctx, scene, params)?;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "jpg" | "jpeg" => {
            // JPEG has no alpha; drop it.
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for px in pixels.chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
            let file = std::fs::File::create(path)?;
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(
                std::io::BufWriter::new(file),
                jpeg_quality.clamp(1, 100),
            );
            enc.encode(&rgb, w, h, image::ExtendedColorType::Rgb8)?;
        }
        "tif" | "tiff" => image::save_buffer(path, &pixels, w, h, image::ColorType::Rgba8)?,
        _ => image::save_buffer(path, &pixels, w, h, image::ColorType::Rgba8)?,
    }
    Ok(())
}

/// Headless render of `scene` with `params` to an 8-bit sRGB PNG.
pub fn export_png(
    ctx: &GpuContext,
    scene: &Scene,
    params: DevelopParams,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h, pixels) = render_rgba8(ctx, scene, params)?;
    image::save_buffer(path, &pixels, w, h, image::ColorType::Rgba8)?;
    Ok(())
}
