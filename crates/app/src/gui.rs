//! egui-based desktop GUI: a Library grid of cataloged photos and a Develop
//! view with live sliders driving the wgpu pipeline. Decoding runs on a
//! background worker so the UI stays responsive; the develop preview is the
//! same shader as headless export (preview == export).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

use catalog::{Catalog, PhotoRow};
use gpu::{DevelopParams, GpuContext, PreviewTarget, Scene};
use recipe::Recipe;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const THUMB_LONG_EDGE: u32 = 256;
const PREVIEW_CAP: u32 = 1600;

/// Launch the GUI. With a `catalog` you get Library → Develop; with a `single`
/// file (path + starting recipe) it opens straight into Develop, no catalog.
pub fn run(catalog: Option<Catalog>, single: Option<(PathBuf, Recipe)>) {
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::Init(Some(InitData { catalog, single }));
    event_loop.run_app(&mut app).expect("run app");
}

struct InitData {
    catalog: Option<Catalog>,
    single: Option<(PathBuf, Recipe)>,
}

enum App {
    Init(Option<InitData>),
    Run(Gui),
}

// ---- background decode worker -------------------------------------------------

enum Job {
    Thumb { id: i64, path: PathBuf },
    Develop { id: i64, path: PathBuf },
}

enum Done {
    Thumb { id: i64, w: u32, h: u32, rgba: Vec<u8> },
    Develop { id: i64, w: u32, h: u32, samples: Vec<u16> },
    Failed { id: i64, err: String },
}

fn spawn_worker() -> (mpsc::Sender<Job>, mpsc::Receiver<Done>) {
    let (job_tx, job_rx) = mpsc::channel::<Job>();
    let (res_tx, res_rx) = mpsc::channel::<Done>();
    std::thread::spawn(move || {
        while let Ok(job) = job_rx.recv() {
            match job {
                Job::Thumb { id, path } => match raw_decode::decode(&path) {
                    Ok(raw) => {
                        let (w, h, rgba) = make_thumb(&raw, THUMB_LONG_EDGE);
                        let _ = res_tx.send(Done::Thumb { id, w, h, rgba });
                    }
                    Err(e) => {
                        let _ = res_tx.send(Done::Failed { id, err: e.to_string() });
                    }
                },
                Job::Develop { id, path } => match raw_decode::decode(&path) {
                    Ok(raw) => {
                        let _ = res_tx.send(Done::Develop {
                            id,
                            w: raw.width,
                            h: raw.height,
                            samples: raw.samples().to_vec(),
                        });
                    }
                    Err(e) => {
                        let _ = res_tx.send(Done::Failed { id, err: e.to_string() });
                    }
                },
            }
        }
    });
    (job_tx, res_rx)
}

fn lin_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Decode a RAW's linear samples to an sRGB 8-bit thumbnail (already oriented by
/// LibRaw). Returns (w, h, RGBA8).
fn make_thumb(raw: &raw_decode::RawImage, long_edge: u32) -> (u32, u32, Vec<u8>) {
    let (w, h) = (raw.width, raw.height);
    let s = raw.samples();
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
        for c in 0..3 {
            let v = lin_to_srgb((s[i * 3 + c] as f32) / 65535.0).clamp(0.0, 1.0);
            rgb[i * 3 + c] = (v * 255.0 + 0.5) as u8;
        }
    }
    let img = image::RgbImage::from_raw(w, h, rgb).expect("rgb buffer size");
    let scale = long_edge as f32 / w.max(h) as f32;
    let tw = ((w as f32 * scale).round() as u32).max(1);
    let th = ((h as f32 * scale).round() as u32).max(1);
    let thumb = image::imageops::resize(&img, tw, th, image::imageops::FilterType::Triangle);
    let rgba = image::DynamicImage::ImageRgb8(thumb).to_rgba8().into_raw();
    (tw, th, rgba)
}

fn preview_dims(w: u32, h: u32, cap: u32) -> (u32, u32) {
    let m = w.max(h);
    if m <= cap {
        (w, h)
    } else {
        let s = cap as f32 / m as f32;
        (((w as f32 * s) as u32).max(1), ((h as f32 * s) as u32).max(1))
    }
}

fn fit(content: egui::Vec2, avail: egui::Vec2) -> egui::Vec2 {
    let s = (avail.x / content.x).min(avail.y / content.y).max(0.001);
    egui::vec2(content.x * s, content.y * s)
}

// ---- the running app ----------------------------------------------------------

#[derive(PartialEq)]
enum Mode {
    Library,
    Develop,
}

struct Gui {
    window: Arc<Window>,
    ctx: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    preview_pipeline: Option<wgpu::RenderPipeline>,

    job_tx: mpsc::Sender<Job>,
    res_rx: mpsc::Receiver<Done>,
    in_flight: usize,

    catalog: Option<Catalog>,
    photos: Vec<PhotoRow>,
    thumbs: HashMap<i64, egui::TextureHandle>,
    requested: HashSet<i64>,

    mode: Mode,
    single_file: bool,
    status: String,

    // develop state
    dev_id: i64,
    dev_path: Option<PathBuf>,
    dev_loading: bool,
    dev_scene: Option<Scene>,
    dev_target: Option<PreviewTarget>,
    dev_tex_id: Option<egui::TextureId>,
    recipe: Recipe,
    preview_dirty: bool,
}

impl Gui {
    fn request_thumb(&mut self, id: i64, path: PathBuf) {
        if id < 0 || self.thumbs.contains_key(&id) || self.requested.contains(&id) {
            return;
        }
        self.requested.insert(id);
        self.in_flight += 1;
        let _ = self.job_tx.send(Job::Thumb { id, path });
    }

    fn open_develop(&mut self, id: i64, path: PathBuf) {
        self.mode = Mode::Develop;
        self.dev_id = id;
        self.dev_path = Some(path.clone());
        self.dev_loading = true;
        self.recipe = match &self.catalog {
            Some(cat) if id >= 0 => cat.master_recipe(id).unwrap_or_default(),
            _ => Recipe::default(),
        };
        self.preview_dirty = true;
        self.in_flight += 1;
        let _ = self.job_tx.send(Job::Develop { id, path });
        let name = self
            .dev_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.status = format!("Loading {name}…");
    }

    fn drain_results(&mut self) {
        while let Ok(done) = self.res_rx.try_recv() {
            self.in_flight = self.in_flight.saturating_sub(1);
            match done {
                Done::Thumb { id, w, h, rgba } => {
                    let img =
                        egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                    let handle =
                        self.egui_ctx
                            .load_texture(format!("thumb{id}"), img, egui::TextureOptions::LINEAR);
                    self.thumbs.insert(id, handle);
                }
                Done::Develop { id, w, h, samples } => {
                    // Ignore a stale decode if the user already moved on.
                    if id != self.dev_id {
                        continue;
                    }
                    let scene = Scene::from_linear_rgb16(&self.ctx, w, h, &samples);
                    if self.preview_pipeline.is_none() {
                        self.preview_pipeline = Some(gpu::make_pipeline(
                            &self.ctx.device,
                            &scene.bind_group_layout,
                            wgpu::TextureFormat::Rgba8Unorm,
                        ));
                    }
                    let (pw, ph) = preview_dims(w, h, PREVIEW_CAP);
                    if let Some(old) = self.dev_tex_id.take() {
                        self.egui_renderer.free_texture(&old);
                    }
                    let target = PreviewTarget::new(&self.ctx, pw, ph);
                    let tex_id = self.egui_renderer.register_native_texture(
                        &self.ctx.device,
                        &target.sample_view,
                        wgpu::FilterMode::Linear,
                    );
                    self.dev_scene = Some(scene);
                    self.dev_target = Some(target);
                    self.dev_tex_id = Some(tex_id);
                    self.dev_loading = false;
                    self.preview_dirty = true;
                    self.status.clear();
                }
                Done::Failed { id, err } => {
                    if id == self.dev_id {
                        self.dev_loading = false;
                    }
                    self.status = format!("Couldn't decode: {err}");
                }
            }
        }
    }

    fn params(&self) -> DevelopParams {
        DevelopParams::from(&self.recipe)
    }

    fn ensure_preview(&mut self) {
        let ready = self.preview_dirty
            && self.dev_scene.is_some()
            && self.dev_target.is_some()
            && self.preview_pipeline.is_some();
        if !ready {
            return;
        }
        let params = self.params();
        {
            let scene = self.dev_scene.as_ref().unwrap();
            let target = self.dev_target.as_ref().unwrap();
            let pipeline = self.preview_pipeline.as_ref().unwrap();
            gpu::render_to_target(&self.ctx, pipeline, scene, params, target);
        }
        self.preview_dirty = false;
    }

    fn do_import(&mut self) {
        let Some(dir) = rfd::FileDialog::new().set_title("Import a folder of RAWs").pick_folder()
        else {
            return;
        };
        if let Some(cat) = &mut self.catalog {
            match cat.import_folder(&dir) {
                Ok(s) => {
                    self.status =
                        format!("Imported {} · skipped {} · failed {}", s.imported, s.skipped, s.failed)
                }
                Err(e) => {
                    self.status = format!("Import failed: {e}");
                    return;
                }
            }
        } else {
            self.status = "No catalog open.".into();
            return;
        }
        if let Some(cat) = &self.catalog {
            self.photos = cat.list_photos().unwrap_or_default();
        }
    }

    fn save_recipe(&mut self) {
        if self.dev_id >= 0 {
            if let Some(cat) = &mut self.catalog {
                self.status = match cat.save_master_recipe(self.dev_id, &self.recipe, "Develop") {
                    Ok(()) => "Saved.".into(),
                    Err(e) => format!("Save failed: {e}"),
                };
                return;
            }
        }
        self.status = "Nothing to save (no catalog).".into();
    }

    fn export_dialog(&mut self) {
        let Some(scene) = &self.dev_scene else {
            return;
        };
        let suggested = self
            .dev_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(|s| format!("{}.png", s.to_string_lossy()))
            .unwrap_or_else(|| "export.png".into());
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name(suggested)
            .add_filter("PNG", &["png"])
            .save_file()
        {
            self.status = match gpu::export_png(&self.ctx, scene, self.params(), &path) {
                Ok(()) => format!("Exported {}", path.display()),
                Err(e) => format!("Export failed: {e}"),
            };
        }
    }

    // ---- UI -------------------------------------------------------------------

    fn library_ui(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("lib-top").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("AdobeMaybeLight");
                ui.separator();
                if ui.button("📁  Import folder…").clicked() {
                    self.do_import();
                }
                ui.label(format!("{} photos", self.photos.len()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(&self.status);
                });
            });
            ui.add_space(4.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let photos = self.photos.clone();
            if photos.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("No photos yet. Click “Import folder…” to catalog a folder of RAWs.");
                });
                return;
            }
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    for p in &photos {
                        self.request_thumb(p.id, p.path.clone());
                        let clicked = ui
                            .allocate_ui(egui::vec2(184.0, 196.0), |ui| {
                                ui.vertical_centered(|ui| {
                                    let clicked = if let Some(h) = self.thumbs.get(&p.id) {
                                        let size = fit(h.size_vec2(), egui::vec2(176.0, 150.0));
                                        ui.add(egui::ImageButton::new(egui::Image::new(
                                            egui::load::SizedTexture::new(h.id(), size),
                                        )))
                                        .clicked()
                                    } else {
                                        ui.allocate_ui(egui::vec2(176.0, 150.0), |ui| {
                                            ui.centered_and_justified(|ui| ui.spinner());
                                        });
                                        false
                                    };
                                    ui.label(egui::RichText::new(&p.filename).small());
                                    clicked
                                })
                                .inner
                            })
                            .inner;
                        if clicked {
                            self.open_develop(p.id, p.path.clone());
                        }
                    }
                });
                ui.add_space(8.0);
            });
        });
    }

    fn develop_ui(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("dev-top").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if !self.single_file && ui.button("←  Library").clicked() {
                    self.mode = Mode::Library;
                }
                ui.separator();
                if let Some(p) = &self.dev_path {
                    ui.label(p.file_name().unwrap_or_default().to_string_lossy());
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(&self.status);
                });
            });
            ui.add_space(4.0);
        });

        egui::SidePanel::right("dev-panel")
            .resizable(false)
            .exact_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut changed = false;
                    let g = &mut self.recipe.globals;

                    ui.add_space(6.0);
                    ui.strong("White Balance");
                    changed |= ui.checkbox(&mut g.white_balance.as_shot, "As shot").changed();
                    ui.add_enabled_ui(!g.white_balance.as_shot, |ui| {
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut g.white_balance.temp_k, 2000.0..=12000.0)
                                    .text("Temp (K)"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut g.white_balance.tint, -100.0..=100.0)
                                    .text("Tint"),
                            )
                            .changed();
                    });

                    ui.add_space(8.0);
                    ui.strong("Tone");
                    changed |= ui
                        .add(egui::Slider::new(&mut g.tone.exposure_ev, -5.0..=5.0).text("Exposure"))
                        .changed();
                    changed |= slider(ui, &mut g.tone.contrast, "Contrast");
                    changed |= slider(ui, &mut g.tone.highlights, "Highlights");
                    changed |= slider(ui, &mut g.tone.shadows, "Shadows");
                    changed |= slider(ui, &mut g.tone.whites, "Whites");
                    changed |= slider(ui, &mut g.tone.blacks, "Blacks");

                    ui.add_space(8.0);
                    ui.strong("Presence");
                    changed |= slider(ui, &mut g.presence.dehaze, "Dehaze");
                    changed |= slider(ui, &mut g.presence.vibrance, "Vibrance");
                    changed |= slider(ui, &mut g.presence.saturation, "Saturation");

                    ui.add_space(8.0);
                    egui::CollapsingHeader::new("Color Mixer (HSL)").show(ui, |ui| {
                        ui.label(egui::RichText::new("Hue").weak());
                        changed |= band_sliders(ui, &mut g.hsl.hue);
                        ui.label(egui::RichText::new("Saturation").weak());
                        changed |= band_sliders(ui, &mut g.hsl.saturation);
                        ui.label(egui::RichText::new("Luminance").weak());
                        changed |= band_sliders(ui, &mut g.hsl.luminance);
                    });

                    ui.add_space(8.0);
                    egui::CollapsingHeader::new("Effects").show(ui, |ui| {
                        ui.label(egui::RichText::new("Post-crop vignette").weak());
                        changed |= slider(ui, &mut g.effects.vignette_amount, "Amount");
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut g.effects.vignette_midpoint, 0.0..=100.0)
                                    .text("Midpoint"),
                            )
                            .changed();
                        changed |= ui
                            .add(
                                egui::Slider::new(&mut g.effects.vignette_feather, 0.0..=100.0)
                                    .text("Feather"),
                            )
                            .changed();
                        ui.label(egui::RichText::new("Grain").weak());
                        changed |= ui
                            .add(egui::Slider::new(&mut g.effects.grain_amount, 0.0..=100.0).text("Amount"))
                            .changed();
                        changed |= ui
                            .add(egui::Slider::new(&mut g.effects.grain_size, 0.0..=100.0).text("Size"))
                            .changed();
                    });

                    ui.add_space(12.0);
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Reset").clicked() {
                            self.recipe = Recipe::default();
                            changed = true;
                        }
                        if ui.button("💾  Save").clicked() {
                            self.save_recipe();
                        }
                        if ui.button("⬇  Export…").clicked() {
                            self.export_dialog();
                        }
                    });

                    if changed {
                        self.preview_dirty = true;
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.dev_loading {
                ui.centered_and_justified(|ui| ui.spinner());
                return;
            }
            if let (Some(id), Some(target)) = (self.dev_tex_id, self.dev_target.as_ref()) {
                let content = egui::vec2(target.width as f32, target.height as f32);
                let size = fit(content, ui.available_size());
                ui.centered_and_justified(|ui| {
                    ui.add(egui::Image::new(egui::load::SizedTexture::new(id, size)));
                });
            }
        });
    }

    fn redraw(&mut self) {
        self.drain_results();
        self.ensure_preview();

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ectx = self.egui_ctx.clone();
        let full = ectx.run(raw_input, |c| match self.mode {
            Mode::Library => self.library_ui(c),
            Mode::Develop => self.develop_ui(c),
        });
        self.egui_state
            .handle_platform_output(&self.window, full.platform_output);

        let tris = ectx.tessellate(full.shapes, full.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: full.pixels_per_point,
        };
        for (id, delta) in &full.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.ctx.device, &self.ctx.queue, *id, delta);
        }

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.ctx.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        self.egui_renderer
            .update_buffers(&self.ctx.device, &self.ctx.queue, &mut enc, &tris, &screen);
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.05,
                                g: 0.05,
                                b: 0.06,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, &tris, &screen);
        }
        self.ctx.queue.submit([enc.finish()]);
        frame.present();
        for id in &full.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

/// A 0-centered [-100, 100] develop slider; returns whether it changed.
fn slider(ui: &mut egui::Ui, v: &mut f32, label: &str) -> bool {
    ui.add(egui::Slider::new(v, -100.0..=100.0).text(label)).changed()
}

const HSL_BANDS: [&str; 8] =
    ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"];

/// Eight per-band [-100, 100] sliders; returns whether any changed.
fn band_sliders(ui: &mut egui::Ui, bands: &mut [f32; 8]) -> bool {
    let mut changed = false;
    for (i, name) in HSL_BANDS.iter().enumerate() {
        changed |= ui.add(egui::Slider::new(&mut bands[i], -100.0..=100.0).text(*name)).changed();
    }
    changed
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let App::Init(data) = self else { return };
        let data = data.take().expect("init data consumed");

        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("AdobeMaybeLight")
                    .with_inner_size(LogicalSize::new(1280.0, 820.0)),
            )
            .expect("create window"),
        );

        let ctx = pollster::block_on(GpuContext::new(None));
        let surface = ctx.instance.create_surface(window.clone()).expect("surface");
        let caps = surface.get_capabilities(&ctx.adapter);
        // egui expects an sRGB framebuffer; pick one so its UI colors are right.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&ctx.device, &config);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&ctx.device, format, None, 1, false);

        let (job_tx, res_rx) = spawn_worker();
        let single = data.single.is_some();
        let photos = data
            .catalog
            .as_ref()
            .map(|c| c.list_photos().unwrap_or_default())
            .unwrap_or_default();

        let mut gui = Gui {
            window,
            ctx,
            surface,
            config,
            egui_ctx,
            egui_state,
            egui_renderer,
            preview_pipeline: None,
            job_tx,
            res_rx,
            in_flight: 0,
            catalog: data.catalog,
            photos,
            thumbs: HashMap::new(),
            requested: HashSet::new(),
            mode: if single { Mode::Develop } else { Mode::Library },
            single_file: single,
            status: String::new(),
            dev_id: -1,
            dev_path: None,
            dev_loading: false,
            dev_scene: None,
            dev_target: None,
            dev_tex_id: None,
            recipe: Recipe::default(),
            preview_dirty: false,
        };

        if let Some((path, rec)) = data.single {
            gui.open_develop(-1, path);
            gui.recipe = rec; // honor a --recipe passed on the command line
        }

        *self = App::Run(gui);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let App::Run(g) = self else { return };
        let resp = g.egui_state.on_window_event(&g.window, &event);
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(sz) => {
                g.config.width = sz.width.max(1);
                g.config.height = sz.height.max(1);
                g.surface.configure(&g.ctx.device, &g.config);
                g.window.request_redraw();
            }
            WindowEvent::RedrawRequested => g.redraw(),
            _ => {}
        }
        if resp.repaint {
            g.window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        // Keep the loop alive while background decodes are in flight.
        if let App::Run(g) = self {
            if g.in_flight > 0 {
                g.window.request_redraw();
            }
        }
    }
}
