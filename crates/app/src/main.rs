//! AdobeMaybeLight CLI / app entry point.
//!
//!   aml import <DIR> [--catalog db]     # catalog a folder of RAWs
//!   aml <RAW_FILE> [--recipe r.json]    # interactive develop preview
//!   aml <RAW_FILE> --export out.png [--recipe r.json]   # headless export
//!   aml --selftest out.png              # GPU pipeline self-test
//!
//! Preview keys:  Up/Down = exposure ±0.25 stop,  [ / ] = warm/cool WB,
//!                S = save current look to the Desktop,  Esc = quit.

use std::sync::Arc;

use gpu::{DevelopParams, GpuContext, Scene};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const RAW_EXTS: &[&str] = &[
    "arw", "cr2", "cr3", "nef", "dng", "raf", "rw2", "orf", "pef", "srw", "raw",
];

fn main() {
    // Ignore the process-serial-number arg Finder passes to GUI apps.
    let args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| !a.starts_with("-psn_"))
        .collect();

    // --selftest: synthetic linear gradient -> develop -> PNG. Proves the GPU
    // path (upload -> WGSL -> readback -> encode) without a RAW file.
    if args.first().map(String::as_str) == Some("--selftest") {
        let out = args.get(1).cloned().unwrap_or_else(|| "selftest.png".into());
        let (w, h) = (512u32, 256u32);
        let mut samples = vec![0u16; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                samples[i] = ((x as f32 / w as f32) * 65535.0) as u16; // R ramp
                samples[i + 1] = ((y as f32 / h as f32) * 65535.0) as u16; // G ramp
                samples[i + 2] = 16384; // constant B (linear)
            }
        }
        let ctx = pollster::block_on(GpuContext::new(None));
        let scene = Scene::from_linear_rgb16(&ctx, w, h, &samples);
        gpu::export_png(&ctx, &scene, DevelopParams::default(), std::path::Path::new(&out))
            .expect("export failed");
        println!("wrote {out}");
        return;
    }

    // `import <DIR>`: catalog a folder of RAWs into the SQLite DB.
    if args.first().map(String::as_str) == Some("import") {
        let mut dir: Option<String> = None;
        let mut db: Option<String> = None;
        let mut it = args[1..].iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--catalog" => db = it.next().cloned(),
                other if !other.starts_with('-') => dir = Some(other.to_string()),
                _ => {}
            }
        }
        let dir = dir.unwrap_or_else(|| {
            eprintln!("usage: aml import <DIR> [--catalog db]");
            std::process::exit(2);
        });
        let db = db.unwrap_or_else(default_catalog_path);
        let mut cat = catalog::Catalog::open(&db).expect("open catalog");
        println!("importing {dir} -> {db} ...");
        let s = cat.import_folder(&dir).expect("import failed");
        println!(
            "scanned {} · imported {} · skipped {} · failed {} · total photos {}",
            s.scanned, s.imported, s.skipped, s.failed,
            cat.photo_count().unwrap_or(0)
        );
        return;
    }

    // Positional RAW path, optional --export <png> and --recipe <json>.
    let mut export: Option<String> = None;
    let mut recipe_file: Option<String> = None;
    let mut positional: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--export" => export = it.next().cloned(),
            "--recipe" => recipe_file = it.next().cloned(),
            other if !other.starts_with('-') => positional = Some(other.to_string()),
            _ => {}
        }
    }

    // Load a recipe (or defaults) and convert to GPU params.
    let recipe = match &recipe_file {
        Some(p) => {
            let txt = std::fs::read_to_string(p).expect("read recipe");
            recipe::Recipe::from_json(&txt).expect("parse recipe")
        }
        None => recipe::Recipe::default(),
    };
    let params = DevelopParams::from(&recipe);

    // No file on the command line (e.g. launched from Finder) -> ask for one.
    let raw_path = match positional {
        Some(p) => p,
        None => match rfd::FileDialog::new()
            .set_title("Open a RAW photo")
            .add_filter("RAW", RAW_EXTS)
            .pick_file()
        {
            Some(p) => p.to_string_lossy().into_owned(),
            None => return, // user cancelled
        },
    };

    println!("decoding {raw_path} ...");
    let raw = match raw_decode::decode(&raw_path) {
        Ok(r) => r,
        Err(e) => {
            // From Finder there's no console, so surface the error visibly.
            rfd::MessageDialog::new()
                .set_title("Couldn't open RAW")
                .set_description(format!("{raw_path}\n\n{e}"))
                .set_level(rfd::MessageLevel::Error)
                .show();
            std::process::exit(1);
        }
    };
    println!("  {}x{} linear RGB16", raw.width, raw.height);

    if let Some(out) = export {
        let ctx = pollster::block_on(GpuContext::new(None));
        let scene = Scene::from_raw(&ctx, &raw);
        gpu::export_png(&ctx, &scene, params, std::path::Path::new(&out))
            .expect("export failed");
        println!("wrote {out}");
        return;
    }

    let event_loop = EventLoop::new().unwrap();
    let mut app = App::Loading { raw: Some(raw), params };
    event_loop.run_app(&mut app).unwrap();
}

/// Default catalog DB location (~/Pictures/AdobeMaybeLight.db, else temp).
fn default_catalog_path() -> String {
    std::env::var_os("HOME")
        .map(|h| std::path::Path::new(&h).join("Pictures"))
        .filter(|p| p.is_dir())
        .unwrap_or_else(std::env::temp_dir)
        .join("AdobeMaybeLight.db")
        .to_string_lossy()
        .into_owned()
}

/// Path on the user's Desktop, falling back to the temp dir.
fn desktop_path(name: &str) -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(|h| std::path::Path::new(&h).join("Desktop"))
        .filter(|p| p.is_dir())
        .unwrap_or_else(std::env::temp_dir)
        .join(name)
}

enum App {
    Loading { raw: Option<raw_decode::RawImage>, params: DevelopParams },
    Running(State),
}

struct State {
    window: Arc<Window>,
    ctx: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    scene: Scene,
    params: DevelopParams,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let App::Loading { raw, params } = self else { return };
        let raw = raw.take().expect("raw already consumed");
        let params = *params;

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("AdobeMaybeLight — spike"))
                .unwrap(),
        );

        let ctx = pollster::block_on(GpuContext::new(None));
        let surface = ctx.instance.create_surface(window.clone()).unwrap();

        let caps = surface.get_capabilities(&ctx.adapter);
        // Use a non-sRGB surface so our shader's explicit sRGB OETF is correct.
        let format = caps.formats[0].remove_srgb_suffix();
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

        let scene = Scene::from_raw(&ctx, &raw);
        scene.set_params(&ctx.queue, params); // start from the loaded recipe
        let pipeline = gpu::make_pipeline(&ctx.device, &scene.bind_group_layout, format);

        *self = App::Running(State {
            window,
            ctx,
            surface,
            config,
            pipeline,
            scene,
            params,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let App::Running(st) = self else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(sz) => {
                st.config.width = sz.width.max(1);
                st.config.height = sz.height.max(1);
                st.surface.configure(&st.ctx.device, &st.config);
                st.window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key, state: ElementState::Pressed, .. },
                ..
            } => {
                let mut dirty = true;
                match logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::ArrowUp) => st.params.exposure += 0.25,
                    Key::Named(NamedKey::ArrowDown) => st.params.exposure -= 0.25,
                    Key::Character("]") => { st.params.wb_r += 0.02; st.params.wb_b -= 0.02; }
                    Key::Character("[") => { st.params.wb_r -= 0.02; st.params.wb_b += 0.02; }
                    Key::Character("s") | Key::Character("S") => {
                        let out = desktop_path("AdobeMaybeLight-export.png");
                        gpu::export_png(&st.ctx, &st.scene, st.params, &out).expect("export failed");
                        println!("wrote {} (exposure {:+.2})", out.display(), st.params.exposure);
                        dirty = false;
                    }
                    _ => dirty = false,
                }
                if dirty {
                    st.scene.set_params(&st.ctx.queue, st.params);
                    st.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => st.render(),
            _ => {}
        }
    }
}

impl State {
    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.ctx.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("preview"),
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.scene.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.ctx.queue.submit([enc.finish()]);
        frame.present();
    }
}
