use crate::{graphics::*, prelude::*, world::World};
use egui::Context;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow},
    window::{Theme, Window, WindowId},
};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use {futures::channel::oneshot::Receiver, wasm_bindgen::prelude::*, web_time::Instant};

pub trait State {
    fn initialize(&mut self, _world: &mut World) {}
    fn receive_event(&mut self, _world: &mut World, _event: &WindowEvent) {}
    fn update(&mut self, _world: &mut World) {}
    fn ui(&mut self, _world: &mut World, _ui: &Context) {}
}

struct WindowData {
    window: Arc<Window>,
    renderer: Option<Renderer>,
    ui_state: egui_winit::State,
    ui_context: Context,
    last_frame_instant: Option<Instant>,
    frame_start_instant: Option<Instant>,
    app_start_instant: Option<Instant>,
    frame_counter: u32,
    #[cfg(target_arch = "wasm32")]
    renderer_rx: Option<Receiver<Renderer>>,
}

pub fn launch(state: impl State + 'static) -> Result<(), winit::error::EventLoopError> {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        let _ = console_log::init();
    }

    let event_loop = winit::event_loop::EventLoop::builder().build()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut Runner::new(state))
}

#[cfg(not(target_arch = "wasm32"))]
fn create_window_data(window: Arc<Window>) -> WindowData {
    let size = window.inner_size();
    let ctx = Context::default();

    WindowData {
        renderer: Some(pollster::block_on(async {
            create_renderer(window.clone(), size.width, size.height).await
        })),
        ui_state: egui_winit::State::new(
            ctx.clone(),
            ctx.viewport_id(),
            &window,
            Some(window.scale_factor() as _),
            Some(Theme::Dark),
            None,
        ),
        ui_context: ctx,
        last_frame_instant: None,
        window,
        frame_start_instant: None,
        app_start_instant: Some(Instant::now()),
        frame_counter: 0,
    }
}

#[cfg(target_arch = "wasm32")]
fn create_window_data(window: Arc<Window>) -> WindowData {
    let size = window.inner_size();
    let ctx = Context::default();
    ctx.set_pixels_per_point(window.scale_factor() as f32);

    let (tx, rx) = futures::channel::oneshot::channel();
    let win = window.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let _ = tx.send(create_renderer(win, size.width, size.height).await);
    });

    WindowData {
        renderer: None,
        ui_state: egui_winit::State::new(
            ctx.clone(),
            ctx.viewport_id(),
            &window,
            Some(window.scale_factor() as _),
            Some(Theme::Dark),
            None,
        ),
        ui_context: ctx,
        last_frame_instant: None,
        frame_start_instant: None,
        app_start_instant: Some(Instant::now()),
        frame_counter: 0,
        window,
        renderer_rx: Some(rx),
    }
}

fn step(world: &mut World, window_data: &mut WindowData, state: &mut Box<dyn State>) {
    #[cfg(target_arch = "wasm32")]
    if let Some(rx) = &mut window_data.renderer_rx {
        if let Ok(Some(renderer)) = rx.try_recv() {
            window_data.renderer = Some(renderer);
            window_data.renderer_rx = None;
        }
    }

    let now = Instant::now();
    world.resources.delta_time = window_data
        .last_frame_instant
        .map_or(0.0, |t| (now - t).as_secs_f32());
    window_data.last_frame_instant = Some(now);
    if window_data.frame_start_instant.is_none() {
        window_data.frame_start_instant = Some(now);
    }
    if let Some(app_start) = window_data.app_start_instant {
        world.resources.uptime_milliseconds = (now - app_start).as_millis();
    }
    gamepad_system(world);
    fps_timer_system(world, window_data, now);
    state.update(world);
    reset_mouse_system(world);

    if let Some(renderer) = &mut window_data.renderer {
        let input = window_data.ui_state.take_egui_input(&window_data.window);
        window_data.ui_context.begin_pass(input);
        state.ui(world, &window_data.ui_context);
        let output = window_data.ui_context.end_pass();
        let size = window_data.window.inner_size();

        render_system(
            renderer,
            world,
            egui_wgpu::ScreenDescriptor {
                size_in_pixels: [size.width, size.height],
                pixels_per_point: window_data.window.scale_factor() as f32,
            },
            window_data
                .ui_context
                .tessellate(output.shapes, output.pixels_per_point),
            output.textures_delta,
        );
    }
}

fn gamepad_system(world: &mut World) {
    if world.resources.gilrs.is_none() {
        match gilrs::Gilrs::new() {
            Ok(gilrs) => {
                world.resources.gilrs = Some(gilrs);
            }
            Err(error) => {
                log::error!("Failed to initialize controller support: {error}");
            }
        }
    }
    let gilrs = world.resources.gilrs.as_mut().unwrap();
    while let Some(gilrs::Event { id, .. }) = gilrs.next_event() {
        world.resources.active_gamepad = Some(id);
    }
}

fn fps_timer_system(world: &mut World, window_data: &mut WindowData, now: Instant) {
    window_data.frame_counter += 1;
    match window_data.frame_start_instant.as_ref() {
        Some(start) => {
            if (now - *start).as_secs_f32() >= 1.0 {
                world.resources.frames_per_second = window_data.frame_counter as f32;
                window_data.frame_counter = 0;
                window_data.frame_start_instant = Some(now);
            }
        }
        None => {
            window_data.frame_start_instant = Some(now);
        }
    }
}

fn process_window_event(
    world: &mut World,
    window_data: &mut WindowData,
    state: &mut Box<dyn State>,
    event: &WindowEvent,
) {
    if window_data
        .ui_state
        .on_window_event(&window_data.window, event)
        .consumed
    {
        return;
    }

    match event {
        WindowEvent::Resized(size) => {
            if let Some(renderer) = &mut window_data.renderer {
                resize_renderer_system(renderer, size.width.max(1), size.height.max(1));
                world.resources.viewport_width = size.width;
                world.resources.viewport_height = size.height;
                world.resources.window_center =
                    nalgebra_glm::Vec2::new(size.width as f32 / 2.0, size.height as f32 / 2.0);
            }
        }
        _ => {
            update_keyboard_system(world, event);
            mouse_system(world, event);
            gamepad_system(world);
            state.receive_event(world, event);
        }
    }
}

struct Runner {
    world: Option<World>,
    window_data: Option<WindowData>,
    state: Option<Box<dyn State>>,
}

impl Runner {
    fn new(state: impl State + 'static) -> Self {
        Self {
            world: None,
            window_data: None,
            state: Some(Box::new(state)),
        }
    }
}

impl ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Ok(window) = create_window(event_loop) else {
            return;
        };
        let Some(mut state) = self.state.take() else {
            return;
        };

        let mut world = World::default();
        let PhysicalSize { width, height } = window.inner_size();
        world.resources.viewport_width = width;
        world.resources.viewport_height = height;
        state.initialize(&mut world);

        let window_data = create_window_data(window.clone());

        self.world = Some(world);
        self.window_data = Some(window_data);
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Self {
            world: Some(world),
            window_data: Some(window_data),
            state: Some(state),
        } = self
        else {
            return;
        };

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key:
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                step(world, window_data, state);
            }
            event => {
                process_window_event(world, window_data, state, &event);
            }
        }

        window_data.window.request_redraw();
    }
}

fn create_window(
    event_loop: &ActiveEventLoop,
) -> Result<Arc<Window>, winit::error::EventLoopError> {
    let attributes = Window::default_attributes();

    #[cfg(not(target_arch = "wasm32"))]
    let attributes = attributes.with_title("Nightshade");

    #[cfg(target_arch = "wasm32")]
    let attributes = {
        use winit::platform::web::WindowAttributesExtWebSys;
        attributes.with_canvas(Some(
            web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("canvas"))
                .and_then(|c| c.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                .expect("Failed to get canvas element"),
        ))
    };

    Ok(event_loop.create_window(attributes).map(Arc::new)?)
}
