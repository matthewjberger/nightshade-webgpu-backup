use nalgebra_glm::{vec4, Mat4, Vec3, Vec4};
use wgpu::{include_wgsl, util::DeviceExt};

use crate::prelude::*;
use std::collections::HashMap;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

// === Resource Types ===
#[derive(Debug)]
pub enum Resource {
    Buffer(wgpu::Buffer),
    BindGroup(wgpu::BindGroup),
    Pipeline(wgpu::RenderPipeline),
    BindGroupLayout(wgpu::BindGroupLayout),
    Texture(wgpu::Texture),
    TextureView(wgpu::TextureView),
    Sampler(wgpu::Sampler),
    MeshCommands(HashMap<String, Vec<DrawCommand>>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(String);

// === Pass Types ===
pub struct RenderPass {
    pub name: String,
    pub inputs: Vec<ResourceId>,
    pub outputs: Vec<ResourceId>,
    pub execute: Box<RenderPassFn>,
}

type RenderPassFn = dyn Fn(&mut wgpu::RenderPass, &Renderer, &[ResourceId], &[ResourceId]);

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub resources: HashMap<String, Resource>,
    pub egui_renderer: egui_wgpu::Renderer,
    pub render_passes: Vec<RenderPass>,
}

// === Resource Functions ===
pub fn add_resource(renderer: &mut Renderer, id: ResourceId, resource: Resource) {
    renderer.resources.insert(id.0, resource);
}

pub fn get_resource<'a>(renderer: &'a Renderer, id: &ResourceId) -> Option<&'a Resource> {
    renderer.resources.get(&id.0)
}

pub fn get_texture_view<'a>(
    renderer: &'a Renderer,
    id: &ResourceId,
) -> Option<&'a wgpu::TextureView> {
    match get_resource(renderer, id) {
        Some(Resource::TextureView(view)) => Some(view),
        _ => None,
    }
}

// === Render Pass Functions ===
pub fn add_render_pass(renderer: &mut Renderer, pass: RenderPass) {
    renderer.render_passes.push(pass);
}

fn sort_passes(renderer: &Renderer) -> Vec<usize> {
    let mut sorted = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut temp = std::collections::HashSet::new();

    for i in 0..renderer.render_passes.len() {
        visit(i, renderer, &mut visited, &mut temp, &mut sorted);
    }

    sorted
}

fn visit(
    pass_idx: usize,
    renderer: &Renderer,
    visited: &mut std::collections::HashSet<usize>,
    temp: &mut std::collections::HashSet<usize>,
    sorted: &mut Vec<usize>,
) {
    if temp.contains(&pass_idx) {
        panic!("Cycle detected in render passes!");
    }
    if visited.contains(&pass_idx) {
        return;
    }

    temp.insert(pass_idx);

    for other_idx in 0..renderer.render_passes.len() {
        if pass_idx == other_idx {
            continue;
        }

        let pass = &renderer.render_passes[pass_idx];
        let other = &renderer.render_passes[other_idx];

        // If other writes to any of our inputs, it must come before us
        if other.outputs.iter().any(|out| pass.inputs.contains(out)) {
            visit(other_idx, renderer, visited, temp, sorted);
        }
    }

    temp.remove(&pass_idx);
    visited.insert(pass_idx);
    sorted.push(pass_idx);
}

// === Core Renderer Functions ===
pub async fn create_renderer(
    window: impl Into<wgpu::SurfaceTarget<'static>>,
    width: u32,
    height: u32,
) -> Renderer {
    let (device, queue, surface, config) = setup_device(window, width, height).await;
    let depth_view = create_depth_texture(&device, width, height);
    let egui_renderer =
        egui_wgpu::Renderer::new(&device, config.format, Some(DEPTH_FORMAT), 1, false);

    let mut renderer = Renderer {
        device,
        queue,
        surface,
        config,
        resources: HashMap::new(),
        egui_renderer,
        render_passes: Vec::new(),
    };

    // Store depth view as a resource
    add_resource(
        &mut renderer,
        ResourceId(String::from("depth")),
        Resource::TextureView(depth_view),
    );

    // Create render passes
    create_sky_pass(&mut renderer);
    create_scene_pass(&mut renderer);
    create_grid_pass(&mut renderer);

    renderer
}

pub fn resize_renderer_system(renderer: &mut Renderer, width: u32, height: u32) {
    renderer.config.width = width;
    renderer.config.height = height;
    renderer
        .surface
        .configure(&renderer.device, &renderer.config);

    let depth_view = create_depth_texture(&renderer.device, width, height);
    add_resource(
        renderer,
        ResourceId(String::from("depth")),
        Resource::TextureView(depth_view),
    );
}

async fn setup_device(
    window: impl Into<wgpu::SurfaceTarget<'static>>,
    width: u32,
    height: u32,
) -> (
    wgpu::Device,
    wgpu::Queue,
    wgpu::Surface<'static>,
    wgpu::SurfaceConfiguration,
) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all),
        ..Default::default()
    });

    let surface = instance.create_surface(window).unwrap();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to request adapter!");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WGPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .await
        .expect("Failed to request device!");

    let surface_caps = surface.get_capabilities(&adapter);
    let format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| !f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    surface.configure(&device, &config);

    (device, queue, surface, config)
}

fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

pub fn render_system(
    renderer: &mut Renderer,
    world: &World,
    screen_descriptor: egui_wgpu::ScreenDescriptor,
    paint_jobs: Vec<egui::epaint::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
) {
    // Update egui textures
    for (id, image_delta) in &textures_delta.set {
        renderer
            .egui_renderer
            .update_texture(&renderer.device, &renderer.queue, *id, image_delta);
    }
    for id in &textures_delta.free {
        renderer.egui_renderer.free_texture(id);
    }

    // Get frame
    let frame = match renderer.surface.get_current_texture() {
        Ok(frame) => frame,
        Err(_) => return,
    };

    update_sky_uniforms(renderer, world);
    update_grid_uniforms(renderer, world);
    update_scene_uniforms(renderer, world);

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = renderer
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

    renderer.egui_renderer.update_buffers(
        &renderer.device,
        &renderer.queue,
        &mut encoder,
        &paint_jobs,
        &screen_descriptor,
    );

    let depth_view = get_texture_view(renderer, &ResourceId(String::from("depth")))
        .expect("Depth texture view not found");

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Main Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.19,
                        g: 0.24,
                        b: 0.42,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Execute passes in dependency order
        let pass_order = sort_passes(renderer);
        for &pass_idx in &pass_order {
            let pass = &renderer.render_passes[pass_idx];
            (pass.execute)(&mut render_pass, renderer, &pass.inputs, &pass.outputs);
        }

        // Render egui last
        renderer.egui_renderer.render(
            &mut render_pass.forget_lifetime(),
            &paint_jobs,
            &screen_descriptor,
        );
    }

    renderer.queue.submit(std::iter::once(encoder.finish()));
    frame.present();
}

// Sky

const SKY_IMAGE_SIZE: u32 = 256;

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniform {
    proj: Mat4,
    proj_inv: Mat4,
    view: Mat4,
    cam_pos: Vec4,
}

pub fn create_sky_pass(renderer: &mut Renderer) {
    // Create uniform buffer
    let uniform_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sky Uniform Buffer"),
        size: std::mem::size_of::<SkyUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create sky texture resources
    let (texture, view, sampler) = create_sky_texture_resources(&renderer.device, &renderer.queue);

    // Create bind group layout
    let bind_group_layout =
        renderer
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sky Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

    // Create bind group
    let bind_group = renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

    // Create pipeline
    let shader = renderer
        .device
        .create_shader_module(include_wgsl!("assets/shaders/sky.wgsl"));
    let pipeline_layout = renderer
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = renderer
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_sky",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sky",
                targets: &[Some(wgpu::ColorTargetState {
                    format: renderer.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    // Store resources
    add_resource(
        renderer,
        ResourceId(String::from("sky_uniform")),
        Resource::Buffer(uniform_buffer),
    );
    add_resource(
        renderer,
        ResourceId(String::from("sky_texture")),
        Resource::Texture(texture),
    );
    add_resource(
        renderer,
        ResourceId(String::from("sky_view")),
        Resource::TextureView(view),
    );
    add_resource(
        renderer,
        ResourceId(String::from("sky_sampler")),
        Resource::Sampler(sampler),
    );
    add_resource(
        renderer,
        ResourceId(String::from("sky_bind_group")),
        Resource::BindGroup(bind_group),
    );
    add_resource(
        renderer,
        ResourceId(String::from("sky_pipeline")),
        Resource::Pipeline(pipeline),
    );

    // Create the render pass
    let pass = RenderPass {
        name: String::from("sky"),
        inputs: vec![
            ResourceId(String::from("depth")),
            ResourceId(String::from("sky_uniform")),
        ],
        outputs: vec![],
        execute: Box::new(render_sky),
    };

    add_render_pass(renderer, pass);
}

fn render_sky(
    render_pass: &mut wgpu::RenderPass,
    renderer: &Renderer,
    _inputs: &[ResourceId],
    _outputs: &[ResourceId],
) {
    let pipeline = match get_resource(renderer, &ResourceId(String::from("sky_pipeline"))) {
        Some(Resource::Pipeline(pipeline)) => pipeline,
        _ => return,
    };

    let bind_group = match get_resource(renderer, &ResourceId(String::from("sky_bind_group"))) {
        Some(Resource::BindGroup(bind_group)) => bind_group,
        _ => return,
    };

    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

fn update_sky_uniforms(renderer: &Renderer, world: &World) {
    // Get camera matrices
    let Some((_entity, matrices)) = query_active_camera_matrices(world) else {
        return;
    };

    let uniform = SkyUniform {
        proj: matrices.projection,
        proj_inv: nalgebra_glm::inverse(&matrices.projection),
        view: matrices.view,
        cam_pos: vec4(
            matrices.camera_position.x,
            matrices.camera_position.y,
            matrices.camera_position.z,
            1.0,
        ),
    };

    if let Some(Resource::Buffer(buffer)) =
        get_resource(renderer, &ResourceId(String::from("sky_uniform")))
    {
        renderer
            .queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
    }
}

fn create_sky_texture_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let bytes = include_bytes!("assets/skyboxes/rgba8.ktx2");
    let reader = ktx2::Reader::new(bytes).unwrap();
    let mut image = Vec::new();
    for level in reader.levels() {
        image.extend_from_slice(level);
    }

    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: SKY_IMAGE_SIZE,
                height: SKY_IMAGE_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("Sky Texture"),
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::MipMajor,
        &image,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    (texture, view, sampler)
}

// Grid
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GridUniform {
    view_proj: Mat4,
    camera_world_pos: Vec3,
    grid_size: f32,
    grid_min_pixels: f32,
    grid_cell_size: f32,
    _padding: [f32; 2],
}

pub fn create_grid_pass(renderer: &mut Renderer) {
    let uniform = GridUniform {
        view_proj: Mat4::identity(),
        camera_world_pos: Vec3::zeros(),
        grid_size: 100.0,
        grid_min_pixels: 2.0,
        grid_cell_size: 0.025,
        _padding: [0.0; 2],
    };

    let uniform_buffer = renderer
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let bind_group_layout =
        renderer
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("Grid Layout"),
            });

    let bind_group = renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("Grid Bind Group"),
        });

    let shader = renderer
        .device
        .create_shader_module(include_wgsl!("assets/shaders/grid.wgsl"));

    let pipeline_layout = renderer
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grid Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = renderer
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grid Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vertex_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fragment_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: renderer.config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    // Store resources
    add_resource(
        renderer,
        ResourceId(String::from("grid_uniform")),
        Resource::Buffer(uniform_buffer),
    );
    add_resource(
        renderer,
        ResourceId(String::from("grid_bind_group")),
        Resource::BindGroup(bind_group),
    );
    add_resource(
        renderer,
        ResourceId(String::from("grid_pipeline")),
        Resource::Pipeline(pipeline),
    );

    // Create the render pass
    let pass = RenderPass {
        name: String::from("grid"),
        inputs: vec![
            ResourceId(String::from("depth")),
            ResourceId(String::from("grid_uniform")),
        ],
        outputs: vec![],
        execute: Box::new(render_grid),
    };

    add_render_pass(renderer, pass);
}

fn render_grid(
    render_pass: &mut wgpu::RenderPass,
    renderer: &Renderer,
    _inputs: &[ResourceId],
    _outputs: &[ResourceId],
) {
    let pipeline = match get_resource(renderer, &ResourceId(String::from("grid_pipeline"))) {
        Some(Resource::Pipeline(pipeline)) => pipeline,
        _ => return,
    };

    let bind_group = match get_resource(renderer, &ResourceId(String::from("grid_bind_group"))) {
        Some(Resource::BindGroup(bind_group)) => bind_group,
        _ => return,
    };

    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.draw(0..6, 0..1);
}

fn update_grid_uniforms(renderer: &Renderer, world: &World) {
    let Some((_entity, matrices)) = query_active_camera_matrices(world) else {
        return;
    };

    let uniform = GridUniform {
        view_proj: matrices.projection * matrices.view,
        camera_world_pos: matrices.camera_position.xyz(),
        grid_size: 100.0,
        grid_min_pixels: 2.0,
        grid_cell_size: 0.025,
        _padding: [0.0; 2],
    };

    if let Some(Resource::Buffer(buffer)) =
        get_resource(renderer, &ResourceId(String::from("grid_uniform")))
    {
        renderer
            .queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&[uniform]));
    }
}

// === Scene Data Types ===
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: Vec3,
    normal: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceBinding {
    model_matrix_0: Vec4,
    model_matrix_1: Vec4,
    model_matrix_2: Vec4,
    model_matrix_3: Vec4,
    color: Vec4,
}

#[derive(Debug, PartialEq)]
pub struct DrawCommand {
    vertex_offset: u32,
    index_offset: u32,
    index_count: u32,
    instance_offset: u32,
    instance_count: u32,
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneUniform {
    view: Mat4,
    projection: Mat4,
    camera_position: Vec4,
}

pub const MAX_LIGHTS: usize = 16;

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LightingUniform {
    lights: [GpuLight; MAX_LIGHTS],
    ambient_color: Vec4,
    num_lights: u32,
    padding: [u32; 3],
    _padding: [u32; 4],
}

#[repr(C)]
#[derive(Default, Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLight {
    position: Vec4,  // xyz = position, w = type (0=dir, 1=point, 2=spot)
    direction: Vec4, // xyz = direction, w unused
    ambient: Vec4,   // xyz = ambient color, w unused
    diffuse: Vec4,   // xyz = diffuse color, w unused
    specular: Vec4,  // xyz = specular color, w unused
    params: Vec4,    // x=constant, y=linear, z=quadratic, w=shininess
    cutoffs: Vec4,   // x=cutoff, y=outerCutoff, z/w unused
}

impl GpuLight {
    pub fn from_light(light: &Light, transform: &GlobalTransform) -> Self {
        match light {
            Light::Point(point) => {
                let position = transform.transform_point(&Vec3::zeros().into());
                let intensity = point.intensity;
                GpuLight {
                    position: vec4(position.x, position.y, position.z, 1.0),
                    direction: Vec4::zeros(),
                    ambient: vec4(point.color.x, point.color.y, point.color.z, 1.0) * intensity,
                    diffuse: vec4(point.color.x, point.color.y, point.color.z, 1.0) * intensity,
                    specular: vec4(point.color.x, point.color.y, point.color.z, 1.0) * intensity,
                    params: vec4(
                        1.0,
                        4.5 / point.range,
                        75.0 / (point.range * point.range),
                        32.0,
                    ),
                    cutoffs: Vec4::zeros(),
                }
            }
            Light::Directional(dir) => {
                let direction = -transform.transform_vector(&Vec3::z_axis()).normalize();
                GpuLight {
                    position: vec4(0.0, 0.0, 0.0, 0.0),
                    direction: vec4(direction.x, direction.y, direction.z, 0.0),
                    ambient: vec4(
                        dir.color.x * dir.intensity,
                        dir.color.y * dir.intensity,
                        dir.color.z * dir.intensity,
                        1.0,
                    ),
                    diffuse: vec4(
                        dir.color.x * dir.intensity,
                        dir.color.y * dir.intensity,
                        dir.color.z * dir.intensity,
                        1.0,
                    ),
                    specular: vec4(
                        dir.color.x * dir.intensity,
                        dir.color.y * dir.intensity,
                        dir.color.z * dir.intensity,
                        1.0,
                    ),
                    params: vec4(1.0, 0.0, 0.0, 32.0),
                    cutoffs: Vec4::zeros(),
                }
            }
            Light::Spot(spot) => {
                let position = transform.transform_point(&Vec3::zeros().into());
                let direction = -transform.transform_vector(&Vec3::z_axis()).normalize();

                let cos_inner = spot.inner_cutoff.cos();
                let cos_outer = spot.outer_cutoff.cos();

                GpuLight {
                    position: vec4(position.x, position.y, position.z, 2.0),
                    direction: vec4(direction.x, direction.y, direction.z, 0.0),
                    ambient: vec4(
                        spot.color.x * spot.intensity,
                        spot.color.y * spot.intensity,
                        spot.color.z * spot.intensity,
                        1.0,
                    ),
                    diffuse: vec4(
                        spot.color.x * spot.intensity,
                        spot.color.y * spot.intensity,
                        spot.color.z * spot.intensity,
                        1.0,
                    ),
                    specular: vec4(
                        spot.color.x * spot.intensity,
                        spot.color.y * spot.intensity,
                        spot.color.z * spot.intensity,
                        1.0,
                    ),
                    params: vec4(
                        1.0,
                        4.5 / spot.range,
                        75.0 / (spot.range * spot.range),
                        32.0,
                    ),
                    cutoffs: vec4(cos_inner, cos_outer, 0.0, 0.0),
                }
            }
        }
    }
}

// === Scene Creation ===
pub fn create_scene_pass(renderer: &mut Renderer) {
    // Create empty vertex and index buffers initially
    let vertex_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scene Vertex Buffer"),
        size: 1024, // Initial size, will grow as needed
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let index_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scene Index Buffer"),
        size: 1024, // Initial size, will grow as needed
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let instance_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scene Instance Buffer"),
        size: 1024, // Initial size, will grow as needed
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniform_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scene Uniform Buffer"),
        size: std::mem::size_of::<SceneUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let lighting_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Lighting Uniform Buffer"),
        size: std::mem::size_of::<LightingUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout =
        renderer
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Scene Bind Group Layout"),
            });

    let bind_group = renderer
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lighting_buffer.as_entire_binding(),
                },
            ],
            label: Some("Scene Bind Group"),
        });

    let shader = renderer
        .device
        .create_shader_module(include_wgsl!("assets/shaders/scene.wgsl"));

    let pipeline_layout = renderer
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Scene Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

    let pipeline = renderer
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Scene Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vertex_main",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x3,  // position
                            1 => Float32x3   // normal
                        ],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<InstanceBinding>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            2 => Float32x4,  // model_matrix_0
                            3 => Float32x4,  // model_matrix_1
                            4 => Float32x4,  // model_matrix_2
                            5 => Float32x4,  // model_matrix_3
                            6 => Float32x4   // color
                        ],
                    },
                ],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fragment_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: renderer.config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    // Store mesh commands
    add_resource(
        renderer,
        ResourceId(String::from("scene_mesh_commands")),
        Resource::MeshCommands(HashMap::<String, Vec<DrawCommand>>::new()),
    );

    // Store resources
    add_resource(
        renderer,
        ResourceId(String::from("scene_vertex_buffer")),
        Resource::Buffer(vertex_buffer),
    );
    add_resource(
        renderer,
        ResourceId(String::from("scene_index_buffer")),
        Resource::Buffer(index_buffer),
    );
    add_resource(
        renderer,
        ResourceId(String::from("scene_instance_buffer")),
        Resource::Buffer(instance_buffer),
    );
    add_resource(
        renderer,
        ResourceId(String::from("scene_uniform_buffer")),
        Resource::Buffer(uniform_buffer),
    );
    add_resource(
        renderer,
        ResourceId(String::from("scene_lighting_buffer")),
        Resource::Buffer(lighting_buffer),
    );
    add_resource(
        renderer,
        ResourceId(String::from("scene_bind_group")),
        Resource::BindGroup(bind_group),
    );
    add_resource(
        renderer,
        ResourceId(String::from("scene_pipeline")),
        Resource::Pipeline(pipeline),
    );

    // Create the render pass
    let pass = RenderPass {
        name: String::from("scene"),
        inputs: vec![
            ResourceId(String::from("depth")),
            ResourceId(String::from("scene_vertex_buffer")),
            ResourceId(String::from("scene_index_buffer")),
            ResourceId(String::from("scene_instance_buffer")),
            ResourceId(String::from("scene_uniform_buffer")),
            ResourceId(String::from("scene_lighting_buffer")),
        ],
        outputs: vec![],
        execute: Box::new(render_scene),
    };

    add_render_pass(renderer, pass);
}

fn render_scene(
    render_pass: &mut wgpu::RenderPass,
    renderer: &Renderer,
    _inputs: &[ResourceId],
    _outputs: &[ResourceId],
) {
    let pipeline = match get_resource(renderer, &ResourceId(String::from("scene_pipeline"))) {
        Some(Resource::Pipeline(pipeline)) => pipeline,
        _ => return,
    };

    let bind_group = match get_resource(renderer, &ResourceId(String::from("scene_bind_group"))) {
        Some(Resource::BindGroup(bind_group)) => bind_group,
        _ => return,
    };

    let vertex_buffer =
        match get_resource(renderer, &ResourceId(String::from("scene_vertex_buffer"))) {
            Some(Resource::Buffer(buffer)) => buffer,
            _ => return,
        };

    let index_buffer = match get_resource(renderer, &ResourceId(String::from("scene_index_buffer")))
    {
        Some(Resource::Buffer(buffer)) => buffer,
        _ => return,
    };

    let instance_buffer =
        match get_resource(renderer, &ResourceId(String::from("scene_instance_buffer"))) {
            Some(Resource::Buffer(buffer)) => buffer,
            _ => return,
        };

    let mesh_commands =
        match get_resource(renderer, &ResourceId(String::from("scene_mesh_commands"))) {
            Some(Resource::MeshCommands(commands)) => commands,
            _ => return,
        };

    render_pass.set_pipeline(pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    render_pass.set_vertex_buffer(1, instance_buffer.slice(..));
    render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);

    for commands in mesh_commands.values() {
        for cmd in commands {
            if cmd.instance_count > 0 {
                render_pass.draw_indexed(
                    cmd.index_offset..(cmd.index_offset + cmd.index_count),
                    cmd.vertex_offset as i32,
                    cmd.instance_offset..(cmd.instance_offset + cmd.instance_count),
                );
            }
        }
    }
}

fn update_scene_uniforms(renderer: &mut Renderer, world: &World) {
    sync_mesh_data(renderer, world); // Add this line at the start

    let Some((_entity, matrices)) = query_active_camera_matrices(world) else {
        return;
    };

    // Update scene uniform
    let scene_uniform = SceneUniform {
        view: matrices.view,
        projection: matrices.projection,
        camera_position: vec4(
            matrices.camera_position.x,
            matrices.camera_position.y,
            matrices.camera_position.z,
            1.0,
        ),
    };

    if let Some(Resource::Buffer(buffer)) =
        get_resource(renderer, &ResourceId(String::from("scene_uniform_buffer")))
    {
        renderer
            .queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&[scene_uniform]));
    }

    // Update lighting uniform
    let lights = query_scene_lights(world);
    let mut lighting_uniform = LightingUniform {
        ambient_color: vec4(0.1, 0.1, 0.1, 1.0),
        num_lights: lights.len().min(MAX_LIGHTS) as u32,
        lights: [GpuLight::default(); MAX_LIGHTS],
        padding: [0; 3],
        _padding: [0; 4],
    };

    for (i, (light, _)) in lights.iter().take(MAX_LIGHTS).enumerate() {
        lighting_uniform.lights[i] = *light;
    }

    if let Some(Resource::Buffer(buffer)) =
        get_resource(renderer, &ResourceId(String::from("scene_lighting_buffer")))
    {
        renderer
            .queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&[lighting_uniform]));
    }
}

fn query_scene_lights(world: &World) -> Vec<(GpuLight, EntityId)> {
    query_entities(world, LIGHT | GLOBAL_TRANSFORM)
        .into_iter()
        .filter_map(|entity| {
            let light = get_component::<Light>(world, entity, LIGHT)?;
            let global_transform =
                get_component::<GlobalTransform>(world, entity, GLOBAL_TRANSFORM)?;
            let gpu_light = GpuLight::from_light(light, global_transform);
            Some((gpu_light, entity))
        })
        .collect()
}

fn sync_mesh_data(renderer: &mut Renderer, world: &World) {
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();
    let mut mesh_commands = HashMap::new();

    // Collect all mesh data
    for (mesh_name, primitives) in &world.resources.meshes {
        let mut commands = Vec::new();

        for primitive in primitives {
            let vertex_offset = all_vertices.len() as u32;
            let index_offset = all_indices.len() as u32;

            all_vertices.extend(primitive.vertices.iter().map(|v| Vertex {
                position: v.position,
                normal: v.normal,
            }));

            all_indices.extend(primitive.indices.iter().map(|i| *i as u16));

            commands.push(DrawCommand {
                vertex_offset,
                index_offset,
                index_count: primitive.indices.len() as u32,
                instance_offset: 0,
                instance_count: 0,
            });
        }

        mesh_commands.insert(mesh_name.clone(), commands);
    }

    // Process instances
    let mut instance_bindings = Vec::new();
    let mut mesh_instances: HashMap<String, Vec<usize>> = HashMap::new();

    // Collect instances for each mesh
    for entity in query_entities(world, GLOBAL_TRANSFORM | RENDER_MESH | VISIBLE) {
        let Some(global_transform) =
            get_component::<GlobalTransform>(world, entity, GLOBAL_TRANSFORM)
        else {
            continue;
        };

        let Some(RenderMesh { mesh_name }) =
            get_component::<RenderMesh>(world, entity, RENDER_MESH)
        else {
            continue;
        };

        let instance_idx = instance_bindings.len();
        mesh_instances
            .entry(mesh_name.to_string())
            .or_default()
            .push(instance_idx);

        instance_bindings.push(InstanceBinding {
            model_matrix_0: global_transform.column(0).into(),
            model_matrix_1: global_transform.column(1).into(),
            model_matrix_2: global_transform.column(2).into(),
            model_matrix_3: global_transform.column(3).into(),
            color: vec4(1.0, 1.0, 1.0, 1.0),
        });
    }

    // Update draw commands with instance information
    for (mesh_name, commands) in mesh_commands.iter_mut() {
        if let Some(instance_indices) = mesh_instances.get(mesh_name) {
            let instance_count = instance_indices.len();
            if let Some(&first_instance) = instance_indices.first() {
                for cmd in commands {
                    cmd.instance_offset = first_instance as u32;
                    cmd.instance_count = instance_count as u32;
                }
            }
        }
    }

    // Update buffers
    if !all_vertices.is_empty() {
        let vertex_buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Scene Vertex Buffer"),
                contents: bytemuck::cast_slice(&all_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        add_resource(
            renderer,
            ResourceId(String::from("scene_vertex_buffer")),
            Resource::Buffer(vertex_buffer),
        );
    }

    if !all_indices.is_empty() {
        let index_buffer = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Scene Index Buffer"),
                contents: bytemuck::cast_slice(&all_indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });
        add_resource(
            renderer,
            ResourceId(String::from("scene_index_buffer")),
            Resource::Buffer(index_buffer),
        );
    }

    if !instance_bindings.is_empty() {
        let instance_buffer =
            renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Scene Instance Buffer"),
                    contents: bytemuck::cast_slice(&instance_bindings),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
        add_resource(
            renderer,
            ResourceId(String::from("scene_instance_buffer")),
            Resource::Buffer(instance_buffer),
        );
    }

    // Update mesh commands
    add_resource(
        renderer,
        ResourceId(String::from("scene_mesh_commands")),
        Resource::MeshCommands(mesh_commands),
    );
}
