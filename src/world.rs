use crate::prelude::*;
use freecs::{has_components, world};
use nalgebra_glm::{vec3, Mat3};

world! {
    World {
        components {
            active_camera: ActiveCamera => ACTIVE_CAMERA,
            camera: Camera => CAMERA,
            collision_shape: CollisionShape => COLLISION_SHAPE,
            debug_line: DebugLine => DEBUG_LINE,
            global_transform: GlobalTransform => GLOBAL_TRANSFORM,
            light: Light => LIGHT,
            local_transform: LocalTransform => LOCAL_TRANSFORM,
            name: Name => NAME,
            parent: Parent => PARENT,
            player: Player => PLAYER,
            render_mesh: RenderMesh => RENDER_MESH,
            visible: Visible => VISIBLE,
        },
        Resources {
            frames_per_second: f32,
            delta_time: f32,
            uptime_milliseconds: u128,
            meshes: std::collections::HashMap<String, Vec<MeshPrimitive>>,
            keyboard: Keyboard,
            mouse: Mouse,
            window_center: nalgebra_glm::Vec2,
            viewport_width: u32,
            viewport_height: u32,
            gilrs: Option<gilrs::Gilrs>,
            active_gamepad: Option<gilrs::GamepadId>,
        }
    }
}

// -- Components

#[derive(Default, Debug, Copy, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Parent(pub EntityId);

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Name(pub String);

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RenderMesh {
    pub mesh_name: String,
}

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ActiveCamera;

#[repr(C)]
#[derive(
    Default,
    Debug,
    Copy,
    Clone,
    bytemuck::Pod,
    bytemuck::Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct DebugLine {
    pub start: nalgebra_glm::Vec3,
    pub end: nalgebra_glm::Vec3,
    pub color: nalgebra_glm::Vec4,
    pub thickness: f32,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct Camera {
    pub projection: Projection,
    pub sensitivity: nalgebra_glm::Vec2,
}

impl Camera {
    pub fn projection_matrix(&self, aspect_ratio: f32) -> nalgebra_glm::Mat4 {
        match &self.projection {
            Projection::Perspective(camera) => camera.matrix(aspect_ratio),
            Projection::Orthographic(camera) => camera.matrix(),
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            projection: Projection::default(),
            sensitivity: nalgebra_glm::vec2(1.0, 1.0),
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub enum Projection {
    Perspective(PerspectiveCamera),
    Orthographic(OrthographicCamera),
}

impl Default for Projection {
    fn default() -> Self {
        Self::Perspective(PerspectiveCamera::default())
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct PerspectiveCamera {
    pub aspect_ratio: Option<f32>,
    pub y_fov_rad: f32,
    pub z_far: Option<f32>,
    pub z_near: f32,
}
impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            aspect_ratio: None,
            y_fov_rad: 90_f32.to_radians(),
            z_far: None,
            z_near: 0.01,
        }
    }
}

impl PerspectiveCamera {
    pub fn matrix(&self, viewport_aspect_ratio: f32) -> nalgebra_glm::Mat4 {
        let aspect_ratio = if let Some(aspect_ratio) = self.aspect_ratio {
            aspect_ratio
        } else {
            viewport_aspect_ratio
        };

        if let Some(z_far) = self.z_far {
            nalgebra_glm::perspective_zo(aspect_ratio, self.y_fov_rad, self.z_near, z_far)
        } else {
            nalgebra_glm::infinite_perspective_rh_zo(aspect_ratio, self.y_fov_rad, self.z_near)
        }
    }
}

#[derive(Default, Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct OrthographicCamera {
    pub x_mag: f32,
    pub y_mag: f32,
    pub z_far: f32,
    pub z_near: f32,
}

impl OrthographicCamera {
    pub fn matrix(&self) -> nalgebra_glm::Mat4 {
        let z_sum = self.z_near + self.z_far;
        let z_diff = self.z_near - self.z_far;
        nalgebra_glm::Mat4::new(
            1.0 / self.x_mag,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / self.y_mag,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / z_diff,
            0.0,
            0.0,
            0.0,
            z_sum / z_diff,
            1.0,
        )
    }
}

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Player(pub u8);

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Visible;

pub type GlobalTransform = nalgebra_glm::Mat4;
pub type LocalTransform = Transform;

#[derive(Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Transform {
    pub translation: nalgebra_glm::Vec3,
    pub rotation: nalgebra_glm::Quat,
    pub scale: nalgebra_glm::Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: nalgebra_glm::Vec3::new(0.0, 0.0, 0.0),
            rotation: nalgebra_glm::Quat::identity(),
            scale: nalgebra_glm::Vec3::new(1.0, 1.0, 1.0),
        }
    }
}

impl From<Transform> for nalgebra_glm::Mat4 {
    fn from(transform: Transform) -> Self {
        nalgebra_glm::translation(&transform.translation)
            * nalgebra_glm::quat_to_mat4(&transform.rotation)
            * nalgebra_glm::scaling(&transform.scale)
    }
}

impl Transform {
    pub fn matrix(&self) -> nalgebra_glm::Mat4 {
        nalgebra_glm::Mat4::from(*self)
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Light {
    Point(PointLight),
    Directional(DirectionalLight),
    Spot(SpotLight),
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct PointLight {
    pub color: nalgebra_glm::Vec3,
    pub intensity: f32,
    pub range: f32,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct DirectionalLight {
    pub color: nalgebra_glm::Vec3,
    pub intensity: f32,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SpotLight {
    pub color: nalgebra_glm::Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_cutoff: f32, // In radians
    pub outer_cutoff: f32, // In radians
}

impl Default for Light {
    fn default() -> Self {
        Light::Point(PointLight {
            color: nalgebra_glm::vec3(1.0, 1.0, 1.0),
            intensity: 1.0,
            range: 10.0,
        })
    }
}

#[repr(C)]
#[derive(
    Default, Copy, Clone, Debug, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
pub enum CollisionShape {
    #[default]
    Empty,
    Box {
        center: nalgebra_glm::Vec3,
        size: nalgebra_glm::Vec3,
        orientation: nalgebra_glm::Vec3,
    },
}

// -- Resources

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Keyboard {
    pub keystates: std::collections::HashMap<winit::keyboard::KeyCode, winit::event::ElementState>,
}

impl Keyboard {
    pub fn is_key_pressed(&self, keycode: winit::keyboard::KeyCode) -> bool {
        self.keystates.contains_key(&keycode)
            && self.keystates[&keycode] == winit::event::ElementState::Pressed
    }
}

bitflags::bitflags! {
    #[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct MouseButtons: u8 {
        const LEFT_CLICKED = 0b0000_0001;
        const MIDDLE_CLICKED = 0b0000_0010;
        const RIGHT_CLICKED = 0b0000_0100;
        const MOVED = 0b0000_1000;
        const SCROLLED = 0b0001_0000;
    }
}

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mouse {
    pub buttons: MouseButtons,
    pub position: nalgebra_glm::Vec2,
    pub position_delta: nalgebra_glm::Vec2,
    pub offset_from_center: nalgebra_glm::Vec2,
    pub wheel_delta: nalgebra_glm::Vec2,
}

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeshPrimitive {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

#[repr(C)]
#[derive(
    Debug, Copy, Clone, serde::Serialize, serde::Deserialize, bytemuck::Pod, bytemuck::Zeroable,
)]
pub struct Vertex {
    pub position: nalgebra_glm::Vec3,
    pub normal: nalgebra_glm::Vec3,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: nalgebra_glm::Vec3::default(),
            normal: nalgebra_glm::vec3(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Image {
    pub pixels: Vec<u8>,
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ImageFormat {
    R8,
    R8G8,
    R8G8B8,
    R8G8B8A8,
    B8G8R8,
    B8G8R8A8,
    R16,
    R16G16,
    R16G16B16,
    R16G16B16A16,
    R16F,
    R16G16F,
    R16G16B16F,
    R16G16B16A16F,
    R32,
    R32G32,
    R32G32B32,
    R32G32B32A32,
    R32F,
    R32G32F,
    R32G32B32F,
    R32G32B32A32F,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Sampler {
    pub min_filter: MinFilter,
    pub mag_filter: MagFilter,
    pub wrap_s: WrappingMode,
    pub wrap_t: WrappingMode,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MagFilter {
    Nearest = 1,
    #[default]
    Linear,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MinFilter {
    Nearest = 1,
    #[default]
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum WrappingMode {
    ClampToEdge,
    MirroredRepeat,
    #[default]
    Repeat,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Texture {
    pub image_index: usize,
    pub sampler_index: Option<usize>,
}

pub fn query_root_nodes(world: &World) -> Vec<EntityId> {
    use rayon::prelude::*;
    let mut root_entities: Vec<EntityId> = world
        .tables
        .par_iter()
        .filter_map(|table| {
            if has_components!(table, PARENT) {
                return None;
            }
            Some(table.entity_indices.to_vec())
        })
        .flatten()
        .collect();
    root_entities.dedup();
    root_entities
}

pub fn activate_camera(world: &mut World, entity: EntityId) {
    for entity in query_entities(world, ACTIVE_CAMERA) {
        remove_components(world, entity, ACTIVE_CAMERA);
    }
    add_components(world, entity, ACTIVE_CAMERA);
}

pub fn query_children(world: &mut World, target_entity: EntityId) -> Vec<EntityId> {
    let mut child_entities = Vec::new();
    query_entities(world, PARENT)
        .into_iter()
        .for_each(|entity| {
            if let Some(Parent(parent_entity)) = get_component(world, entity, PARENT) {
                if *parent_entity != target_entity {
                    return;
                }
                child_entities.push(entity);
            }
        });
    child_entities
}

#[derive(Default, Debug, Copy, Clone, serde::Serialize, serde::Deserialize)]
pub struct CameraMatrices {
    pub camera_position: nalgebra_glm::Vec3,
    pub projection: nalgebra_glm::Mat4,
    pub view: nalgebra_glm::Mat4,
}

pub fn query_active_camera_matrices(world: &World) -> Option<(EntityId, CameraMatrices)> {
    let camera_entity = query_first_entity(world, ACTIVE_CAMERA | CAMERA | LOCAL_TRANSFORM)?;

    let (Some(camera), Some(local_transform), Some(global_transform)) = (
        get_component::<Camera>(world, camera_entity, CAMERA),
        get_component::<LocalTransform>(world, camera_entity, LOCAL_TRANSFORM),
        get_component::<GlobalTransform>(world, camera_entity, GLOBAL_TRANSFORM),
    ) else {
        return None;
    };

    let normalized_rotation = local_transform.rotation.normalize();
    let camera_translation = global_transform.column(3).xyz();
    let target = camera_translation
        + nalgebra_glm::quat_rotate_vec3(&normalized_rotation, &(-nalgebra_glm::Vec3::z()));
    let up = nalgebra_glm::quat_rotate_vec3(&normalized_rotation, &nalgebra_glm::Vec3::y());
    let aspect_ratio =
        world.resources.viewport_width as f32 / world.resources.viewport_height.max(1) as f32;

    Some((
        camera_entity,
        CameraMatrices {
            camera_position: camera_translation,
            projection: camera.projection_matrix(aspect_ratio),
            view: nalgebra_glm::look_at(&camera_translation, &target, &up),
        },
    ))
}

pub fn query_global_transform(world: &World, entity: EntityId) -> nalgebra_glm::Mat4 {
    let Some(local_transform) = get_component::<LocalTransform>(world, entity, LOCAL_TRANSFORM)
    else {
        return nalgebra_glm::Mat4::identity();
    };
    if let Some(Parent(parent)) = get_component::<Parent>(world, entity, PARENT) {
        query_global_transform(world, *parent) * local_transform.matrix()
    } else {
        local_transform.matrix()
    }
}

pub fn query_render_meshes(world: &World) -> Vec<(EntityId, RenderMesh)> {
    let mut entities = Vec::new();
    let render_mesh_tables = world
        .tables
        .iter()
        .filter(|table| !has_components!(table, RENDER_MESH));
    render_mesh_tables.for_each(|table| {
        table.entity_indices.iter().for_each(|entity| {
            if let Some(render_mesh) = get_component::<RenderMesh>(world, *entity, RENDER_MESH) {
                entities.push((*entity, render_mesh.clone()));
            }
        });
    });
    entities
}

pub fn gamepad_controls_system(world: &mut World) {
    if world.resources.gilrs.is_none() {
        return;
    }
    query_entities(world, PLAYER | LOCAL_TRANSFORM)
        .into_iter()
        .for_each(|entity| {
            let speed = 10.0 * world.resources.delta_time;

            let (
                left_trigger2_pressed,
                right_trigger2_pressed,
                left_stick_x_axis_data,
                left_stick_y_axis_data,
                right_stick_x_axis_data,
                right_stick_y_axis_data,
            ) = {
                let Some(gilrs) = &mut world.resources.gilrs else {
                    return;
                };
                let Some(gamepad) = world.resources.active_gamepad.map(|id| gilrs.gamepad(id))
                else {
                    return;
                };
                (
                    gamepad.is_pressed(gilrs::Button::LeftTrigger2),
                    gamepad.is_pressed(gilrs::Button::RightTrigger2),
                    gamepad.axis_data(gilrs::Axis::LeftStickX).cloned(),
                    gamepad.axis_data(gilrs::Axis::LeftStickY).cloned(),
                    gamepad.axis_data(gilrs::Axis::RightStickX).cloned(),
                    gamepad.axis_data(gilrs::Axis::RightStickY).cloned(),
                )
            };

            let Some(local_transform) =
                get_component_mut::<LocalTransform>(world, entity, LOCAL_TRANSFORM)
            else {
                return;
            };

            let local_transform_matrix = local_transform.matrix();
            let forward = extract_forward_vector(&local_transform_matrix);
            let right = extract_right_vector(&local_transform_matrix);
            let up = extract_up_vector(&local_transform_matrix);

            if right_trigger2_pressed {
                local_transform.translation += up * speed;
            }

            if left_trigger2_pressed {
                local_transform.translation -= up * speed;
            }

            if let Some(axis_data) = left_stick_x_axis_data {
                local_transform.translation += right * axis_data.value() * speed;
            }

            if let Some(axis_data) = left_stick_y_axis_data {
                local_transform.translation += forward * axis_data.value() * speed;
            }

            if let Some(axis_data) = right_stick_x_axis_data {
                let yaw = nalgebra_glm::quat_angle_axis(
                    axis_data.value() * speed * -1.0,
                    &nalgebra_glm::Vec3::y(),
                );
                local_transform.rotation = yaw * local_transform.rotation;
            }

            if let Some(axis_data) = right_stick_y_axis_data {
                let forward = extract_forward_vector(&local_transform.matrix());
                let current_pitch = forward.y.asin();
                let new_pitch = (current_pitch + axis_data.value()) * -1.0;
                if new_pitch.abs() <= 89_f32.to_radians() {
                    let pitch = nalgebra_glm::quat_angle_axis(
                        axis_data.value() * speed,
                        &nalgebra_glm::Vec3::x(),
                    );
                    local_transform.rotation *= pitch;
                }
            }
        });
}

pub fn wasd_keyboard_controls_system(world: &mut World) {
    query_entities(world, PLAYER | LOCAL_TRANSFORM)
        .into_iter()
        .for_each(|entity| {
            let speed = 10.0 * world.resources.delta_time;

            let (
                left_key_pressed,
                right_key_pressed,
                forward_key_pressed,
                backward_key_pressed,
                up_key_pressed,
                down_key_pressed,
            ) = {
                let keyboard = &world.resources.keyboard;
                (
                    keyboard.is_key_pressed(winit::keyboard::KeyCode::KeyA),
                    keyboard.is_key_pressed(winit::keyboard::KeyCode::KeyD),
                    keyboard.is_key_pressed(winit::keyboard::KeyCode::KeyW),
                    keyboard.is_key_pressed(winit::keyboard::KeyCode::KeyS),
                    keyboard.is_key_pressed(winit::keyboard::KeyCode::Space),
                    keyboard.is_key_pressed(winit::keyboard::KeyCode::ShiftLeft),
                )
            };

            let Some(local_transform) =
                get_component_mut::<LocalTransform>(world, entity, LOCAL_TRANSFORM)
            else {
                return;
            };
            let local_transform_matrix = local_transform.matrix();
            let forward = extract_forward_vector(&local_transform_matrix);
            let right = extract_right_vector(&local_transform_matrix);
            let up = extract_up_vector(&local_transform_matrix);

            if forward_key_pressed {
                local_transform.translation += forward * speed;
            }
            if backward_key_pressed {
                local_transform.translation -= forward * speed;
            }

            if left_key_pressed {
                local_transform.translation -= right * speed;
            }
            if right_key_pressed {
                local_transform.translation += right * speed;
            }

            if up_key_pressed {
                local_transform.translation += up * speed;
            }
            if down_key_pressed {
                local_transform.translation -= up * speed;
            }
        });
}

pub fn update_global_transforms_system(world: &mut World) {
    query_entities(world, LOCAL_TRANSFORM | GLOBAL_TRANSFORM)
        .into_iter()
        .for_each(|entity| {
            let new_global_transform = query_global_transform(world, entity);
            let Some(global_transform) =
                get_component_mut::<GlobalTransform>(world, entity, GLOBAL_TRANSFORM)
            else {
                return;
            };
            *global_transform = new_global_transform;
        });
}

pub fn fps_camera_controls_system(world: &mut World) {
    query_entities(world, ACTIVE_CAMERA | LOCAL_TRANSFORM | PLAYER)
        .into_iter()
        .for_each(|entity| {
            let (forward, right, up) = {
                let Some(local_transform) =
                    get_component_mut::<LocalTransform>(world, entity, LOCAL_TRANSFORM)
                else {
                    return;
                };
                let local_transform_matrix = local_transform.matrix();
                let forward = extract_forward_vector(&local_transform_matrix);
                let right = extract_right_vector(&local_transform_matrix);
                let up = extract_up_vector(&local_transform_matrix);
                (forward, right, up)
            };

            if world.resources.mouse.wheel_delta.y.abs() > 0.0 {
                let speed = 10.0 * world.resources.mouse.wheel_delta.y * world.resources.delta_time;
                let Some(local_transform) =
                    get_component_mut::<LocalTransform>(world, entity, LOCAL_TRANSFORM)
                else {
                    return;
                };
                local_transform.translation += forward * speed;
            }

            if world
                .resources
                .mouse
                .buttons
                .contains(MouseButtons::RIGHT_CLICKED)
            {
                let mut delta = world.resources.mouse.position_delta * world.resources.delta_time;
                delta.x *= -1.0;
                delta.y *= -1.0;

                let Some(local_transform) =
                    get_component_mut::<LocalTransform>(world, entity, LOCAL_TRANSFORM)
                else {
                    return;
                };

                let yaw = nalgebra_glm::quat_angle_axis(delta.x, &nalgebra_glm::Vec3::y());
                local_transform.rotation = yaw * local_transform.rotation;

                let forward = extract_forward_vector(&local_transform.matrix());
                let current_pitch = forward.y.asin();

                let new_pitch = current_pitch + delta.y;
                if new_pitch.abs() <= 89_f32.to_radians() {
                    let pitch = nalgebra_glm::quat_angle_axis(delta.y, &nalgebra_glm::Vec3::x());
                    local_transform.rotation *= pitch;
                }
            }

            if world
                .resources
                .mouse
                .buttons
                .contains(MouseButtons::MIDDLE_CLICKED)
            {
                let mut delta = world.resources.mouse.position_delta * world.resources.delta_time;
                delta.x *= -1.0;
                delta.y *= -1.0;

                let Some(local_transform) =
                    get_component_mut::<LocalTransform>(world, entity, LOCAL_TRANSFORM)
                else {
                    return;
                };
                local_transform.translation += right * delta.x;
                local_transform.translation += up * delta.y;
            }
        });
}

pub fn query_debug_lines(world: &World) -> Vec<(EntityId, DebugLine)> {
    let mut entities = Vec::new();
    let debug_line_tables = world
        .tables
        .iter()
        .filter(|table| !has_components!(table, DEBUG_LINE));
    debug_line_tables.for_each(|table| {
        table.entity_indices.iter().for_each(|entity| {
            let Some(debug_line) = get_component::<DebugLine>(world, *entity, DEBUG_LINE) else {
                return;
            };

            let Some(global_transform) =
                get_component::<GlobalTransform>(world, *entity, GLOBAL_TRANSFORM)
            else {
                return;
            };

            let DebugLine {
                start,
                end,
                color,
                thickness,
            } = *debug_line;

            // Extract rotation by converting the upper 3x3 part of the matrix to Mat3
            let rotation = Mat3::new(
                global_transform[(0, 0)],
                global_transform[(0, 1)],
                global_transform[(0, 2)],
                global_transform[(1, 0)],
                global_transform[(1, 1)],
                global_transform[(1, 2)],
                global_transform[(2, 0)],
                global_transform[(2, 1)],
                global_transform[(2, 2)],
            );

            // Extract translation from the fourth column
            let translation = vec3(
                global_transform[(0, 3)],
                global_transform[(1, 3)],
                global_transform[(2, 3)],
            );

            // Transform the points
            let transformed_start = rotation * start + translation;
            let transformed_end = rotation * end + translation;

            entities.push((
                *entity,
                DebugLine {
                    start: transformed_start,
                    end: transformed_end,
                    color,
                    thickness,
                },
            ));
        });
    });
    entities
}

pub fn extract_right_vector(transform: &nalgebra_glm::Mat4) -> nalgebra_glm::Vec3 {
    nalgebra_glm::vec3(transform[(0, 0)], transform[(1, 0)], transform[(2, 0)])
}

pub fn extract_up_vector(transform: &nalgebra_glm::Mat4) -> nalgebra_glm::Vec3 {
    nalgebra_glm::vec3(transform[(0, 1)], transform[(1, 1)], transform[(2, 1)])
}

pub fn extract_forward_vector(transform: &nalgebra_glm::Mat4) -> nalgebra_glm::Vec3 {
    nalgebra_glm::vec3(-transform[(0, 2)], -transform[(1, 2)], -transform[(2, 2)])
}

pub fn reset_mouse_system(world: &mut World) {
    let mouse = &mut world.resources.mouse;
    if mouse.buttons.contains(MouseButtons::SCROLLED) {
        mouse.wheel_delta = nalgebra_glm::vec2(0.0, 0.0);
    }
    mouse.buttons.set(MouseButtons::MOVED, false);
    if !mouse.buttons.contains(MouseButtons::MOVED) {
        mouse.position_delta = nalgebra_glm::vec2(0.0, 0.0);
    }
    mouse.buttons.set(MouseButtons::MOVED, false);
}

pub fn update_keyboard_system(world: &mut World, event: &winit::event::WindowEvent) {
    if let winit::event::WindowEvent::KeyboardInput {
        event:
            winit::event::KeyEvent {
                physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                state,
                ..
            },
        ..
    } = event
    {
        *world
            .resources
            .keyboard
            .keystates
            .entry(*key_code)
            .or_insert(*state) = *state;
    }
}

pub fn mouse_system(world: &mut World, event: &winit::event::WindowEvent) {
    let mouse = &mut world.resources.mouse;
    let window_center = world.resources.window_center;
    match event {
        winit::event::WindowEvent::MouseInput { button, state, .. } => {
            let clicked = *state == winit::event::ElementState::Pressed;
            match button {
                winit::event::MouseButton::Left => {
                    mouse.buttons.set(MouseButtons::LEFT_CLICKED, clicked);
                }
                winit::event::MouseButton::Middle => {
                    mouse.buttons.set(MouseButtons::MIDDLE_CLICKED, clicked);
                }
                winit::event::MouseButton::Right => {
                    mouse.buttons.set(MouseButtons::RIGHT_CLICKED, clicked);
                }
                _ => {}
            }
        }
        winit::event::WindowEvent::CursorMoved { position, .. } => {
            let last_position = mouse.position;
            let current_position = nalgebra_glm::vec2(position.x as _, position.y as _);
            mouse.position = current_position;
            mouse.position_delta = current_position - last_position;
            mouse.offset_from_center =
                window_center - nalgebra_glm::vec2(position.x as _, position.y as _);
            mouse.buttons.set(MouseButtons::MOVED, true);
        }
        winit::event::WindowEvent::MouseWheel {
            delta: winit::event::MouseScrollDelta::LineDelta(h_lines, v_lines),
            ..
        } => {
            mouse.wheel_delta = nalgebra_glm::vec2(*h_lines, *v_lines);
            mouse.buttons.set(MouseButtons::SCROLLED, true);
        }
        _ => {}
    }
}

pub mod meshgen {
    use super::*;

    pub fn create_cube_mesh_primitive() -> MeshPrimitive {
        let positions = vec![
            // Front face
            vec3(-0.5, -0.5, 0.5),
            vec3(0.5, -0.5, 0.5),
            vec3(0.5, 0.5, 0.5),
            vec3(-0.5, 0.5, 0.5),
            // Back face
            vec3(-0.5, -0.5, -0.5),
            vec3(-0.5, 0.5, -0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(0.5, -0.5, -0.5),
            // Right face
            vec3(0.5, -0.5, -0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(0.5, 0.5, 0.5),
            vec3(0.5, -0.5, 0.5),
            // Left face
            vec3(-0.5, -0.5, -0.5),
            vec3(-0.5, -0.5, 0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(-0.5, 0.5, -0.5),
            // Top face
            vec3(-0.5, 0.5, -0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(0.5, 0.5, 0.5),
            vec3(0.5, 0.5, -0.5),
            // Bottom face
            vec3(-0.5, -0.5, -0.5),
            vec3(0.5, -0.5, -0.5),
            vec3(0.5, -0.5, 0.5),
            vec3(-0.5, -0.5, 0.5),
        ];

        let normals = vec![
            // Front face
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            vec3(0.0, 0.0, 1.0),
            // Back face
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, -1.0),
            vec3(0.0, 0.0, -1.0),
            // Right face
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            // Left face
            vec3(-1.0, 0.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
            vec3(-1.0, 0.0, 0.0),
            // Top face
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 1.0, 0.0),
            // Bottom face
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, -1.0, 0.0),
            vec3(0.0, -1.0, 0.0),
        ];

        let indices = vec![
            0, 1, 2, 2, 3, 0, // front
            4, 5, 6, 6, 7, 4, // back
            8, 9, 10, 10, 11, 8, // right
            12, 13, 14, 14, 15, 12, // left
            16, 17, 18, 18, 19, 16, // top
            20, 21, 22, 22, 23, 20, // bottom
        ];

        create_mesh_primitive_from_arrays(&positions, &normals, &indices)
    }

    pub fn create_sphere_mesh_primitive(radius: f32, segments: u32) -> MeshPrimitive {
        let mut positions = Vec::new();
        let mut normals = Vec::new();

        // Generate vertices
        for i in 0..=segments {
            let lat = std::f32::consts::PI * (i as f32) / segments as f32;
            for j in 0..=segments {
                let lon = 2.0 * std::f32::consts::PI * (j as f32) / segments as f32;

                let x = radius * lat.sin() * lon.cos();
                let y = radius * lat.cos();
                let z = radius * lat.sin() * lon.sin();

                let pos = vec3(x, y, z);
                positions.push(pos);
                // Normal points outward from sphere center
                normals.push(nalgebra_glm::normalize(&pos));
            }
        }

        let mut indices = Vec::new();
        for i in 0..segments {
            for j in 0..segments {
                let first = i * (segments + 1) + j;
                let second = first + segments + 1;

                indices.extend_from_slice(&[
                    first,
                    first + 1,
                    second,
                    second,
                    first + 1,
                    second + 1,
                ]);
            }
        }

        create_mesh_primitive_from_arrays(&positions, &normals, &indices)
    }

    pub fn create_plane_mesh_primitive(size: f32) -> MeshPrimitive {
        let half_size = size / 2.0;
        let positions = vec![
            vec3(-half_size, 0.0, -half_size),
            vec3(half_size, 0.0, -half_size),
            vec3(half_size, 0.0, half_size),
            vec3(-half_size, 0.0, half_size),
        ];

        let normal = vec3(0.0, 1.0, 0.0);
        let normals = vec![normal; 4];

        let indices = vec![0, 2, 1, 2, 0, 3];

        create_mesh_primitive_from_arrays(&positions, &normals, &indices)
    }

    pub fn create_cylinder_mesh_primitive(
        radius: f32,
        height: f32,
        segments: u32,
    ) -> MeshPrimitive {
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        let half_height = height / 2.0;

        // Side vertices
        for i in 0..=segments {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / segments as f32;
            let x = radius * angle.cos();
            let z = radius * angle.sin();

            positions.push(vec3(x, half_height, z)); // Top
            positions.push(vec3(x, -half_height, z)); // Bottom
        }

        // Side faces
        for i in 0..segments {
            let start = 2 * i;
            indices.extend_from_slice(&[
                start,
                start + 2,
                start + 1, // First triangle
                start + 1,
                start + 2,
                start + 3, // Second triangle
            ]);
        }

        // Add cap centers
        let top_center_idx = positions.len() as u32;
        positions.push(vec3(0.0, half_height, 0.0)); // Top center
        positions.push(vec3(0.0, -half_height, 0.0)); // Bottom center

        // Add cap vertices
        for i in 0..segments {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / segments as f32;
            let x = radius * angle.cos();
            let z = radius * angle.sin();

            positions.push(vec3(x, half_height, z)); // Top cap vertex
            positions.push(vec3(x, -half_height, z)); // Bottom cap vertex
        }

        // Cap faces
        for i in 0..segments {
            let top_idx = top_center_idx + 2 + (i * 2);
            let next_top_idx = top_center_idx + 2 + (((i + 1) % segments) * 2);
            let bottom_idx = top_center_idx + 3 + (i * 2);
            let next_bottom_idx = top_center_idx + 3 + (((i + 1) % segments) * 2);

            // Top triangle
            indices.extend_from_slice(&[top_center_idx, next_top_idx, top_idx]);

            // Bottom triangle
            indices.extend_from_slice(&[top_center_idx + 1, bottom_idx, next_bottom_idx]);
        }

        create_mesh_primitive_with_normals(&positions, &indices)
    }

    pub fn create_cone_mesh_primitive(radius: f32, height: f32, segments: u32) -> MeshPrimitive {
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Add tip vertex
        positions.push(vec3(0.0, height, 0.0));

        // Generate base vertices
        for i in 0..segments {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / segments as f32;
            let x = radius * angle.cos();
            let z = radius * angle.sin();
            positions.push(vec3(x, 0.0, z));
        }

        // Side indices
        for i in 1..=segments {
            let next_index = if i == segments { 1 } else { i + 1 };
            indices.extend_from_slice(&[0, next_index, i]);
        }

        // Base center vertex
        let base_center_idx = positions.len() as u32;
        positions.push(vec3(0.0, 0.0, 0.0));

        // Add base vertices (duplicated for correct normal generation)
        for i in 0..segments {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / segments as f32;
            let x = radius * angle.cos();
            let z = radius * angle.sin();
            positions.push(vec3(x, 0.0, z));
        }

        // Base indices
        for i in 0..segments {
            let base_vertex = base_center_idx + 1 + i;
            let next_vertex = base_center_idx + 1 + ((i + 1) % segments);
            indices.extend_from_slice(&[base_center_idx, base_vertex, next_vertex]);
        }

        create_mesh_primitive_with_normals(&positions, &indices)
    }

    pub fn create_torus_mesh_primitive(
        major_radius: f32,
        minor_radius: f32,
        major_segments: u32,
        minor_segments: u32,
    ) -> MeshPrimitive {
        let mut positions = Vec::new();
        let mut normals = Vec::new();

        // Generate vertices
        for i in 0..=major_segments {
            let major_angle = 2.0 * std::f32::consts::PI * (i as f32) / major_segments as f32;
            let center_x = major_radius * major_angle.cos();
            let center_z = major_radius * major_angle.sin();

            for j in 0..=minor_segments {
                let minor_angle = 2.0 * std::f32::consts::PI * (j as f32) / minor_segments as f32;

                // Position on minor circle
                let x = minor_angle.cos() * minor_radius;
                let y = minor_angle.sin() * minor_radius;

                // Final position
                let pos = vec3(
                    center_x + x * major_angle.cos(),
                    y,
                    center_z + x * major_angle.sin(),
                );

                // Normal points outward from the minor circle center
                let normal = vec3(
                    major_angle.cos() * minor_angle.cos(),
                    minor_angle.sin(),
                    major_angle.sin() * minor_angle.cos(),
                );

                positions.push(pos);
                normals.push(normal);
            }
        }

        // Generate indices with correct winding order
        let mut indices = Vec::new();
        for i in 0..major_segments {
            for j in 0..minor_segments {
                let current = i * (minor_segments + 1) + j;
                let next_major = ((i + 1) % major_segments) * (minor_segments + 1) + j;

                indices.extend_from_slice(&[
                    current,
                    current + 1,
                    next_major,
                    next_major,
                    current + 1,
                    next_major + 1,
                ]);
            }
        }

        create_mesh_primitive_from_arrays(&positions, &normals, &indices)
    }

    /// Creates a mesh primitive from raw position and normal data
    pub fn create_mesh_primitive_from_arrays(
        positions: &[nalgebra_glm::Vec3],
        normals: &[nalgebra_glm::Vec3],
        indices: &[u32],
    ) -> MeshPrimitive {
        let vertices = positions
            .iter()
            .zip(normals.iter())
            .map(|(&position, &normal)| Vertex { position, normal })
            .collect();

        MeshPrimitive {
            vertices,
            indices: indices.to_vec(),
        }
    }

    /// Creates a mesh primitive from positions only, calculating normals
    pub fn create_mesh_primitive_with_normals(
        positions: &[nalgebra_glm::Vec3],
        indices: &[u32],
    ) -> MeshPrimitive {
        let mut vertex_normals = vec![nalgebra_glm::Vec3::zeros(); positions.len()];
        let mut face_counts = vec![0u32; positions.len()];

        // Calculate face normals and accumulate them for shared vertices
        for triangle in indices.chunks(3) {
            let i0 = triangle[0] as usize;
            let i1 = triangle[1] as usize;
            let i2 = triangle[2] as usize;

            let v0 = positions[i0];
            let v1 = positions[i1];
            let v2 = positions[i2];

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let face_normal = nalgebra_glm::normalize(&nalgebra_glm::cross(&edge1, &edge2));

            vertex_normals[i0] += face_normal;
            vertex_normals[i1] += face_normal;
            vertex_normals[i2] += face_normal;

            face_counts[i0] += 1;
            face_counts[i1] += 1;
            face_counts[i2] += 1;
        }

        let vertices = positions
            .iter()
            .enumerate()
            .map(|(i, &position)| {
                let normal = if face_counts[i] > 0 {
                    nalgebra_glm::normalize(&(vertex_normals[i] / face_counts[i] as f32))
                } else {
                    nalgebra_glm::Vec3::y()
                };

                Vertex { position, normal }
            })
            .collect();

        MeshPrimitive {
            vertices,
            indices: indices.to_vec(),
        }
    }
}
