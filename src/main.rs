use egui::{Context, RichText, SidePanel, TopBottomPanel};
use nalgebra_glm::*;
use nightshade_core::prelude::*;

mod gltf;
mod graphics;
mod launch;
mod world;

pub mod prelude {
    pub use crate::{gltf::*, graphics::*, launch::*, world::*};

    pub use egui;
    pub use gilrs;
    pub use log;
    pub use nalgebra_glm as math;
    pub use winit;
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    launch(App::default())?;
    Ok(())
}

#[derive(Default)]
pub struct App {
    selected_entity: Option<EntityId>,
}

impl State for App {
    fn initialize(&mut self, world: &mut World) {
        let _world = import_gltf_bytes(include_bytes!("assets/gltf/Lantern.glb"))
            .expect("Failed to load GLTF asset!");
        for (mesh_name, mesh_primitives) in _world.resources.meshes.iter() {
            log::info!("Mesh name: {mesh_name}");
            world
                .resources
                .meshes
                .insert(mesh_name.clone(), mesh_primitives.clone());
        }

        let sun_entity = spawn_entities(
            world,
            LOCAL_TRANSFORM | GLOBAL_TRANSFORM | NAME | VISIBLE | LIGHT,
            1,
        )[0];

        if let Some(light) = get_component_mut::<Light>(world, sun_entity, LIGHT) {
            *light = Light::Spot(SpotLight {
                color: Vec3::new(1.0, 0.95, 0.8), // Warm sunlight color
                intensity: 50.0,
                range: 100.0,       // Longer range
                inner_cutoff: 45.0, // Wider beam
                outer_cutoff: 50.0,
            })
        }
        if let Some(name) = get_component_mut::<Name>(world, sun_entity, NAME) {
            *name = Name("Sun".to_string());
        }
        if let Some(local_transform) =
            get_component_mut::<LocalTransform>(world, sun_entity, LOCAL_TRANSFORM)
        {
            local_transform.translation = vec3(-50.0, 50.0, -50.0); // Position sun high and to the side
        }

        let camera_entity = spawn_entities(
            world,
            ACTIVE_CAMERA | CAMERA | LOCAL_TRANSFORM | GLOBAL_TRANSFORM | NAME | VISIBLE | PLAYER,
            1,
        )[0];

        if let Some(name) = get_component_mut::<Name>(world, camera_entity, NAME) {
            *name = Name("Main Camera".to_string());
        }

        if let Some(local_transform) =
            get_component_mut::<LocalTransform>(world, camera_entity, LOCAL_TRANSFORM)
        {
            local_transform.translation = vec3(0.0, 0.0, 10.0);
        }

        load_primitive_meshes(world);
    }

    fn update(&mut self, world: &mut World) {
        gamepad_controls_system(world);
        fps_camera_controls_system(world);
        wasd_keyboard_controls_system(world);
        update_global_transforms_system(world);
    }

    fn ui(&mut self, world: &mut World, context: &Context) {
        TopBottomPanel::top("menu").show(context, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("FPS: {}", world.resources.frames_per_second));
                ui.separator();
                ui.label(format!(
                    "Lights: {}/{}",
                    query_entities(world, LIGHT).len(),
                    MAX_LIGHTS
                ));
                ui.separator();
                ui.label(format!("Entities: {}", query_entities(world, ALL).len()));
                ui.separator();
            })
        });

        SidePanel::left("Scene Tree").show(context, |ui| {
            egui::ScrollArea::vertical()
                .id_salt(ui.next_auto_id())
                .show(ui, |ui| {
                    ui.collapsing("Scene Tree", |ui| {
                        egui::ScrollArea::vertical()
                            .id_salt(ui.next_auto_id())
                            .show(ui, |ui| {
                                ui.group(|ui| {
                                    if ui.button("New Entity").clicked() {
                                        spawn_entities(world, VISIBLE, 1);
                                    }
                                    query_root_nodes(world).into_iter().for_each(|entity| {
                                        self.render_entity_tree(world, ui, entity);
                                    });
                                });
                            });
                    });
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .id_salt(ui.next_auto_id())
                        .show(ui, |ui| {
                            if let Some(selected_entity) = self.selected_entity {
                                ui.label(format!("Selected Entity: {selected_entity}"));
                                inspector_ui(selected_entity, ui, world);
                            }
                        });
                });
        });
    }
}

fn load_primitive_meshes(world: &mut World) {
    vec![
        (
            "Torus",
            meshgen::create_torus_mesh_primitive(2.0, 0.5, 32, 16),
        ),
        ("Cube", meshgen::create_cube_mesh_primitive()),
        ("Sphere", meshgen::create_sphere_mesh_primitive(1.5, 12)),
        (
            "Cylinder",
            meshgen::create_cylinder_mesh_primitive(0.5, 1.0, 12),
        ),
        ("Plane", meshgen::create_plane_mesh_primitive(4.0)),
        ("Cone", meshgen::create_cone_mesh_primitive(1.0, 2.0, 12)),
    ]
    .into_iter()
    .for_each(|(name, mesh)| {
        world.resources.meshes.insert(name.to_string(), vec![mesh]);
    });
}

impl App {
    fn render_entity_tree(&mut self, world: &mut World, ui: &mut egui::Ui, entity: EntityId) {
        let name = match get_component::<Name>(world, entity, NAME) {
            Some(Name(name)) if !name.is_empty() => name.to_string(),
            _ => "Entity".to_string(),
        };

        let selected = self.selected_entity == Some(entity);

        let id = ui.make_persistent_id(ui.next_auto_id());
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
            .show_header(ui, |ui| {
                ui.horizontal(|ui| {
                    let has_camera = get_component::<Camera>(world, entity, CAMERA).is_some();
                    let prefix = if has_camera { "🎥" } else { "" }.to_string();
                    let response = ui.selectable_label(selected, format!("{prefix}{name}"));

                    if response.clicked() {
                        self.selected_entity = Some(entity);
                    }

                    response.context_menu(|ui| {
                        if ui.button("Add Child").clicked() {
                            let child = spawn_entities(world, PARENT | VISIBLE, 1)[0];
                            if let Some(parent) = get_component_mut::<Parent>(world, child, PARENT)
                            {
                                *parent = Parent(entity);
                            }
                            ui.close_menu();
                        }
                        if ui.button("Remove").clicked() {
                            despawn_entities(world, &[entity]);
                            ui.close_menu();
                        }
                        if has_camera && ui.button("View Camera").clicked() {
                            activate_camera(world, entity);
                        }
                    });
                });
            })
            .body(|ui| {
                egui::ScrollArea::horizontal()
                    .id_salt(ui.next_auto_id())
                    .show(ui, |ui| {
                        query_children(world, entity).into_iter().for_each(|child| {
                            self.render_entity_tree(world, ui, child);
                        });
                    });
            });
    }
}

use inspector::*;
mod inspector {
    use world::RENDER_MESH;

    use super::*;

    pub fn inspector_ui(entity: EntityId, ui: &mut egui::Ui, world: &mut World) {
        ui.collapsing("Name", |ui| {
            name_inspector_ui(ui, entity, world);
        });
        ui.collapsing("Local Transform", |ui| {
            local_transform_inspector_ui(ui, entity, world);
        });
        ui.collapsing("Camera", |ui| {
            camera_inspector_ui(ui, entity, world);
        });
        ui.collapsing("Render Mesh", |ui| {
            render_mesh_inspector_ui(ui, entity, world);
        });
        ui.collapsing("Light", |ui| {
            light_inspector_ui(ui, entity, world);
        });
    }

    pub fn name_inspector_ui(ui: &mut egui::Ui, selected_entity: EntityId, world: &mut World) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label("Name");
                match get_component_mut::<Name>(world, selected_entity, NAME) {
                    Some(Name(name)) => {
                        ui.text_edit_singleline(name);
                        if ui.button("Remove").clicked() {
                            remove_components(world, selected_entity, NAME);
                        }
                    }
                    None => {
                        if ui.button("Add Name").clicked() {
                            add_components(world, selected_entity, NAME);
                        }
                    }
                }
            });
        });
    }

    pub fn local_transform_inspector_ui(
        ui: &mut egui::Ui,
        selected_entity: EntityId,
        world: &mut World,
    ) {
        ui.group(|ui| {
            ui.label("Local Transform");
            match get_component_mut::<LocalTransform>(world, selected_entity, LOCAL_TRANSFORM) {
                Some(local_transform) => {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label("Translation");
                            ui.label("x");
                            ui.add(
                                egui::DragValue::new(&mut local_transform.translation.x).speed(0.1),
                            );
                            ui.label("y");
                            ui.add(
                                egui::DragValue::new(&mut local_transform.translation.y).speed(0.1),
                            );
                            ui.label("z");
                            ui.add(
                                egui::DragValue::new(&mut local_transform.translation.z).speed(0.1),
                            );
                            ui.label("");
                            ui.label("");
                        });
                    });
                    if ui.button("Remove").clicked() {
                        remove_components(world, selected_entity, LOCAL_TRANSFORM);
                    }
                    if get_component::<GlobalTransform>(world, selected_entity, GLOBAL_TRANSFORM)
                        .is_none()
                        && ui.button("Add Global Transform").clicked()
                    {
                        add_components(world, selected_entity, GLOBAL_TRANSFORM);
                    }
                }
                None => {
                    if ui.button("Add Local Transform").clicked() {
                        add_components(world, selected_entity, LOCAL_TRANSFORM);
                    }
                }
            }
        });
    }

    pub fn camera_inspector_ui(ui: &mut egui::Ui, selected_entity: EntityId, world: &mut World) {
        ui.group(|ui| {
            ui.label("Camera");
            match get_component_mut::<Camera>(world, selected_entity, CAMERA) {
                Some(camera) => {
                    match &camera.projection {
                        Projection::Perspective(_perspective_camera) => {
                            ui.label("Projection is `Perspective`");
                        }
                        Projection::Orthographic(_orthographic_camera) => {
                            ui.label("Projection is `Orhographic`");
                        }
                    }

                    if ui.button("Remove").clicked() {
                        remove_components(world, selected_entity, CAMERA);
                    }
                }
                None => {
                    if ui.button("Add Camera").clicked() {
                        add_components(world, selected_entity, CAMERA);
                    }
                }
            }
        });
    }

    pub fn render_mesh_inspector_ui(ui: &mut egui::Ui, entity: EntityId, world: &mut World) {
        ui.group(|ui| {
            ui.label("Render Mesh");

            if get_component::<LocalTransform>(world, entity, LOCAL_TRANSFORM).is_none() {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("⚠ No local transform! ⚠").color(ui.visuals().warn_fg_color),
                    )
                    .on_hover_text("A local transform is required for a render mesh to appear");
                    if ui.button("Add Local Transform").clicked() {
                        add_components(world, entity, LOCAL_TRANSFORM);
                    }
                });
            }

            if let Some(render_mesh) = get_component_mut::<RenderMesh>(world, entity, RENDER_MESH) {
                ui.horizontal(|ui| {
                    ui.label("Mesh Name");
                    ui.text_edit_singleline(&mut render_mesh.mesh_name);
                });
            }

            if get_component_mut::<RenderMesh>(world, entity, RENDER_MESH).is_some() {
                let mut next_mesh = None;
                ui.group(|ui| {
                    ui.label("Meshes");
                    egui::ScrollArea::vertical()
                        .id_salt(ui.next_auto_id())
                        .show(ui, |ui| {
                            world.resources.meshes.keys().for_each(|mesh| {
                                if ui.button(mesh).clicked() {
                                    next_mesh = Some(mesh.to_string());
                                }
                            });
                        });
                });
                if let Some(mesh_name) = next_mesh.take() {
                    if let Some(render_mesh) =
                        get_component_mut::<RenderMesh>(world, entity, RENDER_MESH)
                    {
                        render_mesh.mesh_name = mesh_name.to_string();
                    }
                }
            }

            if get_component_mut::<RenderMesh>(world, entity, RENDER_MESH).is_none() {
                if ui.button("Add Render Mesh").clicked() {
                    add_components(world, entity, RENDER_MESH);
                }
            } else if ui.button("Remove").clicked() {
                remove_components(world, entity, RENDER_MESH);
            }
        });
    }

    // TODO: make these return `Message` which are either `Command` or `Event`
    pub fn light_inspector_ui(ui: &mut egui::Ui, entity: EntityId, world: &mut World) {
        ui.group(|ui| {
            ui.label("Light");
            match get_component_mut::<Light>(world, entity, LIGHT) {
                Some(light) => {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label("Type:");
                            if ui.button("Point").clicked() {
                                *light = Light::Point(PointLight {
                                    color: math::vec3(1.0, 1.0, 1.0),
                                    intensity: 1.0,
                                    range: 10.0,
                                });
                            }
                            if ui.button("Directional").clicked() {
                                *light = Light::Directional(DirectionalLight {
                                    color: math::vec3(1.0, 1.0, 1.0),
                                    intensity: 1.0,
                                });
                            }
                            if ui.button("Spot").clicked() {
                                *light = Light::Spot(SpotLight {
                                    color: math::vec3(1.0, 1.0, 1.0),
                                    intensity: 1.0,
                                    range: 10.0,
                                    inner_cutoff: 30.0_f32.to_radians(),
                                    outer_cutoff: 45.0_f32.to_radians(),
                                });
                            }
                        });

                        match light {
                            Light::Point(point) => {
                                ui.label("Point Light");
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Color");
                                        ui.label("R");
                                        ui.add(egui::DragValue::new(&mut point.color.x).speed(0.1));
                                        ui.label("G");
                                        ui.add(egui::DragValue::new(&mut point.color.y).speed(0.1));
                                        ui.label("B");
                                        ui.add(egui::DragValue::new(&mut point.color.z).speed(0.1));
                                    });
                                });
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Intensity");
                                        ui.add(
                                            egui::DragValue::new(&mut point.intensity).speed(0.1),
                                        );
                                    });
                                });
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Range");
                                        ui.add(egui::DragValue::new(&mut point.range).speed(0.1));
                                    });
                                });
                            }
                            Light::Directional(dir) => {
                                ui.label("Directional Light");
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Color");
                                        ui.label("R");
                                        ui.add(egui::DragValue::new(&mut dir.color.x).speed(0.1));
                                        ui.label("G");
                                        ui.add(egui::DragValue::new(&mut dir.color.y).speed(0.1));
                                        ui.label("B");
                                        ui.add(egui::DragValue::new(&mut dir.color.z).speed(0.1));
                                    });
                                });
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Intensity");
                                        ui.add(egui::DragValue::new(&mut dir.intensity).speed(0.1));
                                    });
                                });
                            }
                            Light::Spot(spot) => {
                                ui.label("Spot Light");
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Color");
                                        ui.label("R");
                                        ui.add(egui::DragValue::new(&mut spot.color.x).speed(0.1));
                                        ui.label("G");
                                        ui.add(egui::DragValue::new(&mut spot.color.y).speed(0.1));
                                        ui.label("B");
                                        ui.add(egui::DragValue::new(&mut spot.color.z).speed(0.1));
                                    });
                                });
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Intensity");
                                        ui.add(
                                            egui::DragValue::new(&mut spot.intensity).speed(0.1),
                                        );
                                    });
                                });
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Range");
                                        ui.add(egui::DragValue::new(&mut spot.range).speed(0.1));
                                    });
                                });
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        let mut inner = spot.inner_cutoff.to_degrees();
                                        let mut outer = spot.outer_cutoff.to_degrees();
                                        ui.label("Inner Cutoff");
                                        if ui
                                            .add(egui::DragValue::new(&mut inner).speed(1.0))
                                            .changed()
                                        {
                                            spot.inner_cutoff = inner.to_radians();
                                        }
                                        ui.label("Outer Cutoff");
                                        if ui
                                            .add(egui::DragValue::new(&mut outer).speed(1.0))
                                            .changed()
                                        {
                                            spot.outer_cutoff = outer.to_radians();
                                        }
                                    });
                                });
                            }
                        }
                    });
                    if ui.button("Remove").clicked() {
                        remove_components(world, entity, LIGHT);
                    }
                }
                None => {
                    if ui.button("Add Light").clicked() {
                        add_components(world, entity, LIGHT);
                    }
                }
            }
        });
    }
}
