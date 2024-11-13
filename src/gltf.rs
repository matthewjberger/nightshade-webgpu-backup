#![allow(dead_code)]

use crate::prelude::*;

pub fn import_gltf_bytes(bytes: &[u8]) -> Result<World, gltf::Error> {
    let (gltf, buffers, raw_images) = gltf::import_slice(bytes)?;

    let _images = raw_images
        .into_iter()
        .map(load_gltf_image)
        .collect::<Vec<_>>();

    let _sampler = gltf.samplers().map(load_gltf_sampler).collect::<Vec<_>>();

    let _textures = gltf
        .textures()
        .map(|texture| Texture {
            image_index: texture.source().index(),
            sampler_index: texture.sampler().index(),
        })
        .collect::<Vec<_>>();

    log::info!("Has {} meshes", gltf.meshes().len());
    let mut render_meshes = Vec::new();
    gltf.meshes().for_each(|mesh| {
        let mesh_name = mesh.name().unwrap_or("Unnamed Mesh");
        let mut render_mesh = Vec::new();
        mesh.primitives().for_each(|primitive| {
            let reader = primitive.reader(|buffer| Some(&*buffers[buffer.index()]));

            let mut positions = Vec::new();
            let read_positions = reader
                .read_positions()
                .expect("Failed to read gltf vertex positions");
            read_positions.for_each(|position| {
                positions.push(nalgebra_glm::Vec3::from(position));
            });

            let number_of_vertices = positions.len();
            let normals = reader.read_normals().map_or(
                vec![nalgebra_glm::vec3(0.0, 0.0, 0.0); number_of_vertices],
                |normals| normals.map(nalgebra_glm::Vec3::from).collect::<Vec<_>>(),
            );

            let map_to_vec2 = |coords: gltf::mesh::util::ReadTexCoords| -> Vec<nalgebra_glm::Vec2> {
                coords
                    .into_f32()
                    .map(nalgebra_glm::Vec2::from)
                    .collect::<Vec<_>>()
            };

            let _uv_0 = reader.read_tex_coords(0).map_or(
                vec![nalgebra_glm::vec2(0.0, 0.0); number_of_vertices],
                map_to_vec2,
            );

            let _uv_1 = reader.read_tex_coords(1).map_or(
                vec![nalgebra_glm::vec2(0.0, 0.0); number_of_vertices],
                map_to_vec2,
            );

            let convert_joints = |joints: gltf::mesh::util::ReadJoints| -> Vec<nalgebra_glm::Vec4> {
                joints
                    .into_u16()
                    .map(|joint| {
                        nalgebra_glm::vec4(
                            joint[0] as _,
                            joint[1] as _,
                            joint[2] as _,
                            joint[3] as _,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let _joints_0 = reader.read_joints(0).map_or(
                vec![nalgebra_glm::vec4(0.0, 0.0, 0.0, 0.0); number_of_vertices],
                convert_joints,
            );
            let convert_weights =
                |weights: gltf::mesh::util::ReadWeights| -> Vec<nalgebra_glm::Vec4> {
                    weights.into_f32().map(nalgebra_glm::Vec4::from).collect()
                };
            let _weights_0 = reader.read_weights(0).map_or(
                vec![nalgebra_glm::vec4(1.0, 0.0, 0.0, 0.0); number_of_vertices],
                convert_weights,
            );
            let convert_colors = |colors: gltf::mesh::util::ReadColors| -> Vec<nalgebra_glm::Vec3> {
                colors
                    .into_rgb_f32()
                    .map(nalgebra_glm::Vec3::from)
                    .collect::<Vec<_>>()
            };
            let _colors_0 = reader.read_colors(0).map_or(
                vec![nalgebra_glm::vec3(1.0, 1.0, 1.0); number_of_vertices],
                convert_colors,
            );

            let vertices = positions
                .into_iter()
                .enumerate()
                .map(|(index, position)| crate::world::Vertex {
                    position,
                    normal: normals[index],
                    // uv_0: uv_0[index],
                    // uv_1: uv_1[index],
                    // joint_0: joints_0[index],
                    // weight_0: weights_0[index],
                    // color_0: colors_0[index],
                })
                .collect::<Vec<_>>();

            let indices: Vec<u32> = primitive
                .reader(|buffer| Some(&*buffers[buffer.index()]))
                .read_indices()
                .take()
                .map(|read_indices| read_indices.into_u32().collect())
                .unwrap_or_default();

            let primitive = MeshPrimitive { vertices, indices };

            render_mesh.push(primitive);
        });
        render_meshes.push((mesh_name, render_mesh));
    });

    let mut world = World::default();
    render_meshes.into_iter().for_each(|(name, mesh)| {
        world.resources.meshes.insert(name.to_string(), mesh);
    });

    Ok(world)
}

pub fn load_gltf_image(data: gltf::image::Data) -> Image {
    let img = match data.format {
        gltf::image::Format::R8 => image::DynamicImage::ImageLuma8(
            image::ImageBuffer::from_raw(data.width, data.height, data.pixels.to_vec()).unwrap(),
        ),
        gltf::image::Format::R8G8 => image::DynamicImage::ImageLumaA8(
            image::ImageBuffer::from_raw(data.width, data.height, data.pixels.to_vec()).unwrap(),
        ),
        gltf::image::Format::R8G8B8 => image::DynamicImage::ImageRgb8(
            image::ImageBuffer::from_raw(data.width, data.height, data.pixels.to_vec()).unwrap(),
        ),
        gltf::image::Format::R8G8B8A8 => image::DynamicImage::ImageRgba8(
            image::ImageBuffer::from_raw(data.width, data.height, data.pixels.to_vec()).unwrap(),
        ),
        _ => panic!("Unsupported image format!"),
    };
    let rgba_img = img.to_rgba8();
    let pixels = rgba_img.into_raw();
    Image {
        pixels,
        format: ImageFormat::R8G8B8A8,
        width: data.width,
        height: data.height,
    }
}

pub fn load_gltf_sampler(sampler: gltf::texture::Sampler) -> Sampler {
    let min_filter = sampler
        .min_filter()
        .map(|filter| match filter {
            gltf::texture::MinFilter::Nearest => MinFilter::Nearest,
            gltf::texture::MinFilter::NearestMipmapNearest => MinFilter::NearestMipmapNearest,
            gltf::texture::MinFilter::LinearMipmapNearest => MinFilter::LinearMipmapNearest,
            gltf::texture::MinFilter::Linear => MinFilter::Linear,
            gltf::texture::MinFilter::LinearMipmapLinear => MinFilter::LinearMipmapLinear,
            gltf::texture::MinFilter::NearestMipmapLinear => MinFilter::NearestMipmapLinear,
        })
        .unwrap_or_default();

    let mag_filter = sampler
        .mag_filter()
        .map(|filter| match filter {
            gltf::texture::MagFilter::Linear => MagFilter::Linear,
            gltf::texture::MagFilter::Nearest => MagFilter::Nearest,
        })
        .unwrap_or_default();

    let wrap_s = match sampler.wrap_s() {
        gltf::texture::WrappingMode::ClampToEdge => WrappingMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => WrappingMode::MirroredRepeat,
        gltf::texture::WrappingMode::Repeat => WrappingMode::Repeat,
    };

    let wrap_t = match sampler.wrap_t() {
        gltf::texture::WrappingMode::ClampToEdge => WrappingMode::ClampToEdge,
        gltf::texture::WrappingMode::MirroredRepeat => WrappingMode::MirroredRepeat,
        gltf::texture::WrappingMode::Repeat => WrappingMode::Repeat,
    };

    Sampler {
        min_filter,
        mag_filter,
        wrap_s,
        wrap_t,
    }
}
