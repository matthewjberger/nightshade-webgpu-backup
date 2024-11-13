struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct InstanceInput {
    @location(2) model_matrix_0: vec4<f32>,
    @location(3) model_matrix_1: vec4<f32>,
    @location(4) model_matrix_2: vec4<f32>,
    @location(5) model_matrix_3: vec4<f32>,
    @location(6) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frag_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

struct SceneUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_position: vec4<f32>,
}

struct GpuLight {
    position: vec4<f32>,    // w component used for type: 0=dir, 1=point, 2=spot
    direction: vec4<f32>,   // xyz = direction, w unused
    ambient: vec4<f32>,     // xyz = ambient color, w unused
    diffuse: vec4<f32>,     // xyz = diffuse color, w unused
    specular: vec4<f32>,    // xyz = specular color, w unused
    params: vec4<f32>,      // x=constant, y=linear, z=quadratic, w=shininess
    cutoffs: vec4<f32>,     // x=cutoff, y=outerCutoff, z/w unused
}

struct LightingUniform {
    lights: array<GpuLight, 16>,
    ambient_color: vec4<f32>,
    num_lights: u32,
}

@group(0) @binding(0) var<uniform> scene: SceneUniform;
@group(0) @binding(1) var<uniform> lighting: LightingUniform;

@vertex
fn vertex_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    let normal_matrix = mat3x3<f32>(
        model[0].xyz,
        model[1].xyz,
        model[2].xyz,
    );

    var out: VertexOutput;
    let world_pos = (model * vec4<f32>(vertex.position, 1.0)).xyz;
    out.clip_position = scene.projection * scene.view * vec4<f32>(world_pos, 1.0);
    out.frag_pos = world_pos;
    out.normal = normalize(normal_matrix * vertex.normal);
    out.color = instance.color;
    return out;
}

fn calculate_directional_light(light: GpuLight, normal: vec3<f32>, view_dir: vec3<f32>, material_color: vec4<f32>) -> vec3<f32> {
    let light_dir = normalize(-light.direction.xyz);

    // Diffuse
    let diff = max(dot(normal, light_dir), 0.0);

    // Specular
    let reflect_dir = reflect(-light_dir, normal);
    let spec = pow(max(dot(view_dir, reflect_dir), 0.0), light.params.w);

    // Combine
    let ambient = light.ambient.xyz * material_color.rgb;
    let diffuse = light.diffuse.xyz * diff * material_color.rgb;
    let specular = light.specular.xyz * spec * material_color.rgb;

    return ambient + diffuse + specular;
}

fn calculate_point_light(light: GpuLight, normal: vec3<f32>, frag_pos: vec3<f32>, view_dir: vec3<f32>, material_color: vec4<f32>) -> vec3<f32> {
    let light_dir = normalize(light.position.xyz - frag_pos);

    // Diffuse shading
    let diff = max(dot(normal, light_dir), 0.0);

    // Specular shading
    let reflect_dir = reflect(-light_dir, normal);
    let spec = pow(max(dot(view_dir, reflect_dir), 0.0), light.params.w);

    // Attenuation - using precalculated constants based on range
    let distance = length(light.position.xyz - frag_pos);
    let attenuation = 1.0 / (light.params.x + light.params.y * distance + light.params.z * (distance * distance));

    // Combine results (ambient small, diffuse main, specular highlight)
    let ambient = light.ambient.xyz * 0.1 * material_color.rgb;
    let diffuse = light.diffuse.xyz * diff * material_color.rgb;
    let specular = light.specular.xyz * spec * material_color.rgb;

    return (ambient + diffuse + specular) * attenuation;
}

fn calculate_spot_light(light: GpuLight, normal: vec3<f32>, frag_pos: vec3<f32>, view_dir: vec3<f32>, material_color: vec4<f32>) -> vec3<f32> {
    let light_dir = normalize(light.position.xyz - frag_pos);

    // Spot angle calculation
    let theta = dot(light_dir, normalize(-light.direction.xyz));
    let epsilon = light.cutoffs.x - light.cutoffs.y;
    let intensity = clamp((theta - light.cutoffs.y) / epsilon, 0.0, 1.0);

    // Diffuse shading
    let diff = max(dot(normal, light_dir), 0.0);

    // Specular shading
    let reflect_dir = reflect(-light_dir, normal);
    let spec = pow(max(dot(view_dir, reflect_dir), 0.0), light.params.w);

    // Attenuation - using precalculated constants based on range
    let distance = length(light.position.xyz - frag_pos);
    let attenuation = 1.0 / (light.params.x + light.params.y * distance + light.params.z * (distance * distance));

    // Combine results (ambient small, diffuse main, specular highlight)
    let ambient = light.ambient.xyz * 0.1 * material_color.rgb;
    let diffuse = light.diffuse.xyz * diff * material_color.rgb;
    let specular = light.specular.xyz * spec * material_color.rgb;

    return (ambient + diffuse + specular) * attenuation * intensity;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.normal);
    let view_dir = normalize(scene.camera_position.xyz - in.frag_pos);

    var result = vec3<f32>(0.0, 0.0, 0.0);

    for(var i = 0u; i < lighting.num_lights; i++) {
        let light = lighting.lights[i];

        switch(u32(light.position.w)) {
            case 0u: { // Directional light
                result += calculate_directional_light(light, normal, view_dir, in.color);
            }
            case 1u: { // Point light
                result += calculate_point_light(light, normal, in.frag_pos, view_dir, in.color);
            }
            case 2u: { // Spot light
                result += calculate_spot_light(light, normal, in.frag_pos, view_dir, in.color);
            }
            default: {}
        }
    }

    return vec4<f32>(result, in.color.a);
}
