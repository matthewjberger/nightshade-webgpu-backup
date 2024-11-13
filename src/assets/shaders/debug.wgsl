struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) instance_start: vec3<f32>,
    @location(2) instance_end: vec3<f32>,
    @location(3) instance_color: vec4<f32>,
    @location(4) instance_thickness: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

struct Uniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    camera_position: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vertex_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Calculate line direction in world space
    let line_dir = normalize(in.instance_end - in.instance_start);

    // Get the base position along the line
    let base_pos = mix(in.instance_start, in.instance_end, in.position.x);

    // Calculate view direction from the current point to the camera
    let to_camera = normalize(uniforms.camera_position.xyz - base_pos);

    // Calculate the perpendicular direction for thickness that faces the camera
    var thickness_dir = normalize(cross(line_dir, to_camera));

    // If the line is nearly parallel to the view direction, use a different approach
    if (abs(dot(line_dir, to_camera)) > 0.99) {
        // In this case, use any perpendicular vector to the line
        let temp = vec3<f32>(1.0, 0.0, 0.0);
        if (abs(dot(line_dir, temp)) > 0.99) {
            thickness_dir = normalize(cross(line_dir, vec3<f32>(0.0, 1.0, 0.0)));
        } else {
            thickness_dir = normalize(cross(line_dir, temp));
        }
    }

    // Apply thickness offset
    let offset = thickness_dir * in.instance_thickness * in.position.y;
    let final_pos = base_pos + offset;

    // Transform to clip space
    out.clip_position = uniforms.projection * uniforms.view * vec4<f32>(final_pos, 1.0);
    out.color = in.instance_color;

    return out;
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
