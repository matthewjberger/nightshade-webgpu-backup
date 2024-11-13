struct Uniform {
    view_proj: mat4x4<f32>,
    camera_world_pos: vec3<f32>,
    grid_size: f32,
    grid_min_pixels: f32,
    grid_cell_size: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> ubo: Uniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = vec3<f32>(0.0);

    switch vertex_index {
        case 0u: { pos = vec3<f32>(-10.0, 0.0, -10.0); }
        case 1u: { pos = vec3<f32>(10.0, 0.0, -10.0); }
        case 2u: { pos = vec3<f32>(-10.0, 0.0, 10.0); }
        case 3u: { pos = vec3<f32>(-10.0, 0.0, 10.0); }
        case 4u: { pos = vec3<f32>(10.0, 0.0, -10.0); }
        case 5u: { pos = vec3<f32>(10.0, 0.0, 10.0); }
        default: {}
    }

    pos = pos * ubo.grid_size;
    let world_pos = vec3<f32>(
        pos.x + ubo.camera_world_pos.x,
        0.0,
        pos.z + ubo.camera_world_pos.z
    );

    var output: VertexOutput;
    output.clip_position = ubo.view_proj * vec4<f32>(world_pos, 1.0);
    output.world_pos = world_pos;
    return output;
}

fn mod_pos(pos: f32, size: f32) -> f32 {
    return pos - size * floor(pos / size);
}


@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dvx = vec2<f32>(dpdx(in.world_pos.x), dpdy(in.world_pos.x));
    let dvy = vec2<f32>(dpdx(in.world_pos.z), dpdy(in.world_pos.z));
    let lx = length(dvx);
    let ly = length(dvy);
    let dudv = vec2<f32>(lx, ly);
    let l = length(dudv);

    let lod = max(0.0, log10(l * ubo.grid_min_pixels / ubo.grid_cell_size) + 1.0);
    let cell_size_lod0 = ubo.grid_cell_size * pow(10.0, floor(lod));
    let cell_size_lod1 = cell_size_lod0 * 10.0;
    let cell_size_lod2 = cell_size_lod1 * 10.0;

    let dudv4 = dudv * 8.0;  // Increased to make lines thinner

    let mod_lod0 = vec2<f32>(
        mod_pos(in.world_pos.x, cell_size_lod0),
        mod_pos(in.world_pos.z, cell_size_lod0)
    ) / dudv4;
    let lod0_alpha = max2(vec2<f32>(1.0) - abs(saturate(mod_lod0) * 2.0 - vec2<f32>(1.0)));

    let mod_lod1 = vec2<f32>(
        mod_pos(in.world_pos.x, cell_size_lod1),
        mod_pos(in.world_pos.z, cell_size_lod1)
    ) / dudv4;
    let lod1_alpha = max2(vec2<f32>(1.0) - abs(saturate(mod_lod1) * 2.0 - vec2<f32>(1.0)));

    let mod_lod2 = vec2<f32>(
        mod_pos(in.world_pos.x, cell_size_lod2),
        mod_pos(in.world_pos.z, cell_size_lod2)
    ) / dudv4;
    let lod2_alpha = max2(vec2<f32>(1.0) - abs(saturate(mod_lod2) * 2.0 - vec2<f32>(1.0)));

    let lod_fade = fract(lod);

    // Much more subtle base grid
    let grid_color_thin = vec4<f32>(0.75, 0.75, 0.75, 0.25);
    // More subtle major lines with reduced opacity
    let grid_color_thick = vec4<f32>(0.2, 0.4, 0.8, 0.4);

    var color: vec4<f32>;
    if (lod2_alpha > 0.0) {
        color = grid_color_thick;
        color.a *= lod2_alpha * 0.8;
    } else if (lod1_alpha > 0.0) {
        let fade = smoothstep(0.2, 0.8, lod_fade);
        color = mix(grid_color_thick, grid_color_thin, fade);
        color.a *= lod1_alpha * 0.6;
    } else {
        color = grid_color_thin;
        color.a *= (lod0_alpha * (1.0 - lod_fade)) * 0.5;
    }

    // Gentler, longer falloff
    let dist = length(in.world_pos.xz - ubo.camera_world_pos.xz);
    let opacity_falloff = 1.0 - smoothstep(0.8 * ubo.grid_size, ubo.grid_size * 3.0, dist);
    color.a *= opacity_falloff;

    // Check for axis proximity with much thinner detection
    let x_axis_nearby = abs(in.world_pos.z) < 0.03;
    let z_axis_nearby = abs(in.world_pos.x) < 0.03;

    // Add axes on top of grid
    if (x_axis_nearby) {
        color = mix(color, vec4<f32>(0.87, 0.26, 0.24, 0.5), 0.5);
    }
    if (z_axis_nearby) {
        color = mix(color, vec4<f32>(0.24, 0.7, 0.29, 0.5), 0.5);
    }

    if (color.a < 0.02) {
        discard;
    }

    return color;
}


fn log10(x: f32) -> f32 {
    return log2(x) / log2(10.0);
}

fn saturate(x: vec2<f32>) -> vec2<f32> {
    return clamp(x, vec2<f32>(0.0), vec2<f32>(1.0));
}

fn max2(v: vec2<f32>) -> f32 {
    return max(v.x, v.y);
}
