struct Camera {
    view_proj: mat4x4<f32>,
    eye_position: vec4<f32>,
    params: vec4<f32>,
}

struct Agent {
    position: vec4<f32>,
    velocity: vec4<f32>,
    species_id: u32,
    random_seed: u32,
    energy: f32,
    _pad: u32,
}

struct Species {
    color: vec4<f32>,
    params_0: vec4<f32>,
    params_1: vec4<f32>,
    info: vec4<u32>,
}

struct SpeciesBuffer {
    items: array<Species, 8>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(0)
var<storage, read> agents: array<Agent>;

@group(1) @binding(1)
var<storage, read> species_buffer: SpeciesBuffer;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @builtin(instance_index) instance_index: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let agent = agents[input.instance_index];
    let species = species_buffer.items[min(agent.species_id, 7u)];
    let render_size = bitcast<f32>(species.info.w);
    let local = shape_local(input.position, species.info.z) * render_size;
    let basis = orientation_basis(agent.velocity.xyz);
    let world_position =
        agent.position.xyz +
        basis[0] * local.x +
        basis[1] * local.y +
        basis[2] * local.z;
    let world_normal = normalize(
        basis[0] * input.normal.x +
        basis[1] * input.normal.y +
        basis[2] * input.normal.z
    );

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.world_position = world_position;
    out.normal = world_normal;
    out.color = species.color * vec4<f32>(0.75 + agent.energy * 0.35, 0.75 + agent.energy * 0.35, 0.75 + agent.energy * 0.35, 1.0);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(input.normal);
    let light = normalize(vec3<f32>(-0.32, 0.78, 0.46));
    let view_dir = normalize(camera.eye_position.xyz - input.world_position);
    let diffuse = max(dot(normal, light), 0.0);
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.2);
    let half_vec = normalize(light + view_dir);
    let spec = pow(max(dot(normal, half_vec), 0.0), 42.0);
    let pulse = 0.92 + 0.08 * sin(camera.params.x * 1.7 + input.world_position.y * 0.05);
    var rgb = input.color.rgb * (0.22 + diffuse * 0.78) * pulse;
    rgb += vec3<f32>(0.22, 0.55, 1.0) * rim * 0.28;
    rgb += vec3<f32>(1.0, 0.92, 0.76) * spec * 0.34;
    let distance = length(camera.eye_position.xyz - input.world_position);
    let fog = smoothstep(camera.params.y * 1.3, camera.params.y * 3.0, distance);
    let fog_color = vec3<f32>(0.018, 0.024, 0.038);
    return vec4<f32>(mix(rgb, fog_color, fog), input.color.a);
}

fn shape_local(position: vec3<f32>, shape_id: u32) -> vec3<f32> {
    var p = position;
    if (shape_id == 1u) {
        p.x *= 1.85;
        p.z *= 0.82;
        p.y *= 0.72;
    } else if (shape_id == 2u) {
        p.x *= 1.28;
        p.y *= 1.16;
        p.z *= 1.75;
    } else if (shape_id == 3u) {
        p.x *= 1.45;
        p.y *= 1.45;
        p.z *= 0.72;
    } else if (shape_id == 4u) {
        p.y *= 1.65;
        p.z *= 1.28;
    }
    return p;
}

fn orientation_basis(velocity: vec3<f32>) -> mat3x3<f32> {
    let forward = select(vec3<f32>(0.0, 0.0, 1.0), normalize(velocity), dot(velocity, velocity) > 0.0001);
    let up_seed = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(forward.y) > 0.95);
    let right = normalize(cross(up_seed, forward));
    let up = normalize(cross(forward, right));
    return mat3x3<f32>(right, up, forward);
}
