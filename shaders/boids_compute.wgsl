const WORKGROUP_SIZE: u32 = 128u;
const MAX_SPECIES: u32 = 8u;

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
    params_0: vec4<f32>, // max_speed, max_force, separation, alignment
    params_1: vec4<f32>, // cohesion, wander, predator/prey, reserved
    info: vec4<u32>,     // enabled, trail length, shape id, render_size bits
}

struct Params {
    scalars_0: vec4<f32>, // dt, time, bounds, neighbor radius
    scalars_1: vec4<f32>, // separation radius, cell size, goal x, goal y
    scalars_2: vec4<f32>, // goal z, global speed scale, goal weight, obstacle weight
    counts: vec4<u32>,    // agents, species, max neighbours, use spatial hash
    grid: vec4<u32>,      // grid resolution, cell count, max agents per cell, frame index
}

struct SpeciesBuffer {
    items: array<Species, 8>,
}

struct InteractionBuffer {
    rules: array<i32, 64>,
}

@group(0) @binding(0) var<uniform> params: Params;

// Ping-pong storage buffers: the current frame reads `source_agents` and writes
// `destination_agents`. Rust renders the destination buffer, then swaps roles.
@group(0) @binding(1) var<storage, read> source_agents: array<Agent>;
@group(0) @binding(2) var<storage, read_write> destination_agents: array<Agent>;

@group(0) @binding(3) var<storage, read> species_buffer: SpeciesBuffer;
@group(0) @binding(4) var<storage, read> interactions: InteractionBuffer;

// Uniform grid buffers. `grid_cells[cell]` is an atomic occupancy counter;
// `grid_indices[cell * max_cell_agents + slot]` stores agent indices in that cell.
@group(0) @binding(5) var<storage, read_write> grid_cells: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> grid_indices: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn clear_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell = id.x;
    if (cell >= params.grid.y) {
        return;
    }
    atomicStore(&grid_cells[cell], 0u);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn build_grid(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.counts.x) {
        return;
    }

    let agent = source_agents[index];
    let cell = flatten_cell(world_to_cell(agent.position.xyz));
    let slot = atomicAdd(&grid_cells[cell], 1u);
    if (slot < params.grid.z) {
        grid_indices[cell * params.grid.z + slot] = index;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn update_agents(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.counts.x) {
        return;
    }

    var agent = source_agents[index];
    let species_id = min(agent.species_id, max(params.counts.y, 1u) - 1u);
    let species = species_buffer.items[species_id];
    let dt = params.scalars_0.x;
    let bounds = params.scalars_0.z;
    let neighbor_radius = params.scalars_0.w;
    let separation_radius = params.scalars_1.x;
    let goal = vec3<f32>(params.scalars_1.z, params.scalars_1.w, params.scalars_2.x);

    var separation = vec3<f32>(0.0);
    var alignment = vec3<f32>(0.0);
    var cohesion = vec3<f32>(0.0);
    var neighbor_count = 0.0;
    var sampled = 0u;

    if (params.counts.w == 1u) {
        let origin = world_to_cell(agent.position.xyz);
        let cell_radius = max(1i, i32(ceil(neighbor_radius / max(params.scalars_1.y, 0.001))));
        for (var z = -cell_radius; z <= cell_radius; z = z + 1i) {
            for (var y = -cell_radius; y <= cell_radius; y = y + 1i) {
                for (var x = -cell_radius; x <= cell_radius; x = x + 1i) {
                    let cell = clamp_cell(origin + vec3<i32>(x, y, z));
                    let flat = flatten_cell(cell);
                    let count = min(atomicLoad(&grid_cells[flat]), params.grid.z);
                    for (var slot = 0u; slot < count; slot = slot + 1u) {
                        if (sampled >= params.counts.z) {
                            break;
                        }
                        accumulate_neighbor(
                            &separation,
                            &alignment,
                            &cohesion,
                            &neighbor_count,
                            &sampled,
                            index,
                            agent,
                            grid_indices[flat * params.grid.z + slot],
                            neighbor_radius,
                            separation_radius,
                            species,
                        );
                    }
                }
            }
        }
    } else {
        // Debug comparison path: intentionally naive all-pairs scan.
        for (var candidate = 0u; candidate < params.counts.x; candidate = candidate + 1u) {
            if (sampled >= params.counts.z) {
                break;
            }
            accumulate_neighbor(
                &separation,
                &alignment,
                &cohesion,
                &neighbor_count,
                &sampled,
                index,
                agent,
                candidate,
                neighbor_radius,
                separation_radius,
                species,
            );
        }
    }

    var force = vec3<f32>(0.0);
    if (neighbor_count > 0.0) {
        let inv_count = 1.0 / neighbor_count;
        let max_speed = species.params_0.x * params.scalars_2.y;
        let desired_alignment = safe_normalize(alignment * inv_count) * max_speed;
        let center = cohesion * inv_count;
        force += (desired_alignment - agent.velocity.xyz) * species.params_0.w;
        force += safe_normalize(center - agent.position.xyz) * species.params_1.x * max_speed;
    }

    force += safe_normalize(separation) * species.params_0.z * species.params_0.x;
    force += safe_normalize(goal - agent.position.xyz) * params.scalars_2.z * species.params_0.x;
    force += boundary_force(agent.position.xyz, bounds) * species.params_0.x;
    force += wander(agent.random_seed, params.scalars_0.y) * species.params_1.y * species.params_0.y;
    force = clamp_length(force, species.params_0.y);

    var velocity = clamp_length(agent.velocity.xyz + force * dt, species.params_0.x * params.scalars_2.y);
    var position = agent.position.xyz + velocity * dt;

    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        if (position[axis] > bounds) {
            position[axis] = bounds;
            velocity[axis] = -abs(velocity[axis]);
        } else if (position[axis] < -bounds) {
            position[axis] = -bounds;
            velocity[axis] = abs(velocity[axis]);
        }
    }

    agent.position = vec4<f32>(position, 1.0);
    agent.velocity = vec4<f32>(velocity, 0.0);
    agent.energy = clamp(agent.energy + dot(wander(agent.random_seed, params.scalars_0.y), velocity) * 0.0002, 0.2, 1.0);
    destination_agents[index] = agent;
}

fn accumulate_neighbor(
    separation: ptr<function, vec3<f32>>,
    alignment: ptr<function, vec3<f32>>,
    cohesion: ptr<function, vec3<f32>>,
    neighbor_count: ptr<function, f32>,
    sampled: ptr<function, u32>,
    self_index: u32,
    agent: Agent,
    candidate_index: u32,
    neighbor_radius: f32,
    separation_radius: f32,
    species: Species,
) {
    if (candidate_index == self_index || candidate_index >= params.counts.x) {
        return;
    }

    let other = source_agents[candidate_index];
    let offset = other.position.xyz - agent.position.xyz;
    let distance_sq = dot(offset, offset);
    if (distance_sq <= 0.000001 || distance_sq > neighbor_radius * neighbor_radius) {
        return;
    }

    let other_species = min(other.species_id, max(params.counts.y, 1u) - 1u);
    let rule = interactions.rules[agent.species_id * MAX_SPECIES + other_species];
    let distance = max(sqrt(distance_sq), 0.001);
    let direction = offset / distance;
    *sampled = *sampled + 1u;

    if (rule == 1) {
        *alignment = *alignment + other.velocity.xyz;
        *cohesion = *cohesion + other.position.xyz;
        *neighbor_count = *neighbor_count + 1.0;
    } else if (rule == 2) {
        *separation = *separation - direction / distance;
    } else if (rule == 3) {
        *cohesion = *cohesion + other.position.xyz * 1.8;
        *neighbor_count = *neighbor_count + 1.8;
    } else if (rule == 5) {
        *alignment = *alignment + safe_normalize(cross(direction, vec3<f32>(0.0, 1.0, 0.0))) * species.params_0.x;
        *cohesion = *cohesion + other.position.xyz;
        *neighbor_count = *neighbor_count + 1.0;
    }

    if (distance_sq < separation_radius * separation_radius) {
        *separation = *separation - direction / distance;
    }
}

fn world_to_cell(position: vec3<f32>) -> vec3<i32> {
    let resolution = f32(params.grid.x);
    let normalized = floor((position + vec3<f32>(params.scalars_0.z)) / max(params.scalars_1.y, 0.001));
    return clamp_cell(vec3<i32>(normalized));
}

fn clamp_cell(cell: vec3<i32>) -> vec3<i32> {
    let max_cell = i32(params.grid.x) - 1i;
    return clamp(cell, vec3<i32>(0), vec3<i32>(max_cell));
}

fn flatten_cell(cell: vec3<i32>) -> u32 {
    let c = vec3<u32>(cell);
    return c.x + c.y * params.grid.x + c.z * params.grid.x * params.grid.x;
}

fn boundary_force(position: vec3<f32>, bounds: f32) -> vec3<f32> {
    let margin = bounds * 0.18;
    var force = vec3<f32>(0.0);
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let value = position[axis];
        let distance_to_wall = bounds - abs(value);
        if (distance_to_wall < margin) {
            force[axis] = -sign(value) * (1.0 - distance_to_wall / margin);
        }
    }
    return force;
}

fn wander(seed: u32, time: f32) -> vec3<f32> {
    let s = f32(seed & 65535u) * 0.00013;
    return safe_normalize(vec3<f32>(
        sin(time * 1.73 + s),
        cos(time * 1.19 + s * 0.7) * 0.35,
        cos(time * 1.41 + s * 1.3),
    ));
}

fn safe_normalize(value: vec3<f32>) -> vec3<f32> {
    let length_sq = dot(value, value);
    if (length_sq <= 0.000001) {
        return vec3<f32>(0.0);
    }
    return value * inverseSqrt(length_sq);
}

fn clamp_length(value: vec3<f32>, max_length: f32) -> vec3<f32> {
    let length_sq = dot(value, value);
    let max_sq = max_length * max_length;
    if (length_sq > max_sq && length_sq > 0.000001) {
        return value * (max_length * inverseSqrt(length_sq));
    }
    if (length_sq < 0.64 && length_sq > 0.000001) {
        return value * (0.8 * inverseSqrt(length_sq));
    }
    return value;
}
