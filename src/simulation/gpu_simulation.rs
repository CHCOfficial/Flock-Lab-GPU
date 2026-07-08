use crate::simulation::{
    agent::{Agent, GpuAgent},
    cpu_simulation::{SimulationSettings, SimulationStats},
    species::{gpu_interaction_array, gpu_species_array, MAX_SPECIES},
};
use anyhow::{bail, Result};
use bytemuck::{Pod, Zeroable};
use std::time::Instant;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 128;
const MAX_CELL_AGENTS: u32 = 64;
const MAX_GRID_RESOLUTION: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationBackend {
    Cpu,
    Gpu,
}

impl SimulationBackend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU spatial hash",
            Self::Gpu => "GPU compute",
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    scalars_0: [f32; 4],
    scalars_1: [f32; 4],
    scalars_2: [f32; 4],
    counts: [u32; 4],
    grid: [u32; 4],
}

pub struct GpuSimulation {
    pipelines: ComputePipelines,
    params_buffer: wgpu::Buffer,
    species_buffer: wgpu::Buffer,
    interaction_buffer: wgpu::Buffer,
    grid_cells: wgpu::Buffer,
    grid_indices: wgpu::Buffer,
    agent_buffers: [wgpu::Buffer; 2],
    bind_groups: [wgpu::BindGroup; 2],
    source_index: usize,
    agent_capacity: usize,
    agent_count: usize,
    grid_resolution: u32,
    total_cells: u32,
    frame_index: u32,
    pending_swap: bool,
    pub stats: SimulationStats,
}

impl GpuSimulation {
    pub fn new(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: &SimulationSettings,
        agents: &[Agent],
    ) -> Result<Self> {
        let downlevel = adapter.get_downlevel_capabilities();
        if !downlevel.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
            bail!("this adapter does not expose compute shaders");
        }

        let pipelines = ComputePipelines::new(device);
        let params = params_for(settings, agents.len(), 0.0, 0.0, 0);
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu simulation params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let species = gpu_species_array(&settings.species);
        let species_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu simulation species"),
            contents: bytemuck::cast_slice(&species),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let interactions = gpu_interaction_array(&settings.interactions);
        let interaction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu simulation interactions"),
            contents: bytemuck::cast_slice(&interactions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let (grid_resolution, total_cells) = grid_dimensions(settings);
        let (grid_cells, grid_indices) = create_grid_buffers(device, total_cells);
        let agent_capacity = agents.len().max(1).next_power_of_two();
        let initial_agents = padded_agents(agents, agent_capacity);

        let agent_buffers = [
            create_agent_buffer(device, "gpu agents a", &initial_agents),
            create_agent_buffer(device, "gpu agents b", &initial_agents),
        ];
        let bind_groups = create_bind_groups(
            device,
            &pipelines.layout,
            &params_buffer,
            &agent_buffers,
            &species_buffer,
            &interaction_buffer,
            &grid_cells,
            &grid_indices,
        );

        queue.write_buffer(&grid_cells, 0, &vec![0u8; total_cells as usize * 4]);

        Ok(Self {
            pipelines,
            params_buffer,
            species_buffer,
            interaction_buffer,
            grid_cells,
            grid_indices,
            agent_buffers,
            bind_groups,
            source_index: 0,
            agent_capacity,
            agent_count: agents.len(),
            grid_resolution,
            total_cells,
            frame_index: 0,
            pending_swap: false,
            stats: SimulationStats {
                grid_cell_count: total_cells as usize,
                ..SimulationStats::default()
            },
        })
    }

    pub fn reseed(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: &SimulationSettings,
        agents: &[Agent],
    ) {
        let (grid_resolution, total_cells) = grid_dimensions(settings);
        let needs_grid = grid_resolution != self.grid_resolution;
        let needs_agents = agents.len().max(1) > self.agent_capacity;

        if needs_grid {
            let buffers = create_grid_buffers(device, total_cells);
            self.grid_cells = buffers.0;
            self.grid_indices = buffers.1;
            self.grid_resolution = grid_resolution;
            self.total_cells = total_cells;
        }

        if needs_agents {
            self.agent_capacity = agents.len().max(1).next_power_of_two();
            let padded = padded_agents(agents, self.agent_capacity);
            self.agent_buffers = [
                create_agent_buffer(device, "gpu agents a", &padded),
                create_agent_buffer(device, "gpu agents b", &padded),
            ];
        } else {
            let padded = padded_agents(agents, self.agent_capacity);
            for buffer in &self.agent_buffers {
                // Reseeding writes both ping-pong buffers so either one can be the next source.
                // There is no GPU readback; CPU data only flows into the GPU on reset/species resize.
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&padded));
            }
        }

        if needs_grid || needs_agents {
            self.bind_groups = create_bind_groups(
                device,
                &self.pipelines.layout,
                &self.params_buffer,
                &self.agent_buffers,
                &self.species_buffer,
                &self.interaction_buffer,
                &self.grid_cells,
                &self.grid_indices,
            );
        }

        self.source_index = 0;
        self.agent_count = agents.len();
        self.pending_swap = false;
    }

    pub fn update_resources(&mut self, queue: &wgpu::Queue, settings: &SimulationSettings) {
        let species = gpu_species_array(&settings.species);
        queue.write_buffer(&self.species_buffer, 0, bytemuck::cast_slice(&species));
        let interactions = gpu_interaction_array(&settings.interactions);
        queue.write_buffer(&self.interaction_buffer, 0, bytemuck::cast_slice(&interactions));
    }

    pub fn encode_update(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        settings: &SimulationSettings,
        dt: f32,
        elapsed: f32,
    ) {
        if settings.pause || self.agent_count == 0 {
            return;
        }

        let started_at = Instant::now();
        let params = params_for(settings, self.agent_count, dt, elapsed, self.frame_index);
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let source = self.source_index;
        let workgroups_agents = div_ceil(self.agent_count as u32, WORKGROUP_SIZE);
        let workgroups_cells = div_ceil(self.total_cells, WORKGROUP_SIZE);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("boids compute pass"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_groups[source], &[]);

            // The first dispatch clears one atomic counter per grid cell. The second dispatch
            // inserts every source agent into that grid. The third dispatch reads nearby cells
            // and writes updated agents into the opposite storage buffer.
            pass.set_pipeline(&self.pipelines.clear_grid);
            pass.dispatch_workgroups(workgroups_cells, 1, 1);
            pass.set_pipeline(&self.pipelines.build_grid);
            pass.dispatch_workgroups(workgroups_agents, 1, 1);
            pass.set_pipeline(&self.pipelines.update_agents);
            pass.dispatch_workgroups(workgroups_agents, 1, 1);
        }

        self.pending_swap = true;
        self.frame_index = self.frame_index.wrapping_add(1);
        self.stats = SimulationStats {
            average_speed: 0.0,
            neighbor_samples: 0,
            average_neighbors: 0.0,
            max_neighbors_seen: 0,
            grid_cell_count: self.total_cells as usize,
            simulation_ms: started_at.elapsed().as_secs_f32() * 1_000.0,
        };
    }

    pub fn render_buffer(&self) -> &wgpu::Buffer {
        if self.pending_swap {
            &self.agent_buffers[self.destination_index()]
        } else {
            &self.agent_buffers[self.source_index]
        }
    }

    pub fn agent_count(&self) -> usize {
        self.agent_count
    }

    pub fn needs_reseed(&self, settings: &SimulationSettings, agent_count: usize) -> bool {
        let (grid_resolution, _) = grid_dimensions(settings);
        self.agent_count != agent_count
            || self.agent_capacity < agent_count.max(1)
            || self.grid_resolution != grid_resolution
    }

    pub fn finish_frame(&mut self) {
        if self.pending_swap {
            // Ping-ponging: after rendering the freshly written destination buffer, that buffer
            // becomes the source buffer for the next frame.
            self.source_index = self.destination_index();
            self.pending_swap = false;
        }
    }

    fn destination_index(&self) -> usize {
        1 - self.source_index
    }
}

struct ComputePipelines {
    layout: wgpu::BindGroupLayout,
    clear_grid: wgpu::ComputePipeline,
    build_grid: wgpu::ComputePipeline,
    update_agents: wgpu::ComputePipeline,
}

impl ComputePipelines {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("boids compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/boids_compute.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("boids compute bind group layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("boids compute pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let clear_grid = create_compute_pipeline(device, &pipeline_layout, &shader, "clear_grid");
        let build_grid = create_compute_pipeline(device, &pipeline_layout, &shader, "build_grid");
        let update_agents = create_compute_pipeline(device, &pipeline_layout, &shader, "update_agents");

        Self {
            layout,
            clear_grid,
            build_grid,
            update_agents,
        }
    }
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(layout),
        module: shader,
        entry_point,
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_bind_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params_buffer: &wgpu::Buffer,
    agent_buffers: &[wgpu::Buffer; 2],
    species_buffer: &wgpu::Buffer,
    interaction_buffer: &wgpu::Buffer,
    grid_cells: &wgpu::Buffer,
    grid_indices: &wgpu::Buffer,
) -> [wgpu::BindGroup; 2] {
    [
        create_bind_group(
            device,
            layout,
            "boids compute a to b",
            params_buffer,
            &agent_buffers[0],
            &agent_buffers[1],
            species_buffer,
            interaction_buffer,
            grid_cells,
            grid_indices,
        ),
        create_bind_group(
            device,
            layout,
            "boids compute b to a",
            params_buffer,
            &agent_buffers[1],
            &agent_buffers[0],
            species_buffer,
            interaction_buffer,
            grid_cells,
            grid_indices,
        ),
    ]
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    params_buffer: &wgpu::Buffer,
    source_agents: &wgpu::Buffer,
    destination_agents: &wgpu::Buffer,
    species_buffer: &wgpu::Buffer,
    interaction_buffer: &wgpu::Buffer,
    grid_cells: &wgpu::Buffer,
    grid_indices: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            bind_entry(0, params_buffer),
            bind_entry(1, source_agents),
            bind_entry(2, destination_agents),
            bind_entry(3, species_buffer),
            bind_entry(4, interaction_buffer),
            bind_entry(5, grid_cells),
            bind_entry(6, grid_indices),
        ],
    })
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_agent_buffer(device: &wgpu::Device, label: &str, agents: &[GpuAgent]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(agents),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_grid_buffers(device: &wgpu::Device, total_cells: u32) -> (wgpu::Buffer, wgpu::Buffer) {
    let grid_cells = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu grid cell counters"),
        size: total_cells as u64 * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let grid_indices = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu grid cell indices"),
        size: total_cells as u64 * MAX_CELL_AGENTS as u64 * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (grid_cells, grid_indices)
}

fn padded_agents(agents: &[Agent], capacity: usize) -> Vec<GpuAgent> {
    let fallback = agents.first().copied().unwrap_or(Agent {
        position: glam::Vec3::ZERO,
        velocity: glam::Vec3::X,
        species_id: 0,
        energy: 1.0,
        random_seed: 1,
    });
    let mut gpu_agents = agents.iter().copied().map(GpuAgent::from).collect::<Vec<_>>();
    gpu_agents.resize(capacity, GpuAgent::from(fallback));
    gpu_agents
}

fn params_for(
    settings: &SimulationSettings,
    agent_count: usize,
    dt: f32,
    elapsed: f32,
    frame_index: u32,
) -> GpuParams {
    let (grid_resolution, total_cells) = grid_dimensions(settings);
    let goal = [
        (elapsed * 0.31).cos() * settings.bounds * 0.36,
        (elapsed * 0.21).sin() * settings.bounds * 0.18,
        (elapsed * 0.27).sin() * settings.bounds * 0.36,
    ];
    GpuParams {
        scalars_0: [dt.clamp(0.0, 1.0 / 20.0), elapsed, settings.bounds, settings.neighbor_radius],
        scalars_1: [
            settings.separation_radius,
            settings.cell_size,
            goal[0],
            goal[1],
        ],
        scalars_2: [
            goal[2],
            settings.max_speed / 18.0,
            settings.goal_strength,
            settings.obstacle_avoidance_strength,
        ],
        counts: [
            agent_count as u32,
            settings.species.len().min(MAX_SPECIES) as u32,
            settings.max_neighbors as u32,
            settings.use_spatial_hash as u32,
        ],
        grid: [grid_resolution, total_cells, MAX_CELL_AGENTS, frame_index],
    }
}

fn grid_dimensions(settings: &SimulationSettings) -> (u32, u32) {
    let resolution = ((settings.bounds * 2.0) / settings.cell_size)
        .ceil()
        .clamp(4.0, MAX_GRID_RESOLUTION as f32) as u32;
    (resolution, resolution * resolution * resolution)
}

fn div_ceil(value: u32, divisor: u32) -> u32 {
    (value + divisor - 1) / divisor
}
