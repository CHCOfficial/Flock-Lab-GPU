use crate::simulation::{
    agent::{Agent, GpuAgent},
    species::gpu_species_array,
    SimulationSettings,
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AgentVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl AgentVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub struct AgentRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    species_buffer: wgpu::Buffer,
    agent_bind_group_layout: wgpu::BindGroupLayout,
    index_count: u32,
}

impl AgentRenderer {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_layout: &wgpu::BindGroupLayout,
        depth_format: wgpu::TextureFormat,
        settings: &SimulationSettings,
    ) -> Self {
        let agent_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("agent storage bind group layout"),
                entries: &[
                    storage_entry(0),
                    storage_entry(1),
                ],
            });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agents shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/agents.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agents pipeline layout"),
            bind_group_layouts: &[camera_layout, &agent_bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("agents pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[AgentVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let vertices = dart_vertices();
        let indices = dart_indices();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("agent vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("agent index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let species = gpu_species_array(&settings.species);
        let species_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render species buffer"),
            contents: bytemuck::cast_slice(&species),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            species_buffer,
            agent_bind_group_layout,
            index_count: indices.len() as u32,
        }
    }

    pub fn update_species(&self, queue: &wgpu::Queue, settings: &SimulationSettings) {
        let species = gpu_species_array(&settings.species);
        queue.write_buffer(&self.species_buffer, 0, bytemuck::cast_slice(&species));
    }

    pub fn create_agent_buffer(device: &wgpu::Device, agents: &[Agent]) -> wgpu::Buffer {
        let gpu_agents = agents.iter().copied().map(GpuAgent::from).collect::<Vec<_>>();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cpu fallback render agents"),
            contents: bytemuck::cast_slice(&gpu_agents),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn update_agent_buffer(queue: &wgpu::Queue, buffer: &wgpu::Buffer, agents: &[Agent]) {
        let gpu_agents = agents.iter().copied().map(GpuAgent::from).collect::<Vec<_>>();
        if !gpu_agents.is_empty() {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&gpu_agents));
        }
    }

    pub fn create_agent_bind_group(
        &self,
        device: &wgpu::Device,
        agent_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agent render bind group"),
            layout: &self.agent_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.species_buffer.as_entire_binding(),
                },
            ],
        })
    }

    pub fn render<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        agent_bind_group: &'a wgpu::BindGroup,
        instance_count: u32,
    ) {
        if instance_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera_bind_group, &[]);
        pass.set_bind_group(1, agent_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..instance_count);
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn dart_vertices() -> [AgentVertex; 7] {
    [
        AgentVertex {
            position: [0.0, 0.0, 1.35],
            normal: [0.0, 0.2, 1.0],
        },
        AgentVertex {
            position: [-0.48, -0.08, -0.35],
            normal: [-0.6, 0.2, 0.3],
        },
        AgentVertex {
            position: [0.48, -0.08, -0.35],
            normal: [0.6, 0.2, 0.3],
        },
        AgentVertex {
            position: [0.0, 0.22, -0.84],
            normal: [0.0, 0.9, -0.2],
        },
        AgentVertex {
            position: [0.0, -0.32, -0.72],
            normal: [0.0, -1.0, -0.2],
        },
        AgentVertex {
            position: [-0.16, 0.03, -1.12],
            normal: [-0.4, 0.3, -0.7],
        },
        AgentVertex {
            position: [0.16, 0.03, -1.12],
            normal: [0.4, 0.3, -0.7],
        },
    ]
}

fn dart_indices() -> [u16; 24] {
    [
        0, 1, 3, 0, 3, 2, 0, 4, 1, 0, 2, 4, 1, 5, 3, 2, 3, 6, 1, 4, 5, 2, 6, 4,
    ]
}
