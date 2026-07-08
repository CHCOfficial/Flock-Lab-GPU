use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct Agent {
    pub position: Vec3,
    pub velocity: Vec3,
    pub species_id: u32,
    pub energy: f32,
    pub random_seed: u32,
}

impl Agent {
    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuAgent {
    pub position: [f32; 4],
    pub velocity: [f32; 4],
    pub species_id: u32,
    pub random_seed: u32,
    pub energy: f32,
    pub _pad: u32,
}

impl From<Agent> for GpuAgent {
    fn from(agent: Agent) -> Self {
        Self {
            position: [agent.position.x, agent.position.y, agent.position.z, 1.0],
            velocity: [agent.velocity.x, agent.velocity.y, agent.velocity.z, 0.0],
            species_id: agent.species_id,
            random_seed: agent.random_seed,
            energy: agent.energy,
            _pad: 0,
        }
    }
}

impl From<GpuAgent> for Agent {
    fn from(agent: GpuAgent) -> Self {
        Self {
            position: Vec3::new(agent.position[0], agent.position[1], agent.position[2]),
            velocity: Vec3::new(agent.velocity[0], agent.velocity[1], agent.velocity[2]),
            species_id: agent.species_id,
            energy: agent.energy,
            random_seed: agent.random_seed,
        }
    }
}
