pub mod agent;
pub mod cpu_simulation;
pub mod gpu_simulation;
pub mod spatial_hash;
pub mod species;

pub use agent::{Agent, GpuAgent};
pub use cpu_simulation::{apply_bounds, clamp_velocity, FlockSimulation, Obstacle, SimulationSettings, SimulationStats};
pub use gpu_simulation::{GpuSimulation, SimulationBackend};
pub use spatial_hash::SpatialHash;
pub use species::{built_in_species, default_interactions, InterSpeciesRule, SpeciesDefinition, SpeciesPreset};
