use bytemuck::{Pod, Zeroable};

pub const MAX_SPECIES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterSpeciesRule {
    Attract,
    Repel,
    Hunt,
    Ignore,
    Orbit,
}

impl InterSpeciesRule {
    pub fn as_i32(self) -> i32 {
        match self {
            Self::Attract => 1,
            Self::Repel => 2,
            Self::Hunt => 3,
            Self::Ignore => 4,
            Self::Orbit => 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpeciesDefinition {
    pub name: String,
    pub color: [f32; 4],
    pub enabled: bool,
    pub agent_count: usize,
    pub max_speed: f32,
    pub max_force: f32,
    pub separation_weight: f32,
    pub alignment_weight: f32,
    pub cohesion_weight: f32,
    pub wander_weight: f32,
    pub predator_weight: f32,
    pub prey_weight: f32,
    pub trail_length: usize,
    pub render_size: f32,
    pub shape_id: u32,
}

impl SpeciesDefinition {
    pub fn validated(mut self) -> Self {
        self.agent_count = self.agent_count.clamp(0, 40_000);
        self.max_speed = self.max_speed.clamp(0.5, 90.0);
        self.max_force = self.max_force.clamp(0.1, 80.0);
        self.separation_weight = self.separation_weight.clamp(0.0, 16.0);
        self.alignment_weight = self.alignment_weight.clamp(0.0, 16.0);
        self.cohesion_weight = self.cohesion_weight.clamp(0.0, 16.0);
        self.wander_weight = self.wander_weight.clamp(0.0, 12.0);
        self.predator_weight = self.predator_weight.clamp(0.0, 20.0);
        self.prey_weight = self.prey_weight.clamp(0.0, 20.0);
        self.trail_length = self.trail_length.clamp(0, 96);
        self.render_size = self.render_size.clamp(0.25, 8.0);
        self
    }
}

#[derive(Debug, Clone)]
pub struct SpeciesPreset {
    pub name: &'static str,
    pub species: Vec<SpeciesDefinition>,
    pub interactions: [[InterSpeciesRule; MAX_SPECIES]; MAX_SPECIES],
}

impl SpeciesPreset {
    pub fn built_in() -> Vec<Self> {
        vec![Self {
            name: "Aurora Lab",
            species: built_in_species(),
            interactions: default_interactions(),
        }]
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSpecies {
    pub color: [f32; 4],
    pub params_0: [f32; 4],
    pub params_1: [f32; 4],
    pub meta: [u32; 4],
}

impl GpuSpecies {
    pub fn from_definition(species: &SpeciesDefinition) -> Self {
        Self {
            color: species.color,
            params_0: [
                species.max_speed,
                species.max_force,
                species.separation_weight,
                species.alignment_weight,
            ],
            params_1: [
                species.cohesion_weight,
                species.wander_weight,
                species.predator_weight,
                species.prey_weight,
            ],
            meta: [
                species.enabled as u32,
                species.trail_length as u32,
                species.shape_id,
                species.render_size.to_bits(),
            ],
        }
    }

    pub fn disabled() -> Self {
        Self {
            color: [0.0, 0.0, 0.0, 0.0],
            params_0: [1.0, 1.0, 0.0, 0.0],
            params_1: [0.0, 0.0, 0.0, 0.0],
            meta: [0, 0, 0, 1.0f32.to_bits()],
        }
    }
}

pub fn built_in_species() -> Vec<SpeciesDefinition> {
    vec![
        SpeciesDefinition {
            name: "Swarm".to_string(),
            color: [0.20, 0.82, 1.00, 0.96],
            enabled: true,
            agent_count: 2_400,
            max_speed: 22.0,
            max_force: 18.0,
            separation_weight: 2.4,
            alignment_weight: 1.0,
            cohesion_weight: 0.62,
            wander_weight: 0.55,
            predator_weight: 5.0,
            prey_weight: 0.0,
            trail_length: 20,
            render_size: 0.85,
            shape_id: 0,
        },
        SpeciesDefinition {
            name: "Gliders".to_string(),
            color: [0.62, 0.90, 0.42, 0.96],
            enabled: true,
            agent_count: 760,
            max_speed: 15.0,
            max_force: 10.0,
            separation_weight: 1.3,
            alignment_weight: 1.8,
            cohesion_weight: 0.8,
            wander_weight: 0.22,
            predator_weight: 4.0,
            prey_weight: 0.0,
            trail_length: 26,
            render_size: 1.25,
            shape_id: 1,
        },
        SpeciesDefinition {
            name: "Predators".to_string(),
            color: [1.00, 0.16, 0.08, 1.0],
            enabled: true,
            agent_count: 18,
            max_speed: 27.0,
            max_force: 24.0,
            separation_weight: 0.7,
            alignment_weight: 0.1,
            cohesion_weight: 0.0,
            wander_weight: 0.3,
            predator_weight: 0.0,
            prey_weight: 6.5,
            trail_length: 32,
            render_size: 3.0,
            shape_id: 2,
        },
        SpeciesDefinition {
            name: "Drifters".to_string(),
            color: [0.80, 0.56, 1.00, 0.84],
            enabled: true,
            agent_count: 420,
            max_speed: 7.5,
            max_force: 5.0,
            separation_weight: 1.0,
            alignment_weight: 0.35,
            cohesion_weight: 0.35,
            wander_weight: 1.4,
            predator_weight: 2.4,
            prey_weight: 0.0,
            trail_length: 40,
            render_size: 1.65,
            shape_id: 3,
        },
        SpeciesDefinition {
            name: "Leaders".to_string(),
            color: [1.00, 0.78, 0.22, 1.0],
            enabled: true,
            agent_count: 12,
            max_speed: 19.0,
            max_force: 15.0,
            separation_weight: 0.4,
            alignment_weight: 0.0,
            cohesion_weight: 0.0,
            wander_weight: 0.75,
            predator_weight: 1.5,
            prey_weight: 0.0,
            trail_length: 34,
            render_size: 2.25,
            shape_id: 4,
        },
    ]
}

pub fn default_interactions() -> [[InterSpeciesRule; MAX_SPECIES]; MAX_SPECIES] {
    let mut rules = [[InterSpeciesRule::Ignore; MAX_SPECIES]; MAX_SPECIES];
    for i in 0..MAX_SPECIES {
        rules[i][i] = InterSpeciesRule::Attract;
    }

    let swarm = 0;
    let gliders = 1;
    let predators = 2;
    let drifters = 3;
    let leaders = 4;

    rules[swarm][gliders] = InterSpeciesRule::Attract;
    rules[swarm][predators] = InterSpeciesRule::Repel;
    rules[swarm][leaders] = InterSpeciesRule::Attract;
    rules[gliders][swarm] = InterSpeciesRule::Orbit;
    rules[gliders][predators] = InterSpeciesRule::Repel;
    rules[predators][swarm] = InterSpeciesRule::Hunt;
    rules[predators][gliders] = InterSpeciesRule::Hunt;
    rules[predators][drifters] = InterSpeciesRule::Hunt;
    rules[drifters][predators] = InterSpeciesRule::Repel;
    rules[drifters][leaders] = InterSpeciesRule::Orbit;
    rules[leaders][swarm] = InterSpeciesRule::Attract;
    rules[leaders][gliders] = InterSpeciesRule::Attract;

    rules
}

pub fn gpu_species_array(species: &[SpeciesDefinition]) -> [GpuSpecies; MAX_SPECIES] {
    let mut output = [GpuSpecies::disabled(); MAX_SPECIES];
    for (index, species) in species.iter().take(MAX_SPECIES).enumerate() {
        output[index] = GpuSpecies::from_definition(species);
    }
    output
}

pub fn gpu_interaction_array(
    rules: &[[InterSpeciesRule; MAX_SPECIES]; MAX_SPECIES],
) -> [[i32; MAX_SPECIES]; MAX_SPECIES] {
    let mut output = [[InterSpeciesRule::Ignore.as_i32(); MAX_SPECIES]; MAX_SPECIES];
    for y in 0..MAX_SPECIES {
        for x in 0..MAX_SPECIES {
            output[y][x] = rules[y][x].as_i32();
        }
    }
    output
}
