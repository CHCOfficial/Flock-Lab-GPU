use crate::simulation::{
    agent::Agent,
    spatial_hash::SpatialHash,
    species::{default_interactions, built_in_species, InterSpeciesRule, SpeciesDefinition, MAX_SPECIES},
};
use glam::Vec3;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

const MIN_SPEED: f32 = 0.8;

#[derive(Debug, Clone)]
pub struct SimulationSettings {
    pub max_speed: f32,
    pub separation_strength: f32,
    pub alignment_strength: f32,
    pub cohesion_strength: f32,
    pub predator_avoidance_strength: f32,
    pub goal_strength: f32,
    pub obstacle_avoidance_strength: f32,
    pub obstacle_count: usize,
    pub trail_length: usize,
    pub bounds: f32,
    pub neighbor_radius: f32,
    pub separation_radius: f32,
    pub cell_size: f32,
    pub max_neighbors: usize,
    pub use_spatial_hash: bool,
    pub pause: bool,
    pub species: Vec<SpeciesDefinition>,
    pub interactions: [[InterSpeciesRule; MAX_SPECIES]; MAX_SPECIES],
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            max_speed: 18.0,
            separation_strength: 1.0,
            alignment_strength: 1.0,
            cohesion_strength: 1.0,
            predator_avoidance_strength: 1.0,
            goal_strength: 0.12,
            obstacle_avoidance_strength: 2.8,
            obstacle_count: 10,
            trail_length: 24,
            bounds: 90.0,
            neighbor_radius: 8.5,
            separation_radius: 3.0,
            cell_size: 10.0,
            max_neighbors: 64,
            use_spatial_hash: true,
            pause: false,
            species: built_in_species(),
            interactions: default_interactions(),
        }
    }
}

impl SimulationSettings {
    pub fn validated(mut self) -> Self {
        self.max_speed = self.max_speed.clamp(MIN_SPEED, 80.0);
        self.separation_strength = self.separation_strength.clamp(0.0, 12.0);
        self.alignment_strength = self.alignment_strength.clamp(0.0, 12.0);
        self.cohesion_strength = self.cohesion_strength.clamp(0.0, 12.0);
        self.predator_avoidance_strength = self.predator_avoidance_strength.clamp(0.0, 20.0);
        self.goal_strength = self.goal_strength.clamp(0.0, 5.0);
        self.obstacle_avoidance_strength = self.obstacle_avoidance_strength.clamp(0.0, 20.0);
        self.obstacle_count = self.obstacle_count.clamp(0, 128);
        self.trail_length = self.trail_length.clamp(0, 96);
        self.bounds = self.bounds.clamp(20.0, 300.0);
        self.neighbor_radius = self.neighbor_radius.clamp(1.0, 60.0);
        self.separation_radius = self.separation_radius.clamp(0.25, self.neighbor_radius);
        self.cell_size = self.cell_size.clamp(1.0, 80.0);
        self.max_neighbors = self.max_neighbors.clamp(1, 512);
        self.species = self
            .species
            .into_iter()
            .take(MAX_SPECIES)
            .map(SpeciesDefinition::validated)
            .collect();
        self
    }

    pub fn agent_count(&self) -> usize {
        self.species
            .iter()
            .filter(|species| species.enabled)
            .map(|species| species.agent_count)
            .sum()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Obstacle {
    pub position: Vec3,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SimulationStats {
    pub average_speed: f32,
    pub neighbor_samples: usize,
    pub average_neighbors: f32,
    pub max_neighbors_seen: usize,
    pub grid_cell_count: usize,
    pub simulation_ms: f32,
}

#[derive(Debug)]
pub struct FlockSimulation {
    pub settings: SimulationSettings,
    pub agents: Vec<Agent>,
    pub obstacles: Vec<Obstacle>,
    pub trails: Vec<Vec<Vec3>>,
    pub goal: Vec3,
    pub stats: SimulationStats,
    rng: StdRng,
    spatial_hash: SpatialHash,
    time: f32,
}

impl FlockSimulation {
    pub fn new(settings: SimulationSettings) -> Self {
        let settings = settings.validated();
        let mut simulation = Self {
            spatial_hash: SpatialHash::new(settings.cell_size),
            settings,
            agents: Vec::new(),
            obstacles: Vec::new(),
            trails: Vec::new(),
            goal: Vec3::ZERO,
            stats: SimulationStats::default(),
            rng: StdRng::seed_from_u64(0xF10C_AB1E),
            time: 0.0,
        };
        simulation.randomize();
        simulation
    }

    pub fn randomize(&mut self) {
        self.settings = self.settings.clone().validated();
        self.spatial_hash.set_cell_size(self.settings.cell_size);
        self.agents.clear();
        self.obstacles.clear();
        self.trails.clear();
        self.goal = Vec3::ZERO;
        self.time = 0.0;

        let species = self.settings.species.clone();
        for (species_id, species) in species.iter().enumerate() {
            if !species.enabled {
                continue;
            }
            for _ in 0..species.agent_count {
                let position = self.random_in_bounds(0.72);
                let velocity = self.random_unit() * self.rng.gen_range(4.0..species.max_speed);
                self.agents.push(Agent {
                    position,
                    velocity,
                    species_id: species_id as u32,
                    energy: self.rng.gen_range(0.55..1.0),
                    random_seed: self.rng.gen(),
                });
            }
        }

        self.trails = self
            .agents
            .iter()
            .map(|agent| vec![agent.position])
            .collect::<Vec<_>>();
        self.ensure_obstacle_count();
    }

    pub fn reset(&mut self) {
        self.randomize();
    }

    pub fn update(&mut self, dt: f32) {
        let started_at = Instant::now();
        self.settings = self.settings.clone().validated();
        self.resize_agents_for_species();
        self.ensure_obstacle_count();
        self.spatial_hash.set_cell_size(self.settings.cell_size);

        if self.settings.pause {
            return;
        }

        let dt = dt.clamp(0.0, 1.0 / 20.0);
        self.time += dt;
        self.goal = Vec3::new(
            (self.time * 0.31).cos() * self.settings.bounds * 0.36,
            (self.time * 0.21).sin() * self.settings.bounds * 0.18,
            (self.time * 0.27).sin() * self.settings.bounds * 0.36,
        );

        self.spatial_hash
            .rebuild(self.agents.iter().enumerate().map(|(i, agent)| (i, agent.position)));

        let previous = self.agents.clone();
        let mut next = previous.clone();
        let mut total_speed = 0.0;
        let mut neighbor_samples = 0;
        let mut max_neighbors_seen = 0;

        for (index, agent) in previous.iter().enumerate() {
            let (force, sampled_neighbors) = self.agent_force_and_neighbor_count(index, &previous);
            let species = self.species_for(agent.species_id);
            let max_speed = species.max_speed * self.settings.max_speed / 18.0;
            let mut velocity = agent.velocity + force * dt;
            velocity = clamp_velocity(velocity, max_speed);

            let mut position = agent.position + velocity * dt;
            apply_bounds(&mut position, &mut velocity, self.settings.bounds);

            next[index].position = position;
            next[index].velocity = velocity;
            next[index].energy = (agent.energy + self.wander(index as u32).x * 0.002).clamp(0.2, 1.0);
            total_speed += velocity.length();
            neighbor_samples += sampled_neighbors;
            max_neighbors_seen = max_neighbors_seen.max(sampled_neighbors);
        }

        self.agents = next;
        self.stats = SimulationStats {
            average_speed: total_speed / self.agents.len().max(1) as f32,
            neighbor_samples,
            average_neighbors: neighbor_samples as f32 / self.agents.len().max(1) as f32,
            max_neighbors_seen,
            grid_cell_count: self.spatial_hash.cell_count(),
            simulation_ms: started_at.elapsed().as_secs_f32() * 1_000.0,
        };
        self.update_trails();
    }

    pub fn agent_force(&self, index: usize, snapshot: &[Agent]) -> Vec3 {
        self.agent_force_and_neighbor_count(index, snapshot).0
    }

    fn agent_force_and_neighbor_count(&self, index: usize, snapshot: &[Agent]) -> (Vec3, usize) {
        let agent = snapshot[index];
        let species = self.species_for(agent.species_id);
        let mut separation = Vec3::ZERO;
        let mut alignment = Vec3::ZERO;
        let mut cohesion = Vec3::ZERO;
        let mut neighbor_count = 0.0;
        let mut sampled_neighbors = 0;
        let neighbor_radius_sq = self.settings.neighbor_radius * self.settings.neighbor_radius;
        let separation_radius_sq = self.settings.separation_radius * self.settings.separation_radius;

        let candidates: Vec<usize> = if self.settings.use_spatial_hash {
            self.spatial_hash
                .nearby_indices_limited(
                    agent.position,
                    self.settings.neighbor_radius,
                    self.settings.max_neighbors,
                )
                .collect()
        } else {
            (0..snapshot.len()).take(self.settings.max_neighbors + 1).collect()
        };

        for candidate in candidates {
            if candidate == index {
                continue;
            }
            let other = snapshot[candidate];
            let other_species = self.species_for(other.species_id);
            if !other_species.enabled {
                continue;
            }

            let offset = other.position - agent.position;
            let distance_sq = offset.length_squared();
            if distance_sq > neighbor_radius_sq || distance_sq <= f32::EPSILON {
                continue;
            }

            let rule = self.settings.interactions[agent.species_id as usize][other.species_id as usize];
            let direction = offset.normalize_or_zero();
            let distance = distance_sq.sqrt().max(0.001);
            sampled_neighbors += 1;

            match rule {
                InterSpeciesRule::Ignore => {}
                InterSpeciesRule::Attract => {
                    neighbor_count += 1.0;
                    alignment += other.velocity;
                    cohesion += other.position;
                }
                InterSpeciesRule::Repel => {
                    separation -= direction / distance;
                }
                InterSpeciesRule::Hunt => {
                    cohesion += other.position * 1.8;
                    neighbor_count += 1.8;
                }
                InterSpeciesRule::Orbit => {
                    cohesion += other.position;
                    alignment += direction.cross(Vec3::Y).normalize_or_zero() * species.max_speed;
                    neighbor_count += 1.0;
                }
            }

            if distance_sq < separation_radius_sq {
                separation -= direction / distance;
            }

            if sampled_neighbors >= self.settings.max_neighbors {
                break;
            }
        }

        let mut force = Vec3::ZERO;
        if neighbor_count > 0.0 {
            let inv_count = 1.0 / neighbor_count;
            let desired_alignment = (alignment * inv_count).normalize_or_zero() * species.max_speed;
            let center = cohesion * inv_count;
            force += (desired_alignment - agent.velocity)
                * species.alignment_weight
                * self.settings.alignment_strength;
            force += (center - agent.position).normalize_or_zero()
                * species.cohesion_weight
                * self.settings.cohesion_strength
                * species.max_speed;
        }

        force += separation.normalize_or_zero()
            * species.separation_weight
            * self.settings.separation_strength
            * species.max_speed;
        force += (self.goal - agent.position).normalize_or_zero()
            * self.settings.goal_strength
            * species.max_speed;
        force += self.obstacle_avoidance(agent.position);
        force += self.boundary_force(agent.position);
        force += self.wander(agent.random_seed) * species.wander_weight * species.max_force;
        force = clamp_force(force, species.max_force);
        (force, sampled_neighbors)
    }

    fn obstacle_avoidance(&self, position: Vec3) -> Vec3 {
        let mut force = Vec3::ZERO;
        for obstacle in &self.obstacles {
            let offset = position - obstacle.position;
            let safe_radius = obstacle.radius + self.settings.neighbor_radius;
            let distance = offset.length();
            if distance < safe_radius && distance > f32::EPSILON {
                let falloff = 1.0 - distance / safe_radius;
                force += offset.normalize_or_zero()
                    * falloff
                    * self.settings.obstacle_avoidance_strength
                    * self.settings.max_speed;
            }
        }
        force
    }

    fn boundary_force(&self, position: Vec3) -> Vec3 {
        let bound = self.settings.bounds;
        let margin = bound * 0.18;
        let mut force = Vec3::ZERO;
        for axis in 0..3 {
            let value = position[axis];
            let distance_to_wall = bound - value.abs();
            if distance_to_wall < margin {
                force[axis] -= value.signum() * (1.0 - distance_to_wall / margin) * self.settings.max_speed;
            }
        }
        force
    }

    fn resize_agents_for_species(&mut self) {
        let species = self.settings.species.clone();
        let mut rebuilt = Vec::with_capacity(self.settings.agent_count());
        for (species_id, species) in species.iter().enumerate() {
            if !species.enabled {
                continue;
            }
            let existing = self
                .agents
                .iter()
                .copied()
                .filter(|agent| agent.species_id == species_id as u32)
                .take(species.agent_count);
            rebuilt.extend(existing);
            while rebuilt
                .iter()
                .filter(|agent| agent.species_id == species_id as u32)
                .count()
                < species.agent_count
            {
                let position = self.random_in_bounds(0.65);
                let velocity = self.random_unit() * self.rng.gen_range(4.0..species.max_speed);
                rebuilt.push(Agent {
                    position,
                    velocity,
                    species_id: species_id as u32,
                    energy: self.rng.gen_range(0.55..1.0),
                    random_seed: self.rng.gen(),
                });
            }
        }

        self.agents = rebuilt;
        self.trails.resize_with(self.agents.len(), Vec::new);
        for (trail, agent) in self.trails.iter_mut().zip(self.agents.iter()) {
            if trail.is_empty() {
                trail.push(agent.position);
            }
        }
        self.trails.truncate(self.agents.len());
    }

    fn ensure_obstacle_count(&mut self) {
        while self.obstacles.len() < self.settings.obstacle_count {
            let position = self.random_in_bounds(0.55);
            let radius = self.rng.gen_range(4.0..11.0);
            self.obstacles.push(Obstacle { position, radius });
        }
        self.obstacles.truncate(self.settings.obstacle_count);
    }

    fn update_trails(&mut self) {
        if self.settings.trail_length == 0 {
            for trail in &mut self.trails {
                trail.clear();
            }
            return;
        }

        let species_trail_lengths = self
            .settings
            .species
            .iter()
            .map(|species| species.trail_length)
            .collect::<Vec<_>>();
        for (trail, agent) in self.trails.iter_mut().zip(self.agents.iter()) {
            trail.push(agent.position);
            let species_len = species_trail_lengths
                .get(agent.species_id as usize)
                .copied()
                .unwrap_or(self.settings.trail_length);
            let max_len = self.settings.trail_length.min(species_len).max(1);
            let overflow = trail.len().saturating_sub(max_len);
            if overflow > 0 {
                trail.drain(0..overflow);
            }
        }
    }

    fn species_for(&self, id: u32) -> &SpeciesDefinition {
        self.settings
            .species
            .get(id as usize)
            .unwrap_or_else(|| &self.settings.species[0])
    }

    fn random_in_bounds(&mut self, scale: f32) -> Vec3 {
        let radius = self.settings.bounds * scale;
        Vec3::new(
            self.rng.gen_range(-radius..radius),
            self.rng.gen_range(-radius * 0.55..radius * 0.55),
            self.rng.gen_range(-radius..radius),
        )
    }

    fn random_unit(&mut self) -> Vec3 {
        loop {
            let candidate = Vec3::new(
                self.rng.gen_range(-1.0..1.0),
                self.rng.gen_range(-1.0..1.0),
                self.rng.gen_range(-1.0..1.0),
            );
            if candidate.length_squared() > 0.001 {
                return candidate.normalize();
            }
        }
    }

    fn wander(&self, seed: u32) -> Vec3 {
        let t = self.time + seed as f32 * 0.000_013;
        Vec3::new(
            (t * 1.73).sin(),
            (t * 1.19).cos() * 0.35,
            (t * 1.41).cos(),
        )
        .normalize_or_zero()
    }
}

pub fn clamp_velocity(velocity: Vec3, max_speed: f32) -> Vec3 {
    let speed = velocity.length();
    if speed > max_speed {
        velocity / speed * max_speed
    } else if speed < MIN_SPEED && speed > f32::EPSILON {
        velocity / speed * MIN_SPEED
    } else if speed <= f32::EPSILON {
        Vec3::X * MIN_SPEED
    } else {
        velocity
    }
}

pub fn clamp_force(force: Vec3, max_force: f32) -> Vec3 {
    let length = force.length();
    if length > max_force && length > f32::EPSILON {
        force / length * max_force
    } else {
        force
    }
}

pub fn apply_bounds(position: &mut Vec3, velocity: &mut Vec3, bounds: f32) {
    for axis in 0..3 {
        if position[axis] > bounds {
            position[axis] = bounds;
            velocity[axis] = -velocity[axis].abs();
        } else if position[axis] < -bounds {
            position[axis] = -bounds;
            velocity[axis] = velocity[axis].abs();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn settings_validation_clamps_unsafe_values() {
        let mut settings = SimulationSettings {
            max_speed: 0.01,
            separation_radius: 99.0,
            neighbor_radius: 2.0,
            trail_length: 9_999,
            bounds: 2.0,
            cell_size: 0.0,
            max_neighbors: 0,
            ..SimulationSettings::default()
        };
        settings.species[0].agent_count = 999_999;
        let settings = settings.validated();

        assert_relative_eq!(settings.max_speed, MIN_SPEED);
        assert_relative_eq!(settings.separation_radius, settings.neighbor_radius);
        assert_eq!(settings.trail_length, 96);
        assert_relative_eq!(settings.bounds, 20.0);
        assert_relative_eq!(settings.cell_size, 1.0);
        assert_eq!(settings.max_neighbors, 1);
        assert_eq!(settings.species[0].agent_count, 40_000);
    }

    #[test]
    fn separation_pushes_agents_apart() {
        let mut simulation = FlockSimulation::new(SimulationSettings::default());
        simulation.settings.alignment_strength = 0.0;
        simulation.settings.cohesion_strength = 0.0;
        simulation.settings.goal_strength = 0.0;
        simulation.settings.obstacle_count = 0;
        simulation.settings.neighbor_radius = 10.0;
        simulation.settings.separation_radius = 5.0;
        simulation.agents = vec![
            Agent {
                position: Vec3::ZERO,
                velocity: Vec3::X,
                species_id: 0,
                energy: 1.0,
                random_seed: 1,
            },
            Agent {
                position: Vec3::X,
                velocity: Vec3::X,
                species_id: 0,
                energy: 1.0,
                random_seed: 2,
            },
        ];
        simulation
            .spatial_hash
            .rebuild(simulation.agents.iter().enumerate().map(|(i, b)| (i, b.position)));

        let force = simulation.agent_force(0, &simulation.agents);
        assert!(force.x < 0.0, "expected force away from nearby agent, got {force:?}");
    }

    #[test]
    fn bounds_reflect_velocity_and_clamp_position() {
        let mut position = Vec3::new(12.0, -13.0, 2.0);
        let mut velocity = Vec3::new(4.0, -6.0, 1.0);
        apply_bounds(&mut position, &mut velocity, 10.0);

        assert_relative_eq!(position.x, 10.0);
        assert_relative_eq!(position.y, -10.0);
        assert!(velocity.x < 0.0);
        assert!(velocity.y > 0.0);
    }
}
