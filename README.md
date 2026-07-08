# Flock Lab

Flock Lab is a native Rust/wgpu flocking and crowd simulation demo for macOS. It renders thousands of multi-species agents as lit 3D drones, with GPU compute simulation, storage-buffer rendering, spatial grid debug overlays, and an egui control panel.

<img width="2962" height="2200" alt="image" src="https://github.com/user-attachments/assets/eb189b92-2a2a-4d92-94a2-63e09168f33f" />


## Run

```sh
cargo run --release
```

The app uses `winit` for the native window, `wgpu` for compute/rendering, WGSL shaders, `glam`, `bytemuck`, and `egui`.

## Controls

- Drag with the left mouse button to orbit or look around.
- Mouse wheel zooms the orbit camera.
- `Tab` toggles orbit/fly camera.
- Fly mode: `W/A/S/D` move, `Q/E` descend/ascend, `Shift` boosts.
- `Space` pauses, `R` resets, `Enter` randomises.
- The right panel controls GPU/CPU mode, spatial grid lookup, neighbour radius, cell size, max neighbours, species enable/count/speed/force/weights, bounds, and debug overlays.

## GPU Simulation

The default path uses `shaders/boids_compute.wgsl`:

1. Clear one atomic counter per spatial grid cell.
2. Insert every source agent into the uniform grid.
3. Query nearby cells and write updated agents into the destination buffer.
4. Render agents directly from that updated storage buffer.
5. Swap source/destination buffers for the next frame.

The two agent buffers are ping-ponged every frame, so normal GPU mode does not read agent data back to the CPU. If compute shaders are unavailable, the app logs the reason and uses the CPU spatial-hash fallback.

Each agent stores `position`, `velocity`, `species_id`, `energy`, and `random_seed`. The renderer reads the same agent storage buffer in the vertex shader and uses the species buffer for colour, render size, and shape variation.

## Spatial Grid

Both CPU and GPU paths support a uniform grid / spatial hash. Agents query neighbouring cells for separation, alignment, cohesion, hunt/repel/orbit rules, and the old naive all-pairs scan remains behind the `Spatial grid lookup` debug toggle for comparison.

Debug stats include FPS, active backend, agent count, average/max neighbours in CPU mode, grid cell count, and CPU update/GPU encode time. GPU neighbour counts are intentionally not read back during normal simulation.

## Species

The built-in preset defines five data-driven species:

- Swarm
- Gliders
- Predators
- Drifters
- Leaders

Each species has colour, count, max speed, max force, flocking weights, wander weight, predator/prey weights, trail length, render size, and shape id. Inter-species rules support attract, repel, hunt, ignore, and orbit. New species can be added by extending the preset data in `src/simulation/species.rs`.

## Project Layout

```text
src/
  simulation/
    agent.rs
    cpu_simulation.rs
    gpu_simulation.rs
    spatial_hash.rs
    species.rs
  renderer/
    agent_renderer.rs
    trail_renderer.rs
  camera/
  gpu/
  ui/
  utils/

shaders/
  agents.wgsl
  boids_compute.wgsl
  trails.wgsl
```

## Tests

```sh
cargo test
```

The tests cover CPU force behaviour, settings validation, spatial hash indexing/querying/hash consistency/negative coordinates, and WGSL parsing plus validation.
