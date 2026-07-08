use crate::simulation::{SimulationBackend, SimulationSettings, SimulationStats, SpeciesPreset};

#[derive(Debug, Clone)]
pub struct UiState {
    pub prefer_gpu: bool,
    pub show_trails: bool,
    pub show_neighbor_radius: bool,
    pub show_bounds: bool,
    pub show_spatial_grid: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            prefer_gpu: true,
            show_trails: true,
            show_neighbor_radius: false,
            show_bounds: true,
            show_spatial_grid: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UiFrameInfo {
    pub backend: SimulationBackend,
    pub gpu_available: bool,
    pub fps: f32,
    pub adapter_name: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct UiAction {
    pub reset: bool,
    pub randomize: bool,
    pub apply_default_preset: bool,
}

pub fn draw(
    ctx: &egui::Context,
    settings: &mut SimulationSettings,
    stats: SimulationStats,
    frame: &UiFrameInfo,
    ui_state: &mut UiState,
) -> UiAction {
    let mut action = UiAction::default();

    egui::TopBottomPanel::top("top_status")
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_unmultiplied(8, 11, 18, 196))
                .inner_margin(egui::Margin::symmetric(14.0, 8.0)),
        )
        .show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("Flock Lab");
                ui.separator();
                ui.label(format!("{} agents", settings.agent_count()));
                ui.label(format!("{:.0} FPS", frame.fps));
                ui.label(frame.backend.label());
                ui.label(format!("backend: {}", frame.adapter_name));
                ui.separator();
                if ui
                    .button(if settings.pause { "Resume" } else { "Pause" })
                    .clicked()
                {
                    settings.pause = !settings.pause;
                }
                if ui.button("Reset").clicked() {
                    action.reset = true;
                }
                if ui.button("Randomise").clicked() {
                    action.randomize = true;
                }
            });
        });

    egui::SidePanel::right("controls")
        .resizable(false)
        .default_width(340.0)
        .frame(
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_unmultiplied(9, 13, 22, 224))
                .inner_margin(egui::Margin::same(16.0)),
        )
        .show(ctx, |ui| {
            ui.heading("Simulation");
            ui.add_space(8.0);
            ui.add_enabled_ui(frame.gpu_available, |ui| {
                ui.checkbox(&mut ui_state.prefer_gpu, "GPU compute simulation");
            });
            if !frame.gpu_available {
                ui.label("GPU compute unavailable; CPU fallback active.");
            }
            ui.checkbox(&mut settings.use_spatial_hash, "Spatial grid lookup");
            ui.add(egui::Slider::new(&mut settings.max_speed, 1.0..=60.0).text("Global speed"));
            ui.add(egui::Slider::new(&mut settings.bounds, 30.0..=220.0).text("Bounds"));
            ui.add(egui::Slider::new(&mut settings.trail_length, 0..=64).text("Trail length"));

            ui.add_space(12.0);
            ui.heading("Neighbour Grid");
            ui.add(egui::Slider::new(&mut settings.neighbor_radius, 2.0..=36.0).text("Neighbour radius"));
            settings.separation_radius = settings.separation_radius.min(settings.neighbor_radius);
            ui.add(
                egui::Slider::new(&mut settings.separation_radius, 0.5..=settings.neighbor_radius)
                    .text("Separation radius"),
            );
            ui.add(egui::Slider::new(&mut settings.cell_size, 2.0..=40.0).text("Cell size"));
            ui.add(egui::Slider::new(&mut settings.max_neighbors, 4..=256).text("Max neighbours"));

            ui.add_space(12.0);
            ui.heading("Behaviour");
            ui.add(egui::Slider::new(&mut settings.separation_strength, 0.0..=8.0).text("Separation"));
            ui.add(egui::Slider::new(&mut settings.alignment_strength, 0.0..=6.0).text("Alignment"));
            ui.add(egui::Slider::new(&mut settings.cohesion_strength, 0.0..=6.0).text("Cohesion"));
            ui.add(egui::Slider::new(&mut settings.goal_strength, 0.0..=2.0).text("Goal seeking"));

            ui.add_space(12.0);
            ui.heading("Species");
            if ui.button("Load Aurora Lab preset").clicked() {
                action.apply_default_preset = true;
            }
            for species in &mut settings.species {
                ui.collapsing(species.name.clone(), |ui| {
                    ui.checkbox(&mut species.enabled, "Enabled");
                    ui.add(egui::Slider::new(&mut species.agent_count, 0..=8_000).text("Count"));
                    ui.add(egui::Slider::new(&mut species.max_speed, 0.5..=50.0).text("Max speed"));
                    ui.add(egui::Slider::new(&mut species.max_force, 0.1..=40.0).text("Max force"));
                    ui.add(egui::Slider::new(&mut species.separation_weight, 0.0..=10.0).text("Separate"));
                    ui.add(egui::Slider::new(&mut species.alignment_weight, 0.0..=10.0).text("Align"));
                    ui.add(egui::Slider::new(&mut species.cohesion_weight, 0.0..=10.0).text("Cohere"));
                    ui.add(egui::Slider::new(&mut species.wander_weight, 0.0..=6.0).text("Wander"));
                    ui.add(egui::Slider::new(&mut species.render_size, 0.25..=5.0).text("Render size"));
                });
            }

            ui.add_space(12.0);
            ui.heading("Debug Stats");
            ui.label(format!("Avg speed: {:.1}", stats.average_speed));
            ui.label(format!("Avg neighbours: {:.1}", stats.average_neighbors));
            ui.label(format!("Max neighbours: {}", stats.max_neighbors_seen));
            ui.label(format!("Grid cells: {}", stats.grid_cell_count));
            ui.label(format!("Sim encode/update: {:.2} ms", stats.simulation_ms));
            if frame.backend == SimulationBackend::Gpu {
                ui.label("GPU neighbour stats avoid readback; CPU mode shows exact counts.");
            }

            ui.add_space(12.0);
            ui.heading("View");
            ui.checkbox(&mut ui_state.show_trails, "CPU trails");
            ui.checkbox(&mut ui_state.show_bounds, "Bounds");
            ui.checkbox(&mut ui_state.show_neighbor_radius, "Neighbour radius debug");
            ui.checkbox(&mut ui_state.show_spatial_grid, "Spatial grid debug overlay");
        });

    action
}

pub fn apply_default_preset(settings: &mut SimulationSettings) {
    if let Some(preset) = SpeciesPreset::built_in().into_iter().next() {
        settings.species = preset.species;
        settings.interactions = preset.interactions;
    }
}
