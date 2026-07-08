#[test]
fn wgsl_shaders_parse() {
    for (name, source) in [
        ("agents.wgsl", include_str!("../shaders/agents.wgsl")),
        ("trails.wgsl", include_str!("../shaders/trails.wgsl")),
        ("boids_compute.wgsl", include_str!("../shaders/boids_compute.wgsl")),
    ] {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{name} failed to parse: {error}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{name} failed validation: {error}"));
    }
}
