use glam::Vec3;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SpatialHash {
    cell_size: f32,
    cells: HashMap<[i32; 3], Vec<usize>>,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size: cell_size.max(0.001),
            cells: HashMap::new(),
        }
    }

    pub fn set_cell_size(&mut self, cell_size: f32) {
        self.cell_size = cell_size.max(0.001);
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }

    pub fn rebuild(&mut self, positions: impl Iterator<Item = (usize, Vec3)>) {
        self.clear();
        for (index, position) in positions {
            self.insert(index, position);
        }
    }

    pub fn insert(&mut self, index: usize, position: Vec3) {
        self.cells
            .entry(self.cell_for(position))
            .or_default()
            .push(index);
    }

    pub fn cell_for(&self, position: Vec3) -> [i32; 3] {
        [
            (position.x / self.cell_size).floor() as i32,
            (position.y / self.cell_size).floor() as i32,
            (position.z / self.cell_size).floor() as i32,
        ]
    }

    pub fn hash_cell(cell: [i32; 3]) -> u64 {
        let x = cell[0] as i64 as u64;
        let y = cell[1] as i64 as u64;
        let z = cell[2] as i64 as u64;
        x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ z.wrapping_mul(83_492_791)
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    pub fn nearby_indices(&self, position: Vec3) -> impl Iterator<Item = usize> + '_ {
        self.nearby_indices_limited(position, self.cell_size, usize::MAX)
    }

    pub fn nearby_indices_limited(
        &self,
        position: Vec3,
        radius: f32,
        max_results: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        let origin = self.cell_for(position);
        let cell_radius = (radius / self.cell_size).ceil().max(1.0) as i32;
        let mut emitted = 0usize;
        (-cell_radius..=cell_radius).flat_map(move |x| {
            (-cell_radius..=cell_radius).flat_map(move |y| {
                (-cell_radius..=cell_radius).flat_map(move |z| {
                    let key = [origin[0] + x, origin[1] + y, origin[2] + z];
                    self.cells
                        .get(&key)
                        .into_iter()
                        .flat_map(|indices| indices.iter().copied())
                })
            })
        }).filter(move |_| {
            if emitted >= max_results {
                false
            } else {
                emitted += 1;
                true
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_negative_and_positive_positions() {
        let hash = SpatialHash::new(4.0);
        assert_eq!(hash.cell_for(Vec3::new(0.0, 3.9, -0.1)), [0, 0, -1]);
        assert_eq!(hash.cell_for(Vec3::new(8.2, -4.1, 4.0)), [2, -2, 1]);
    }

    #[test]
    fn world_to_cell_handles_boundaries() {
        let hash = SpatialHash::new(10.0);
        assert_eq!(hash.cell_for(Vec3::new(9.999, 0.0, 0.0)), [0, 0, 0]);
        assert_eq!(hash.cell_for(Vec3::new(10.0, 0.0, 0.0)), [1, 0, 0]);
        assert_eq!(hash.cell_for(Vec3::new(-0.001, 0.0, 0.0)), [-1, 0, 0]);
        assert_eq!(hash.cell_for(Vec3::new(-10.0, 0.0, 0.0)), [-1, 0, 0]);
    }

    #[test]
    fn hash_is_consistent_for_same_cell() {
        let a = SpatialHash::hash_cell([3, -2, 9]);
        let b = SpatialHash::hash_cell([3, -2, 9]);
        let c = SpatialHash::hash_cell([3, -2, 10]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn query_returns_adjacent_cell_indices() {
        let mut hash = SpatialHash::new(5.0);
        hash.insert(7, Vec3::new(1.0, 0.0, 0.0));
        hash.insert(9, Vec3::new(6.0, 0.0, 0.0));
        hash.insert(11, Vec3::new(25.0, 0.0, 0.0));

        let mut found = hash.nearby_indices(Vec3::ZERO).collect::<Vec<_>>();
        found.sort_unstable();
        assert_eq!(found, vec![7, 9]);
    }

    #[test]
    fn limited_query_caps_results() {
        let mut hash = SpatialHash::new(5.0);
        for index in 0..8 {
            hash.insert(index, Vec3::new(index as f32 * 0.1, 0.0, 0.0));
        }
        assert_eq!(hash.nearby_indices_limited(Vec3::ZERO, 5.0, 3).count(), 3);
    }
}
