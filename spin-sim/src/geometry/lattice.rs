use super::offsets::hypercubic;

/// Periodic Bravais lattice with precomputed neighbor table.
///
/// Sites are indexed in row-major (C) order. Couplings are stored in a flat
/// array of length `n_spins * n_neighbors`, where element `i * n_neighbors + d`
/// is the coupling on the bond from site `i` to its forward neighbor in
/// direction `d`.
pub struct Lattice {
    /// Extent along each dimension (e.g. `[8, 8, 8]`).
    pub shape: Vec<usize>,
    /// Row-major strides: `strides[d] = product of shape[d+1..]`.
    pub strides: Vec<usize>,
    /// Total number of sites (`shape.iter().product()`).
    pub n_spins: usize,
    /// Number of spatial dimensions (`shape.len()`).
    pub n_dims: usize,
    /// Number of forward neighbor directions per site.
    pub n_neighbors: usize,
    /// Forward neighbors, length `n_spins * n_neighbors`.
    /// Layout: `fwd_neighbors[i * n_neighbors + d]`.
    fwd_neighbors: Vec<u32>,
    /// Backward neighbors, length `n_spins * n_neighbors`.
    /// Layout: `bwd_neighbors[i * n_neighbors + d]`.
    bwd_neighbors: Vec<u32>,
}

impl Lattice {
    /// Create a hypercubic lattice with the given shape (e.g. `vec![16, 16]`).
    pub fn new(shape: Vec<usize>) -> Self {
        let n_dims = shape.len();
        Self::with_offsets(shape, hypercubic(n_dims))
    }

    /// Create a lattice with arbitrary neighbor offsets.
    ///
    /// Each offset is a vector of length `n_dims` specifying a displacement in
    /// lattice coordinates. The backward neighbor is the negation of the offset.
    /// Periodic boundary conditions are applied via `rem_euclid`.
    pub fn with_offsets(shape: Vec<usize>, offsets: Vec<Vec<isize>>) -> Self {
        let n_dims = shape.len();
        let n_neighbors = offsets.len();
        let n_spins: usize = shape.iter().product();

        for (idx, off) in offsets.iter().enumerate() {
            assert_eq!(
                off.len(),
                n_dims,
                "offset {idx} has length {}, expected {n_dims}",
                off.len(),
            );
        }

        let mut strides = vec![1usize; n_dims];
        for d in (0..n_dims.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        let mut fwd_neighbors = vec![0u32; n_spins * n_neighbors];
        let mut bwd_neighbors = vec![0u32; n_spins * n_neighbors];

        for i in 0..n_spins {
            let coords: Vec<usize> = (0..n_dims).map(|d| (i / strides[d]) % shape[d]).collect();

            for (d, off) in offsets.iter().enumerate() {
                for (sign, table) in [(1isize, &mut fwd_neighbors), (-1isize, &mut bwd_neighbors)] {
                    let mut flat = 0usize;
                    for dim in 0..n_dims {
                        let c = (coords[dim] as isize + sign * off[dim])
                            .rem_euclid(shape[dim] as isize)
                            as usize;
                        flat += c * strides[dim];
                    }
                    table[i * n_neighbors + d] = flat as u32;
                }
            }
        }

        Self {
            shape,
            strides,
            n_spins,
            n_dims,
            n_neighbors,
            fwd_neighbors,
            bwd_neighbors,
        }
    }

    #[inline]
    pub fn neighbor_fwd(&self, flat_idx: usize, dim: usize) -> usize {
        self.fwd_neighbors[flat_idx * self.n_neighbors + dim] as usize
    }

    #[inline]
    pub fn neighbor_bwd(&self, flat_idx: usize, dim: usize) -> usize {
        self.bwd_neighbors[flat_idx * self.n_neighbors + dim] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_neighbors() {
        // 3x4 lattice
        let lat = Lattice::new(vec![3, 4]);
        assert_eq!(lat.n_spins, 12);
        assert_eq!(lat.strides, vec![4, 1]);

        // Spin 0 = (0,0): forward in dim 0 -> (1,0)=4, forward in dim 1 -> (0,1)=1
        assert_eq!(lat.neighbor_fwd(0, 0), 4);
        assert_eq!(lat.neighbor_fwd(0, 1), 1);

        // Spin 0 = (0,0): backward in dim 0 -> (2,0)=8 (wrap), backward in dim 1 -> (0,3)=3 (wrap)
        assert_eq!(lat.neighbor_bwd(0, 0), 8);
        assert_eq!(lat.neighbor_bwd(0, 1), 3);

        // Spin 11 = (2,3): forward in dim 0 -> (0,3)=3 (wrap), forward in dim 1 -> (2,0)=8 (wrap)
        assert_eq!(lat.neighbor_fwd(11, 0), 3);
        assert_eq!(lat.neighbor_fwd(11, 1), 8);
    }

    #[test]
    fn test_3d_neighbors() {
        let lat = Lattice::new(vec![2, 3, 4]);
        assert_eq!(lat.n_spins, 24);
        assert_eq!(lat.strides, vec![12, 4, 1]);

        // Spin 0 = (0,0,0)
        assert_eq!(lat.neighbor_fwd(0, 0), 12); // (1,0,0)
        assert_eq!(lat.neighbor_fwd(0, 1), 4); // (0,1,0)
        assert_eq!(lat.neighbor_fwd(0, 2), 1); // (0,0,1)
    }

    #[test]
    fn test_triangular_neighbors() {
        use super::super::offsets::triangular;

        // 4x4 triangular lattice: offsets [1,0], [0,1], [1,-1]
        let lat = Lattice::with_offsets(vec![4, 4], triangular());
        assert_eq!(lat.n_neighbors, 3);
        assert_eq!(lat.n_spins, 16);

        // Site 0 = (0,0)
        // offset [1,0]  -> (1,0) = 4
        assert_eq!(lat.neighbor_fwd(0, 0), 4);
        // offset [0,1]  -> (0,1) = 1
        assert_eq!(lat.neighbor_fwd(0, 1), 1);
        // offset [1,-1] -> (1, -1 mod 4) = (1,3) = 7
        assert_eq!(lat.neighbor_fwd(0, 2), 7);

        // backward of [1,0] from (0,0) -> (-1 mod 4, 0) = (3,0) = 12
        assert_eq!(lat.neighbor_bwd(0, 0), 12);
        // backward of [0,1] from (0,0) -> (0, -1 mod 4) = (0,3) = 3
        assert_eq!(lat.neighbor_bwd(0, 1), 3);
        // backward of [1,-1] from (0,0) -> (-1 mod 4, 1 mod 4) = (3,1) = 13
        assert_eq!(lat.neighbor_bwd(0, 2), 13);

        // Site 5 = (1,1)
        // offset [1,-1] -> (2, 0) = 8
        assert_eq!(lat.neighbor_fwd(5, 2), 8);
        // backward of [1,-1] from (1,1) -> (0, 2) = 2
        assert_eq!(lat.neighbor_bwd(5, 2), 2);

        // Site 15 = (3,3): all forward neighbors wrap
        // offset [1,0]  -> (0,3) = 3
        assert_eq!(lat.neighbor_fwd(15, 0), 3);
        // offset [0,1]  -> (3,0) = 12
        assert_eq!(lat.neighbor_fwd(15, 1), 12);
        // offset [1,-1] -> (0, 2) = 2
        assert_eq!(lat.neighbor_fwd(15, 2), 2);
    }
}
