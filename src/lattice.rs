/// Lattice geometry with on-the-fly neighbor computation.
pub struct Lattice {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub n_spins: usize,
    pub n_dims: usize,
}

impl Lattice {
    pub fn new(shape: Vec<usize>) -> Self {
        let n_dims = shape.len();
        let n_spins: usize = shape.iter().product();

        // Row-major strides: stride[d] = product of shape[d+1..]
        let mut strides = vec![1usize; n_dims];
        for d in (0..n_dims.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        Self {
            shape,
            strides,
            n_spins,
            n_dims,
        }
    }

    /// Compute the flat index of the neighbor of `flat_idx` in dimension `dim`.
    /// `forward = true` means +1 direction, `forward = false` means -1 direction.
    #[inline]
    pub fn neighbor(&self, flat_idx: usize, dim: usize, forward: bool) -> usize {
        let stride = self.strides[dim];
        let size = self.shape[dim];

        // Extract the coordinate in this dimension
        let coord = (flat_idx / stride) % size;

        let new_coord = if forward {
            if coord + 1 == size {
                0
            } else {
                coord + 1
            }
        } else if coord == 0 {
            size - 1
        } else {
            coord - 1
        };

        // flat_idx - coord*stride strips this dimension's contribution (always >= 0),
        // then we add back new_coord*stride.
        flat_idx - coord * stride + new_coord * stride
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
        assert_eq!(lat.neighbor(0, 0, true), 4);
        assert_eq!(lat.neighbor(0, 1, true), 1);

        // Spin 0 = (0,0): backward in dim 0 -> (2,0)=8 (wrap), backward in dim 1 -> (0,3)=3 (wrap)
        assert_eq!(lat.neighbor(0, 0, false), 8);
        assert_eq!(lat.neighbor(0, 1, false), 3);

        // Spin 11 = (2,3): forward in dim 0 -> (0,3)=3 (wrap), forward in dim 1 -> (2,0)=8 (wrap)
        assert_eq!(lat.neighbor(11, 0, true), 3);
        assert_eq!(lat.neighbor(11, 1, true), 8);
    }

    #[test]
    fn test_3d_neighbors() {
        let lat = Lattice::new(vec![2, 3, 4]);
        assert_eq!(lat.n_spins, 24);
        assert_eq!(lat.strides, vec![12, 4, 1]);

        // Spin 0 = (0,0,0)
        assert_eq!(lat.neighbor(0, 0, true), 12); // (1,0,0)
        assert_eq!(lat.neighbor(0, 1, true), 4); // (0,1,0)
        assert_eq!(lat.neighbor(0, 2, true), 1); // (0,0,1)
    }
}
