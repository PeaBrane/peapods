/// Hypercubic neighbor offsets: one unit vector per dimension.
///
/// For `n_dims = 3` this returns `[[1,0,0], [0,1,0], [0,0,1]]`.
pub fn hypercubic(n_dims: usize) -> Vec<Vec<isize>> {
    (0..n_dims)
        .map(|d| {
            let mut v = vec![0isize; n_dims];
            v[d] = 1;
            v
        })
        .collect()
}

/// Triangular-lattice neighbor offsets (2D only).
///
/// Returns `[[1,0], [0,1], [1,-1]]` — three forward directions giving
/// coordination number 6.
pub fn triangular() -> Vec<Vec<isize>> {
    vec![vec![1, 0], vec![0, 1], vec![1, -1]]
}
