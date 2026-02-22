use crate::lattice::Lattice;
use crate::parallel::par_over_replicas;
use rand::Rng;
use rand_xoshiro::Xoshiro256StarStar;

// --- Union-Find ---

#[inline]
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

#[inline]
fn union(parent: &mut [u32], rank: &mut [u8], x: u32, y: u32) {
    let rx = find(parent, x);
    let ry = find(parent, y);
    if rx == ry {
        return;
    }
    if rank[rx as usize] < rank[ry as usize] {
        parent[rx as usize] = ry;
    } else {
        parent[ry as usize] = rx;
        if rank[rx as usize] == rank[ry as usize] {
            rank[rx as usize] += 1;
        }
    }
}

// --- Wolff helpers ---

/// Try to add a neighbor to the Wolff cluster.
#[inline]
#[allow(clippy::too_many_arguments)]
fn wolff_try_add(
    neighbor: usize,
    si: f32,
    spin_slice: &[i8],
    coupling: f32,
    temp: f32,
    in_cluster: &mut [bool],
    stack: &mut Vec<usize>,
    rng: &mut Xoshiro256StarStar,
) {
    if in_cluster[neighbor] {
        return;
    }
    let interaction = si * spin_slice[neighbor] as f32 * coupling;
    if interaction > 0.0 {
        let p = 1.0 - (-2.0 * interaction / temp).exp();
        if rng.gen::<f32>() < p {
            in_cluster[neighbor] = true;
            stack.push(neighbor);
        }
    }
}

/// Swendsen-Wang cluster update, parallelized over replicas.
pub fn sw_update(
    lattice: &Lattice,
    spins: &mut [i8],
    interactions: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
) {
    let n_spins = lattice.n_spins;
    let n_dims = lattice.n_dims;

    par_over_replicas(
        spins,
        rngs,
        temperatures,
        system_ids,
        n_spins,
        |spin_slice, rng, temp, system_id| {
            let inter_base = system_id * n_spins * n_dims;

            // Initialize union-find
            let mut parent: Vec<u32> = (0..n_spins as u32).collect();
            let mut rank = vec![0u8; n_spins];

            // Activate bonds and union
            for i in 0..n_spins {
                for d in 0..n_dims {
                    let inter = interactions[inter_base + i * n_dims + d];
                    if inter <= 0.0 {
                        continue;
                    }
                    let p = 1.0 - (-2.0 * inter / temp).exp();
                    if rng.gen::<f32>() < p {
                        let j = lattice.neighbor(i, d, true);
                        union(&mut parent, &mut rank, i as u32, j as u32);
                    }
                }
            }

            // Flatten parent
            for i in 0..n_spins {
                parent[i] = find(&mut parent, i as u32);
            }

            // Decide flip for each cluster root
            let mut flip_decision = vec![2u8; n_spins]; // 2 = undecided
            for i in 0..n_spins {
                let root = parent[i] as usize;
                if flip_decision[root] == 2 {
                    flip_decision[root] = u8::from(rng.gen::<f32>() < 0.5);
                }
                if flip_decision[root] == 1 {
                    spin_slice[i] = -spin_slice[i];
                }
            }
        },
    );
}

/// Wolff single-cluster update, parallelized over replicas.
pub fn wolff_update(
    lattice: &Lattice,
    spins: &mut [i8],
    couplings: &[f32],
    temperatures: &[f32],
    system_ids: &[usize],
    rngs: &mut [Xoshiro256StarStar],
) {
    let n_spins = lattice.n_spins;
    let n_dims = lattice.n_dims;

    par_over_replicas(
        spins,
        rngs,
        temperatures,
        system_ids,
        n_spins,
        |spin_slice, rng, temp, _| {
            let seed = rng.gen_range(0..n_spins);
            let mut in_cluster = vec![false; n_spins];
            let mut stack = Vec::with_capacity(n_spins);

            in_cluster[seed] = true;
            stack.push(seed);

            while let Some(spin_id) = stack.pop() {
                let si = spin_slice[spin_id] as f32;
                for d in 0..n_dims {
                    // Forward neighbor
                    let j_fwd = lattice.neighbor(spin_id, d, true);
                    let c_fwd = couplings[spin_id * n_dims + d];
                    wolff_try_add(
                        j_fwd,
                        si,
                        spin_slice,
                        c_fwd,
                        temp,
                        &mut in_cluster,
                        &mut stack,
                        rng,
                    );

                    // Backward neighbor (coupling is forward coupling of that neighbor)
                    let j_back = lattice.neighbor(spin_id, d, false);
                    let c_back = couplings[j_back * n_dims + d];
                    wolff_try_add(
                        j_back,
                        si,
                        spin_slice,
                        c_back,
                        temp,
                        &mut in_cluster,
                        &mut stack,
                        rng,
                    );
                }
            }

            // Flip the cluster
            for i in 0..n_spins {
                if in_cluster[i] {
                    spin_slice[i] = -spin_slice[i];
                }
            }
        },
    );
}
