use crate::spins::Lattice;
use crate::statistics::{Statistics, SweepResult};
use crate::{clusters, mcmc, spins};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

/// Mutable state for one disorder realization.
///
/// Holds the coupling array (fixed after construction), spin configurations for
/// every replica at every temperature, and bookkeeping for parallel tempering.
///
/// With `n_replicas` replicas and `n_temps` temperatures there are
/// `n_systems = n_replicas * n_temps` independent spin configurations.
/// Spins are stored in a single flat `Vec` of length `n_systems * n_spins`,
/// where system `i` occupies `spins[i*n_spins .. (i+1)*n_spins]`.
pub struct Realization {
    /// Forward couplings, length `n_spins * n_dims`.
    pub couplings: Vec<f32>,
    /// All spin configurations, length `n_systems * n_spins` (+1/−1).
    pub spins: Vec<i8>,
    /// Temperature assigned to each system slot, length `n_systems`.
    pub temperatures: Vec<f32>,
    /// Parallel-tempering permutation: `system_ids[slot]` is the system index
    /// currently occupying temperature slot `slot`.
    pub system_ids: Vec<usize>,
    /// One PRNG per system.
    pub rngs: Vec<Xoshiro256StarStar>,
    /// One PRNG per overlap-update pair slot, length `n_temps * (n_replicas / 2)`.
    pub pair_rngs: Vec<Xoshiro256StarStar>,
    /// Cached total energy per system (E / N), length `n_systems`.
    pub energies: Vec<f32>,
}

impl Realization {
    /// Initialize a realization with random ±1 spins.
    ///
    /// Seeds replica RNGs deterministically as `base_seed, base_seed+1, …`.
    pub fn new(
        lattice: &Lattice,
        couplings: Vec<f32>,
        temps: &[f32],
        n_replicas: usize,
        base_seed: u64,
    ) -> Self {
        let n_spins = lattice.n_spins;
        let n_temps = temps.len();
        let n_systems = n_replicas * n_temps;

        let temperatures = temps.repeat(n_replicas);

        let mut rngs = Vec::with_capacity(n_systems);
        for i in 0..n_systems {
            rngs.push(Xoshiro256StarStar::seed_from_u64(base_seed + i as u64));
        }

        let mut spins = vec![0i8; n_systems * n_spins];
        for (i, rng) in rngs.iter_mut().enumerate() {
            for j in 0..n_spins {
                spins[i * n_spins + j] = if rng.gen::<f32>() < 0.5 { -1 } else { 1 };
            }
        }

        let system_ids: Vec<usize> = (0..n_systems).collect();

        let n_pairs = n_replicas / 2;
        let mut pair_rngs = Vec::with_capacity(n_temps * n_pairs);
        for i in 0..n_temps * n_pairs {
            pair_rngs.push(Xoshiro256StarStar::seed_from_u64(
                base_seed + n_systems as u64 + i as u64,
            ));
        }

        let (energies, _) =
            spins::energy::compute_energies(lattice, &spins, &couplings, n_systems, false);

        Self {
            couplings,
            spins,
            temperatures,
            system_ids,
            rngs,
            pair_rngs,
            energies,
        }
    }

    /// Re-randomize all spins and reset the tempering permutation.
    pub fn reset(&mut self, lattice: &Lattice, n_replicas: usize, n_temps: usize, base_seed: u64) {
        let n_spins = lattice.n_spins;
        let n_systems = n_replicas * n_temps;

        for i in 0..n_systems {
            self.rngs[i] = Xoshiro256StarStar::seed_from_u64(base_seed + i as u64);
            for j in 0..n_spins {
                self.spins[i * n_spins + j] = if self.rngs[i].gen::<f32>() < 0.5 {
                    -1
                } else {
                    1
                };
            }
        }

        self.system_ids = (0..n_systems).collect();

        let n_pairs = n_replicas / 2;
        for i in 0..n_temps * n_pairs {
            self.pair_rngs[i] =
                Xoshiro256StarStar::seed_from_u64(base_seed + n_systems as u64 + i as u64);
        }

        let (energies, _) = spins::energy::compute_energies(
            lattice,
            &self.spins,
            &self.couplings,
            n_systems,
            false,
        );
        self.energies = energies;
    }
}

/// Run the full Monte Carlo loop (warmup + measurement) for one [`Realization`].
///
/// Each sweep consists of:
/// 1. A full single-spin pass (`sweep_mode`: `"metropolis"` or `"gibbs"`)
/// 2. An optional cluster update (`cluster_mode`: `"wolff"` or `"sw"`,
///    every `cluster_update_interval` sweeps)
/// 3. Measurement (after `warmup_sweeps`)
/// 4. Optional Houdayer ICM (every `houdayer_interval` sweeps, requires `n_replicas ≥ 2`)
/// 5. Optional parallel tempering (every `pt_interval` sweeps)
///
/// `on_sweep` is called once per sweep (useful for progress bars).
#[allow(clippy::too_many_arguments)]
pub fn run_sweep_loop(
    lattice: &Lattice,
    real: &mut Realization,
    n_replicas: usize,
    n_temps: usize,
    n_sweeps: usize,
    warmup_sweeps: usize,
    sweep_mode: &str,
    cluster_update_interval: Option<usize>,
    cluster_mode: &str,
    pt_interval: Option<usize>,
    houdayer_interval: Option<usize>,
    houdayer_mode: &str,
    overlap_cluster_mode: &str,
    collect_csd: bool,
    on_sweep: &(dyn Fn() + Sync),
) -> SweepResult {
    let n_spins = lattice.n_spins;
    let n_systems = n_replicas * n_temps;
    let overlap_wolff = overlap_cluster_mode == "wolff";

    let (stochastic, restrict_to_negative) = match houdayer_mode {
        "houdayer" => (false, true),
        "jorg" => (true, true),
        "cmr" => (true, false),
        _ => unreachable!(),
    };

    let n_pairs = n_replicas / 2;

    let mut fk_csd_accum: Vec<Vec<u64>> = (0..n_temps).map(|_| vec![0u64; n_spins + 1]).collect();
    let mut sw_csd_buf: Vec<Vec<u64>> = (0..n_systems).map(|_| vec![0u64; n_spins + 1]).collect();

    let mut overlap_csd_accum: Vec<Vec<u64>> =
        (0..n_temps).map(|_| vec![0u64; n_spins + 1]).collect();
    let mut overlap_csd_buf: Vec<Vec<u64>> = (0..n_temps * n_pairs)
        .map(|_| vec![0u64; n_spins + 1])
        .collect();

    let mut mags_stat = Statistics::new(n_temps, 1);
    let mut mags2_stat = Statistics::new(n_temps, 1);
    let mut mags4_stat = Statistics::new(n_temps, 1);
    let mut energies_stat = Statistics::new(n_temps, 1);
    let mut energies2_stat = Statistics::new(n_temps, 2);
    let mut overlap_stat = Statistics::new(n_temps, 1);
    let mut overlap2_stat = Statistics::new(n_temps, 1);
    let mut overlap4_stat = Statistics::new(n_temps, 1);

    for sweep_id in 0..n_sweeps {
        on_sweep();
        let record = sweep_id >= warmup_sweeps;

        match sweep_mode {
            "metropolis" => mcmc::sweep::metropolis_sweep(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
            ),
            "gibbs" => mcmc::sweep::gibbs_sweep(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
            ),
            _ => unreachable!(),
        }

        let do_cluster = cluster_update_interval.is_some_and(|interval| sweep_id % interval == 0);

        if do_cluster {
            let wolff = cluster_mode == "wolff";
            let csd_out = if collect_csd && record {
                for buf in sw_csd_buf.iter_mut() {
                    buf.fill(0);
                }
                Some(sw_csd_buf.as_mut_slice())
            } else {
                None
            };

            clusters::fk_update(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
                wolff,
                csd_out,
            );

            if collect_csd && record {
                for (slot, buf) in sw_csd_buf.iter().enumerate() {
                    let accum = &mut fk_csd_accum[slot % n_temps];
                    for (a, &b) in accum.iter_mut().zip(buf.iter()) {
                        *a += b;
                    }
                }
            }

            (real.energies, _) = spins::energy::compute_energies(
                lattice,
                &real.spins,
                &real.couplings,
                n_systems,
                false,
            );
        } else {
            (real.energies, _) = spins::energy::compute_energies(
                lattice,
                &real.spins,
                &real.couplings,
                n_systems,
                false,
            );
        }

        if record {
            let mut mags = vec![0.0f32; n_temps];
            let mut mags2 = vec![0.0f32; n_temps];
            let mut mags4 = vec![0.0f32; n_temps];
            let mut energies_ordered = vec![0.0f32; n_temps];

            for r in 0..n_replicas {
                let offset = r * n_temps;
                for t in 0..n_temps {
                    let system_id = real.system_ids[offset + t];
                    let spin_base = system_id * n_spins;
                    let mut sum = 0i64;
                    for j in 0..n_spins {
                        sum += real.spins[spin_base + j] as i64;
                    }
                    let mag = sum as f32 / n_spins as f32;
                    let m2 = mag * mag;
                    mags[t] = mag;
                    mags2[t] = m2;
                    mags4[t] = m2 * m2;
                    energies_ordered[t] = real.energies[system_id];
                }

                mags_stat.update(&mags);
                mags2_stat.update(&mags2);
                mags4_stat.update(&mags4);
                energies_stat.update(&energies_ordered);
                energies2_stat.update(&energies_ordered);
            }

            for pair_idx in 0..n_pairs {
                let r_a = 2 * pair_idx;
                let r_b = 2 * pair_idx + 1;
                let mut overlaps = vec![0.0f32; n_temps];
                let mut overlaps2 = vec![0.0f32; n_temps];
                let mut overlaps4 = vec![0.0f32; n_temps];

                for t in 0..n_temps {
                    let sys_a = real.system_ids[r_a * n_temps + t];
                    let sys_b = real.system_ids[r_b * n_temps + t];
                    let base_a = sys_a * n_spins;
                    let base_b = sys_b * n_spins;
                    let mut dot = 0i64;
                    for j in 0..n_spins {
                        dot += (real.spins[base_a + j] as i64) * (real.spins[base_b + j] as i64);
                    }
                    let q = dot as f32 / n_spins as f32;
                    let q2 = q * q;
                    overlaps[t] = q;
                    overlaps2[t] = q2;
                    overlaps4[t] = q2 * q2;
                }

                overlap_stat.update(&overlaps);
                overlap2_stat.update(&overlaps2);
                overlap4_stat.update(&overlaps4);
            }
        }

        if let Some(interval) = houdayer_interval {
            if sweep_id % interval == 0 && n_replicas >= 2 {
                let ov_csd_out = if collect_csd && record {
                    for buf in overlap_csd_buf.iter_mut() {
                        buf.fill(0);
                    }
                    Some(overlap_csd_buf.as_mut_slice())
                } else {
                    None
                };

                clusters::overlap_update(
                    lattice,
                    &mut real.spins,
                    &real.couplings,
                    &real.temperatures,
                    &real.system_ids,
                    n_replicas,
                    n_temps,
                    &mut real.pair_rngs,
                    stochastic,
                    restrict_to_negative,
                    overlap_wolff,
                    ov_csd_out,
                );

                if collect_csd && record {
                    for (slot, buf) in overlap_csd_buf.iter().enumerate() {
                        let accum = &mut overlap_csd_accum[slot / n_pairs];
                        for (a, &b) in accum.iter_mut().zip(buf.iter()) {
                            *a += b;
                        }
                    }
                }

                (real.energies, _) = spins::energy::compute_energies(
                    lattice,
                    &real.spins,
                    &real.couplings,
                    n_systems,
                    false,
                );
            }
        }

        if let Some(interval) = pt_interval {
            if sweep_id % interval == 0 {
                for r in 0..n_replicas {
                    let offset = r * n_temps;
                    let sid_slice = &mut real.system_ids[offset..offset + n_temps];
                    let temp_slice = &real.temperatures[offset..offset + n_temps];
                    mcmc::tempering::parallel_tempering(
                        &real.energies,
                        temp_slice,
                        sid_slice,
                        n_spins,
                        &mut real.rngs[offset],
                    );
                }
            }
        }
    }

    SweepResult {
        mags: mags_stat.average(),
        mags2: mags2_stat.average(),
        mags4: mags4_stat.average(),
        energies: energies_stat.average(),
        energies2: energies2_stat.average(),
        overlap: if n_pairs > 0 {
            overlap_stat.average()
        } else {
            vec![]
        },
        overlap2: if n_pairs > 0 {
            overlap2_stat.average()
        } else {
            vec![]
        },
        overlap4: if n_pairs > 0 {
            overlap4_stat.average()
        } else {
            vec![]
        },
        fk_csd: fk_csd_accum,
        overlap_csd: overlap_csd_accum,
    }
}
