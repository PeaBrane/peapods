use crate::lattice::Lattice;
use crate::stats::Statistics;
use crate::{clusters, energy, sweep, tempering};
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
    /// Cached total energy per system (E / N), length `n_systems`.
    pub energies: Vec<f32>,
    /// Cached per-bond interactions (s_i * s_j * J_ij), length `n_systems * n_spins * n_dims`.
    pub interactions: Vec<f32>,
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

        let (energies, interactions) =
            energy::compute_energies(lattice, &spins, &couplings, n_systems, true);
        let interactions = interactions.unwrap();

        Self {
            couplings,
            spins,
            temperatures,
            system_ids,
            rngs,
            energies,
            interactions,
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

        let (energies, interactions) =
            energy::compute_energies(lattice, &self.spins, &self.couplings, n_systems, true);
        self.energies = energies;
        self.interactions = interactions.unwrap();
    }
}

/// Per-temperature observables averaged over measurement sweeps and replicas.
///
/// All vectors are indexed by temperature index and have length `n_temps`.
/// Overlap vectors are empty when `n_replicas < 2`.
pub struct SweepResult {
    /// ⟨m⟩ — mean magnetization per spin.
    pub mags: Vec<f64>,
    /// ⟨m²⟩.
    pub mags2: Vec<f64>,
    /// ⟨m⁴⟩.
    pub mags4: Vec<f64>,
    /// ⟨E⟩ — mean energy per spin.
    pub energies: Vec<f64>,
    /// ⟨E²⟩.
    pub energies2: Vec<f64>,
    /// ⟨q⟩ — mean replica overlap.
    pub overlap: Vec<f64>,
    /// ⟨q²⟩.
    pub overlap2: Vec<f64>,
    /// ⟨q⁴⟩.
    pub overlap4: Vec<f64>,
    /// Raw FK cluster sizes per temperature, concatenated across sweeps/replicas.
    pub csd_sizes: Vec<Vec<usize>>,
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
    collect_csd: bool,
    on_sweep: &(dyn Fn() + Sync),
) -> SweepResult {
    let n_spins = lattice.n_spins;
    let n_systems = n_replicas * n_temps;

    let mut csd_accum: Vec<Vec<usize>> = (0..n_temps).map(|_| Vec::new()).collect();

    let mut mags_stat = Statistics::new(n_temps, 1);
    let mut mags2_stat = Statistics::new(n_temps, 1);
    let mut mags4_stat = Statistics::new(n_temps, 1);
    let mut energies_stat = Statistics::new(n_temps, 1);
    let mut energies2_stat = Statistics::new(n_temps, 2);

    let n_pairs = n_replicas / 2;
    let mut overlap_stat = Statistics::new(n_temps, 1);
    let mut overlap2_stat = Statistics::new(n_temps, 1);
    let mut overlap4_stat = Statistics::new(n_temps, 1);

    for sweep_id in 0..n_sweeps {
        on_sweep();
        let record = sweep_id >= warmup_sweeps;

        match sweep_mode {
            "metropolis" => sweep::metropolis_sweep(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
            ),
            "gibbs" => sweep::gibbs_sweep(
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
            match cluster_mode {
                "wolff" => {
                    clusters::wolff_update(
                        lattice,
                        &mut real.spins,
                        &real.couplings,
                        &real.temperatures,
                        &real.system_ids,
                        &mut real.rngs,
                    );
                    (real.energies, _) = energy::compute_energies(
                        lattice,
                        &real.spins,
                        &real.couplings,
                        n_systems,
                        false,
                    );
                }
                "sw" => {
                    let (energies, interactions) = energy::compute_energies(
                        lattice,
                        &real.spins,
                        &real.couplings,
                        n_systems,
                        true,
                    );
                    real.energies = energies;
                    real.interactions = interactions.unwrap();

                    clusters::sw_update(
                        lattice,
                        &mut real.spins,
                        &real.interactions,
                        &real.temperatures,
                        &real.system_ids,
                        &mut real.rngs,
                    );

                    (real.energies, _) = energy::compute_energies(
                        lattice,
                        &real.spins,
                        &real.couplings,
                        n_systems,
                        false,
                    );
                }
                _ => unreachable!(),
            }
        } else {
            (real.energies, _) =
                energy::compute_energies(lattice, &real.spins, &real.couplings, n_systems, false);
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

            if collect_csd {
                for r in 0..n_replicas {
                    let offset = r * n_temps;
                    for (t, accum) in csd_accum.iter_mut().enumerate() {
                        let system_id = real.system_ids[offset + t];
                        let sizes = clusters::fk_cluster_sizes(
                            lattice,
                            &real.spins,
                            &real.couplings,
                            real.temperatures[offset + t],
                            system_id * n_spins,
                            &mut real.rngs[system_id],
                        );
                        accum.extend(sizes);
                    }
                }
            }
        }

        if let Some(interval) = houdayer_interval {
            if sweep_id % interval == 0 && n_replicas >= 2 {
                clusters::houdayer_update(
                    lattice,
                    &mut real.spins,
                    &real.system_ids,
                    n_replicas,
                    n_temps,
                    &mut real.rngs[0],
                );
                (real.energies, _) = energy::compute_energies(
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
                    tempering::parallel_tempering(
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
        csd_sizes: csd_accum,
    }
}

/// Average [`SweepResult`]s across disorder realizations.
pub fn aggregate_results(results: &[SweepResult]) -> SweepResult {
    let n = results.len() as f64;
    let n_temps = results[0].mags.len();
    let n_overlap = results[0].overlap.len();
    let n_csd = results[0].csd_sizes.len();

    let mut agg = SweepResult {
        mags: vec![0.0; n_temps],
        mags2: vec![0.0; n_temps],
        mags4: vec![0.0; n_temps],
        energies: vec![0.0; n_temps],
        energies2: vec![0.0; n_temps],
        overlap: vec![0.0; n_overlap],
        overlap2: vec![0.0; n_overlap],
        overlap4: vec![0.0; n_overlap],
        csd_sizes: (0..n_csd).map(|_| Vec::new()).collect(),
    };

    for r in results {
        for (a, &v) in agg.mags.iter_mut().zip(r.mags.iter()) {
            *a += v;
        }
        for (a, &v) in agg.mags2.iter_mut().zip(r.mags2.iter()) {
            *a += v;
        }
        for (a, &v) in agg.mags4.iter_mut().zip(r.mags4.iter()) {
            *a += v;
        }
        for (a, &v) in agg.energies.iter_mut().zip(r.energies.iter()) {
            *a += v;
        }
        for (a, &v) in agg.energies2.iter_mut().zip(r.energies2.iter()) {
            *a += v;
        }
        for (a, &v) in agg.overlap.iter_mut().zip(r.overlap.iter()) {
            *a += v;
        }
        for (a, &v) in agg.overlap2.iter_mut().zip(r.overlap2.iter()) {
            *a += v;
        }
        for (a, &v) in agg.overlap4.iter_mut().zip(r.overlap4.iter()) {
            *a += v;
        }
        for (a, s) in agg.csd_sizes.iter_mut().zip(r.csd_sizes.iter()) {
            a.extend(s);
        }
    }

    for v in agg
        .mags
        .iter_mut()
        .chain(agg.mags2.iter_mut())
        .chain(agg.mags4.iter_mut())
        .chain(agg.energies.iter_mut())
        .chain(agg.energies2.iter_mut())
        .chain(agg.overlap.iter_mut())
        .chain(agg.overlap2.iter_mut())
        .chain(agg.overlap4.iter_mut())
    {
        *v /= n;
    }

    agg
}
