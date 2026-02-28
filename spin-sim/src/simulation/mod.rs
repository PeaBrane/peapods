pub mod realization;

pub use realization::Realization;

use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::{OverlapClusterBuildMode, OverlapUpdateMode, SimConfig, SweepMode};
use crate::geometry::Lattice;
use crate::statistics::{
    sokal_tau, AutocorrAccum, ClusterStats, Diagnostics, EquilDiagnosticAccum, Statistics,
    SweepResult, OVERLAP_HIST_BINS,
};
use crate::{clusters, mcmc, spins};
use rayon::prelude::*;
use validator::Validate;

/// Run the full Monte Carlo loop (warmup + measurement) for one [`Realization`].
///
/// Each sweep consists of:
/// 1. A full single-spin pass (`sweep_mode`: Metropolis or Gibbs)
/// 2. An optional cluster update (every `cluster_update.interval` sweeps)
/// 3. Measurement (after `warmup_sweeps`)
/// 4. Optional overlap cluster move (every `overlap_cluster.interval` sweeps, requires `n_replicas ≥ 2`)
/// 5. Optional parallel tempering (every `pt_interval` sweeps)
///
/// `on_sweep` is called once per sweep (useful for progress bars).
pub fn run_sweep_loop(
    lattice: &Lattice,
    real: &mut Realization,
    n_replicas: usize,
    n_temps: usize,
    config: &SimConfig,
    interrupted: &AtomicBool,
    on_sweep: &(dyn Fn() + Sync),
) -> Result<SweepResult, String> {
    config.validate().map_err(|e| format!("{e}"))?;

    let n_spins = lattice.n_spins;
    let n_systems = n_replicas * n_temps;
    let n_sweeps = config.n_sweeps;
    let warmup_sweeps = config.warmup_sweeps;

    let overlap_wolff = config
        .overlap_cluster
        .as_ref()
        .is_some_and(|h| h.cluster_mode == crate::config::ClusterMode::Wolff);

    let (stochastic, restrict_to_negative) =
        config
            .overlap_cluster
            .as_ref()
            .map_or((false, true), |h| match h.mode {
                OverlapClusterBuildMode::Houdayer => (false, true),
                OverlapClusterBuildMode::Jorg => (true, true),
                OverlapClusterBuildMode::Cmr(_) => (true, false),
            });

    let group_size = config
        .overlap_cluster
        .as_ref()
        .map_or(2, |h| h.mode.group_size());

    if config.overlap_cluster.is_some() && n_replicas < group_size {
        return Err(format!(
            "overlap cluster requires n_replicas >= group_size ({n_replicas} < {group_size})"
        ));
    }

    let free_assign = config
        .overlap_cluster
        .as_ref()
        .is_some_and(|h| h.update_mode == OverlapUpdateMode::Free);

    let n_pairs = n_replicas / 2;

    let mut fk_csd_accum: Vec<Vec<u64>> = (0..n_temps).map(|_| vec![0u64; n_spins + 1]).collect();
    let mut sw_csd_buf: Vec<Vec<u64>> = (0..n_systems).map(|_| vec![0u64; n_spins + 1]).collect();

    let mut overlap_csd_accum: Vec<Vec<u64>> =
        (0..n_temps).map(|_| vec![0u64; n_spins + 1]).collect();
    let mut overlap_csd_buf: Vec<Vec<u64>> = (0..n_temps * n_pairs)
        .map(|_| vec![0u64; n_spins + 1])
        .collect();

    let collect_top = config
        .overlap_cluster
        .as_ref()
        .is_some_and(|h| h.collect_top_clusters)
        && n_pairs > 0;

    let mut top4_accum: Vec<[f64; 4]> = vec![[0.0; 4]; n_temps];
    let mut top4_n: usize = 0;
    let mut top4_buf: Vec<[u32; 4]> = if collect_top {
        vec![[0u32; 4]; n_temps * n_pairs]
    } else {
        vec![]
    };

    let mut mags_stat = Statistics::new(n_temps, 1);
    let mut mags2_stat = Statistics::new(n_temps, 1);
    let mut mags4_stat = Statistics::new(n_temps, 1);
    let mut energies_stat = Statistics::new(n_temps, 1);
    let mut energies2_stat = Statistics::new(n_temps, 2);
    let mut overlap_stat = Statistics::new(n_temps, 1);
    let mut overlap2_stat = Statistics::new(n_temps, 1);
    let mut overlap4_stat = Statistics::new(n_temps, 1);

    let n_measurement_sweeps = n_sweeps.saturating_sub(warmup_sweeps);
    let ac_max_lag = config
        .autocorrelation_max_lag
        .map(|k| k.min(n_measurement_sweeps / 4).max(1));
    let mut m2_accum = ac_max_lag.map(|k| AutocorrAccum::new(k, n_temps));
    let mut q2_accum = if ac_max_lag.is_some() && n_pairs > 0 {
        ac_max_lag.map(|k| AutocorrAccum::new(k, n_temps))
    } else {
        None
    };
    let collect_ac = ac_max_lag.is_some();
    let collect_q2_ac = q2_accum.is_some();
    let mut m2_ac_buf = if collect_ac {
        vec![0.0f64; n_temps]
    } else {
        vec![]
    };
    let mut q2_ac_buf = if collect_q2_ac {
        vec![0.0f64; n_temps]
    } else {
        vec![]
    };

    let equil_diag = config.equilibration_diagnostic;
    let mut equil_accum = if equil_diag {
        Some(EquilDiagnosticAccum::new(n_temps, n_sweeps))
    } else {
        None
    };
    let mut diag_e_buf = if equil_diag {
        vec![0.0f32; n_temps]
    } else {
        vec![]
    };

    let mut overlap_hist: Vec<Vec<u64>> = if n_pairs > 0 {
        (0..n_temps)
            .map(|_| vec![0u64; OVERLAP_HIST_BINS])
            .collect()
    } else {
        vec![]
    };

    let mut mags_buf = vec![0.0f32; n_temps];
    let mut mags2_buf = vec![0.0f32; n_temps];
    let mut mags4_buf = vec![0.0f32; n_temps];
    let mut energies_buf = vec![0.0f32; n_temps];
    let mut overlaps_buf = vec![0.0f32; n_temps];
    let mut overlaps2_buf = vec![0.0f32; n_temps];
    let mut overlaps4_buf = vec![0.0f32; n_temps];

    for sweep_id in 0..n_sweeps {
        if interrupted.load(Ordering::Relaxed) {
            return Err("interrupted".to_string());
        }
        on_sweep();
        let record = sweep_id >= warmup_sweeps;

        match config.sweep_mode {
            SweepMode::Metropolis => mcmc::sweep::metropolis_sweep(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
                config.sequential,
            ),
            SweepMode::Gibbs => mcmc::sweep::gibbs_sweep(
                lattice,
                &mut real.spins,
                &real.couplings,
                &real.temperatures,
                &real.system_ids,
                &mut real.rngs,
                config.sequential,
            ),
        }

        let do_cluster = config
            .cluster_update
            .as_ref()
            .is_some_and(|c| sweep_id % c.interval == 0);

        if do_cluster {
            let cluster_cfg = config.cluster_update.as_ref().unwrap();
            let wolff = cluster_cfg.mode == crate::config::ClusterMode::Wolff;
            let csd_out = if cluster_cfg.collect_csd && record {
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
                config.sequential,
            );

            if cluster_cfg.collect_csd && record {
                for (slot, buf) in sw_csd_buf.iter().enumerate() {
                    let accum = &mut fk_csd_accum[slot % n_temps];
                    for (a, &b) in accum.iter_mut().zip(buf.iter()) {
                        *a += b;
                    }
                }
            }
        }

        let pt_this_sweep = config
            .pt_interval
            .is_some_and(|interval| sweep_id % interval == 0);

        if record || pt_this_sweep || equil_diag {
            (real.energies, _) = spins::energy::compute_energies(
                lattice,
                &real.spins,
                &real.couplings,
                n_systems,
                false,
            );
        }

        if equil_diag {
            diag_e_buf.fill(0.0);
            #[allow(clippy::needless_range_loop)]
            for r in 0..n_replicas {
                let offset = r * n_temps;
                for t in 0..n_temps {
                    let system_id = real.system_ids[offset + t];
                    diag_e_buf[t] += real.energies[system_id];
                }
            }
            let inv = 1.0 / n_replicas as f32;
            for v in diag_e_buf.iter_mut() {
                *v *= inv;
            }

            let link_overlaps = if n_pairs > 0 {
                spins::energy::compute_link_overlaps(
                    lattice,
                    &real.spins,
                    &real.system_ids,
                    n_replicas,
                    n_temps,
                )
            } else {
                vec![0.0f32; n_temps]
            };

            equil_accum
                .as_mut()
                .unwrap()
                .push(&diag_e_buf, &link_overlaps);
        }

        if record {
            for t in 0..n_temps {
                mags_buf[t] = 0.0;
                mags2_buf[t] = 0.0;
                mags4_buf[t] = 0.0;
                energies_buf[t] = 0.0;
            }

            if collect_ac {
                m2_ac_buf.fill(0.0);
            }

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
                    mags_buf[t] = mag;
                    mags2_buf[t] = m2;
                    mags4_buf[t] = m2 * m2;
                    energies_buf[t] = real.energies[system_id];
                }

                if collect_ac {
                    for t in 0..n_temps {
                        m2_ac_buf[t] += mags2_buf[t] as f64;
                    }
                }

                mags_stat.update(&mags_buf);
                mags2_stat.update(&mags2_buf);
                mags4_stat.update(&mags4_buf);
                energies_stat.update(&energies_buf);
                energies2_stat.update(&energies_buf);
            }

            if let Some(ref mut acc) = m2_accum {
                let inv = 1.0 / n_replicas as f64;
                for v in m2_ac_buf.iter_mut() {
                    *v *= inv;
                }
                acc.push(&m2_ac_buf);
            }

            if collect_q2_ac {
                q2_ac_buf.fill(0.0);
            }

            for pair_idx in 0..n_pairs {
                let r_a = 2 * pair_idx;
                let r_b = 2 * pair_idx + 1;
                for t in 0..n_temps {
                    overlaps_buf[t] = 0.0;
                    overlaps2_buf[t] = 0.0;
                    overlaps4_buf[t] = 0.0;
                }

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
                    overlaps_buf[t] = q;
                    overlaps2_buf[t] = q2;
                    overlaps4_buf[t] = q2 * q2;
                    let bin = (((q + 1.0) * 0.5 * OVERLAP_HIST_BINS as f32) as usize)
                        .min(OVERLAP_HIST_BINS - 1);
                    overlap_hist[t][bin] += 1;
                }

                if collect_q2_ac {
                    for t in 0..n_temps {
                        q2_ac_buf[t] += overlaps2_buf[t] as f64;
                    }
                }

                overlap_stat.update(&overlaps_buf);
                overlap2_stat.update(&overlaps2_buf);
                overlap4_stat.update(&overlaps4_buf);
            }

            if let Some(ref mut acc) = q2_accum {
                let inv = 1.0 / n_pairs as f64;
                for v in q2_ac_buf.iter_mut() {
                    *v *= inv;
                }
                acc.push(&q2_ac_buf);
            }
        }

        if let Some(ref oc_cfg) = config.overlap_cluster {
            if sweep_id % oc_cfg.interval == 0 {
                let ov_csd_out = if oc_cfg.collect_csd && record {
                    for buf in overlap_csd_buf.iter_mut() {
                        buf.fill(0);
                    }
                    Some(overlap_csd_buf.as_mut_slice())
                } else {
                    None
                };

                let top4_out = if collect_top && record {
                    for slot in top4_buf.iter_mut() {
                        *slot = [0u32; 4];
                    }
                    Some(top4_buf.as_mut_slice())
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
                    free_assign,
                    group_size,
                    ov_csd_out,
                    top4_out,
                    config.sequential,
                );

                if oc_cfg.collect_csd && record {
                    for (slot, buf) in overlap_csd_buf.iter().enumerate() {
                        let accum = &mut overlap_csd_accum[slot / n_pairs];
                        for (a, &b) in accum.iter_mut().zip(buf.iter()) {
                            *a += b;
                        }
                    }
                }

                if collect_top && record {
                    for t in 0..n_temps {
                        for p in 0..n_pairs {
                            let raw = top4_buf[t * n_pairs + p];
                            for (k, &v) in raw.iter().enumerate() {
                                top4_accum[t][k] += v as f64 / n_spins as f64;
                            }
                        }
                    }
                    top4_n += 1;
                }
            }
        }

        if pt_this_sweep {
            if config.overlap_cluster.is_some() {
                (real.energies, _) = spins::energy::compute_energies(
                    lattice,
                    &real.spins,
                    &real.couplings,
                    n_systems,
                    false,
                );
            }
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

    let top_cluster_sizes = if collect_top && top4_n > 0 {
        let denom = (top4_n * n_pairs) as f64;
        top4_accum
            .iter()
            .map(|arr| {
                [
                    arr[0] / denom,
                    arr[1] / denom,
                    arr[2] / denom,
                    arr[3] / denom,
                ]
            })
            .collect()
    } else {
        vec![]
    };

    let mags2_tau = m2_accum
        .as_ref()
        .map(|acc| acc.finish().iter().map(|g| sokal_tau(g)).collect())
        .unwrap_or_default();
    let overlap2_tau = q2_accum
        .as_ref()
        .map(|acc| acc.finish().iter().map(|g| sokal_tau(g)).collect())
        .unwrap_or_default();

    let equil_checkpoints = equil_accum.map(|acc| acc.finish()).unwrap_or_default();

    Ok(SweepResult {
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
        overlap_histogram: overlap_hist,
        cluster_stats: ClusterStats {
            fk_csd: fk_csd_accum,
            overlap_csd: overlap_csd_accum,
            top_cluster_sizes,
        },
        diagnostics: Diagnostics {
            mags2_tau,
            overlap2_tau,
            equil_checkpoints,
        },
    })
}

/// Run the sweep loop in parallel over multiple disorder realizations.
///
/// Each realization is processed by [`run_sweep_loop`], then results are
/// averaged via [`SweepResult::aggregate`]. For a single realization the
/// call is made directly, skipping rayon thread-pool overhead.
pub fn run_sweep_parallel(
    lattice: &Lattice,
    realizations: &mut [Realization],
    n_replicas: usize,
    n_temps: usize,
    config: &SimConfig,
    interrupted: &AtomicBool,
    on_sweep: &(dyn Fn() + Sync),
) -> Result<SweepResult, String> {
    if realizations.len() == 1 {
        return run_sweep_loop(
            lattice,
            &mut realizations[0],
            n_replicas,
            n_temps,
            config,
            interrupted,
            on_sweep,
        );
    }

    let results: Vec<Result<SweepResult, String>> = realizations
        .par_iter_mut()
        .map(|real| {
            run_sweep_loop(
                lattice,
                real,
                n_replicas,
                n_temps,
                config,
                interrupted,
                on_sweep,
            )
        })
        .collect();

    let results: Vec<SweepResult> = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(SweepResult::aggregate(&results))
}
