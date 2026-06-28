pub mod realization;

pub use realization::Realization;

use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::{SimConfig, SweepMode};
use crate::geometry::Lattice;
use crate::statistics::{
    sokal_tau, AutocorrAccum, ClusterSnapshot, ClusterStats, Diagnostics, EquilDiagnosticAccum,
    OverlapAccum, Statistics, SweepResult,
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
#[allow(clippy::too_many_arguments)]
pub fn run_sweep_loop(
    lattice: &Lattice,
    real: &mut Realization,
    n_replicas: usize,
    n_temps: usize,
    config: &SimConfig,
    interrupted: &AtomicBool,
    on_sweep: &(dyn Fn() + Sync),
    realization_idx: usize,
) -> Result<SweepResult, String> {
    run_sweep_loop_impl(
        lattice,
        real,
        n_replicas,
        n_temps,
        config,
        interrupted,
        on_sweep,
        realization_idx,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_sweep_loop_impl(
    lattice: &Lattice,
    real: &mut Realization,
    n_replicas: usize,
    n_temps: usize,
    config: &SimConfig,
    interrupted: &AtomicBool,
    on_sweep: &(dyn Fn() + Sync),
    realization_idx: usize,
    materialize_disabled_stats: bool,
) -> Result<SweepResult, String> {
    config.validate().map_err(|e| format!("{e}"))?;

    let n_spins = lattice.n_spins;
    let n_systems = n_replicas * n_temps;
    let n_sweeps = config.n_sweeps;
    let warmup_sweeps = config.warmup_sweeps;

    let n_modes = config.overlap_cluster.as_ref().map_or(0, |h| h.modes.len());

    if let Some(ref oc_cfg) = config.overlap_cluster {
        let max_gs = oc_cfg.max_group_size();
        if n_replicas < max_gs {
            return Err(format!(
                "overlap cluster requires n_replicas >= max group_size ({n_replicas} < {max_gs})"
            ));
        }
    }

    let n_pairs = n_replicas / 2;
    let collect_fk = config
        .cluster_update
        .as_ref()
        .is_some_and(|c| c.collect_stats);
    let collect_overlap = config
        .overlap_cluster
        .as_ref()
        .is_some_and(|h| h.collect_stats)
        && n_pairs > 0;

    let mut fk_csd_accum: Vec<Vec<u64>> = if collect_fk || materialize_disabled_stats {
        (0..n_temps).map(|_| vec![0u64; n_spins + 1]).collect()
    } else {
        vec![]
    };
    let mut sw_csd_buf: Vec<Vec<u64>> = if collect_fk {
        (0..n_systems).map(|_| vec![0u64; n_spins + 1]).collect()
    } else {
        vec![]
    };

    let mut overlap_csd_accum: Vec<Vec<Vec<u64>>> =
        if collect_overlap || (materialize_disabled_stats && n_modes > 0) {
            (0..n_modes)
                .map(|_| (0..n_temps).map(|_| vec![0u64; n_spins + 1]).collect())
                .collect()
        } else {
            vec![]
        };
    let mut overlap_csd_buf: Vec<Vec<u64>> = if collect_overlap {
        (0..n_temps * n_pairs)
            .map(|_| vec![0u64; n_spins + 1])
            .collect()
    } else {
        vec![]
    };

    let collect_top = collect_overlap;

    let mut top4_accum: Vec<Vec<[f64; 4]>> = if collect_top {
        (0..n_modes).map(|_| vec![[0.0; 4]; n_temps]).collect()
    } else {
        vec![]
    };
    let mut top4_n: Vec<usize> = if collect_top {
        vec![0; n_modes]
    } else {
        vec![]
    };
    let mut top4_buf: Vec<[u32; 4]> = if collect_top {
        vec![[0u32; 4]; n_temps * n_pairs]
    } else {
        vec![]
    };

    let mut overlap_call_count: usize = 0;

    let snapshot_interval = if realization_idx == 0 {
        config
            .overlap_cluster
            .as_ref()
            .and_then(|oc| oc.snapshot_interval)
    } else {
        None
    };
    let n_pair_slots = n_temps * n_pairs;
    let mut snap_buf: Vec<Vec<u32>> = if snapshot_interval.is_some() {
        (0..n_pair_slots)
            .map(|_| Vec::with_capacity(n_spins))
            .collect()
    } else {
        vec![]
    };
    let mut blue_snap_buf: Vec<Vec<u32>> = if snapshot_interval.is_some() {
        (0..n_pair_slots)
            .map(|_| Vec::with_capacity(n_spins))
            .collect()
    } else {
        vec![]
    };
    let mut spin_snap_buf: Vec<Vec<[Vec<i8>; 2]>> = if snapshot_interval.is_some() {
        (0..n_pair_slots).map(|_| Vec::new()).collect()
    } else {
        vec![]
    };
    let mut sid_snap_buf: Vec<Vec<[usize; 2]>> = if snapshot_interval.is_some() {
        (0..n_pair_slots).map(|_| Vec::new()).collect()
    } else {
        vec![]
    };
    let mut cluster_snapshots: Vec<ClusterSnapshot> = Vec::new();

    let mut mags_stat = Statistics::new(n_temps, 1);
    let mut mags2_stat = Statistics::new(n_temps, 1);
    let mut mags4_stat = Statistics::new(n_temps, 1);
    let mut energies_stat = Statistics::new(n_temps, 1);
    let mut energies2_stat = Statistics::new(n_temps, 2);
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

    let mut ov_accum = OverlapAccum::new(
        n_temps,
        n_spins,
        n_pairs,
        lattice.n_neighbors,
        equil_diag,
        collect_q2_ac,
    );

    let mut mags_buf = vec![0.0f32; n_temps];
    let mut mags2_buf = vec![0.0f32; n_temps];
    let mut mags4_buf = vec![0.0f32; n_temps];
    let mut energies_buf = vec![0.0f32; n_temps];
    let mut magnetization_sums = if n_measurement_sweeps > 0 {
        vec![0i64; n_systems]
    } else {
        vec![]
    };

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
            let csd_out = if cluster_cfg.collect_stats && record {
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

            if cluster_cfg.collect_stats && record {
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
            if record {
                spins::energy::compute_energies_and_magnetizations_into(
                    lattice,
                    &real.spins,
                    &real.couplings,
                    &mut real.energies,
                    &mut magnetization_sums,
                );
            } else {
                spins::energy::compute_energies_into(
                    lattice,
                    &real.spins,
                    &real.couplings,
                    &mut real.energies,
                );
            }
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
        }

        if (equil_diag || record) && n_pairs > 0 {
            ov_accum.collect(lattice, &real.spins, &real.system_ids, record);
        }

        if equil_diag {
            if n_pairs > 0 {
                equil_accum
                    .as_mut()
                    .unwrap()
                    .push(&diag_e_buf, &ov_accum.diag_ql_buf);
            } else {
                let zeros = vec![0.0f32; n_temps];
                equil_accum.as_mut().unwrap().push(&diag_e_buf, &zeros);
            }
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
                    let mag = magnetization_sums[system_id] as f32 / n_spins as f32;
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

            if let Some(ref mut acc) = q2_accum {
                let inv = 1.0 / n_pairs as f64;
                for v in ov_accum.q2_ac_buf.iter_mut() {
                    *v *= inv;
                }
                acc.push(&ov_accum.q2_ac_buf);
            }
        }

        let mut did_overlap_update = false;
        if let Some(ref oc_cfg) = config.overlap_cluster {
            if sweep_id % oc_cfg.interval == 0 {
                did_overlap_update = true;
                let mode_idx = overlap_call_count % n_modes;
                let mode = &oc_cfg.modes[mode_idx];

                let ov_csd_out = if oc_cfg.collect_stats && record {
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

                let take_snapshot =
                    snapshot_interval.is_some_and(|si| sweep_id % si == 0) && record;

                let is_cmr = matches!(mode, crate::config::OverlapClusterBuildMode::Cmr);

                let snap = if take_snapshot {
                    for buf in spin_snap_buf.iter_mut() {
                        buf.clear();
                    }
                    for buf in sid_snap_buf.iter_mut() {
                        buf.clear();
                    }
                    Some(snap_buf.as_mut_slice())
                } else {
                    None
                };
                let blue_snap = if take_snapshot && is_cmr {
                    Some(blue_snap_buf.as_mut_slice())
                } else {
                    None
                };
                let spin_snap = if take_snapshot {
                    Some(spin_snap_buf.as_mut_slice())
                } else {
                    None
                };
                let sid_snap = if take_snapshot {
                    Some(sid_snap_buf.as_mut_slice())
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
                    mode,
                    oc_cfg.cluster_mode,
                    ov_csd_out,
                    top4_out,
                    config.sequential,
                    snap,
                    blue_snap,
                    spin_snap,
                    sid_snap,
                );

                if take_snapshot {
                    let ids: Vec<Vec<u32>> = (0..n_temps)
                        .map(|t| snap_buf[t * n_pairs].clone())
                        .collect();
                    let blue = if is_cmr {
                        Some(
                            (0..n_temps)
                                .map(|t| blue_snap_buf[t * n_pairs].clone())
                                .collect(),
                        )
                    } else {
                        None
                    };
                    let spins: Vec<[Vec<i8>; 2]> = (0..n_temps)
                        .map(|t| {
                            spin_snap_buf[t * n_pairs]
                                .first()
                                .cloned()
                                .unwrap_or_else(|| [vec![], vec![]])
                        })
                        .collect();
                    let sids: Vec<[usize; 2]> = (0..n_temps)
                        .map(|t| sid_snap_buf[t * n_pairs].first().copied().unwrap_or([0, 0]))
                        .collect();
                    cluster_snapshots.push(ClusterSnapshot {
                        sweep_id,
                        mode_idx,
                        cluster_ids: ids,
                        blue_ids: blue,
                        spins,
                        system_ids: sids,
                    });
                }

                if oc_cfg.collect_stats && record {
                    for (slot, buf) in overlap_csd_buf.iter().enumerate() {
                        let accum = &mut overlap_csd_accum[mode_idx][slot / n_pairs];
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
                                top4_accum[mode_idx][t][k] += v as f64 / n_spins as f64;
                            }
                        }
                    }
                    top4_n[mode_idx] += 1;
                }

                overlap_call_count += 1;
            }
        }

        if pt_this_sweep {
            if did_overlap_update {
                spins::energy::compute_energies_into(
                    lattice,
                    &real.spins,
                    &real.couplings,
                    &mut real.energies,
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

    let top_cluster_sizes: Vec<Vec<[f64; 4]>> = if collect_top {
        top4_accum
            .iter()
            .zip(top4_n.iter())
            .map(|(mode_accum, &count)| {
                if count == 0 {
                    return vec![];
                }
                let denom = (count * n_pairs) as f64;
                mode_accum
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
        overlap_stats: ov_accum.finish(),
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
        cluster_snapshots,
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
            0,
        );
    }

    let results: Vec<Result<SweepResult, String>> = realizations
        .par_iter_mut()
        .enumerate()
        .map(|(idx, real)| {
            run_sweep_loop_impl(
                lattice,
                real,
                n_replicas,
                n_temps,
                config,
                interrupted,
                on_sweep,
                idx,
                false,
            )
        })
        .collect();

    let mut results: Vec<SweepResult> = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    let snapshots = std::mem::take(&mut results[0].cluster_snapshots);
    let mut agg = SweepResult::aggregate_without_overlap_samples(&results);
    if !agg.overlap_stats.overlap.is_empty() {
        agg.overlap_stats.per_sample_histogram = results
            .iter_mut()
            .map(|r| std::mem::take(&mut r.overlap_stats.histogram))
            .collect();
        agg.overlap_stats.per_sample_ql_at_q_sum = results
            .iter_mut()
            .map(|r| std::mem::take(&mut r.overlap_stats.ql_at_q_sum))
            .collect();
        agg.overlap_stats.per_sample_ql2_at_q_sum = results
            .iter_mut()
            .map(|r| std::mem::take(&mut r.overlap_stats.ql2_at_q_sum))
            .collect();
    }
    if agg.cluster_stats.fk_csd.is_empty() {
        agg.cluster_stats.fk_csd = (0..n_temps)
            .map(|_| vec![0u64; lattice.n_spins + 1])
            .collect();
    }
    let n_modes = config.overlap_cluster.as_ref().map_or(0, |h| h.modes.len());
    if n_modes > 0 && agg.cluster_stats.overlap_csd.is_empty() {
        agg.cluster_stats.overlap_csd = (0..n_modes)
            .map(|_| {
                (0..n_temps)
                    .map(|_| vec![0u64; lattice.n_spins + 1])
                    .collect()
            })
            .collect();
    }
    agg.cluster_snapshots = snapshots;
    Ok(agg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ClusterConfig, ClusterMode, OverlapClusterBuildMode, OverlapClusterConfig,
    };

    fn run_with_cluster_stats(collect_stats: bool) -> SweepResult {
        let lattice = Lattice::new(vec![4, 4]);
        let couplings = vec![1.0; lattice.n_spins * lattice.n_neighbors];
        let mut realization = Realization::new(&lattice, couplings, &[2.0], 1, 42);
        let config = SimConfig {
            n_sweeps: 1,
            warmup_sweeps: 0,
            sweep_mode: SweepMode::Metropolis,
            cluster_update: Some(ClusterConfig {
                interval: 1,
                mode: ClusterMode::Sw,
                collect_stats,
            }),
            pt_interval: None,
            overlap_cluster: None,
            autocorrelation_max_lag: None,
            sequential: true,
            equilibration_diagnostic: false,
        };

        run_sweep_loop(
            &lattice,
            &mut realization,
            1,
            1,
            &config,
            &AtomicBool::new(false),
            &|| {},
            0,
        )
        .unwrap()
    }

    #[test]
    fn disabled_cluster_histograms_preserve_public_shape() {
        let without_stats = run_with_cluster_stats(false);
        assert_eq!(without_stats.cluster_stats.fk_csd.len(), 1);
        assert_eq!(without_stats.cluster_stats.fk_csd[0].iter().sum::<u64>(), 0);

        let with_stats = run_with_cluster_stats(true);
        assert_eq!(with_stats.cluster_stats.fk_csd.len(), 1);
        assert!(with_stats.cluster_stats.fk_csd[0].iter().sum::<u64>() > 0);
    }

    #[test]
    fn disabled_overlap_histograms_preserve_public_shape() {
        let lattice = Lattice::new(vec![4, 4]);
        let couplings = vec![1.0; lattice.n_spins * lattice.n_neighbors];

        for collect_stats in [false, true] {
            let mut realization = Realization::new(&lattice, couplings.clone(), &[2.0], 2, 42);
            let config = SimConfig {
                n_sweeps: 1,
                warmup_sweeps: 0,
                sweep_mode: SweepMode::Metropolis,
                cluster_update: None,
                pt_interval: None,
                overlap_cluster: Some(OverlapClusterConfig {
                    interval: 1,
                    modes: vec![OverlapClusterBuildMode::Houdayer(2)],
                    cluster_mode: ClusterMode::Sw,
                    collect_stats,
                    snapshot_interval: None,
                }),
                autocorrelation_max_lag: None,
                sequential: true,
                equilibration_diagnostic: false,
            };

            let result = run_sweep_loop(
                &lattice,
                &mut realization,
                2,
                1,
                &config,
                &AtomicBool::new(false),
                &|| {},
                0,
            )
            .unwrap();

            if collect_stats {
                assert_eq!(result.cluster_stats.overlap_csd.len(), 1);
                assert!(result.cluster_stats.overlap_csd[0][0].iter().sum::<u64>() > 0);
            } else {
                assert_eq!(result.cluster_stats.overlap_csd.len(), 1);
                assert_eq!(
                    result.cluster_stats.overlap_csd[0][0].iter().sum::<u64>(),
                    0
                );
            }
        }
    }
}
