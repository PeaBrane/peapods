---
name: analyze
description: Analyze peapods Monte Carlo output correctly - check equilibration, autocorrelation and parallel-tempering health, compute disorder-averaged observables with jackknife error bars, and locate phase transitions from Binder or correlation-length crossings. Use when a user asks whether a run is equilibrated, how many sweeps or disorder samples they need, how to get error bars, or how to estimate a critical temperature from peapods results.
---

# Analyzing peapods runs

An index of what peapods measures and how to turn it into trustworthy numbers. The
reference implementation of everything below is
[`validation/spin_glass_3d/`](../../validation/spin_glass_3d/) (`run.py` produces data,
`analyze.py` reduces it); reuse its functions rather than rederiving them. For other
temperature grids or sizes, put that directory on `sys.path`, set `run.TEMPERATURES`,
call `run.run_batch`, and pass its dict to `analyze.window_observables`, `jackknife` and
`crossing`. Estimates use the last log window (the second half of the chain).

## Equilibration

| Check | How |
|---|---|
| Log-binning (any model) | Call `sample()` repeatedly on one model with `warmup_ratio=0.0` and lengths `2^w, 2^w, 2^(w+1), ...`; each call measures the window `[2^k, 2^(k+1))` of one chain. Equilibrated once the last two or three windows agree. Compare windows on the same samples with a paired jackknife of their difference: it is far tighter than the windows' own errors (`analyze.py`, `xi_over_L_minus_last`) |
| Gaussian couplings only | `equilibration_diagnostic=True`, then `model.equilibration_delta()`: the identity Δ = e − J²β(N_b/N)(1 − q_l) → 0 holds at equilibrium for Gaussian couplings. Not valid for ±J |
| Autocorrelation | `autocorrelation_max_lag=k`: `mags2_tau`, and with replicas `overlap2_tau`, the integrated time in measured sweeps. `overlap2_tau` uses q² averaged over all replica pairs, so replica relabeling does not fake decorrelation. `autocorrelation_backend="fft"` is faster but keeps the full history. This sizes thermal error bars; it is not an equilibration test (under PT it can be a few sweeps while equilibration takes thousands). The automatic window closes after a few lags when the lag-1 correlation is near zero, which full-ladder PT produces, so a weak long tail goes uncounted (0.5 sweeps reported where batch means give 4–10); cross-check with batch means. The value is a mean over disorder realizations |
| Parallel tempering | `result["per_disorder"]["parallel_tempering"]`, counted per `sample()` call (sum over windows for a whole run): acceptance `edge_acceptances / edge_attempts` per adjacent pair (keep the minimum above ~0.2; add temperatures where it drops) and `round_trips` per walker (each walker should cross the ladder several times per run). Prefer `pt_schedule="full_ladder"`: it moves walkers along the ladder far faster than `single_random_edge` |

Reading window comparisons: with many temperatures, expect occasional 2–3σ differences
that move together across T; drift is a same-sign trend toward the last window at the
lowest T. Compare derived results (a crossing temperature) between the last two windows
the same way.

Choose sweep counts from a pilot: run ~100 samples per size with generous length, find
the window where results stop drifting at the lowest temperature, then use at least ~8×
that in production. Time the pilot: cost per disorder sample and sweep scales as
L^d × n_temps × n_replicas, and the largest size usually dominates the error budget,
so give it the samples the budget allows.

## Disorder averages and errors

- Average over disorder after the thermal average. Per-realization values: Ising
  `result["physics"]["per_disorder"]` (with `collect_physics=True`); XY
  `result["per_disorder"]`, which mirrors every key with a leading disorder axis.
- For a disorder mean of a per-sample quantity (energy, heat capacity, helicity), the
  error is `std(ddof=1) / sqrt(n_disorder)`. With few samples the error bar is itself
  uncertain, by roughly 1/√(2(n − 1)) relative; say so.
- Ratios of disorder averages (Binder ratios, ξ/L, U4) need a jackknife over disorder
  realizations. Do not propagate errors of the averages separately. `analyze.py`:
  `block_means` and `jackknife`, delete-one-block with `min(64, n_disorder)` blocks.
- `per_disorder["sg_binder"]` (and other per-sample ratios) are ratios within one
  sample; never average them. The quoted `sg_binder` (`model.sg_binder`, equal to
  `physics["sg_binder"]`) is 1 − [⟨q⁴⟩]/(3[⟨q²⟩]²) of disorder averages; jackknife it from
  `per_disorder["overlap2"]` and `["overlap4"]`, shape `(n_disorder, n_temps)`.
- Thermal errors within one realization come from blocked sums: `block_size=` (Ising
  `collect_physics`) or `collect_blocks=True` (XY, blocks of 128 sweeps by default) give
  `per_disorder["blocks"]` with raw-moment `sums`, `counts` and `sweeps`; recompute
  derived quantities from summed moments (`docs/xy.md`, "Blocks, errors and
  autocorrelation"). `validation/xy_finite_size.py` shows a block jackknife that respects
  chain boundaries.
- Heat capacity: use `heat_capacity` / `energy_variance`, the thermal variance inside
  each realization. `energies2 - energies**2` of disorder averages adds the disorder
  variance of the mean energy.

## Locating a transition

| Model | Observable | Where |
|---|---|---|
| Ising ferromagnet | Binder cumulant `binder_cumulant`; second-moment `correlation_length_ratio` | attributes; `result["physics"]` |
| Ising spin glass | `sg_binder` (= 1 − U4/3); ξ_SG/L; U4 = `overlap4 / overlap2**2` | `result["physics"]` with `n_replicas >= 2`. `sg_correlation_length_ratio` is per axis and has no error bar: for crossings, form ξ/L from disorder averages of the per-disorder `overlap2`, `overlap4` and `overlap_structure_factor_min` (average its last axis) with `analyze.estimators`, and jackknife. Never average per-disorder ξ/L |
| XY | helicity modulus, `correlation_length_ratio` | XY `result`, `docs/xy.md` |

Curves of ξ/L (or Binder) against T for several L cross near T_c. Take crossings of
pairs of sizes, ideally (L, 2L), by interpolating the difference on the temperature
grid, jackknife the crossing temperature (deleting block j from both sizes, so both need
the same block count), and extrapolate the crossings in L^-(ω+1/ν) when the exponents
are known (`analyze.py`: `crossing`, `extrapolation`). Check for a single sign change
and for no NaN among the deleted-block crossings; if one side of the crossing is not
resolved, the error bar is only indicative. Small lattices cross well above T_c in 3D
spin glasses; expect drift and say so rather than reading T_c off one pair.
`tests/utils.py` (`assert_crossing`) is a quick spread-at-T_c check for known T_c.

## Conventions worth checking before comparing numbers

- Ising legacy `energies` use the opposite sign of the physical `H/N` reported in
  `physics` and by XY.
- Ising Binder uses one-component normalization `1 - <m^4>/(3<m^2>^2)`; XY uses its
  two-component form. Spin-glass papers often quote `g = (3 - U4)/2` instead of
  `sg_binder`.
- Second-moment ξ uses k_min = 2π/L per axis: ξ = √(χ(0)/χ(k_min) − 1)/(2 sin(k_min/2)).
- Overlap statistics pair replicas (2p, 2p + 1); with more replicas, more pairs enter.
