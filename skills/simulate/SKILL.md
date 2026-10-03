---
name: simulate
description: Set up and run peapods Monte Carlo simulations of Ising ferromagnets, Ising spin glasses and XY models from Python, the peapods CLI or the spin-sim Rust crate. Use when a user wants to simulate a spin model, use a custom lattice, custom or diluted couplings, choose update moves (Metropolis, Gibbs, Swendsen-Wang, Wolff, parallel tempering, Houdayer, Jörg, CMR, replica Monte Carlo), read observables from a run, or extract spin configurations and cluster snapshots.
---

# Running peapods simulations

An index of entry points. Read the referenced docstring or file for full detail; the
public API is `peapods.Ising`, `peapods.XY`, the `peapods` CLI and the `spin-sim` crate.
Install with `uv pip install peapods` (wheels for Linux, macOS, Windows) or build from a
checkout with `maturin develop --release` (see the develop skill).

## Models

| Need | Entry point | Notes |
|---|---|---|
| Ising ferromagnet or spin glass | `Ising(lattice_shape, couplings=..., temperatures=..., n_replicas=1, n_disorder=1, seed=None)` | `python/peapods/spin_models.py` |
| XY (O(2)) model, signed bonds, dilution | `XY(lattice_shape, couplings=..., temperatures=..., n_replicas=1, n_disorder=1, occupation=None, seed=None)` | Hypercubic only, every extent ≥ 3; guide in `docs/xy.md` |
| No Python | `spin_sim::{Lattice, Realization, run_sweep_parallel}`, `spin_sim::simulation::xy::XySimulation`, `spin_sim::config::SimConfig` | Crate `spin-sim` (docs.rs); `spin-sim/examples/bench.rs` is a complete driver |

Every model holds `n_disorder` coupling realizations, each with `n_replicas` copies at
every temperature. All lattices are periodic. `seed` fixes couplings and dynamics;
`None` draws fresh entropy.

## Lattices and couplings

| Need | How |
|---|---|
| Hypercubic in any dimension | `lattice_shape=(L,) * d` |
| Triangular, FCC, BCC (Ising) | `geometry="triangular"` / `"fcc"` / `"bcc"`; offsets are in primitive-cell coordinates (`GEOMETRIES` in `spin_models.py`), so an `L^d` lattice is one connected lattice of `L^d` sites |
| Any Bravais lattice (Ising) | `neighbor_offsets=[[1, 0], [0, 1], [1, -1]]`: forward offsets only, one per bond direction; the backward neighbor is the negated offset. Write offsets in primitive-cell coordinates: conventional-cell offsets on a cubic grid split it into disconnected copies (two for FCC, four for BCC at even L). Keep each extent larger than twice the largest offset component, or bonds wrap onto themselves |
| Honeycomb (Ising) | Brick wall: `lattice_shape=(L0, L1)` with both even, default square offsets; keep every axis-1 bond and the axis-0 bond only where `(i + j) % 2 == 0` (zero the others in a custom couplings array). Every site then has 3 bonds |
| Other non-Bravais lattices | Only by zeroing bonds of a Bravais lattice whose every site belongs to the target lattice: Ising has no site dilution, so a site with all bonds zeroed is still a free spin in `m` and `q`. Lattices that need removed sites (kagome on a triangular grid) are not supported |
| Built-in disorder | `couplings="ferro"`, `"bimodal"` (±1), `"gaussian"` |
| Custom couplings | Array of shape `(*lattice_shape, n_neighbors)` or `(n_disorder, *lattice_shape, n_neighbors)`. `J[(k,) i, j, ..., d]` couples site `(i, j, ...)` to `(i, j, ...) + offsets[d]` modulo the shape; without `neighbor_offsets`, `offsets[d]` is the unit vector along axis `d`. Zero removes a bond |
| Site dilution (XY) | `occupation=` Boolean array, `True` = occupied, of `lattice_shape` (one pattern reused for every disorder sample) or `(n_disorder, *lattice_shape)` (independent patterns). Bonds touching vacancies are zeroed; XY observables stay normalized by the full `L^d` including vacancies |
| Temperatures | Any increasing array; geometric spacing suits parallel tempering. Ising uses float32, XY float64 |

## Sampling: `model.sample(n_sweeps, ...)`

`n_sweeps` includes warmup (`warmup_ratio`, default 0.25). Successive `sample()` calls
continue the same chain; `reset(seed)` restarts it. Each call reports statistics of that
call only.

| Goal | Arguments |
|---|---|
| Local updates | `sweep_mode="metropolis"` or `"gibbs"` (heat bath) |
| Ferromagnet near or below T_c | `cluster_update_interval=1, cluster_mode="sw"` or `"wolff"` |
| Spin glass, rough landscapes | `pt_interval=1, pt_schedule="full_ladder"`, `n_replicas >= 2` for overlaps |
| Replica cluster moves | `overlap_cluster_update_interval=k, overlap_cluster_build_mode=` `"houdayer"`, `"jorg"`, `"cmr"`, `"rmc"`, `"pairN"`, `"jorgN"`, or several joined by `+`; `overlap_cluster_mode="wolff"`/`"sw"`. Read `docs/overlap_moves.md` "Which move to use" first: in its benchmark only `rmc` with `"wolff"` clearly paid for itself (measured before a parallel-tempering fix that made PT alone much faster), and `houdN` with N > 2 violates detailed balance |
| Restrict same-temperature moves to low T | `overlap_cluster_max_temperature=T` |
| Measure a cluster graph without flipping | `cluster_action="observe"` or `overlap_cluster_action="observe"` |
| XY | defaults to Metropolis plus embedded SW each sweep; `cluster_update_interval=None` for local only, `overrelaxation_sweeps=n`, `sweep_mode="gibbs"`, `pt_interval` |
| Many disorder samples on many cores | `n_disorder >= threads` parallelizes over realizations and turns off inner parallelism automatically; set `RAYON_NUM_THREADS` to cap threads |

## Reading results

`sample()` returns a dict. Ising also sets attributes, and its `sample` docstring lists
the keys; for XY, the "Observable definitions" table of `docs/xy.md` lists every key.
Shapes: aggregates are `(n_temps[, n_dims])`, per-disorder arrays
`(n_disorder, n_temps[, n_dims])`, block sums `(n_disorder, n_blocks, n_temps[, ...])`.

| Quantity | Where |
|---|---|
| Energy, magnetization moments, Binder, heat capacity | Ising: `energies_avg`, `mags2`, `binder_cumulant`, `heat_capacity`. XY: `result["energies"]`, `result["binder_cumulant"]`, ... |
| Spin-glass overlaps | Ising with `n_replicas >= 2`: `overlap2`, `overlap4`, `sg_binder`, `link_overlap*`, `overlap_histogram` |
| Per-disorder values, correlation lengths, blocks | Ising `collect_physics=True`: `result["physics"]` with `per_disorder`, `correlation_length[_ratio]`, and with replicas `overlap2`, `overlap4`, `overlap_structure_factor_min` `(n_disorder, n_temps, n_dims)` and `sg_correlation_length[_ratio]`; add `block_size=` for blocked sums and `displacements=[[1, 0], ...]` for correlations. XY returns these at the top level of `result`, with per-sample copies in `result["per_disorder"]` (`collect_blocks=True` adds blocks of `block_size=128`) |
| XY helicity modulus | `result["helicity_modulus"]`, shape `(n_temps, n_dims)`, one value per direction; per sample in `result["per_disorder"]["helicity_modulus"]` |
| Parallel-tempering diagnostics | `result["per_disorder"]["parallel_tempering"]` for this call: `edge_attempts` and `edge_acceptances` `(n_disorder, n_temps - 1)`, summed over replica rows (`full_ladder` attempts every edge once per row per event), and `round_trips` `(n_disorder, n_replicas, n_temps)` per walker |
| Cluster statistics | `collect_cluster_stats=True`: `fk_csd[t]`, `overlap_csd[mode][t]` (for `cmr`, blue clusters), `top_cluster_sizes` |
| Cluster snapshots | `snapshot_interval=k` with an overlap move: `cluster_snapshots`, a list of dicts with `spins`, `cluster_ids`, `blue_ids` (CMR); plot with `validation/plot/cluster_snapshots.py` |
| Raw spins (internal) | `model._sim.get_spins()` returns the current state, sites in row-major order. Ising: flat `int8`, disorder realization 0 only, `n_replicas * n_temps` systems of `N` sites in system-id order (system `r * n_temps + t` starts at temperature `t`; tempering then permutes them, so use one temperature or no PT to read a known slot). XY: `(disorder, replica, temperature, site, 2)` with tempering resolved; the last axis is `(cos, sin)`, and vacancies still hold vectors, so mask them with `occupation` |

On embedded lattices (zeroed bonds), observables that assume the grid's geometry are
off: `equilibration_delta` uses `n_neighbors` bonds per site, and correlation lengths use
the grid's axes.

Heat capacity must use the thermal variance inside each realization (`energy_variance`
or `heat_capacity`), never `energies2 - energies**2` of disorder averages. Ising legacy
`energies` use the opposite sign to the physical `H/N` in `physics` and XY.

## Command line

| Command | Use |
|---|---|
| `peapods simulate --shape 16 16 --temp-min 1.5 --temp-max 3 --n-sweeps 5000 [flags] -o run.npz` | One run; flags mirror `sample()` (`--couplings`, `--geometry`, `--neighbor-offsets '[[1,0],[0,1]]'`, `--pt-interval`, `--overlap-cluster-build-mode`, ...) |
| `peapods sweep --config sweep_config.toml` | Grid over sizes, couplings and moves with plots; template in [`sweep_config.toml`](sweep_config.toml) |
| `peapods bench ...` | Timing; see the benchmark skill |

Run `peapods <command> --help` for every flag. The CLI covers Ising only.

## Before a long run

Use the analyze skill to choose sweeps and disorder counts: equilibrate with log
windows, check PT acceptance, and get error bars from disorder jackknife.
`validation/spin_glass_3d/run.py` is a complete batched, resumable production driver.
