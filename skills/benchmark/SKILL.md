---
name: benchmark
description: Measure and compare peapods performance - time sweeps from the CLI or Rust bench examples, verify behavior-neutral changes with the state checksum, compare two commits fairly, and pick fast settings for a production run. Use when a user asks how fast peapods is, whether a change slowed it down or sped it up, which update moves or thread settings to use for throughput, or how to profile a hot path.
---

# Benchmarking peapods

## Entry points

| Need | Command | Output |
|---|---|---|
| Time a configuration from Python's build | `peapods bench --shape 64 64 --temp-min 0.1 --temp-max 10 --n-sweeps 1000 [simulate flags]` | Total seconds and ms/sweep; takes the `peapods simulate` sampling flags (not `-o` or `--warmup-ratio`: every sweep is timed) |
| Time the Ising kernels without Python | `cargo run --release --example bench -p spin-sim` | ms/sweep and `State checksum: <hex>` |
| Time the XY kernels | `cargo run --release --example bench_xy -p spin-sim` | ms/sweep; options in the file header |
| Compare spin-glass moves | `peapods bench --shape 16 16 --couplings bimodal --n-replicas 4 --n-disorder 8 --n-temps 16 --temp-min 0.3 --temp-max 1.5 --n-sweeps 1000 --pt-interval 1 --pt-schedule full_ladder [--overlap-cluster-build-mode rmc --overlap-cluster-update-interval 1]` | Defaults are ferro couplings, one replica, one disorder sample, 32 temperatures and no PT; the output does not echo them |
| Throughput of a real workload | Time `sample()` calls in a script; `validation/spin_glass_3d/run.py` records seconds per window | spin updates/s = N × replicas × temperatures × sweeps × samples / s |

`bench.rs` is configured by environment variables (defaults in parentheses):
`PEAPODS_L` (128), `PEAPODS_DIM` (2), `PEAPODS_TEMPS` (16), `PEAPODS_REPLICAS` (2),
`PEAPODS_SWEEPS` (50), `PEAPODS_NREAL` (100), `PEAPODS_TMIN`/`PEAPODS_TMAX` (0.1/5.0,
geometric), `PEAPODS_COUPLINGS` (bimodal), `PEAPODS_SWEEP` (metropolis | gibbs),
`PEAPODS_MODE` (cmr; also metropolis, pt, sw, wolff, sw_pt, houdayer, jorg, or any
overlap build mode string, which runs with PT every sweep), flags (set to any value) `PEAPODS_OVERLAP_WOLFF`, `PEAPODS_SEQUENTIAL`,
`PEAPODS_GENERIC_LATTICE` (forces the generic neighbor table instead of the hypercubic
fast path). Read the top of `spin-sim/examples/bench.rs` before relying on a default.

## Behavior-neutral changes

A refactor or optimization that should not change results must leave the bench state
checksum unchanged for the modes it touches. Run the same `PEAPODS_*` settings on the
old and new commit and compare the printed checksum; a mismatch means the random stream
or the dynamics changed. New observables measured only on request (for example
`collect_physics`) must not change it either.

## Fair comparisons

- Fix `RAYON_NUM_THREADS`, the machine, the build profile and the inputs; close other
  heavy processes.
- Interleave runs (old, new, old, new, ...) and report the median of at least seven.
  Single runs vary by several percent on a quiet Linux machine and up to 2× on a busy
  laptop.
- Record the git SHA and the extension's build time (`_core*.so` modification time) with
  every result: a `maturin develop` during a measurement silently mixes builds.
- To compare two commits of the Python package, build a wheel per commit with
  `maturin build --release --out <dir>` and install each into its own venv;
  `maturin develop` installs editable from the source tree, so building the second
  commit replaces the first.
- The baseline table in `AGENTS.md` (64×64, 16 temperatures, 50 sweeps, 128
  realizations) is the reference point for regressions; refresh it on a quiet machine
  when kernels change.

## Choosing fast settings

- Parallelism is over disorder realizations and, inside one realization, over
  replicas and temperatures. With `n_disorder` at least the thread count, inner
  parallelism switches off automatically (`sequential=True` forces it).
- Throughput is bound by memory traffic, not arithmetic: larger lattices and more
  temperatures lower updates per second. Measure at the production size.
- Measurement costs time every recorded sweep; the opt-in collectors
  (`collect_physics`, `collect_cluster_stats`, autocorrelation) add more. Leave them off
  in throughput tests unless production uses them.
- For spin glasses, compare moves by cost × τ, the integrated autocorrelation time of
  Q2 (q² averaged over replica pairs), not by ms/sweep. Use one `Ising(..., n_disorder=1,
  seed=s)` per sample with the same seeds for every configuration, warm up, then
  `sample(n, warmup_ratio=0.0, collect_physics=True, block_size=1)`; Q2 per sweep is
  `physics["per_disorder"]["blocks"]["sums"]["overlap2"] / counts`. Estimate τ by batch
  means (τ = B·Var(batch means)/(2·Var) at two batch lengths B ≫ τ), report the median
  over seeds, and take the cost from `peapods bench` with the same flags.
  `docs/overlap_moves.md` explains the observable; its table is partly stale (see there).
- `block_size=1` stores about 5 KB per measured sweep (~190 MB for 40k sweeps): one
  model per process, or `del` it before the next.

## Profiling

Build with debug symbols in release (`CARGO_PROFILE_RELEASE_DEBUG=true`) and profile
`cargo run --release --example bench -p spin-sim` with a sampling profiler such as
`samply` or `perf`. Kernels live in `spin-sim/src/mcmc/` (sweeps, tempering),
`spin-sim/src/clusters/` (FK and overlap clusters) and `spin-sim/src/statistics/`
(measurement).
