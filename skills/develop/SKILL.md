---
name: develop
description: Work on the peapods source tree - build the Rust extension, find where a feature lives, add an observable, update move or CLI option end to end, and run the same formatting, lint, test and physics checks as CI. Use when a user wants to modify, extend, debug or contribute to peapods, or asks why CI failed on their change.
---

# Developing peapods

## Layout

| Path | Contents |
|---|---|
| `spin-sim/` | Pure Rust crate (published as `spin-sim`): `geometry/` lattices and neighbor tables, `spins/` spin types and energies, `mcmc/` sweeps and parallel tempering, `clusters/` FK (SW, Wolff) and overlap moves (Houdayer, Jörg, CMR, replica Monte Carlo), `simulation/` sweep loop, schedule and XY driver, `statistics/` accumulators and the physics collector, `config.rs` |
| `src/` | PyO3 bindings: `lib.rs` (Ising), `xy.rs` (XY), `execution.rs` (shared conversion, physics dicts, PT counters) |
| `python/peapods/` | `spin_models.py` (`Ising`, `XY` wrappers and docstrings), `cli.py` (`peapods` command), `sweep.py` (parameter sweeps) |
| `tests/` | `test_*.py` (pytest) and physics scripts run by CI: `binder_crossings.py`, `spin_glass_crossings.py`, `overlap_histogram.py`, `autocorrelation_scaling.py`, `heat_capacity_consistency.py` |
| `validation/` | Comparisons with published results, plotting scripts; `xy_finite_size.py` runs in CI |
| `docs/` | mkdocs site (`mkdocs.yml`); API reference from Google-style docstrings |
| `skills/` | This skill pack; `.claude-plugin/` makes it a Claude Code plugin |

## Build

```sh
uv venv && uv pip install maturin numpy
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
# Dev tools: tests, README checks, hooks, and matplotlib for physics scripts and plots
uv pip install --python .venv/bin/python pytest pytest-codeblocks pre-commit matplotlib
```

Rebuild after any Rust change; Python-only edits need no rebuild (editable install).
Build artifacts go to `target/`, which can grow to several hundred MB.

## Checks (what CI runs)

| Check | Command |
|---|---|
| Format | Rust: `cargo fmt --check` (CI). Python: `pre-commit run --all-files` runs ruff format and check; CI does not run ruff, so run it locally |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` with the latest stable toolchain. CI does not pin Rust, so new lints appear when the runner's stable updates; an older local toolchain can pass where CI fails (`cargo +stable clippy ...`) |
| Rust tests | `cargo test --workspace --all-targets`, then `cargo test --workspace --doc` (`--all-targets` skips doctests) |
| Python tests | `.venv/bin/python -m pytest tests/` |
| README code | `pytest --codeblocks README.md` (needs `pytest-codeblocks`); mark blocks that must not run with `<!--pytest-codeblocks:skip-->`. CI also runs hand-copied `peapods simulate` commands from the README (`ci.yml`, "README CLI examples"); update them with the README |
| Links | lychee checks every `*.md` link in CI; new external URLs must resolve |
| Physics | `cd tests && python binder_crossings.py` (and the other scripts listed above), `python validation/xy_finite_size.py`; minutes each |

CI's pytest job installs no matplotlib: do not import `validation/plot` modules or
plotting helpers from `tests/test_*.py`.

## Changing behavior safely

- Performance or refactoring changes must keep the bench state checksum unchanged
  (benchmark skill). Changes to the random stream must be deliberate and said so.
- New measurements should be opt-in so the default `sample()` cost and outputs stay
  the same.
- Every overlap move must pass the exact one-step stationarity tests described in
  `docs/overlap_moves.md` ("Validation"); add the new mode there.

## Adding things end to end

| Change | Touch, in order |
|---|---|
| New observable | Rust accumulator (`statistics/physics.rs` for per-disorder physics with blocks, or `statistics/overlap.rs`), wiring in `simulation/mod.rs` (Ising) or `simulation/xy.rs`, binding output in `src/lib.rs` / `src/execution.rs` (vector-valued physics fields must be listed in `is_vector`), Python attribute and docstring, docs, and two tests: one against a direct definition, one against an existing estimator through the sweep loop |
| New update move | `clusters/` or `mcmc/`, config enum and parser in `config.rs`, schedule in `simulation/driver.rs` or the sweep loop, CLI flags in `cli.py` (`_add_common_args` serves `simulate` and `bench`; `sweep` has its own `_add_sweep_common_args` and `_add_sweep_args`, so add the flag to both), stationarity test, `docs/overlap_moves.md` |
| New sample() option | Rust `SimConfig` or physics options, binding signature, wrapper argument and docstring, CLI flags in both argument builders, README if user-facing |
| Expose an existing option in the CLI or saved output | First `rg -n <option> spin-sim/src/config.rs src/lib.rs python/peapods/` to see which layers already have it. `simulate -o` is written by `run_simulate` (`cli.py`), sweep data by `sweep._save_data`, and `validation/plot/*.py` read those key names: change all three together. Output-only flags go on the `simulate` subparser, not `_add_common_args`, because `bench` shares `sample_kwargs` |
| New lattice geometry | `GEOMETRIES` in `spin_models.py` (forward neighbor offsets) and the `--geometry` choices, which both CLI argument builders hard-code; non-hypercubic lattices use the generic neighbor table |

## Docs and examples

Keep README code blocks runnable and short. Results compared against the literature
belong in `validation/` with their command lines, reference values and the source
paper; figures go to `docs/assets/`.
