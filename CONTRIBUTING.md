# Contributing to peapods

## Dev environment setup

You need:
- **Rust toolchain** (stable) — install via [rustup](https://rustup.rs/)
- **Python 3.11+** with [uv](https://docs.astral.sh/uv/)
- **Maturin** for building the Rust extension

```bash
git clone https://github.com/PeaBrane/peapods.git
cd peapods
uv venv
uv pip install maturin numpy pytest
```

## Building from source

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
```

This compiles the Rust core and installs the package into the local venv.

## Running tests

```bash
cargo test --workspace --all-targets
.venv/bin/python -m pytest tests/
```

CI also runs the physics checks in `tests/` (for example
`cd tests && ../.venv/bin/python binder_crossings.py`) and
`validation/xy_finite_size.py`; they take minutes each.

## Running benchmarks

```bash
.venv/bin/peapods bench --shape 32 32 \
    --temp-min 0.1 --temp-max 10 --temp-scale log --n-sweeps 1000
```

For the Rust kernels, the bench example prints a state checksum that
behavior-neutral changes must leave unchanged (configure it with `PEAPODS_*`
environment variables, see the example):

```bash
cargo run --release --example bench -p spin-sim
```

## Code style

- **Python**: formatted and linted with [ruff](https://docs.astral.sh/ruff/)
- **Rust**: `cargo fmt` and `cargo clippy --workspace --all-targets -- -D warnings`
  from the repository root, with the latest stable toolchain (CI does not pin one)
- `pre-commit run --all-files` runs both formatters and ruff's lints

## Pull requests

1. Fork the repo and create a feature branch
2. Make your changes
3. Run tests and lints
4. Open a PR against `main`

## Reporting issues

Open an issue on [GitHub](https://github.com/PeaBrane/peapods/issues) with:
- What you expected vs. what happened
- Minimal reproduction steps
- Python/Rust versions and OS
