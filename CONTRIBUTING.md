# Contributing to peapods

## Dev environment setup

You need:
- **Rust toolchain** (stable) — install via [rustup](https://rustup.rs/)
- **Python 3.10+** with [uv](https://docs.astral.sh/uv/)
- **Maturin** for building the Rust extension

```bash
git clone https://github.com/PeaBrane/peapods.git
cd peapods
uv venv
uv pip install maturin numpy
```

## Building from source

```bash
VIRTUAL_ENV=.venv .venv/bin/maturin develop --release
```

This compiles the Rust core and installs the package into the local venv.

## Running tests

```bash
.venv/bin/python -m pytest tests/
```

## Running benchmarks

```bash
.venv/bin/python -m peapods bench --shape 32 32 \
    --temp-min 0.1 --temp-max 10 --temp-scale log --n-sweeps 1000
```

Or the full benchmark suite:

```bash
.venv/bin/python benchmarks/sweep_modes.py
```

## Code style

- **Python**: formatted and linted with [ruff](https://docs.astral.sh/ruff/)
- **Rust**: `cargo fmt` and `cargo clippy` in `peapods_core/`

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
