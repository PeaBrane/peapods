# Project: Peapods

- Build Rust: `VIRTUAL_ENV=.venv .venv/bin/maturin develop --release`
- Paper PDFs and algorithm notes in `refs/` (gitignored) — see `refs/README.md`

## Benchmark baseline (v0.1.4, 32x32, 16 temps, 1000 sweeps, Apple Silicon)

| Mode | ms/sweep |
|------|----------|
| metropolis | 0.082 |
| gibbs | 0.090 |
| metropolis + SW cluster | 0.212 |
| metropolis + Wolff cluster | 0.144 |
| metropolis + PT | 0.081 |

Run: `.venv/bin/python benchmarks/sweep_modes.py`

## Performance notes

- Bottleneck is cache pressure + dependent load chains, not compute — see `refs/cache-optimization.md`
- Sweep ordering options (checkerboard, typewriter, random, etc.) — see `refs/sweep-orderings.md`
