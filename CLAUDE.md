# Project: Peapods

- Build Rust: `VIRTUAL_ENV=.venv .venv/bin/maturin develop --release`
- Paper PDFs and algorithm notes in `refs/` (gitignored) — see `refs/README.md`
- When unsure about an algorithm or physics claim, check `refs/` first; only fall back to web search if refs don't cover it

## Benchmark baseline (v0.1.4, 64x64, 16 temps, 50 sweeps, 128 realizations, Apple Silicon)

| Mode | ms/sweep |
|------|----------|
| metropolis | 11.72 |
| gibbs | 13.14 |
| metropolis + SW cluster | 38.46 |
| metropolis + Wolff cluster | 21.54 |
| metropolis + PT | 12.20 |

Run: `.venv/bin/python benchmarks/sweep_modes.py`

## Performance notes

- Bottleneck is cache pressure + dependent load chains, not compute — see `refs/cache-optimization.md`
- Sweep ordering options (checkerboard, typewriter, random, etc.) — see `refs/sweep-orderings.md`
