# Project: Peapods

- Build Rust: `VIRTUAL_ENV=.venv .venv/bin/maturin develop --release`
- Paper PDFs and algorithm notes in `refs/`, own write-ups in `notes/` (both gitignored) — see `refs/README.md`
- When unsure about an algorithm or physics claim, check `refs/` and `notes/` first; only fall back to web search if they don't cover it

## Benchmark baseline (main, 2026-09-29; 64x64, 16 temps, 50 sweeps, 128 realizations; Intel Core Ultra 9 285K, 24 threads, Linux)

| Mode | ms/sweep | v0.2.1, same machine |
|------|----------|----------------------|
| metropolis | 1.47 | 3.49 |
| gibbs | 1.37 | 4.83 |
| metropolis + SW cluster | 7.63 | 14.82 |
| metropolis + Wolff cluster | 3.99 | 7.10 |
| metropolis + PT | 1.35 | 3.19 |

Median of 5 interleaved runs of `benchmarks/sweep_modes.py` (removed 2026-10-03; in git history).

Rust bench: `cargo run --release --example bench -p spin-sim` (configure with `PEAPODS_*` env vars, see the example). It prints a state checksum; behavior-neutral changes must leave it unchanged.

## Performance notes

- Bottleneck is cache pressure + dependent load chains, not compute — see `refs/cache-optimization.md`
- Sweep ordering options (checkerboard, typewriter, random, etc.) — see `refs/sweep-orderings.md`
- Measured without gain (2026-09-29): Z-order site layout (working set is cache-resident; the stride fast paths matter more), LTO/codegen-units=1, Rem's union-find (+7% on 3D CMR), pooled O(|C|) Wolff buffers (slower for large clusters)

## Local Overrides

- In chat responses, prefer inline math rendering for short expressions.
- If an expression is too long for inline math or hurts readability, place it on its own line instead.
