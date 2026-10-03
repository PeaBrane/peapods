"""Simulate the 3D ±J Edwards-Anderson spin glass for the T_c reproducer.

Each batch of disorder realizations runs parallel tempering (Metropolis sweeps
plus a full-ladder swap pass every sweep) with two replicas. Successive sample()
calls cover the log-spaced windows [0, 2^w), [2^w, 2^(w+1)), ..., [2^(K-1), 2^K)
of one continuous chain; each window's per-disorder overlap moments are saved,
so equilibration can be checked by comparing the last windows. Finished batches
are skipped on restart.

    python run.py --size 8 --n-disorder 3840 --log2-sweeps 16 --out data
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from peapods import Ising

TEMPERATURES = np.round(np.linspace(1.0, 1.6, 31), 4).astype(np.float32)
FIELDS = ("overlap2", "overlap4", "overlap_structure_factor_min", "energies")


def run_batch(size, n_disorder, seed, log2_sweeps, first_window):
    model = Ising(
        (size, size, size),
        couplings="bimodal",
        temperatures=TEMPERATURES,
        n_replicas=2,
        n_disorder=n_disorder,
        seed=seed,
    )
    lengths = [2**first_window] + [2**k for k in range(first_window, log2_sweeps)]
    windows = {name: [] for name in FIELDS}
    acceptance, round_trips, seconds = [], [], []
    for n_sweeps in lengths:
        start = time.perf_counter()
        result = model.sample(
            n_sweeps,
            pt_interval=1,
            pt_schedule="full_ladder",
            warmup_ratio=0.0,
            collect_physics=True,
        )
        seconds.append(time.perf_counter() - start)
        per_disorder = result["physics"]["per_disorder"]
        for name in FIELDS:
            windows[name].append(per_disorder[name])
        pt = result["per_disorder"]["parallel_tempering"]
        acceptance.append(pt["edge_acceptances"] / np.maximum(pt["edge_attempts"], 1))
        round_trips.append(pt["round_trips"].sum(axis=(1, 2)))
    return {
        **{name: np.stack(values) for name, values in windows.items()},
        "window_end": np.cumsum(lengths),
        "pt_acceptance": np.stack(acceptance),
        "pt_round_trips": np.stack(round_trips),
        "seconds": np.array(seconds),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--n-disorder", type=int, required=True)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--log2-sweeps", type=int, required=True)
    parser.add_argument("--first-window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out / f"L{args.size}"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "size": args.size,
        "temperatures": [round(float(t), 4) for t in TEMPERATURES],
        "log2_sweeps": args.log2_sweeps,
        "first_window": args.first_window,
        "couplings": "bimodal",
        "n_replicas": 2,
        "moves": "metropolis + full-ladder parallel tempering every sweep",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    n_batches = -(-args.n_disorder // args.batch)
    for b in range(n_batches):
        path = out / f"batch{b:04d}.npz"
        if path.exists():
            continue
        n = min(args.batch, args.n_disorder - b * args.batch)
        seed = args.seed + 100_000 * args.size + b
        start = time.perf_counter()
        data = run_batch(args.size, n, seed, args.log2_sweeps, args.first_window)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, seed=seed, **data)
        tmp.rename(path)
        print(
            f"L={args.size} batch {b + 1}/{n_batches} ({n} samples) "
            f"{time.perf_counter() - start:.0f}s, "
            f"min PT acceptance {data['pt_acceptance'][-1].min():.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
