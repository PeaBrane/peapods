import sys
import time

import numpy as np

sys.path.insert(0, "tests")

from peapods import Ising
from utils import plot_bars


LATTICE_SHAPE = (64, 64)
N_TEMPS = 16
N_SWEEPS = 50
N_REALIZATIONS = 128


def make_model():
    temperatures = np.geomspace(0.1, 10, N_TEMPS)
    return Ising(LATTICE_SHAPE, temperatures=temperatures, n_disorder=N_REALIZATIONS)


BENCH_RESULTS = {}


def bench(name, run_fn):
    model = make_model()

    t0 = time.perf_counter()
    run_fn(model)
    elapsed = time.perf_counter() - t0

    per_sweep = elapsed / N_SWEEPS * 1000
    BENCH_RESULTS[name] = per_sweep
    print(f"  {name:<30s}  {elapsed:8.3f} s  ({per_sweep:.3f} ms/sweep)")


def run_benchmarks():
    shape_str = "x".join(str(s) for s in LATTICE_SHAPE)
    print(
        f"Lattice: {shape_str}  |  Temps: {N_TEMPS}  |  Sweeps: {N_SWEEPS}  |  Realizations: {N_REALIZATIONS}"
    )
    print("-" * 64)

    bench(
        "metropolis",
        lambda m: m.sample(N_SWEEPS, sweep_mode="metropolis", warmup_ratio=0.0),
    )

    bench(
        "gibbs",
        lambda m: m.sample(N_SWEEPS, sweep_mode="gibbs", warmup_ratio=0.0),
    )

    bench(
        "metropolis + SW cluster",
        lambda m: m.sample(
            N_SWEEPS,
            sweep_mode="metropolis",
            cluster_update_interval=1,
            cluster_mode="sw",
            warmup_ratio=0.0,
        ),
    )

    bench(
        "metropolis + Wolff cluster",
        lambda m: m.sample(
            N_SWEEPS,
            sweep_mode="metropolis",
            cluster_update_interval=1,
            cluster_mode="wolff",
            warmup_ratio=0.0,
        ),
    )

    bench(
        "metropolis + PT",
        lambda m: m.sample(
            N_SWEEPS, sweep_mode="metropolis", pt_interval=1, warmup_ratio=0.0
        ),
    )


if __name__ == "__main__":
    run_benchmarks()
    plot_bars(
        list(BENCH_RESULTS.keys()),
        list(BENCH_RESULTS.values()),
        xlabel="ms / sweep",
        title=f"Benchmark ({LATTICE_SHAPE[0]}x{LATTICE_SHAPE[1]}, {N_TEMPS} temps, {N_SWEEPS} sweeps)",
        out_path="tests/benchmark.png",
    )
