import time

import numpy as np

from spin_models import Ising


LATTICE_SHAPE = (32, 32)
N_TEMPS = 16
N_SWEEPS = 1000
N_WARMUP_SWEEPS = 100


def make_model():
    temperatures = np.geomspace(0.1, 10, N_TEMPS)
    return Ising(LATTICE_SHAPE, temperatures=temperatures)


def bench(name, run_fn):
    model = make_model()

    # warmup (triggers numba JIT compilation on first call)
    model.sample(N_WARMUP_SWEEPS, sweep_mode="metropolis", warmup_ratio=0.0)

    model.reset()
    t0 = time.perf_counter()
    run_fn(model)
    elapsed = time.perf_counter() - t0

    per_sweep = elapsed / N_SWEEPS * 1000
    print(f"  {name:<30s}  {elapsed:8.3f} s  ({per_sweep:.3f} ms/sweep)")


def run_benchmarks():
    shape_str = "x".join(str(s) for s in LATTICE_SHAPE)
    print(f"Lattice: {shape_str}  |  Temps: {N_TEMPS}  |  Sweeps: {N_SWEEPS}")
    print("-" * 64)

    bench(
        "metropolis",
        lambda m: m.sample(
            N_SWEEPS,
            sweep_mode="metropolis",
            warmup_ratio=0.0,
        ),
    )

    bench(
        "gibbs",
        lambda m: m.sample(
            N_SWEEPS,
            sweep_mode="gibbs",
            warmup_ratio=0.0,
        ),
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
            N_SWEEPS,
            sweep_mode="metropolis",
            pt_interval=1,
            warmup_ratio=0.0,
        ),
    )


if __name__ == "__main__":
    run_benchmarks()
