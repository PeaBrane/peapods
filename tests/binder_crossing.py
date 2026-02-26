"""Verify the 2D Ising binder cumulant crossing at T_c ≈ 2.269."""

from pathlib import Path

import numpy as np

from peapods import Ising
from utils import assert_crossing, plot_crossing

T_C = 2.0 / np.log(1 + np.sqrt(2))  # exact: 2.26918...
TEMPS = np.linspace(T_C - 0.3, T_C + 0.3, 32).astype(np.float32)
SIZES = [8, 16, 32]
N_SWEEPS = 10000


def run():
    results = {}
    for L in SIZES:
        model = Ising((L, L), temperatures=TEMPS, n_replicas=2)
        model.sample(
            N_SWEEPS,
            sweep_mode="metropolis",
            cluster_update_interval=1,
            cluster_mode="sw",
            pt_interval=1,
            warmup_ratio=0.25,
        )
        results[f"L={L}"] = model.binder_cumulant

    assert_crossing(TEMPS, results, T_C)

    plot_crossing(
        TEMPS,
        results,
        T_C,
        ylabel="Binder cumulant",
        title="Binder cumulant crossing",
        out_path=Path(__file__).parent / "binder_crossing.png",
    )


if __name__ == "__main__":
    run()
