"""Verify the triangular-lattice Ising binder cumulant crossing at T_c = 4/ln(3)."""

from pathlib import Path

import numpy as np

from peapods import Ising
from utils import assert_crossing, plot_crossing

T_C = 4.0 / np.log(3)  # exact: 3.6410...
TEMPS = np.linspace(T_C - 0.4, T_C + 0.4, 32).astype(np.float32)
SIZES = [8, 16, 32]
N_SWEEPS = 10000
OFFSETS = [[1, 0], [0, 1], [1, -1]]


def run():
    results = {}
    for L in SIZES:
        model = Ising(
            (L, L),
            temperatures=TEMPS,
            n_replicas=2,
            neighbor_offsets=OFFSETS,
        )
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
        title="Triangular Binder cumulant crossing",
        out_path=Path(__file__).parent / "triangular_binder_crossing.png",
    )


if __name__ == "__main__":
    run()
