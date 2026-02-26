"""Verify the 3D EA spin glass Binder ratio crossing near T_c ≈ 1.1."""

from pathlib import Path

import numpy as np

from peapods import Ising
from utils import assert_crossing, plot_crossing

# T_c ≈ 1.102 for 3D bimodal EA (Baity-Jesi et al. 2013)
T_C = 1.102
TEMPS = np.linspace(0.8, 1.4, 12).astype(np.float32)
SIZES = [8, 10]
N_SWEEPS = 10000
N_DISORDER = 25


def run():
    results = {}

    for L in SIZES:
        model = Ising(
            (L, L, L),
            couplings="bimodal",
            temperatures=TEMPS,
            n_replicas=2,
            n_disorder=N_DISORDER,
        )
        model.sample(
            N_SWEEPS,
            sweep_mode="metropolis",
            pt_interval=1,
            houdayer_interval=1,
            warmup_ratio=0.25,
        )
        results[f"L={L}"] = model.sg_binder

    assert_crossing(TEMPS, results, T_C, tol=0.3)

    plot_crossing(
        TEMPS,
        results,
        T_C,
        ylabel="SG Binder ratio",
        title="Spin glass Binder ratio crossing (3D EA)",
        out_path=Path(__file__).parent / "sg_binder_crossing.png",
    )


if __name__ == "__main__":
    run()
