"""Verify the 3D EA spin glass Binder ratio crossing near T_c ≈ 1.1."""

from pathlib import Path

import numpy as np

from peapods import Ising
from plot_utils import plot_crossing

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
        results[L] = model.sg_binder
        binder_at_tc = np.interp(T_C, TEMPS, results[L])
        print(f"L={L:>2d}  sg_binder at T_c: {binder_at_tc:.4f}")

    # Find approximate crossing temperature between smallest and largest size
    diff = results[SIZES[0]] - results[SIZES[-1]]
    cross_idx = np.argmin(np.abs(diff))
    t_cross = TEMPS[cross_idx]
    print(f"\napproximate crossing T: {t_cross:.3f} (expected ~{T_C})")
    assert abs(t_cross - T_C) < 0.3, (
        f"crossing at {t_cross:.3f}, too far from T_c={T_C}"
    )
    print("PASSED")

    plot_crossing(
        TEMPS,
        {f"L={L}": results[L] for L in SIZES},
        T_C,
        ylabel="SG Binder ratio",
        title="Spin glass Binder ratio crossing (3D EA)",
        out_path=Path(__file__).parent / "sg_binder_crossing.png",
    )


if __name__ == "__main__":
    run()
