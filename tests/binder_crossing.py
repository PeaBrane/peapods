"""Verify the 2D Ising binder cumulant crossing at T_c ≈ 2.269."""

from pathlib import Path

import numpy as np

from peapods import Ising
from plot_utils import plot_crossing

T_C = 2.0 / np.log(1 + np.sqrt(2))  # exact: 2.26918...
TEMPS = np.linspace(T_C - 0.3, T_C + 0.3, 32).astype(np.float32)
SIZES = [8, 16, 32]
N_SWEEPS = 5000


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
        results[L] = model.binder_cumulant
        print(
            f"L={L:>2d}  binder at T_c: {np.interp(T_C, TEMPS, model.binder_cumulant):.4f}"
        )

    # check that all sizes cross near T_c with binder ≈ 0.61
    binders_at_tc = [np.interp(T_C, TEMPS, results[L]) for L in SIZES]
    spread = max(binders_at_tc) - min(binders_at_tc)
    mean_binder = np.mean(binders_at_tc)

    print(f"\nbinder values at T_c: {[f'{b:.4f}' for b in binders_at_tc]}")
    print(f"mean: {mean_binder:.4f}, spread: {spread:.4f}")
    print("expected: ~0.61, spread should be < 0.05")

    assert abs(mean_binder - 0.61) < 0.05, (
        f"mean binder {mean_binder:.4f} too far from 0.61"
    )
    assert spread < 0.05, f"spread {spread:.4f} too large, sizes not crossing"
    print("\nPASSED")

    plot_crossing(
        TEMPS,
        {f"L={L}": results[L] for L in SIZES},
        T_C,
        ylabel="Binder cumulant",
        title="Binder cumulant crossing",
        out_path=Path(__file__).parent / "binder_crossing.png",
    )


if __name__ == "__main__":
    run()
