"""Verify Binder cumulant crossings for spin glass models."""

from pathlib import Path

import numpy as np

from peapods import Ising
from utils import assert_crossing, plot_crossing

OUT_DIR = Path(__file__).parent
N_SWEEPS = 10000

TC_EA_3D = 1.102


def spin_glass_3d():
    name = "3D EA spin glass"
    temps = np.linspace(0.8, 1.4, 12).astype(np.float32)
    sizes = [8, 10]

    print(f"\n{'=' * 60}")
    print(f"  {name}  (T_c = {TC_EA_3D:.4f})")
    print(f"{'=' * 60}")

    results = {}
    for L in sizes:
        model = Ising(
            (L, L, L),
            couplings="bimodal",
            temperatures=temps,
            n_replicas=2,
            n_disorder=25,
        )
        model.sample(
            N_SWEEPS,
            sweep_mode="metropolis",
            pt_interval=1,
            houdayer_interval=1,
            warmup_ratio=0.25,
        )
        results[f"L={L}"] = model.sg_binder

    assert_crossing(temps, results, TC_EA_3D, tol=0.3)
    plot_crossing(
        temps,
        results,
        TC_EA_3D,
        ylabel="SG Binder ratio",
        title=f"{name} Binder crossing",
        out_path=OUT_DIR / "3d_ea_spin_glass.png",
    )


def spin_glass_3d_cmr_free():
    name = "3D EA spin glass (CMR free)"
    temps = np.linspace(0.8, 1.4, 12).astype(np.float32)
    sizes = [8, 10]

    print(f"\n{'=' * 60}")
    print(f"  {name}  (T_c = {TC_EA_3D:.4f})")
    print(f"{'=' * 60}")

    results = {}
    for L in sizes:
        model = Ising(
            (L, L, L),
            couplings="bimodal",
            temperatures=temps,
            n_replicas=2,
            n_disorder=25,
        )
        model.sample(
            N_SWEEPS,
            sweep_mode="metropolis",
            pt_interval=1,
            houdayer_interval=1,
            houdayer_mode="cmr",
            overlap_cluster_mode="sw",
            overlap_update_mode="free",
            warmup_ratio=0.25,
        )
        results[f"L={L}"] = model.sg_binder

    assert_crossing(temps, results, TC_EA_3D, tol=0.3)
    plot_crossing(
        temps,
        results,
        TC_EA_3D,
        ylabel="SG Binder ratio",
        title=f"{name} Binder crossing",
        out_path=OUT_DIR / "3d_ea_spin_glass_cmr_free.png",
    )


def run():
    spin_glass_3d()
    spin_glass_3d_cmr_free()


if __name__ == "__main__":
    run()
