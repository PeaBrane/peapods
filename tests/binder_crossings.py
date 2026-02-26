"""Verify Binder cumulant crossings for various lattice geometries."""

from pathlib import Path

import numpy as np

from peapods import Ising
from utils import assert_crossing, plot_crossing

OUT_DIR = Path(__file__).parent
N_SWEEPS = 10000

TC_SQUARE = 2.0 / np.log(1 + np.sqrt(2))  # exact: 2.26918...
TC_TRIANGULAR = 4.0 / np.log(3)  # exact: 3.64096...
TC_CUBIC = 4.511
TC_BCC = 6.235
TC_FCC = 9.792


def ferromagnet(name, sizes, tc, temps, tol=0.05, shape_fn=None, **ising_kwargs):
    print(f"\n{'=' * 60}")
    print(f"  {name}  (T_c = {tc:.4f})")
    print(f"{'=' * 60}")

    def _square(n):
        return (n, n)

    if shape_fn is None:
        shape_fn = _square

    results = {}
    for L in sizes:
        shape = shape_fn(L)
        model = Ising(shape, temperatures=temps, n_replicas=2, **ising_kwargs)
        model.sample(
            N_SWEEPS,
            sweep_mode="metropolis",
            cluster_update_interval=1,
            cluster_mode="sw",
            pt_interval=1,
            warmup_ratio=0.25,
        )
        results[f"L={L}"] = model.binder_cumulant

    assert_crossing(temps, results, tc, tol=tol)
    slug = name.lower().replace(" ", "_")
    plot_crossing(
        temps,
        results,
        tc,
        ylabel="Binder cumulant",
        title=f"{name} Binder crossing",
        out_path=OUT_DIR / f"{slug}.png",
    )


def run():
    ferromagnet(
        "2D square",
        sizes=[8, 16, 32],
        tc=TC_SQUARE,
        temps=np.linspace(TC_SQUARE - 0.3, TC_SQUARE + 0.3, 32).astype(np.float32),
    )

    ferromagnet(
        "2D triangular",
        sizes=[8, 16, 32],
        tc=TC_TRIANGULAR,
        temps=np.linspace(TC_TRIANGULAR - 0.4, TC_TRIANGULAR + 0.4, 32).astype(
            np.float32
        ),
        geometry="tri",
    )

    ferromagnet(
        "3D cubic",
        sizes=[6, 8, 10],
        tc=TC_CUBIC,
        temps=np.linspace(TC_CUBIC - 0.4, TC_CUBIC + 0.4, 24).astype(np.float32),
        shape_fn=lambda n: (n, n, n),
    )

    ferromagnet(
        "3D BCC",
        sizes=[6, 8, 10],
        tc=TC_BCC,
        temps=np.linspace(TC_BCC - 0.5, TC_BCC + 0.5, 24).astype(np.float32),
        geometry="bcc",
        shape_fn=lambda n: (n, n, n),
    )

    ferromagnet(
        "3D FCC",
        sizes=[6, 8, 10],
        tc=TC_FCC,
        temps=np.linspace(TC_FCC - 0.6, TC_FCC + 0.6, 24).astype(np.float32),
        geometry="fcc",
        shape_fn=lambda n: (n, n, n),
    )


if __name__ == "__main__":
    run()
