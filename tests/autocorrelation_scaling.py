"""Verify SW autocorrelation time scaling τ ~ L^z with z ≈ 0.25 for 2D Ising."""

import numpy as np

from peapods import Ising
from utils import TC_SQUARE

N_SWEEPS = 40000
MAX_LAG = 2000
EXPECTED_RATIO = 2**0.25  # ≈ 1.189
TOLERANCE = 0.1


def run():
    print(f"\n{'=' * 60}")
    print("  SW autocorrelation scaling: τ(L=64) / τ(L=32) ≈ 2^0.25")
    print(f"{'=' * 60}")

    temps = np.array([TC_SQUARE], dtype=np.float32)
    taus = {}

    for L in [32, 64]:
        model = Ising([L, L], temperatures=temps, n_disorder=16)
        model.sample(
            N_SWEEPS,
            cluster_update_interval=1,
            cluster_mode="sw",
            autocorrelation_max_lag=MAX_LAG,
            warmup_ratio=0.25,
        )
        tau = model.mags2_tau[0]
        taus[L] = tau
        print(f"  L={L:3d}  τ_int(m²) = {tau:.3f}")

    ratio = taus[64] / taus[32]
    err = abs(ratio - EXPECTED_RATIO)
    print(
        f"  ratio: {ratio:.3f}  (expected {EXPECTED_RATIO:.3f}, err={err:.3f}, tol={TOLERANCE})"
    )
    assert err < TOLERANCE, (
        f"ratio {ratio:.3f} deviates from {EXPECTED_RATIO:.3f} by {err:.3f} >= {TOLERANCE}"
    )
    print("  PASSED")


if __name__ == "__main__":
    run()
