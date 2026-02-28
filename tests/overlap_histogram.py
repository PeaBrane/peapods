"""Verify overlap histogram P(q) symmetry for a paramagnetic spin glass."""

import numpy as np

from peapods import Ising

N_SWEEPS = 40000


def overlap_histogram_symmetry():
    name = "3D EA spin glass P(q) at T=1.4"

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    model = Ising(
        (8, 8, 8),
        couplings="bimodal",
        temperatures=np.array([1.4], dtype=np.float32),
        n_replicas=2,
        n_disorder=64,
    )
    model.sample(
        N_SWEEPS,
        sweep_mode="metropolis",
        pt_interval=1,
        overlap_cluster_update_interval=1,
        warmup_ratio=0.25,
    )

    hist = model.overlap_histogram[0]
    mean_q = model.overlap[0]
    print(f"  |<q>|  = {abs(mean_q):.4f}")
    print(f"  counts = {hist.sum()}")

    assert abs(mean_q) < 0.1, f"|<q>| = {abs(mean_q):.4f} >= 0.1"
    print("  <q> ~ 0: PASSED")

    coarse = hist.reshape(20, 10).sum(axis=1).astype(float)
    left = coarse[:10]
    right = coarse[10:][::-1]
    norm = np.linalg.norm(coarse)
    sym_err = np.linalg.norm(left - right) / norm
    print(f"  coarse symmetry error = {sym_err:.4f}")

    assert sym_err < 0.25, f"coarse symmetry error {sym_err:.4f} >= 0.25"
    print("  P(q) symmetry: PASSED")


def run():
    overlap_histogram_symmetry()


if __name__ == "__main__":
    run()
