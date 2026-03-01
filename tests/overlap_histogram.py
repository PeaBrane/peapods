"""Verify overlap histogram P(q) symmetry and thermalization for a Gaussian spin glass."""

import numpy as np

from peapods import Ising
from peapods.sweep import _cumulative_overlap_ratio

N_SWEEPS = 40000
N_DISORDER = 64


def overlap_histogram_checks():
    name = "3D Gaussian spin glass at T=1.4"

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    model = Ising(
        (8, 8, 8),
        couplings="gaussian",
        temperatures=np.array([1.4], dtype=np.float32),
        n_replicas=2,
        n_disorder=N_DISORDER,
    )
    model.sample(
        N_SWEEPS,
        sweep_mode="metropolis",
        pt_interval=1,
        overlap_cluster_update_interval=1,
        warmup_ratio=0.25,
        equilibration_diagnostic=True,
    )

    # --- <q> ~ 0 ---
    mean_q = model.overlap[0]
    print(f"  |<q>|  = {abs(mean_q):.4f}")
    assert abs(mean_q) < 0.1, f"|<q>| = {abs(mean_q):.4f} >= 0.1"
    print("  <q> ~ 0: PASSED")

    # --- P(q) symmetry ---
    hist = model.overlap_histogram[0]
    print(f"  counts = {hist.sum()}")
    coarse = hist.reshape(20, 10).sum(axis=1).astype(float)
    left = coarse[:10]
    right = coarse[10:][::-1]
    norm = np.linalg.norm(coarse)
    sym_err = np.linalg.norm(left - right) / norm
    print(f"  coarse symmetry error = {sym_err:.4f}")
    assert sym_err < 0.25, f"coarse symmetry error {sym_err:.4f} >= 0.25"
    print("  P(q) symmetry: PASSED")

    # --- per-sample overlap histogram shape ---
    ps_hist = model.per_sample_overlap_histogram
    expected = (N_DISORDER, 1, 200)
    assert ps_hist.shape == expected, f"shape {ps_hist.shape} != {expected}"
    print(f"  per_sample_overlap_histogram shape: {ps_hist.shape}: PASSED")

    # --- thermalization diagnostic ---
    sweeps, delta = model.equilibration_delta(j_squared=1.0)
    final_delta = delta[-1, 0]
    print(f"  final Δ = {final_delta:.4f} (at sweep {sweeps[-1]})")
    assert abs(final_delta) < 0.15, f"|Δ| = {abs(final_delta):.4f} >= 0.15"
    print("  thermalization Δ ~ 0: PASSED")

    # --- I(q)/X(q) ~ 1 in paramagnetic phase ---
    q_grid, ratio, _, _ = _cumulative_overlap_ratio(ps_hist)
    mid = len(q_grid) // 2
    ratio_interior = ratio[0, 1:mid]
    max_dev = np.max(np.abs(ratio_interior - 1.0))
    print(f"  I(q)/X(q) max deviation from 1 = {max_dev:.4f} (interior q bins)")
    assert max_dev < 0.15, f"I(q)/X(q) max dev {max_dev:.4f} >= 0.15"
    print("  I(q)/X(q) ~ 1 (paramagnetic): PASSED")


def run():
    overlap_histogram_checks()


if __name__ == "__main__":
    run()
