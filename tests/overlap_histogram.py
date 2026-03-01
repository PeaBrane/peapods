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
    hist = model.overlap_histogram[0].astype(float)
    print(f"  counts = {hist.sum()}")
    norm = np.linalg.norm(hist)
    sym_err = np.linalg.norm(hist - hist[::-1]) / norm
    print(f"  symmetry error = {sym_err:.4f}")
    assert sym_err < 0.25, f"symmetry error {sym_err:.4f} >= 0.25"
    print("  P(q) symmetry: PASSED")

    # --- per-sample overlap histogram shape ---
    ps_hist = model.per_sample_overlap_histogram
    expected = (N_DISORDER, 1, 513)
    assert ps_hist.shape == expected, f"shape {ps_hist.shape} != {expected}"
    print(f"  per_sample_overlap_histogram shape: {ps_hist.shape}: PASSED")

    # --- thermalization diagnostic ---
    sweeps, delta = model.equilibration_delta(j_squared=1.0)
    final_delta = delta[-1, 0]
    print(f"  final Δ = {final_delta:.4f} (at sweep {sweeps[-1]})")
    assert abs(final_delta) < 0.15, f"|Δ| = {abs(final_delta):.4f} >= 0.15"
    print("  thermalization Δ ~ 0: PASSED")

    # --- A(q) = Var(q_l | q) small in paramagnetic phase ---
    ps_hist_f = ps_hist.astype(float)
    ps_s1 = model.per_sample_ql_at_q_sum
    ps_s2 = model.per_sample_ql2_at_q_sum
    assert ps_s1.shape == expected, (
        f"per_sample_ql_at_q_sum shape {ps_s1.shape} != {expected}"
    )
    print(f"  per_sample_ql_at_q_sum shape: {ps_s1.shape}: PASSED")

    mask = ps_hist_f > 0
    mean_ql = np.where(mask, ps_s1 / np.where(mask, ps_hist_f, 1), 0)
    a_s = np.where(mask, ps_s2 / np.where(mask, ps_hist_f, 1) - mean_ql**2, 0)
    # disorder average weighted by counts: A(q) = Σ_s n_s A_s / Σ_s n_s
    numer = (ps_hist_f * a_s).sum(axis=0)  # (n_temps, n_bins)
    denom = ps_hist_f.sum(axis=0)
    valid = denom > 0
    a_q = np.where(valid, numer / np.where(valid, denom, 1), 0)
    # weighted average over q bins for a single scalar per temperature
    a_mean = (a_q * denom).sum(axis=-1) / denom.sum(axis=-1)
    print(f"  A(q) weighted mean = {a_mean[0]:.6f}")
    assert a_mean[0] >= -1e-6, f"A(q) negative: {a_mean[0]:.6f}"
    assert a_mean[0] < 0.05, (
        f"A(q) = {a_mean[0]:.6f} >= 0.05 (should be small in paramagnetic)"
    )
    print("  A(q) ~ 0 (paramagnetic): PASSED")

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
