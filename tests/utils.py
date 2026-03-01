from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from peapods import Ising

TC_SQUARE = 2.0 / np.log(1 + np.sqrt(2))  # exact: 2.26918...
TC_TRIANGULAR = 4.0 / np.log(3)  # exact: 3.64096...
TC_CUBIC = 4.511
TC_BCC = 6.235
TC_FCC = 9.792
TC_EA_3D = 1.102


def assert_overlap_binder(model: Ising, tol=0.05):
    """Assert SG Binder from histogram matches direct estimate."""
    if not hasattr(model, "sg_binder"):
        return
    N = np.prod(model.lattice_shape)
    n_bins = N + 1
    q_values = np.linspace(-1, 1, n_bins)
    for t in range(model.n_temps):
        hist = model.overlap_histogram[t].astype(np.float64)
        total = hist.sum()
        if total == 0:
            continue
        p = hist / total
        q2_hist = (q_values**2 * p).sum()
        q4_hist = (q_values**4 * p).sum()
        binder_hist = 1 - q4_hist / (3 * q2_hist**2)
        binder_direct = model.sg_binder[t]
        err = abs(binder_hist - binder_direct)
        assert err < tol, (
            f"T[{t}]: histogram Binder {binder_hist:.6f} vs direct {binder_direct:.6f}, "
            f"error {err:.6f} >= {tol}"
        )


def assert_crossing(temps, results, tc, tol=0.05):
    """Assert that Binder curves cross at T_c with spread < tol."""
    binders = [np.interp(tc, temps, curve) for curve in results.values()]
    spread = max(binders) - min(binders)
    for label, b in zip(results.keys(), binders):
        print(f"  {label}  binder at T_c: {b:.4f}")
    print(f"  spread: {spread:.4f} (tol={tol})")
    assert spread < tol, f"spread {spread:.4f} >= {tol}, sizes not crossing"
    print("  PASSED")


def plot_crossing(temps, results, tc, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in results.items():
        ax.plot(temps, values, label=label)
    ax.axvline(tc, color="k", linestyle="--", alpha=0.5, label=f"$T_c$ = {tc:.4f}")
    ax.set_xlabel("Temperature")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title)

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot to {out}")


def plot_bars(names, values, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.5))
    ax.barh(names, values)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.3f}", va="center")

    out = Path(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot to {out}")
