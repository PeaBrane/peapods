from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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
