from pathlib import Path

import matplotlib.pyplot as plt


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
