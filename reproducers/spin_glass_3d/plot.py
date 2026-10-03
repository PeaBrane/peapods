"""Plot xi/L crossings and crossing temperatures from analyze.py's summary.

Writes a light and a dark variant for GitHub's two themes:

    python plot.py summary.json --out ../../docs/assets/spin_glass_3d_tc
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CURVE_SIZES = (4, 6, 8, 10, 12)
# Ordinal blue ramps (light to dark by L), validated per surface.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "sizes": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "sizes": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
        "series": ["#3987e5", "#d95926"],
    },
}


def style(ax, theme):
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelcolor=theme["secondary"], labelsize=9)
    ax.xaxis.label.set_color(theme["secondary"])
    ax.yaxis.label.set_color(theme["secondary"])


def reference_band(ax, tc, err, theme, vertical=True):
    span = ax.axvspan if vertical else ax.axhspan
    line = ax.axvline if vertical else ax.axhline
    span(tc - err, tc + err, color=theme["muted"], alpha=0.18, linewidth=0)
    line(tc, color=theme["muted"], linewidth=1)


def draw(summary, theme_name, path):
    theme = THEMES[theme_name]
    temps = np.array(summary["temperatures"])
    tc, tc_err = summary["reference"]["T_c"]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    fig.patch.set_facecolor(theme["surface"])

    reference_band(left, tc, tc_err, theme)
    for size, color in zip(CURVE_SIZES, theme["sizes"]):
        if str(size) not in summary["sizes"]:
            continue
        values = np.array(summary["sizes"][str(size)]["xi_over_L"])
        left.fill_between(
            temps,
            values[:, 0] - values[:, 1],
            values[:, 0] + values[:, 1],
            color=color,
            alpha=0.25,
            linewidth=0,
        )
        left.plot(temps, values[:, 0], color=color, linewidth=2, label=f"L = {size}")
    style(left, theme)
    left.set_xlim(temps[0], temps[-1])
    left.set_xlabel("Temperature T")
    left.set_ylabel("Spin-glass correlation length  ξ / L")
    legend = left.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(theme["secondary"])
    left.text(
        tc + 0.008,
        0.03,
        "Janus $T_c$",
        transform=left.get_xaxis_transform(),
        color=theme["muted"],
        fontsize=8,
    )

    reference_band(right, tc, tc_err, theme, vertical=False)
    fit = summary.get("extrapolation")
    exponent = fit["exponent"] if fit else 1.51
    small = np.array([c["sizes"][0] for c in summary["crossings"]])
    x = small.astype(float) ** -exponent
    values = np.array([c["T_xi_over_L"] for c in summary["crossings"]])
    x_max = 1.15 * x.max()
    if fit:
        grid = np.linspace(0, x_max, 100)
        line = fit["T_c"][0] + fit["slope"][0] * grid
        right.plot(grid, line, color=theme["series"][0], linewidth=1, alpha=0.6)
        right.errorbar(
            [0],
            [fit["T_c"][0]],
            yerr=[fit["T_c"][1]],
            color=theme["series"][0],
            marker="D",
            markersize=7,
            markerfacecolor=theme["surface"],
            markeredgewidth=2,
            elinewidth=1.5,
            linewidth=0,
            clip_on=False,
            zorder=5,
            label=f"fit: $T_c$ = {fit['T_c'][0]:.3f}({round(fit['T_c'][1] * 1000):d})",
        )
    right.errorbar(
        x,
        values[:, 0],
        yerr=values[:, 1],
        color=theme["series"][0],
        marker="o",
        markersize=8,
        markeredgecolor=theme["surface"],
        markeredgewidth=2,
        linewidth=0,
        elinewidth=1.5,
        label="ξ/L crossing of L and 2L",
    )
    style(right, theme)
    right.set_xlim(0, x_max)
    right.set_xlabel(f"$L^{{-(\\omega + 1/\\nu)}}$,  ω + 1/ν = {exponent:.2f}")
    right.set_ylabel("Crossing temperature")
    top = right.secondary_xaxis("top")
    top.set_xticks(x, [f"L = {s}" for s in small])
    top.tick_params(colors=theme["muted"], labelcolor=theme["secondary"], labelsize=8)
    top.spines["top"].set_visible(False)
    right.text(
        0.97 * x_max,
        tc - tc_err,
        "Janus $T_c$ = 1.1019(29)",
        color=theme["muted"],
        fontsize=8,
        ha="right",
        va="top",
    )
    legend = right.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=theme["surface"])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="path prefix")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        suffix = "" if theme == "light" else "_dark"
        draw(summary, theme, args.out.with_name(args.out.name + suffix + ".png"))


if __name__ == "__main__":
    main()
