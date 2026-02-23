"""Reproduce Figure 1 (top panel) from Zhu et al. (2015).

Plots the fraction p of negative-overlap sites vs temperature T for 2D
Gaussian spin glasses. The horizontal line marks the square-lattice
bond percolation threshold p_c ≈ 0.5927.

Reference: Zhu, Ochoa, Katzgraber, arXiv:1501.05630
"""

import numpy as np
from matplotlib import pyplot as plt

from peapods import Ising

sizes = [16, 24, 32]
temperatures = np.linspace(0.212, 3.0, 30)

fig, ax = plt.subplots(figsize=(6, 4))
for L in sizes:
    ising = Ising(
        lattice_shape=(L, L),
        couplings="gaussian",
        temperatures=temperatures,
        n_replicas=2,
        n_disorder=200,
    )
    results = ising.sample(
        n_sweeps=2**16,
        warmup_ratio=0.25,
        houdayer_interval=1,
        houdayer_mode="houdayer",
        pt_interval=1,
    )
    p = (1 - results["overlap"]) / 2
    ax.plot(temperatures, p, "o-", ms=4, label=f"N = {L * L}")

ax.axhline(0.5927, color="k", ls="--", lw=0.8, label=r"$p_c \approx 0.593$")
ax.set_xlabel("T")
ax.set_ylabel("p")
ax.set_title("Fraction of negative-overlap sites (2D)")
ax.legend()
fig.tight_layout()
plt.show()
