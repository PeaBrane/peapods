"""Blue-bond cluster size distribution for 2D ±J spin glass.

Collects CMR blue-bond CSD at several temperatures and plots the
distribution on a log-log scale. At low T, the distribution develops
a heavy tail with system-spanning clusters (the "infinite blue clusters"
of the graphical representation).

Reference: Pei, Di Ventra, arXiv:2105.01188
"""

import numpy as np
from matplotlib import pyplot as plt

from peapods import Ising

L = 64
temperatures = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

ising = Ising(
    lattice_shape=(L, L),
    couplings="bimodal",
    temperatures=temperatures,
    n_replicas=2,
    n_disorder=100,
)
results = ising.sample(
    n_sweeps=2**14,
    warmup_ratio=0.25,
    overlap_cluster_update_interval=1,
    overlap_cluster_build_mode="houdayer",
    overlap_cluster_mode="wolff",
    pt_interval=1,
    collect_csd=True,
)

fig, ax = plt.subplots(figsize=(6, 4))
for t, temp in enumerate(temperatures):
    csd = results["overlap_csd"][t]  # csd[s] = count of size-s clusters
    sizes = np.arange(len(csd))
    mask = csd > 0
    total = csd[mask].sum()
    ax.scatter(sizes[mask], csd[mask] / total, s=8, label=f"T = {temp:.1f}")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Cluster size s")
ax.set_ylabel("P(s)")
ax.set_title(f"Blue-bond CSD ({L}×{L} ±J spin glass)")
ax.legend()
fig.tight_layout()
fig.savefig("overlap_csd.png", dpi=150)
plt.show()
