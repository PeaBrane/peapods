"""Blue-cluster size distribution of the CMR move in the 2D ±J spin glass.

Collects the Chayes-Machta-Redner blue-cluster size distribution at several
temperatures and plots it on log-log axes. At low T the distribution develops a
heavy tail with system-spanning clusters (the "infinite blue clusters" of the
graphical representation). Reference: Pei and Di Ventra, arXiv:2105.01188.

    python validation/plot/overlap_csd.py --out overlap_csd.png
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from peapods import Ising

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--n-disorder", type=int, default=100)
    parser.add_argument("--n-sweeps", type=int, default=2**14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("overlap_csd.png"))
    args = parser.parse_args()

    temperatures = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    model = Ising(
        (args.size, args.size),
        couplings="bimodal",
        temperatures=temperatures,
        n_replicas=2,
        n_disorder=args.n_disorder,
        seed=args.seed,
    )
    results = model.sample(
        n_sweeps=args.n_sweeps,
        overlap_cluster_update_interval=1,
        overlap_cluster_build_mode="cmr",
        overlap_cluster_mode="wolff",
        pt_interval=1,
        collect_cluster_stats=True,
    )
    # overlap_csd[mode][t][s]: clusters of size s; one build mode here, and for
    # CMR the histogram counts blue clusters.
    csds = results["overlap_csd"][0]

    fig, ax = plt.subplots(figsize=(6, 4))
    for csd, temperature in zip(csds, temperatures):
        sizes = np.arange(len(csd))
        mask = csd > 0
        ax.scatter(
            sizes[mask],
            csd[mask] / csd[mask].sum(),
            s=8,
            label=f"T = {temperature:.1f}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cluster size s")
    ax.set_ylabel("P(s)")
    ax.set_title(f"CMR blue-cluster sizes ({args.size}×{args.size} ±J spin glass)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
