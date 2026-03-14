#!/usr/bin/env python
"""Visualize cluster snapshots from peapods sweep .npz files.

Usage:
    python plot/cluster_snapshots.py results.npz
    python plot/cluster_snapshots.py results.npz -s 3 -t 5
    python plot/cluster_snapshots.py results.npz --all-temps
    python plot/cluster_snapshots.py results.npz --all-snaps -t 8
    python plot/cluster_snapshots.py results.npz -o snapshot.png
"""

import argparse
import sys

import numpy as np


def _find_prefix(keys):
    for key in keys:
        if key.endswith("_snapshot_sweep_ids"):
            return key[: -len("_snapshot_sweep_ids")]
    return None


def load_snapshots(path):
    data = np.load(path, allow_pickle=True)
    prefix = _find_prefix(data.files)
    if prefix is None:
        print("no snapshot data found in npz", file=sys.stderr)
        sys.exit(1)

    result = {
        "sweep_ids": data[f"{prefix}_snapshot_sweep_ids"],
        "mode_idxs": data[f"{prefix}_snapshot_mode_idxs"],
        "cluster_ids": data[f"{prefix}_snapshot_cluster_ids"],
        "spins": data[f"{prefix}_snapshot_spins"],
        "system_ids": data[f"{prefix}_snapshot_system_ids"],
        "shape": tuple(data[f"{prefix}_lattice_shape"]),
    }
    blue_key = f"{prefix}_snapshot_blue_ids"
    if blue_key in data.files:
        result["blue_ids"] = data[blue_key]
    if "temperatures" in data.files:
        result["temperatures"] = data["temperatures"]

    return result


MIN_CLUSTER_SIZE = 10
BLUE = np.array([0.2, 0.5, 1.0])
RED = np.array([0.9, 0.2, 0.2])
GREEN = np.array([0.2, 0.8, 0.3])


def cluster_image(snaps, snap_idx, temp_idx):
    shape = snaps["shape"]
    if len(shape) != 2:
        raise ValueError(f"only 2D lattices supported, got shape {shape}")

    n_spins = int(np.prod(shape))
    grey_ids = snaps["cluster_ids"][snap_idx, temp_idx]
    has_blue = "blue_ids" in snaps

    _, inverse, counts = np.unique(grey_ids, return_inverse=True, return_counts=True)
    in_grey = counts[inverse] >= MIN_CLUSTER_SIZE

    img = np.ones((n_spins, 3))

    if has_blue:
        blue_ids = snaps["blue_ids"][snap_idx, temp_idx]
        _, b_inv, b_counts = np.unique(
            blue_ids, return_inverse=True, return_counts=True
        )
        in_blue = b_counts[b_inv] >= MIN_CLUSTER_SIZE

        img[in_grey] = RED
        img[in_blue] = BLUE
    else:
        img[in_grey] = GREEN

    return img.reshape(*shape, 3)


def plot_single(snaps, snap_idx, temp_idx, ax):
    img = cluster_image(snaps, snap_idx, temp_idx)
    ax.imshow(img, interpolation="nearest", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])

    sweep_id = snaps["sweep_ids"][snap_idx]
    temps = snaps.get("temperatures")
    if temps is not None:
        ax.set_title(f"sweep {sweep_id}, T={temps[temp_idx]:.4f}", fontsize=9)
    else:
        ax.set_title(f"sweep {sweep_id}, t_idx={temp_idx}", fontsize=9)


def main():
    parser = argparse.ArgumentParser(description="Plot cluster snapshots")
    parser.add_argument("npz", help="Path to .npz file")
    parser.add_argument("-s", "--snap", type=int, default=-1)
    parser.add_argument("-t", "--temp", type=int, default=0)
    parser.add_argument("--all-temps", action="store_true")
    parser.add_argument("--all-snaps", action="store_true")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    snaps = load_snapshots(args.npz)
    n_snaps = len(snaps["sweep_ids"])
    n_temps = snaps["cluster_ids"].shape[1]
    mode = "CMR" if "blue_ids" in snaps else "overlap"
    args.snap = args.snap % n_snaps
    args.temp = args.temp % n_temps

    if args.all_temps:
        ncols = min(4, n_temps)
        nrows = (n_temps + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)
        for t in range(n_temps):
            plot_single(snaps, args.snap, t, axes[t // ncols, t % ncols])
        for t in range(n_temps, nrows * ncols):
            axes[t // ncols, t % ncols].axis("off")
        fig.suptitle(f"{mode} clusters — snapshot {args.snap}", fontsize=12)
    elif args.all_snaps:
        ncols = min(4, n_snaps)
        nrows = (n_snaps + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)
        for s in range(n_snaps):
            plot_single(snaps, s, args.temp, axes[s // ncols, s % ncols])
        for s in range(n_snaps, nrows * ncols):
            axes[s // ncols, s % ncols].axis("off")
        temps = snaps.get("temperatures")
        t_label = (
            f"T={temps[args.temp]:.4f}" if temps is not None else f"t_idx={args.temp}"
        )
        fig.suptitle(f"{mode} clusters — {t_label}", fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_single(snaps, args.snap, args.temp, ax)

    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
