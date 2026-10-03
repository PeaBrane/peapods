#!/usr/bin/env python
"""Render the README banner: a 2D XY model quenched from infinite temperature.

Random spins evolve at T = 0.1, well below the BKT temperature (T_BKT ~ 0.89),
under heat-bath sweeps plus two overrelaxation sweeps each. Vortex-antivortex
pairs, the points where every color meets, annihilate until one domain is left.
Color is the spin angle on matplotlib's cyclic "twilight" map. For display only,
each site shows the mean of its 3x3 neighborhood of unit vectors, which removes
single-site thermal jitter but keeps the vortex cores.

    python plot/xy_quench_banner.py --out docs/assets/xy_quench.webp
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from peapods import XY
from PIL import Image

HEIGHT, WIDTH, SCALE = 144, 432, 2


def smooth(vectors):
    return sum(
        np.roll(vectors, (dy, dx), axis=(0, 1))
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/assets/xy_quench.webp"))
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    cmap = matplotlib.colormaps["twilight"]
    lut = (cmap(np.linspace(0, 1, 512, endpoint=False))[:, :3] * 255).round()
    lut = lut.astype(np.uint8)
    model = XY((HEIGHT, WIDTH), temperatures=[0.1], seed=args.seed)
    # Log-spaced frames show the fast early coarsening and the last annihilations.
    times = np.unique(np.round(np.geomspace(1, 1200, 110)).astype(int))
    frames, done = [], 0
    for t in times:
        model.sample(
            int(t - done),
            sweep_mode="gibbs",
            overrelaxation_sweeps=2,
            cluster_update_interval=None,
            warmup_ratio=0.0,
        )
        done = t
        # The extension's raw state: (disorder, replica, temperature, site, 2).
        spins = model._sim.get_spins()[0, 0, 0].reshape(HEIGHT, WIDTH, 2)
        v = smooth(spins)
        angle = np.mod(np.arctan2(v[..., 1], v[..., 0]), 2 * np.pi)
        index = (angle / (2 * np.pi) * len(lut)).astype(int) % len(lut)
        image = lut[index].repeat(SCALE, axis=0).repeat(SCALE, axis=1)
        frames.append(Image.fromarray(image))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    held = frames + [frames[-1]] * 20
    held[0].save(
        args.out,
        save_all=True,
        append_images=held[1:],
        duration=55,
        loop=0,
        quality=75,
        method=6,
    )
    print(
        f"wrote {args.out} ({len(held)} frames, {args.out.stat().st_size / 1e6:.2f} MB)"
    )


if __name__ == "__main__":
    main()
