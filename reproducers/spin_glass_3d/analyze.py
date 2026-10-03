"""Reduce run.py output to xi/L, U4, equilibration checks and crossing temperatures.

Observables follow Baity-Jesi et al. (Janus), PRB 88, 224416 (2013), arXiv:1310.2910:
xi = sqrt(chi(0)/chi(k_min) - 1) / (2 sin(k_min/2)) with chi(0) = N [<q^2>] and
chi(k_min) averaged over the three axes, and U4 = [<q^4>] / [<q^2>]^2. Errors are
delete-one-block jackknife over disorder realizations. Results come from the last
log window, the second half of each chain.

    python analyze.py data --out results/summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

N_BLOCKS = 64
# (L, 2L) pairs for crossing temperatures.
PAIRS = ((4, 8), (6, 12), (8, 16))


def load(directory):
    meta = json.loads((directory / "meta.json").read_text())
    batches = [np.load(p) for p in sorted(directory.glob("batch*.npz"))]
    data = {
        name: np.concatenate([b[name] for b in batches], axis=1)
        for name in ("overlap2", "overlap4", "overlap_structure_factor_min")
    }
    data["window_end"] = batches[0]["window_end"]
    data["pt_acceptance"] = np.concatenate(
        [b["pt_acceptance"] for b in batches], axis=1
    )
    data["pt_round_trips"] = np.concatenate(
        [b["pt_round_trips"] for b in batches], axis=1
    )
    data["wall_seconds"] = sum(float(b["seconds"].sum()) for b in batches)
    return meta, data


def estimators(size, q2, q4, sk):
    """Disorder-averaged moments -> (xi/L, U4), vectorized over leading axes."""
    n = size**3
    xi = np.sqrt(np.maximum(n * q2 / sk - 1, 0)) / (2 * np.sin(np.pi / size))
    return xi / size, q4 / q2**2


def block_means(values, n_blocks):
    """Means over all realizations and with each contiguous block deleted."""
    blocks = np.array_split(values, n_blocks, axis=0)
    sums = np.stack([b.sum(axis=0) for b in blocks])
    counts = np.array([len(b) for b in blocks]).reshape(-1, *[1] * (values.ndim - 1))
    total, count = sums.sum(axis=0), counts.sum()
    return total / count, (total - sums) / (count - counts)


def jackknife(full, deleted):
    n = len(deleted)
    return full, np.sqrt(
        (n - 1) / n * ((deleted - deleted.mean(axis=0)) ** 2).sum(axis=0)
    )


def window_observables(size, data, window, n_blocks=N_BLOCKS):
    """(xi/L, U4) full estimates and per-deleted-block values for one window."""
    q2 = data["overlap2"][window]
    q4 = data["overlap4"][window]
    sk = data["overlap_structure_factor_min"][window].mean(axis=-1)
    (q2f, q2d), (q4f, q4d), (skf, skd) = (
        block_means(v, n_blocks) for v in (q2, q4, sk)
    )
    return estimators(size, q2f, q4f, skf), estimators(size, q2d, q4d, skd)


def crossing(temps, a, b):
    """Highest T where b - a changes sign, by linear interpolation on the grid."""
    diff = b - a
    idx = np.nonzero(np.diff(np.sign(diff)) != 0)[0]
    if len(idx) == 0:
        return np.nan
    i = idx[-1]
    t0, t1, d0, d1 = temps[i], temps[i + 1], diff[i], diff[i + 1]
    return t0 - d0 * (t1 - t0) / (d1 - d0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sizes, temps, runs = [], None, {}
    for directory in sorted(args.data.glob("L*"), key=lambda p: int(p.name[1:])):
        meta, data = load(directory)
        size = meta["size"]
        sizes.append(size)
        temps = np.round(meta["temperatures"], 4)
        runs[size] = (meta, data)

    summary = {
        "reference": {
            "source": "Baity-Jesi et al. (Janus), PRB 88, 224416 (2013), arXiv:1310.2910",
            "T_c": [1.1019, 0.0029],
            "xi_over_L_at_T_c": [0.6516, 0.0032],
            "U4_at_T_c": [1.4899, 0.0028],
        },
        "temperatures": temps.tolist(),
        "sizes": {},
        "crossings": [],
    }
    deleted = {}
    for size in sizes:
        meta, data = runs[size]
        (xi, u4), (xi_d, u4_d) = window_observables(size, data, -1)
        deleted[size] = (xi_d, u4_d)
        # Windows share realizations, so compare each with the last one pairwise.
        equilibration = []
        for w in range(max(0, len(data["window_end"]) - 5), len(data["window_end"])):
            (xw, uw), (xwd, uwd) = window_observables(size, data, w)
            equilibration.append(
                {
                    "window": [
                        int(data["window_end"][w] // 2),
                        int(data["window_end"][w]),
                    ],
                    "xi_over_L_at_T_min": jackknife(xw[0], xwd[:, 0]),
                    "xi_over_L_minus_last": jackknife(
                        xw[0] - xi[0], xwd[:, 0] - xi_d[:, 0]
                    ),
                    "U4_at_T_min": jackknife(uw[0], uwd[:, 0]),
                    "U4_minus_last": jackknife(uw[0] - u4[0], uwd[:, 0] - u4_d[:, 0]),
                }
            )
        acceptance = data["pt_acceptance"][-1]
        summary["sizes"][str(size)] = {
            "n_samples": int(data["overlap2"].shape[1]),
            "sweeps": int(data["window_end"][-1]),
            "wall_hours": data["wall_seconds"] / 3600,
            "xi_over_L": np.stack(jackknife(xi, xi_d)).T.tolist(),
            "U4": np.stack(jackknife(u4, u4_d)).T.tolist(),
            "equilibration": [
                {
                    k: (list(map(float, v)) if isinstance(v, tuple) else v)
                    for k, v in e.items()
                }
                for e in equilibration
            ],
            "pt_min_edge_acceptance": float(acceptance.mean(axis=0).min()),
            "pt_round_trips_per_sample": float(data["pt_round_trips"][-1].mean()),
        }
    for small, large in PAIRS:
        if small not in deleted or large not in deleted:
            continue
        a, b = runs[small][1], runs[large][1]
        (xa, ua), _ = window_observables(small, a, -1)
        (xb, ub), _ = window_observables(large, b, -1)
        n_blocks = N_BLOCKS
        xad, uad = deleted[small]
        xbd, ubd = deleted[large]
        # Independent samples: delete block j from both sizes at once.
        t_xi = jackknife(
            crossing(temps, xa, xb),
            np.array([crossing(temps, xad[j], xbd[j]) for j in range(n_blocks)]),
        )
        t_u4 = jackknife(
            crossing(temps, -ua, -ub),
            np.array([crossing(temps, -uad[j], -ubd[j]) for j in range(n_blocks)]),
        )
        r_at = float(np.interp(t_xi[0], temps, (xa + xb) / 2))
        summary["crossings"].append(
            {
                "sizes": [small, large],
                "T_xi_over_L": list(map(float, t_xi)),
                "xi_over_L_at_crossing": r_at,
                "T_U4": list(map(float, t_u4)),
            }
        )

    # Leading corrections: T*(L, 2L) - T_c ~ L^-(omega + 1/nu), exponents from Janus.
    # The pairs share no sizes, so their errors are independent.
    exponent = 1.12 + 1 / 2.562
    rows = [c for c in summary["crossings"] if np.isfinite(c["T_xi_over_L"][0])]
    if len(rows) >= 2:
        x = np.array([c["sizes"][0] ** -exponent for c in rows])
        y, err = np.array([c["T_xi_over_L"] for c in rows]).T
        design = np.stack([np.ones_like(x), x], axis=1) / err[:, None]
        coef, *_ = np.linalg.lstsq(design, y / err, rcond=None)
        cov = np.linalg.inv(design.T @ design)
        chi2 = float((((design @ coef) - y / err) ** 2).sum())
        summary["extrapolation"] = {
            "model": "T*(L, 2L) = T_c + a L^-(omega + 1/nu), weighted least squares",
            "exponent": exponent,
            "T_c": [float(coef[0]), float(np.sqrt(cov[0, 0]))],
            "slope": [float(coef[1]), float(np.sqrt(cov[1, 1]))],
            "chi2": chi2,
            "dof": len(rows) - 2,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    for size, s in summary["sizes"].items():
        print(
            f"L={size}: {s['n_samples']} samples, 2^{int(np.log2(s['sweeps']))} sweeps, "
            f"{s['wall_hours']:.2f} h wall, min PT acc {s['pt_min_edge_acceptance']:.2f}"
        )
        for e in s["equilibration"]:
            dx, du = e["xi_over_L_minus_last"], e["U4_minus_last"]
            print(
                f"   window {e['window']}: xi/L(T_min) = {e['xi_over_L_at_T_min'][0]:.4f}, "
                f"minus last {dx[0]:+.4f} ± {dx[1]:.4f}; U4 minus last {du[0]:+.4f} ± {du[1]:.4f}"
            )
    for c in summary["crossings"]:
        print(
            f"crossing L={c['sizes']}: T(xi/L) = {c['T_xi_over_L'][0]:.4f} ± "
            f"{c['T_xi_over_L'][1]:.4f} (xi/L = {c['xi_over_L_at_crossing']:.3f}), "
            f"T(U4) = {c['T_U4'][0]:.4f} ± {c['T_U4'][1]:.4f}"
        )
    if "extrapolation" in summary:
        e = summary["extrapolation"]
        print(
            f"extrapolated T_c = {e['T_c'][0]:.4f} ± {e['T_c'][1]:.4f} "
            f"(chi2 = {e['chi2']:.2f}, dof = {e['dof']})"
        )


if __name__ == "__main__":
    main()
