"""Bounded XY validation; generated files live only in an owned TemporaryDirectory.

Run with this worktree's .venv/bin/python after maturin develop --release.
Hasenbusch, arXiv:cond-mat/0502556v2, Table 1 and equations (4)-(10):
https://arxiv.org/pdf/cond-mat/0502556#page=13
The paper's susceptibility is our S(0); its stiffness is beta*Y. Both directional
observables below use axis zero, matching the paper, without pooling directions.
Low-T leading spin-wave stiffness: https://arxiv.org/abs/1811.08734
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

import numpy as np
from peapods import XY

BETA = 1.1199
# (dimensionless stiffness, xi/L, S(0)), each with its published standard error.
REFERENCES = {
    16: ((0.72536, 0.00007), (0.79953, 0.00017), (133.011, 0.009)),
    32: ((0.70883, 0.00007), (0.79231, 0.00018), (452.114, 0.031)),
}
NAMES = ("beta_Y_axis0", "xi_axis0_over_L", "S0")
PRODUCTION = 16384
WARMUP = 4096
CAP_SECONDS = 300


def derived(raw, extent):
    n = extent**2
    s0 = n * raw[..., 0]
    sk = raw[..., 1]
    stiffness = BETA * (raw[..., 2] - BETA * raw[..., 3]) / n
    length = np.sqrt(s0 / sk - 1) / (2 * np.sin(np.pi / extent) * extent)
    return np.stack((stiffness, length, s0), axis=-1)


def block_estimate(blocks, extent, factor):
    """Delete-one-block jackknife, preserving chain boundaries when reblocking."""
    sums = blocks["sums"]
    raw = np.stack(
        (
            sums["mags2"][..., 0],
            sums["structure_factor_min"][..., 0, 0],
            sums["helicity_d"][..., 0, 0],
            sums["helicity_i2"][..., 0, 0],
        ),
        axis=-1,
    )
    n_chains, n_blocks, _ = raw.shape
    assert n_blocks % factor == 0
    grouped = raw.reshape(n_chains, n_blocks // factor, factor, 4).sum(axis=2)
    counts = blocks["counts"].reshape(n_chains, n_blocks // factor, factor).sum(axis=2)
    assert np.all(counts == counts[0, 0])
    means = (grouped / counts[..., None]).reshape(-1, 4)
    total = means.sum(axis=0)
    estimate = derived(total / len(means), extent)
    jackknife = derived((total - means) / (len(means) - 1), extent)
    stderr = np.sqrt(
        (len(means) - 1)
        / len(means)
        * ((jackknife - jackknife.mean(axis=0)) ** 2).sum(axis=0)
    )
    return estimate, stderr


def compare(blocks, extent):
    estimate, se128 = block_estimate(blocks, extent, 1)
    _, se256 = block_estimate(blocks, extent, 2)
    reference = np.array(REFERENCES[extent])
    stderr = np.maximum(se128, se256)
    z = np.abs(estimate - reference[:, 0]) / np.hypot(stderr, reference[:, 1])
    relative_se = stderr / np.abs(estimate)
    stability = se256 / se128
    precision = bool(
        np.all(relative_se < 0.01) and np.all((stability > 0.7) & (stability < 1.3))
    )
    agreement = bool(np.all(z <= 4))
    return {
        "estimate": estimate.tolist(),
        "stderr": stderr.tolist(),
        "reference": reference[:, 0].tolist(),
        "combined_z": z.tolist(),
        "relative_stderr": relative_se.tolist(),
        "se256_over_se128": stability.tolist(),
        "precision_sufficient": precision,
        "status": "pass"
        if precision and agreement
        else "inconclusive"
        if not precision
        else "fail",
    }


def merge_blocks(first, second):
    return {
        "counts": np.concatenate((first["counts"], second["counts"]), axis=1),
        "sweeps": np.concatenate((first["sweeps"], second["sweeps"]), axis=1),
        "sums": {
            k: np.concatenate((first["sums"][k], second["sums"][k]), axis=1)
            for k in first["sums"]
        },
    }


def save_blocks(directory, extent, blocks):
    np.savez_compressed(
        directory / f"L{extent}-blocks.npz",
        counts=blocks["counts"],
        sweeps=blocks["sweeps"],
        **blocks["sums"],
    )


def worker(directory):
    start = time.monotonic()
    report = {
        "references": "Hasenbusch cond-mat/0502556v2 Table 1",
        "observables": NAMES,
        "beta": BETA,
        "threads": 4,
        "sizes": {},
    }

    def checkpoint():
        report["elapsed_seconds"] = time.monotonic() - start
        (directory / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)

    checkpoint()
    for extent in (16, 32):
        # Four independent identical-coupling realizations retain each chain's blocks.
        model = XY(
            (extent, extent),
            temperatures=[1 / BETA],
            n_disorder=4,
            seed=20260912 + extent,
        )
        result = model.sample(
            WARMUP + PRODUCTION,
            warmup_ratio=WARMUP / (WARMUP + PRODUCTION),
            collect_blocks=True,
            block_size=128,
            sequential=True,
        )
        blocks = result["per_disorder"]["blocks"]
        comparison = compare(blocks, extent)
        comparison["production_sweeps_per_chain"] = PRODUCTION
        save_blocks(directory, extent, blocks)
        report["sizes"][str(extent)] = comparison
        checkpoint()
        # One extension only, on the same chains; never retry with alternative seeds.
        elapsed = time.monotonic() - start
        if not comparison["precision_sufficient"] and elapsed < CAP_SECONDS * 0.55:
            extension = model.sample(
                PRODUCTION,
                warmup_ratio=0,
                collect_blocks=True,
                block_size=128,
                sequential=True,
            )
            blocks = merge_blocks(blocks, extension["per_disorder"]["blocks"])
            comparison = compare(blocks, extent)
            comparison["production_sweeps_per_chain"] = 2 * PRODUCTION
            report["sizes"][str(extent)] = comparison
            save_blocks(directory, extent, blocks)
            checkpoint()

    low_high = XY((16, 16), temperatures=[0.2, 2.0], n_disorder=4, seed=20261912)
    data = low_high.sample(9216, warmup_ratio=1 / 9, vortices=True, sequential=True)
    e, y, v, s0 = (
        data[key]
        for key in (
            "energies",
            "helicity_modulus",
            "angle_vortex_density",
            "structure_factor_0",
        )
    )
    # Leading spin waves are approximate at T=.2, not high-precision references.
    low_e = -2 + 0.2 / 2 * (1 - 1 / 256)
    low_y = 1 - 0.2 / 4
    trend_ok = bool(
        abs(e[0] - low_e) < 0.02
        and np.all(np.abs(y[0] - low_y) < 0.025)
        and v[1] > v[0] + 0.05
        and s0[1] < s0[0] / 5
    )
    report["low_high"] = {
        "status": "pass" if trend_ok else "fail",
        "temperatures": [0.2, 2.0],
        "energy": e.tolist(),
        "helicity": y.tolist(),
        "vortex_density": v.tolist(),
        "S0": s0.tolist(),
        "leading_low_T_energy": low_e,
        "leading_low_T_Y": low_y,
    }
    np.savez_compressed(
        directory / "temperature-smoke.npz", energies=e, helicity=y, vortex=v, S0=s0
    )
    cube = XY((4, 4, 4), couplings="bimodal", temperatures=[1.2], seed=20262012).sample(
        1024
    )
    mask = np.random.default_rng(20262112).random((2, 8, 8)) > 0.25
    dilute = XY(
        (8, 8),
        couplings="bimodal",
        temperatures=[0.8, 2.0],
        n_disorder=2,
        occupation=mask,
        seed=20262212,
    ).sample(2048, vortices=True, displacements=[[0, 0]])
    np.testing.assert_allclose(
        dilute["per_disorder"]["correlations"][:, :, 0],
        np.broadcast_to(mask.mean(axis=(1, 2))[:, None], (2, 2)),
        atol=1e-12,
    )
    assert np.all(np.isfinite(cube["energies"]))
    assert np.all(
        (dilute["intact_plaquette_fraction"] > 0)
        & (dilute["intact_plaquette_fraction"] < 1)
    )
    report["geometry_smokes"] = {
        "status": "pass",
        "cubic_energy": cube["energies"].tolist(),
        "diluted_energy": dilute["energies"].tolist(),
        "intact_fraction": dilute["intact_plaquette_fraction"].tolist(),
    }
    report["complete"] = True
    checkpoint()
    return (
        1
        if any(r["status"] == "fail" for r in report["sizes"].values()) or not trend_ok
        else 0
    )


def inspect_outputs(directory):
    """Inspect completed or partial artifacts before the cleanup boundary."""
    log = directory / "run.log"
    if log.exists():
        print(log.read_text())
    for artifact in sorted(directory.glob("*.npz")):
        try:
            with np.load(artifact) as arrays:
                print(
                    f"Inspected {artifact.name}: "
                    + ", ".join(f"{k}{arrays[k].shape}" for k in arrays.files)
                )
        except (OSError, ValueError, BadZipFile) as error:
            print(f"Incomplete artifact {artifact.name}: {error}")
    report_path = directory / "report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text())
    except (OSError, ValueError) as error:
        print(f"Incomplete report: {error}")
        return None
    print("Final report:", json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args.worker)
    environment = dict(os.environ, RAYON_NUM_THREADS="4")
    temporary_path = None
    returncode = 1
    try:
        with TemporaryDirectory(prefix="peapods-xy-validation-") as directory:
            temporary_path = Path(directory)
            print(f"Temporary validation directory: {directory}", flush=True)
            try:
                with (temporary_path / "run.log").open("w") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            directory,
                        ],
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        returncode = process.wait(timeout=CAP_SECONDS)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        print("INCONCLUSIVE: five-minute runtime cap reached.")
                        returncode = 2
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
            finally:
                report = inspect_outputs(temporary_path)
            if (
                returncode == 0
                and report is not None
                and any(r["status"] == "inconclusive" for r in report["sizes"].values())
            ):
                returncode = 2
    finally:
        if temporary_path is not None:
            assert not temporary_path.exists(), (
                f"temporary directory survived cleanup: {temporary_path}"
            )
            print(f"Verified removed: {temporary_path}", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
