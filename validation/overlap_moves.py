"""Measure the overlap-move comparison of docs/overlap_moves.md ("Which move to use").

Every task is one ±J disorder sample with 4 replicas, run with Metropolis sweeps plus a
full-ladder parallel-tempering pass every sweep, and optionally an overlap move. By
default a task first runs `--warmup` sweeps with measurement off, timed for the cost per
sweep, then `--measure` sweeps recording Q2 (q² averaged over every replica pair) each
sweep. τ_int of Q2 at the reported temperatures comes from batch means,
τ = B Var(batch means) / (2 Var), at several batch lengths B.

With `--equil-sweeps`, a task instead times a short separate run for the cost and then
records block means of the energy and Q2 from a random start, for the equilibration
table (systems 2d_L32, 3d_L12, 3d_L16).

    python validation/overlap_moves.py run --system 2d --config pt --seed 0
    python validation/overlap_moves.py all --jobs 96 --out tau.jsonl
    python validation/overlap_moves.py all --jobs 96 --out equil.jsonl \
        --systems 2d_L32 3d_L16 --configs pt gperm2_every4 --equil-sweeps 65536
    python validation/overlap_moves.py summarize tau.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from peapods import Ising

SYSTEMS = {
    "2d": ((16, 16), np.geomspace(0.3, 1.5, 16), (0.30, 0.40)),
    "3d_low": ((8, 8, 8), np.geomspace(0.5, 2.0, 16), (0.50,)),
    "3d": ((8, 8, 8), np.geomspace(0.8, 2.0, 16), (0.80, 0.90)),
    # Harder systems for equilibration from a random start (`--equil-sweeps`).
    "3d_L12": ((12, 12, 12), np.geomspace(0.4, 1.6, 24), (0.40, 0.60)),
    "2d_L32": ((32, 32), np.geomspace(0.2, 1.5, 24), (0.20, 0.30)),
    "3d_L16": ((16, 16, 16), np.geomspace(0.4, 1.6, 32), (0.40, 0.60)),
}
TAU_SYSTEMS = ("2d", "3d_low", "3d")
MOVE = dict(overlap_cluster_update_interval=1)
CONFIGS = {
    "pt": {},
    "houd2": {**MOVE, "overlap_cluster_build_mode": "houdayer"},
    "jorg": {**MOVE, "overlap_cluster_build_mode": "jorg"},
    "pair4": {**MOVE, "overlap_cluster_build_mode": "pair4"},
    "jorg4": {**MOVE, "overlap_cluster_build_mode": "jorg4"},
    "cmr": {**MOVE, "overlap_cluster_build_mode": "cmr"},
    "rmc": {**MOVE, "overlap_cluster_build_mode": "rmc"},
    "rmc_every4": {
        "overlap_cluster_update_interval": 4,
        "overlap_cluster_build_mode": "rmc",
    },
    "perm3": {**MOVE, "overlap_cluster_build_mode": "perm3"},
    "perm4": {**MOVE, "overlap_cluster_build_mode": "perm4"},
    "perm4_sw": {
        **MOVE,
        "overlap_cluster_build_mode": "perm4",
        "overlap_cluster_mode": "sw",
    },
    "gperm2": {**MOVE, "overlap_cluster_build_mode": "gperm2"},
    "gperm2_sw": {
        **MOVE,
        "overlap_cluster_build_mode": "gperm2",
        "overlap_cluster_mode": "sw",
    },
    "gperm4": {**MOVE, "overlap_cluster_build_mode": "gperm4"},
    "gperm4_sw": {
        **MOVE,
        "overlap_cluster_build_mode": "gperm4",
        "overlap_cluster_mode": "sw",
    },
    "houd2_every4": {
        "overlap_cluster_update_interval": 4,
        "overlap_cluster_build_mode": "houdayer",
    },
    "gperm2_every4": {
        "overlap_cluster_update_interval": 4,
        "overlap_cluster_build_mode": "gperm2",
    },
    "gperm4_every4": {
        "overlap_cluster_update_interval": 4,
        "overlap_cluster_build_mode": "gperm4",
    },
}
BATCHES = (250, 500, 1000, 2000)


def model_and_kwargs(system, config, seed):
    shape, temperatures, _ = SYSTEMS[system]
    model = Ising(
        shape,
        couplings="bimodal",
        temperatures=temperatures,
        n_replicas=4,
        seed=seed,
    )
    kwargs = dict(
        pt_interval=1,
        pt_schedule="full_ladder",
        **{"overlap_cluster_mode": "wolff", **CONFIGS[config]},
    )
    if "overlap_cluster_update_interval" not in kwargs:
        kwargs.pop("overlap_cluster_mode")
    return model, kwargs


def run_equil(system, config, seed, sweeps, block):
    """Block means of energy and Q2 from a random start, plus unmeasured cost per sweep."""
    _, temperatures, report = SYSTEMS[system]
    timer, kwargs = model_and_kwargs(system, config, seed + 10_000)
    timing_sweeps = 1024
    start = time.perf_counter()
    with np.errstate(invalid="ignore", divide="ignore"):
        timer.sample(timing_sweeps, warmup_ratio=1.0, **kwargs)
    seconds = time.perf_counter() - start
    del timer
    model, kwargs = model_and_kwargs(system, config, seed)
    result = model.sample(
        sweeps, warmup_ratio=0.0, collect_physics=True, block_size=block, **kwargs
    )
    blocks = result["physics"]["per_disorder"]["blocks"]
    counts = blocks["counts"][0][:, None]
    energy = blocks["sums"]["energies"][0] / counts
    q2 = blocks["sums"]["overlap2_all_pairs"][0] / counts
    series = {}
    for t in report:
        slot = int(np.argmin(np.abs(temperatures - t)))
        series[f"{temperatures[slot]:.3f}"] = {
            "energy": energy[:, slot].tolist(),
            "q2": q2[:, slot].tolist(),
        }
    return {
        "system": system,
        "config": config,
        "seed": seed,
        "block": block,
        "ms_per_sweep": 1000 * seconds / timing_sweeps,
        "series": series,
    }


def run_task(system, config, seed, warmup, measure):
    shape, temperatures, report = SYSTEMS[system]
    model, kwargs = model_and_kwargs(system, config, seed)
    start = time.perf_counter()
    # Warmup records nothing, so the wrapper's Binder ratios divide zero by zero.
    with np.errstate(invalid="ignore", divide="ignore"):
        model.sample(warmup, warmup_ratio=1.0, **kwargs)
    seconds = time.perf_counter() - start
    result = model.sample(
        measure, warmup_ratio=0.0, collect_physics=True, block_size=1, **kwargs
    )
    blocks = result["physics"]["per_disorder"]["blocks"]
    q2 = blocks["sums"]["overlap2_all_pairs"][0] / blocks["counts"][0][:, None]
    slots = [int(np.argmin(np.abs(temperatures - t))) for t in report]
    taus = {}
    for slot in slots:
        series = q2[:, slot]
        variance = series.var()
        per_batch = {}
        for length in BATCHES:
            n = len(series) // length
            means = series[: n * length].reshape(n, length).mean(axis=1)
            per_batch[str(length)] = float(length * means.var(ddof=1) / (2 * variance))
        taus[f"{temperatures[slot]:.3f}"] = {
            "q2": float(series.mean()),
            "tau": per_batch,
        }
    return {
        "system": system,
        "config": config,
        "seed": seed,
        "ms_per_sweep": 1000 * seconds / warmup,
        "temperatures": taus,
    }


def run_all(args):
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["system"], r["config"], r["seed"]))
    # Seed-major order keeps every configuration under similar machine load.
    tasks = [
        (system, config, seed)
        for seed in range(args.seeds)
        for system in args.systems
        for config in args.configs
        if (system, config, seed) not in done
    ]
    env = {**os.environ, "RAYON_NUM_THREADS": "1"}

    def launch(task):
        system, config, seed = task
        command = [
            sys.executable,
            __file__,
            "run",
            "--system",
            system,
            "--config",
            config,
            "--seed",
            str(seed),
            "--warmup",
            str(args.warmup),
            "--measure",
            str(args.measure),
            "--equil-sweeps",
            str(args.equil_sweeps),
            "--equil-block",
            str(args.equil_block),
        ]
        out = subprocess.run(command, capture_output=True, text=True, env=env)
        if out.returncode:
            return task, None, out.stderr[-2000:]
        return task, out.stdout.strip(), None

    with ThreadPoolExecutor(args.jobs) as pool, args.out.open("a") as sink:
        for i, (task, line, error) in enumerate(pool.map(launch, tasks), 1):
            if error:
                print(f"FAILED {task}: {error}", file=sys.stderr, flush=True)
                continue
            sink.write(line + "\n")
            sink.flush()
            if i % 50 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} tasks", flush=True)


def summarize_equil(rows):
    """Seed-averaged energy and Q2 at log-spaced checkpoints, per system and config."""
    out = {}
    for system in sorted({r["system"] for r in rows}):
        sel = [r for r in rows if r["system"] == system]
        block = sel[0]["block"]
        configs = [c for c in CONFIGS if any(r["config"] == c for r in sel)]
        cost_pt = np.median([r["ms_per_sweep"] for r in sel if r["config"] == "pt"])
        for temp in sel[0]["series"]:
            n_blocks = len(sel[0]["series"][temp]["energy"])
            checkpoints = [
                b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256) if b <= n_blocks
            ]
            key = f"{system} T={temp}"
            print(f"\n{key}: seed-mean energy/Q2 of the block ending at sweep")
            header = " ".join(f"{b * block:>13d}" for b in checkpoints)
            print(f"config          cost×PT (n)  {header}")
            for config in configs:
                rs = [r for r in sel if r["config"] == config]
                e = np.mean([r["series"][temp]["energy"] for r in rs], axis=0)
                q = np.mean([r["series"][temp]["q2"] for r in rs], axis=0)
                ratio = np.median([r["ms_per_sweep"] for r in rs]) / cost_pt
                cells = " ".join(f"{e[b - 1]:7.4f}/{q[b - 1]:.3f}" for b in checkpoints)
                print(f"{config:15s} {ratio:6.2f} ({len(rs):2d})  {cells}")
                out.setdefault(key, {})[config] = {
                    "samples": len(rs),
                    "cost_ratio_per_sweep": float(ratio),
                    "energy": e.tolist(),
                    "q2": q.tolist(),
                }
    return out


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if rows and "series" in rows[0]:
        return summarize_equil(rows)
    out = {}
    for system in TAU_SYSTEMS:
        cost_pt = np.median(
            [
                r["ms_per_sweep"]
                for r in rows
                if r["system"] == system and r["config"] == "pt"
            ]
        )
        temps = sorted(
            {t for r in rows if r["system"] == system for t in r["temperatures"]}
        )
        for temp in temps:
            tau_pt = np.median(
                [
                    r["temperatures"][temp]["tau"]["1000"]
                    for r in rows
                    if r["system"] == system and r["config"] == "pt"
                ]
            )
            for config in CONFIGS:
                sel = [
                    r for r in rows if r["system"] == system and r["config"] == config
                ]
                if not sel:
                    continue
                tau = {
                    b: float(
                        np.median([r["temperatures"][temp]["tau"][b] for r in sel])
                    )
                    for b in ("1000", "2000")
                }
                ms = float(np.median([r["ms_per_sweep"] for r in sel]))
                out.setdefault(f"{system} T={temp}", {})[config] = {
                    "samples": len(sel),
                    "ms_per_sweep": ms,
                    "cost_ratio_per_sweep": ms / cost_pt,
                    "tau_B1000": tau["1000"],
                    "tau_B2000": tau["2000"],
                    "cost_x_tau_vs_pt": ms * tau["1000"] / (cost_pt * tau_pt),
                    "q2": float(
                        np.median([r["temperatures"][temp]["q2"] for r in sel])
                    ),
                }
    for key, configs in out.items():
        print(f"\n{key}")
        print("config        n   ms/sweep(×PT)   tau B=1000  tau B=2000  cost×tau/PT")
        for config, v in configs.items():
            print(
                f"{config:12s} {v['samples']:3d}  {v['ms_per_sweep']:6.3f} "
                f"({v['cost_ratio_per_sweep']:4.2f})   {v['tau_B1000']:8.2f}    "
                f"{v['tau_B2000']:8.2f}    {v['cost_x_tau_vs_pt']:6.2f}"
            )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run", help="Run one task and print a JSON line")
    one.add_argument("--system", choices=SYSTEMS, required=True)
    one.add_argument("--config", choices=CONFIGS, required=True)
    one.add_argument("--seed", type=int, required=True)
    many = sub.add_parser("all", help="Run every task in parallel processes")
    many.add_argument("--jobs", type=int, default=os.cpu_count())
    many.add_argument("--seeds", type=int, default=32)
    many.add_argument("--out", type=Path, required=True)
    many.add_argument(
        "--systems", nargs="+", choices=SYSTEMS, default=list(TAU_SYSTEMS)
    )
    many.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    for p in (one, many):
        p.add_argument("--warmup", type=int, default=30_000)
        p.add_argument("--measure", type=int, default=40_000)
        p.add_argument(
            "--equil-sweeps",
            type=int,
            default=0,
            help="If set, record block means from a random start instead of tau",
        )
        p.add_argument("--equil-block", type=int, default=512)
    summary = sub.add_parser("summarize", help="Median tau and cost per configuration")
    summary.add_argument("results", type=Path)
    summary.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "run" and args.equil_sweeps:
        result = run_equil(
            args.system, args.config, args.seed, args.equil_sweeps, args.equil_block
        )
        print(json.dumps(result))
    elif args.command == "run":
        result = run_task(
            args.system, args.config, args.seed, args.warmup, args.measure
        )
        print(json.dumps(result))
    elif args.command == "all":
        run_all(args)
    else:
        out = summarize(args.results)
        if args.json:
            args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
