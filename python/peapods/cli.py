import argparse
import json
import sys
import time

import numpy as np

from peapods import Ising


def add_simulation_args(parser):
    # Model setup
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        required=True,
        help="Lattice dimensions, e.g. --shape 32 32",
    )
    parser.add_argument(
        "--couplings",
        default="ferro",
        choices=["ferro", "bimodal", "gaussian"],
        help="Coupling distribution (default: ferro)",
    )
    parser.add_argument(
        "--geometry",
        choices=["triangular", "tri", "fcc", "bcc"],
        help="Named lattice geometry",
    )
    parser.add_argument(
        "--neighbor-offsets",
        type=str,
        default=None,
        help="JSON list of offset vectors, e.g. '[[1,0],[0,1]]'",
    )
    parser.add_argument("--n-replicas", type=int, default=1)
    parser.add_argument("--n-disorder", type=int, default=1)

    # Temperature grid
    parser.add_argument("--temp-min", type=float, required=True)
    parser.add_argument("--temp-max", type=float, required=True)
    parser.add_argument("--n-temps", type=int, default=32)
    parser.add_argument(
        "--temp-scale",
        default="linear",
        choices=["linear", "log"],
        help="Temperature spacing (default: linear)",
    )

    # Sampling
    parser.add_argument("--n-sweeps", type=int, required=True)
    parser.add_argument(
        "--sweep-mode", default="metropolis", choices=["metropolis", "gibbs"]
    )
    parser.add_argument(
        "--cluster-interval",
        type=int,
        default=None,
        help="Cluster update every N sweeps",
    )
    parser.add_argument("--cluster-mode", default="sw", choices=["sw", "wolff"])
    parser.add_argument(
        "--pt-interval",
        type=int,
        default=None,
        help="Parallel tempering every N sweeps",
    )
    parser.add_argument(
        "--houdayer-interval",
        type=int,
        default=None,
        help="Houdayer moves every N sweeps (requires n_replicas >= 2)",
    )
    parser.add_argument(
        "--houdayer-mode", default="houdayer", choices=["houdayer", "jorg", "cmr"]
    )
    parser.add_argument(
        "--overlap-cluster-mode", default="wolff", choices=["wolff", "sw"]
    )
    parser.add_argument(
        "--overlap-update-mode",
        default="swap",
        choices=["swap", "free"],
        help="Overlap cluster update mode (default: swap)",
    )
    parser.add_argument(
        "--collect-top-clusters",
        action="store_true",
        help="Collect top-4 overlap cluster sizes per temperature",
    )


def build_model(args):
    if args.temp_scale == "linear":
        temperatures = np.linspace(args.temp_min, args.temp_max, args.n_temps)
    else:
        temperatures = np.geomspace(args.temp_min, args.temp_max, args.n_temps)

    neighbor_offsets = None
    if args.neighbor_offsets is not None:
        neighbor_offsets = json.loads(args.neighbor_offsets)

    return Ising(
        tuple(args.shape),
        couplings=args.couplings,
        temperatures=temperatures,
        n_replicas=args.n_replicas,
        n_disorder=args.n_disorder,
        neighbor_offsets=neighbor_offsets,
        geometry=args.geometry,
    )


def sample_kwargs(args):
    return dict(
        sweep_mode=args.sweep_mode,
        cluster_update_interval=args.cluster_interval,
        cluster_mode=args.cluster_mode,
        pt_interval=args.pt_interval,
        houdayer_interval=args.houdayer_interval,
        houdayer_mode=args.houdayer_mode,
        overlap_cluster_mode=args.overlap_cluster_mode,
        overlap_update_mode=args.overlap_update_mode,
        collect_top_clusters=args.collect_top_clusters,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="peapods",
        description="Ising Monte Carlo simulations from the command line.",
    )
    sub = parser.add_subparsers(dest="command")

    # simulate subcommand
    sim = sub.add_parser("simulate", help="Run an Ising simulation")
    add_simulation_args(sim)
    sim.add_argument("--warmup-ratio", type=float, default=0.25)
    sim.add_argument(
        "--collect-csd",
        action="store_true",
        help="Collect FK cluster size distribution",
    )
    sim.add_argument(
        "-o", "--output", type=str, default=None, help="Save full results to .npz file"
    )

    # bench subcommand
    bench = sub.add_parser("bench", help="Benchmark sampling performance")
    add_simulation_args(bench)

    return parser


def run_simulate(args):
    model = build_model(args)

    result = model.sample(
        args.n_sweeps,
        **sample_kwargs(args),
        warmup_ratio=args.warmup_ratio,
        collect_csd=args.collect_csd,
    )

    has_overlap = hasattr(model, "sg_binder")
    has_csd = hasattr(model, "mean_cluster_size")
    print_table(model, has_overlap, has_csd)

    if args.output:
        save_dict = {
            "temperatures": model.temperatures,
            "binder_cumulant": model.binder_cumulant,
            "heat_capacity": model.heat_capacity,
        }
        for key in (
            "mags",
            "mags2",
            "mags4",
            "energies",
            "energies2",
            "overlap",
            "overlap2",
            "overlap4",
        ):
            if key in result:
                save_dict[key] = result[key]
        if has_overlap:
            save_dict["sg_binder"] = model.sg_binder
        if has_csd:
            save_dict["mean_cluster_size"] = model.mean_cluster_size
        if hasattr(model, "fk_csd"):
            save_dict["fk_csd"] = model.fk_csd
        if hasattr(model, "top_cluster_sizes"):
            save_dict["top_cluster_sizes"] = model.top_cluster_sizes
        np.savez(args.output, **save_dict)
        print(f"\nResults saved to {args.output}")


def run_bench(args):
    model = build_model(args)
    shape_str = "x".join(str(s) for s in args.shape)

    t0 = time.perf_counter()
    model.sample(args.n_sweeps, **sample_kwargs(args), warmup_ratio=0.0)
    elapsed = time.perf_counter() - t0

    per_sweep = elapsed / args.n_sweeps * 1000
    print(f"Lattice: {shape_str}  |  Temps: {args.n_temps}  |  Sweeps: {args.n_sweeps}")
    print(f"Total: {elapsed:.3f} s  |  {per_sweep:.3f} ms/sweep")


def print_table(model, has_overlap, has_csd):
    temps = model.temperatures
    energy = model.energies_avg
    binder = model.binder_cumulant
    hcap = model.heat_capacity
    has_top4 = hasattr(model, "top_cluster_sizes")

    cols = [f"{'T':>8}", f"{'E':>10}", f"{'Binder':>10}", f"{'C_v':>10}"]
    if has_overlap:
        cols.append(f"{'Overlap Binder':>15}")
    if has_csd:
        cols.append(f"{'Cluster Size':>14}")
    if has_top4:
        cols.append(f"{'Top-4 Clusters':>30}")

    header = "  ".join(cols)
    print(header)
    print("-" * len(header))

    for i in range(len(temps)):
        row = [
            f"{temps[i]:8.4f}",
            f"{energy[i]:10.6f}",
            f"{binder[i]:10.6f}",
            f"{hcap[i]:10.4f}",
        ]
        if has_overlap:
            row.append(f"{model.sg_binder[i]:15.6f}")
        if has_csd:
            row.append(f"{model.mean_cluster_size[i]:14.2f}")
        if has_top4:
            t = model.top_cluster_sizes[i]
            row.append(f"({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}, {t[3]:.3f})".rjust(30))
        print("  ".join(row))


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "simulate":
        run_simulate(args)
    elif args.command == "bench":
        run_bench(args)


if __name__ == "__main__":
    main()
