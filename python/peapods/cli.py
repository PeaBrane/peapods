import argparse
import json
import sys
import time
import tomllib

import numpy as np

from peapods import Ising
from peapods.sweep import run_sweep

COUPLING_CHOICES = ["ferro", "bimodal", "gaussian"]
BUILD_MODE_CHOICES = ["houdayer", "jorg", "cmr", "cmr3"]
OVERLAP_UPDATE_CHOICES = ["swap", "free"]
OVERLAP_CLUSTER_CHOICES = ["wolff", "sw"]


def _add_common_args(parser):
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
        default="log",
        choices=["linear", "log"],
        help="Temperature spacing (default: log)",
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
        "--overlap-cluster-update-interval",
        type=int,
        default=None,
        help="Overlap cluster move every N sweeps (requires n_replicas >= 2)",
    )
    parser.add_argument(
        "--collect-top-clusters",
        action="store_true",
        help="Collect top-4 overlap cluster sizes per temperature",
    )
    parser.add_argument(
        "--autocorrelation-max-lag",
        type=int,
        default=None,
        help="Max lag for streaming autocorrelation of m² and q²",
    )


def add_simulation_args(parser):
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
        choices=COUPLING_CHOICES,
        help="Coupling distribution (default: ferro)",
    )
    parser.add_argument(
        "--overlap-cluster-build-mode",
        default="houdayer",
        choices=BUILD_MODE_CHOICES,
    )
    parser.add_argument(
        "--overlap-cluster-mode", default="wolff", choices=OVERLAP_CLUSTER_CHOICES
    )
    parser.add_argument(
        "--overlap-update-mode",
        default="swap",
        choices=OVERLAP_UPDATE_CHOICES,
        help="Overlap cluster update mode (default: swap)",
    )
    _add_common_args(parser)


def _add_sweep_common_args(parser):
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
    parser.add_argument("--n-replicas", type=int, default=None)
    parser.add_argument("--n-disorder", type=int, default=None)
    parser.add_argument("--temp-min", type=float, default=None)
    parser.add_argument("--temp-max", type=float, default=None)
    parser.add_argument("--n-temps", type=int, default=None)
    parser.add_argument(
        "--temp-scale",
        default=None,
        choices=["linear", "log"],
        help="Temperature spacing (default: log)",
    )
    parser.add_argument("--n-sweeps", type=int, default=None)
    parser.add_argument("--sweep-mode", default=None, choices=["metropolis", "gibbs"])
    parser.add_argument(
        "--cluster-interval",
        type=int,
        default=None,
        help="Cluster update every N sweeps",
    )
    parser.add_argument("--cluster-mode", default=None, choices=["sw", "wolff"])
    parser.add_argument(
        "--pt-interval",
        type=int,
        default=None,
        help="Parallel tempering every N sweeps",
    )
    parser.add_argument(
        "--overlap-cluster-update-interval",
        type=int,
        default=None,
        help="Overlap cluster move every N sweeps (requires n_replicas >= 2)",
    )
    parser.add_argument(
        "--collect-top-clusters",
        action="store_true",
        default=None,
        help="Collect top-4 overlap cluster sizes per temperature",
    )
    parser.add_argument(
        "--autocorrelation-max-lag",
        type=int,
        default=None,
        help="Max lag for streaming autocorrelation of m² and q²",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        default=None,
        help=(
            "Disable inner-loop parallelism (replicas/temps processed sequentially). "
            "Set n_disorder to number of physical cores (ignore hyperthreading — "
            "this workload is cache-pressure bound) so each realization gets its "
            "own L1 cache."
        ),
    )


def _add_sweep_args(parser):
    parser.add_argument(
        "--config", type=str, default=None, help="Path to TOML config file"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        help="Lattice sizes as comma-separated dims, e.g. --sizes 8,8 16,16 8,8,8",
    )
    parser.add_argument(
        "--couplings",
        nargs="+",
        default=None,
        choices=COUPLING_CHOICES,
        help="Coupling distributions to sweep (default: ferro)",
    )
    parser.add_argument(
        "--overlap-cluster-build-mode",
        nargs="+",
        default=None,
        choices=BUILD_MODE_CHOICES,
    )
    parser.add_argument(
        "--overlap-cluster-mode",
        nargs="+",
        default=None,
        choices=OVERLAP_CLUSTER_CHOICES,
    )
    parser.add_argument(
        "--overlap-update-mode",
        nargs="+",
        default=None,
        choices=OVERLAP_UPDATE_CHOICES,
    )
    _add_sweep_common_args(parser)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument(
        "--collect-csd",
        action="store_true",
        default=None,
        help="Collect FK cluster size distribution",
    )
    parser.add_argument(
        "--autocorrelation-plot-temp",
        type=float,
        default=None,
        help="Temperature at which to plot τ vs L (uses nearest T in grid)",
    )
    parser.add_argument(
        "--save-plots", action="store_true", default=None, help="Save plots to disk"
    )
    parser.add_argument(
        "--save-data", action="store_true", default=None, help="Save data as .npz"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: .)"
    )


def build_model(args):
    temperatures = _build_temperatures(args)

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
        overlap_cluster_update_interval=args.overlap_cluster_update_interval,
        overlap_cluster_build_mode=args.overlap_cluster_build_mode,
        overlap_cluster_mode=args.overlap_cluster_mode,
        overlap_update_mode=args.overlap_update_mode,
        collect_top_clusters=args.collect_top_clusters,
        autocorrelation_max_lag=args.autocorrelation_max_lag,
    )


def _build_temperatures(args):
    if args.temp_scale == "linear":
        return np.linspace(args.temp_min, args.temp_max, args.n_temps)
    return np.geomspace(args.temp_min, args.temp_max, args.n_temps)


_SWEEP_DEFAULTS = dict(
    sizes=None,
    couplings=("ferro",),
    temp_min=None,
    temp_max=None,
    n_temps=32,
    temp_scale="log",
    n_replicas=1,
    n_disorder=1,
    neighbor_offsets=None,
    geometry=None,
    n_sweeps=None,
    sweep_mode="metropolis",
    cluster_interval=None,
    cluster_mode="sw",
    pt_interval=None,
    overlap_cluster_update_interval=None,
    overlap_cluster_build_mode=("houdayer",),
    overlap_cluster_mode=("wolff",),
    overlap_update_mode=("swap",),
    warmup_ratio=0.25,
    collect_csd=False,
    collect_top_clusters=False,
    autocorrelation_max_lag=None,
    autocorrelation_plot_temp=None,
    save_plots=False,
    save_data=False,
    output_dir=".",
    sequential=False,
)


def _load_sweep_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    kw = {}

    if "lattice" in cfg:
        lat = cfg["lattice"]
        if "sizes" in lat:
            kw["sizes"] = [tuple(s) for s in lat["sizes"]]
        if "geometry" in lat:
            kw["geometry"] = lat["geometry"]
        if "neighbor_offsets" in lat:
            kw["neighbor_offsets"] = [list(o) for o in lat["neighbor_offsets"]]

    if "couplings" in cfg.get("lattice", {}):
        kw["couplings"] = tuple(cfg["lattice"]["couplings"])

    if "temperatures" in cfg:
        t = cfg["temperatures"]
        if "min" in t:
            kw["temp_min"] = t["min"]
        if "max" in t:
            kw["temp_max"] = t["max"]
        if "count" in t:
            kw["n_temps"] = t["count"]
        if "scale" in t:
            kw["temp_scale"] = t["scale"]

    if "replicas" in cfg:
        r = cfg["replicas"]
        if "n_replicas" in r:
            kw["n_replicas"] = r["n_replicas"]
        if "n_disorder" in r:
            kw["n_disorder"] = r["n_disorder"]

    if "sampling" in cfg:
        s = cfg["sampling"]
        if "n_sweeps" in s:
            kw["n_sweeps"] = s["n_sweeps"]
        if "sweep_mode" in s:
            kw["sweep_mode"] = s["sweep_mode"]
        if "warmup_ratio" in s:
            kw["warmup_ratio"] = s["warmup_ratio"]
        if "sequential" in s:
            kw["sequential"] = s["sequential"]

    if "cluster" in cfg:
        c = cfg["cluster"]
        if "interval" in c:
            kw["cluster_interval"] = c["interval"]
        if "mode" in c:
            kw["cluster_mode"] = c["mode"]

    if "parallel_tempering" in cfg:
        pt = cfg["parallel_tempering"]
        if "interval" in pt:
            kw["pt_interval"] = pt["interval"]

    if "overlap_cluster" in cfg:
        oc = cfg["overlap_cluster"]
        if "interval" in oc:
            kw["overlap_cluster_update_interval"] = oc["interval"]
        if "build_modes" in oc:
            kw["overlap_cluster_build_mode"] = tuple(oc["build_modes"])
        if "cluster_mode" in oc:
            kw["overlap_cluster_mode"] = tuple(
                oc["cluster_mode"]
                if isinstance(oc["cluster_mode"], list)
                else [oc["cluster_mode"]]
            )
        if "update_modes" in oc:
            kw["overlap_update_mode"] = tuple(oc["update_modes"])

    if "diagnostics" in cfg:
        d = cfg["diagnostics"]
        if "collect_csd" in d:
            kw["collect_csd"] = d["collect_csd"]
        if "collect_top_clusters" in d:
            kw["collect_top_clusters"] = d["collect_top_clusters"]
        if "autocorrelation" in d:
            ac = d["autocorrelation"]
            if "max_lag" in ac:
                kw["autocorrelation_max_lag"] = ac["max_lag"]
            if "plot_temp" in ac:
                kw["autocorrelation_plot_temp"] = ac["plot_temp"]

    if "output" in cfg:
        o = cfg["output"]
        if "save_plots" in o:
            kw["save_plots"] = o["save_plots"]
        if "save_data" in o:
            kw["save_data"] = o["save_data"]
        if "dir" in o:
            kw["output_dir"] = o["dir"]

    return kw


def run_sweep_cli(args):
    kw = {}
    if args.config is not None:
        kw = _load_sweep_config(args.config)

    cli_map = {
        "sizes": args.sizes,
        "couplings": args.couplings,
        "temp_min": args.temp_min,
        "temp_max": args.temp_max,
        "n_temps": args.n_temps,
        "temp_scale": args.temp_scale,
        "n_replicas": args.n_replicas,
        "n_disorder": args.n_disorder,
        "neighbor_offsets": args.neighbor_offsets,
        "geometry": args.geometry,
        "n_sweeps": args.n_sweeps,
        "sweep_mode": args.sweep_mode,
        "cluster_interval": args.cluster_interval,
        "cluster_mode": args.cluster_mode,
        "pt_interval": args.pt_interval,
        "overlap_cluster_update_interval": args.overlap_cluster_update_interval,
        "overlap_cluster_build_mode": args.overlap_cluster_build_mode,
        "overlap_cluster_mode": args.overlap_cluster_mode,
        "overlap_update_mode": args.overlap_update_mode,
        "warmup_ratio": args.warmup_ratio,
        "collect_csd": args.collect_csd,
        "collect_top_clusters": args.collect_top_clusters,
        "autocorrelation_max_lag": args.autocorrelation_max_lag,
        "autocorrelation_plot_temp": args.autocorrelation_plot_temp,
        "save_plots": args.save_plots,
        "save_data": args.save_data,
        "output_dir": args.output_dir,
        "sequential": args.sequential,
    }
    for key, val in cli_map.items():
        if val is not None:
            kw[key] = val

    for key, default in _SWEEP_DEFAULTS.items():
        kw.setdefault(key, default)

    if kw["sizes"] is None:
        print("error: --sizes is required (via CLI or config file)", file=sys.stderr)
        sys.exit(1)
    if kw["temp_min"] is None or kw["temp_max"] is None:
        print(
            "error: --temp-min and --temp-max are required (via CLI or config file)",
            file=sys.stderr,
        )
        sys.exit(1)
    if kw["n_sweeps"] is None:
        print("error: --n-sweeps is required (via CLI or config file)", file=sys.stderr)
        sys.exit(1)

    if isinstance(kw["sizes"][0], str):
        kw["sizes"] = [tuple(int(x) for x in s.split(",")) for s in kw["sizes"]]

    if kw["temp_scale"] == "linear":
        temperatures = np.linspace(kw["temp_min"], kw["temp_max"], kw["n_temps"])
    else:
        temperatures = np.geomspace(kw["temp_min"], kw["temp_max"], kw["n_temps"])

    neighbor_offsets = kw["neighbor_offsets"]
    if isinstance(neighbor_offsets, str):
        neighbor_offsets = json.loads(neighbor_offsets)

    run_sweep(
        kw["sizes"],
        couplings=tuple(kw["couplings"]),
        temperatures=temperatures,
        n_replicas=kw["n_replicas"],
        n_disorder=kw["n_disorder"],
        neighbor_offsets=neighbor_offsets,
        geometry=kw["geometry"],
        n_sweeps=kw["n_sweeps"],
        sweep_mode=kw["sweep_mode"],
        cluster_update_interval=kw["cluster_interval"],
        cluster_mode=kw["cluster_mode"],
        pt_interval=kw["pt_interval"],
        overlap_cluster_update_interval=kw["overlap_cluster_update_interval"],
        overlap_cluster_build_modes=tuple(kw["overlap_cluster_build_mode"]),
        overlap_cluster_modes=tuple(kw["overlap_cluster_mode"]),
        overlap_update_modes=tuple(kw["overlap_update_mode"]),
        warmup_ratio=kw["warmup_ratio"],
        collect_csd=kw["collect_csd"],
        collect_top_clusters=kw["collect_top_clusters"],
        autocorrelation_max_lag=kw["autocorrelation_max_lag"],
        autocorrelation_plot_temp=kw["autocorrelation_plot_temp"],
        save_plots=kw["save_plots"],
        save_data=kw["save_data"],
        output_dir=kw["output_dir"],
        sequential=kw["sequential"],
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

    # sweep subcommand
    sweep = sub.add_parser("sweep", help="Run parameter sweeps with optional plotting")
    _add_sweep_args(sweep)

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
    elif args.command == "sweep":
        run_sweep_cli(args)


if __name__ == "__main__":
    main()
