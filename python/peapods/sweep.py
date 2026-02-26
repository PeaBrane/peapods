import itertools
import sys
import time
from pathlib import Path

import numpy as np

from peapods.spin_models import Ising


def _config_label(coupling, h_mode, ou_mode, oc_mode):
    parts = [coupling]
    if h_mode != "houdayer":
        parts.append(h_mode)
    if ou_mode != "swap":
        parts.append(ou_mode)
    if oc_mode != "wolff":
        parts.append(oc_mode)
    return "_".join(parts)


def _size_label(shape):
    return "x".join(str(s) for s in shape)


def _validate_combo(coupling, h_mode, ou_mode, houdayer_interval):
    if ou_mode == "free" and h_mode != "cmr":
        return False, f"free update requires cmr, got houdayer_mode={h_mode}"
    if h_mode != "houdayer" and houdayer_interval is None:
        return False, f"houdayer_mode={h_mode} set but no --houdayer-interval"
    return True, ""


def _save_data(models, config_label, temperatures, output_dir):
    save_dict = {"temperatures": temperatures}
    for size_label, model in models.items():
        prefix = size_label
        save_dict[f"{prefix}_binder_cumulant"] = model.binder_cumulant
        save_dict[f"{prefix}_heat_capacity"] = model.heat_capacity
        save_dict[f"{prefix}_energies"] = model.energies_avg
        if hasattr(model, "sg_binder"):
            save_dict[f"{prefix}_sg_binder"] = model.sg_binder
        if hasattr(model, "mean_cluster_size"):
            save_dict[f"{prefix}_mean_cluster_size"] = model.mean_cluster_size
        if hasattr(model, "top_cluster_sizes"):
            save_dict[f"{prefix}_top_cluster_sizes"] = model.top_cluster_sizes

    path = Path(output_dir) / f"sweep_{config_label}.npz"
    np.savez(path, **save_dict)
    print(f"  Data saved to {path}")


def _plot_binder(models, config_label, temperatures, output_dir):
    import matplotlib.pyplot as plt

    has_overlap = any(hasattr(m, "sg_binder") for m in models.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    for size_label, model in models.items():
        y = model.sg_binder if has_overlap else model.binder_cumulant
        ax.plot(temperatures, y, label=size_label)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("SG Binder" if has_overlap else "Binder cumulant")
    ax.set_xscale("log")
    ax.legend()
    ax.set_title(config_label)

    path = Path(output_dir) / f"binder_{config_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {path}")


def _plot_heat_capacity(models, config_label, temperatures, output_dir):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for size_label, model in models.items():
        ax.plot(temperatures, model.heat_capacity, label=size_label)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("$C_v$")
    ax.legend()
    ax.set_title(f"Heat capacity — {config_label}")

    path = Path(output_dir) / f"heat_capacity_{config_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {path}")


def _plot_csd(model, size_label, config_label, temperatures, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=temperatures.min(), vmax=temperatures.max())
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6, 4))
    for t_idx, hist in enumerate(model.fk_csd):
        sizes = np.arange(len(hist))
        total = hist.sum()
        if total == 0:
            continue
        mask = hist > 0
        ps = hist[mask] / total
        ax.scatter(
            sizes[mask], ps, s=8, color=cmap(norm(temperatures[t_idx])), alpha=0.7
        )
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Temperature")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cluster size $s$")
    ax.set_ylabel("$P(s)$")
    ax.set_title(f"CSD — {size_label}, {config_label}")

    path = Path(output_dir) / f"csd_{size_label}_{config_label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {path}")


def run_sweep(
    sizes,
    *,
    couplings=("ferro",),
    temperatures,
    n_replicas=1,
    n_disorder=1,
    neighbor_offsets=None,
    geometry=None,
    n_sweeps,
    sweep_mode="metropolis",
    cluster_update_interval=None,
    cluster_mode="sw",
    pt_interval=None,
    houdayer_interval=None,
    houdayer_modes=("houdayer",),
    overlap_cluster_modes=("wolff",),
    overlap_update_modes=("swap",),
    warmup_ratio=0.25,
    collect_csd=False,
    collect_top_clusters=False,
    save_plots=False,
    save_data=False,
    output_dir=".",
):
    """Run a parameter sweep over sizes and configurations.

    Sizes share a plot (as legend entries); each other Cartesian config combo
    produces its own set of plots.

    Returns:
        ``{config_label: {size_label: Ising}}`` mapping.
    """
    if save_plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print(
                "error: matplotlib is required for --save-plots. "
                "Install it with: uv pip install matplotlib",
                file=sys.stderr,
            )
            sys.exit(1)

    output_path = Path(output_dir)
    if save_plots or save_data:
        output_path.mkdir(parents=True, exist_ok=True)

    combos = list(
        itertools.product(
            couplings, houdayer_modes, overlap_update_modes, overlap_cluster_modes
        )
    )

    total_runs = 0
    valid_combos = []
    for coupling, h_mode, ou_mode, oc_mode in combos:
        ok, reason = _validate_combo(coupling, h_mode, ou_mode, houdayer_interval)
        if not ok:
            print(
                f"  skip: {_config_label(coupling, h_mode, ou_mode, oc_mode)} — {reason}",
                file=sys.stderr,
            )
            continue
        valid_combos.append((coupling, h_mode, ou_mode, oc_mode))
        total_runs += len(sizes)

    all_results = {}
    run_idx = 0
    wall_start = time.perf_counter()

    for coupling, h_mode, ou_mode, oc_mode in valid_combos:
        label = _config_label(coupling, h_mode, ou_mode, oc_mode)
        models = {}

        for shape in sizes:
            run_idx += 1
            slabel = _size_label(shape)
            print(f"[{run_idx}/{total_runs}] {slabel}, {label}")

            model = Ising(
                shape,
                couplings=coupling,
                temperatures=temperatures,
                n_replicas=n_replicas,
                n_disorder=n_disorder,
                neighbor_offsets=neighbor_offsets,
                geometry=geometry,
            )

            t0 = time.perf_counter()
            model.sample(
                n_sweeps,
                sweep_mode=sweep_mode,
                cluster_update_interval=cluster_update_interval,
                cluster_mode=cluster_mode,
                pt_interval=pt_interval,
                houdayer_interval=houdayer_interval,
                houdayer_mode=h_mode,
                overlap_cluster_mode=oc_mode,
                warmup_ratio=warmup_ratio,
                collect_csd=collect_csd,
                overlap_update_mode=ou_mode,
                collect_top_clusters=collect_top_clusters,
            )
            elapsed = time.perf_counter() - t0
            print(f"  {elapsed:.2f}s")

            models[slabel] = model

        all_results[label] = models

        if save_data:
            _save_data(models, label, temperatures, output_dir)

        if save_plots:
            _plot_binder(models, label, temperatures, output_dir)
            _plot_heat_capacity(models, label, temperatures, output_dir)
            if collect_csd:
                for slabel, model in models.items():
                    if hasattr(model, "fk_csd"):
                        _plot_csd(model, slabel, label, temperatures, output_dir)

    wall_total = time.perf_counter() - wall_start
    print(f"\nSweep complete: {total_runs} runs in {wall_total:.1f}s")

    return all_results
