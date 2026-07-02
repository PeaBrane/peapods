import numpy as np

from peapods._core import IsingSimulation

GEOMETRIES = {
    "triangular": [[1, 0], [0, 1], [1, -1]],
    "tri": [[1, 0], [0, 1], [1, -1]],
    "fcc": [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, -1, 0], [1, 0, -1], [0, 1, -1]],
    "bcc": [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]],
}


def _seed_material(seed):
    if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a non-negative integer or None")
    root = np.random.SeedSequence(seed)
    coupling_seed, dynamics_seed = root.spawn(2)
    dynamics = int(dynamics_seed.generate_state(1, dtype=np.uint64)[0])
    return coupling_seed, dynamics


def _dynamics_seed(seed):
    return _seed_material(seed)[1]


class Ising:
    """Ising model on a periodic Bravais lattice with Monte Carlo sampling.

    Supports ferromagnets and spin glasses on hypercubic, triangular, FCC, BCC,
    or any custom lattice defined by neighbor offsets. Multiple replicas enable
    overlap-based spin glass order parameters.

    Attributes:
        lattice_shape: Shape of the lattice as a tuple of ints.
        n_dims: Number of spatial dimensions.
        n_neighbors: Number of nearest neighbors per site.
        temperatures: Array of temperatures for parallel tempering.
        n_temps: Number of temperature points.
        n_replicas: Number of replicas per temperature.
        n_disorder: Number of disorder realizations.
        couplings: Coupling array with shape `(*lattice_shape, n_neighbors)`.
        binder_cumulant: Binder cumulant `1 - <m^4> / (3 <m^2>^2)`, set after
            [`sample`][peapods.Ising.sample].
        heat_capacity: Heat capacity `(<E^2> - <E>^2) / T^2`, set after
            [`sample`][peapods.Ising.sample].
        sg_binder: Spin glass Binder parameter `1 - <q^4> / (3 <q^2>^2)`, set
            after [`sample`][peapods.Ising.sample] with `n_replicas >= 2`.
    """

    def __init__(
        self,
        lattice_shape,
        couplings="ferro",
        temperatures=np.geomspace(0.1, 10, 32),
        n_replicas=1,
        n_disorder=1,
        neighbor_offsets=None,
        geometry=None,
        seed=None,
    ):
        """Create an Ising model.

        Args:
            lattice_shape: Shape of the periodic lattice, e.g. `(32, 32)` for a
                2D 32x32 grid.
            couplings: Coupling configuration. One of `"ferro"` (all +1),
                `"bimodal"` (random +/-1), `"gaussian"` (standard normal), or a
                NumPy array of shape `(*lattice_shape, n_neighbors)`.
            temperatures: Array of temperatures for the simulation. Defaults to
                32 points log-spaced from 0.1 to 10.
            n_replicas: Number of independent replicas per temperature. Must be
                >= 2 for overlap statistics and Houdayer moves.
            n_disorder: Number of disorder realizations. Each realization gets
                its own coupling array.
            neighbor_offsets: List of integer offset vectors defining nearest
                neighbors, e.g. `[[1, 0], [0, 1]]` for a square lattice. Mutually
                exclusive with `geometry`.
            geometry: Named lattice geometry. One of `"triangular"` / `"tri"`,
                `"fcc"`, or `"bcc"`. Mutually exclusive with `neighbor_offsets`.
                If neither is given, defaults to a hypercubic lattice.
            seed: Optional non-negative integer controlling built-in random
                couplings and initial dynamics. `None` uses fresh entropy.
        """
        if geometry is not None:
            if neighbor_offsets is not None:
                raise ValueError("Cannot specify both geometry and neighbor_offsets")
            if geometry not in GEOMETRIES:
                raise ValueError(
                    f"Unknown geometry '{geometry}', choose from: {list(GEOMETRIES.keys())}"
                )
            neighbor_offsets = GEOMETRIES[geometry]

        self.lattice_shape = tuple(lattice_shape)
        self.n_spins = int(np.prod(lattice_shape))
        self.n_dims = len(lattice_shape)
        self.n_neighbors = len(neighbor_offsets) if neighbor_offsets else self.n_dims
        self.temperatures = temperatures.copy().astype(np.float32)
        self.n_temps = len(temperatures)
        self.n_replicas = n_replicas
        self.n_disorder = n_disorder
        self.seed = seed
        coupling_seed, self._constructor_dynamics_seed = _seed_material(seed)

        if isinstance(couplings, np.ndarray):
            coup = couplings.astype(np.float32)
        else:
            single_shape = self.lattice_shape + (self.n_neighbors,)
            coupling_children = coupling_seed.spawn(n_disorder)
            realizations = []
            for child in coupling_children:
                rng = np.random.default_rng(child)
                match couplings:
                    case "ferro":
                        realization = np.ones(single_shape, dtype=np.float32)
                    case "bimodal":
                        realization = (
                            2 * rng.integers(0, 2, size=single_shape) - 1
                        ).astype(np.float32)
                    case "gaussian":
                        realization = rng.standard_normal(single_shape).astype(
                            np.float32
                        )
                    case _:
                        raise ValueError("Invalid mode for couplings.")
                realizations.append(realization)
            coup = realizations[0] if n_disorder == 1 else np.stack(realizations)

        self.couplings = coup
        self._sim = IsingSimulation(
            list(lattice_shape),
            coup,
            self.temperatures,
            n_replicas,
            neighbor_offsets,
            self._constructor_dynamics_seed,
        )

    def reset(self, seed=None):
        """Reset dynamics while keeping the model's couplings fixed.

        A bare reset replays the constructor's initial dynamics. Passing a seed
        performs a deterministic one-off reset without replacing that seed.
        """
        self._sim.reset(None if seed is None else _dynamics_seed(seed))

    def sample(
        self,
        n_sweeps,
        sweep_mode="metropolis",
        cluster_update_interval=None,
        cluster_mode="sw",
        cluster_action="update",
        pt_interval=None,
        pt_schedule="single_random_edge",
        overlap_cluster_update_interval=None,
        overlap_cluster_build_mode="houdayer",
        overlap_cluster_mode="wolff",
        overlap_cluster_action="update",
        warmup_ratio=0.25,
        collect_cluster_stats=False,
        autocorrelation_max_lag=None,
        sequential=False,
        equilibration_diagnostic=False,
        snapshot_interval=None,
    ):
        """Run Monte Carlo sampling and compute observables.

        After sampling, the following attributes are set on the instance:

        - `binder_cumulant` — Binder cumulant per temperature.
        - `heat_capacity` — Heat capacity per temperature.
        - `sg_binder` — Spin glass Binder parameter (only with `n_replicas >= 2`).
        - `fk_csd` — FK cluster size distribution (only with
          `collect_cluster_stats=True`).
        - `top_cluster_sizes` — List of arrays (one per overlap mode), each
          shape `(n_temps, 4)`, giving average relative sizes of the 4 largest
          overlap clusters per temperature (only with
          `collect_cluster_stats=True`).

        Args:
            n_sweeps: Total number of Monte Carlo sweeps (including warmup).
            sweep_mode: Single-spin update algorithm. `"metropolis"` or `"gibbs"`.
            cluster_update_interval: If set, perform a cluster update every this
                many sweeps.
            cluster_mode: Cluster algorithm. `"sw"` (Swendsen-Wang) or `"wolff"`.
            cluster_action: `"update"` to mutate spins or `"observe"` to
                measure a full SW/FK graph without flipping spins.
            pt_interval: If set, attempt parallel tempering swaps every this many
                sweeps.
            pt_schedule: `"single_random_edge"` for legacy PT or
                `"full_ladder"` to attempt every adjacent edge per event.
            overlap_cluster_update_interval: If set, attempt overlap cluster
                moves every this many sweeps. Requires `n_replicas >= 2`.
            overlap_cluster_build_mode: Overlap cluster algorithm. `"houdayer"`
                (deterministic, group_size=2), `"houdN"` where N is even >= 2
                (e.g. `"houd4"`, `"houd6"` — isoenergetic balanced-site
                criterion, requires `n_replicas >= N`;
                **experimental for N > 2: very likely does not satisfy
                detailed balance**), `"jorg"` (stochastic
                FK bonds, group_size=2), or `"cmr"` (two-phase grey+blue,
                group_size=2). Multiple modes can be alternated with `+`,
                e.g. `"cmr+houdayer"` round-robins each overlap update call.
            overlap_cluster_mode: Cluster type used inside the overlap move.
                `"wolff"` or `"sw"`.
            overlap_cluster_action: `"update"` to perform the move or
                `"observe"` to record the full graph without acting on replicas.
            warmup_ratio: Fraction of sweeps discarded as warmup before
                collecting statistics. Default 0.25.
            collect_cluster_stats: If `True`, collect FK cluster size
                distribution and top-4 overlap cluster sizes.
            sequential: If `True`, disable inner-loop parallelism over
                replicas/temperatures. Use when outer-level parallelism over
                disorder realizations already saturates all physical cores.

        Returns:
            Raw results dictionary with keys like `"mags"`, `"energies"`, etc.
        """
        if cluster_action not in {"update", "observe"}:
            raise ValueError("cluster_action must be 'update' or 'observe'")
        if overlap_cluster_action not in {"update", "observe"}:
            raise ValueError("overlap_cluster_action must be 'update' or 'observe'")
        if pt_schedule not in {"single_random_edge", "full_ladder"}:
            raise ValueError(
                "pt_schedule must be 'single_random_edge' or 'full_ladder'"
            )
        if cluster_action == "observe" and cluster_update_interval is None:
            raise ValueError(
                "cluster_action='observe' requires cluster_update_interval"
            )
        if (
            overlap_cluster_action == "observe"
            and overlap_cluster_update_interval is None
        ):
            raise ValueError(
                "overlap_cluster_action='observe' requires "
                "overlap_cluster_update_interval"
            )

        oci = overlap_cluster_update_interval
        result = self._sim.sample(
            n_sweeps,
            sweep_mode,
            cluster_update_interval=cluster_update_interval,
            cluster_mode=cluster_mode if cluster_update_interval else None,
            cluster_action=cluster_action if cluster_update_interval else None,
            pt_interval=pt_interval,
            pt_schedule=pt_schedule,
            overlap_cluster_update_interval=oci,
            overlap_cluster_build_mode=overlap_cluster_build_mode if oci else None,
            overlap_cluster_mode=overlap_cluster_mode if oci else None,
            overlap_cluster_action=overlap_cluster_action if oci else None,
            warmup_ratio=warmup_ratio,
            collect_cluster_stats=collect_cluster_stats,
            autocorrelation_max_lag=autocorrelation_max_lag,
            sequential=sequential,
            equilibration_diagnostic=equilibration_diagnostic,
            snapshot_interval=snapshot_interval if oci else None,
        )
        self.mags = result["mags"]
        self.mags2 = result["mags2"]
        self.mags4 = result["mags4"]
        self.energies_avg = result["energies"]
        self.energies2_avg = result["energies2"]

        self.binder_cumulant = 1 - self.mags4 / (3 * self.mags2**2)
        self.heat_capacity = (
            self.n_spins
            * (self.energies2_avg - self.energies_avg**2)
            / self.temperatures**2
        )

        if "overlap2" in result:
            self.overlap = result["overlap"]
            self.overlap2 = result["overlap2"]
            self.overlap4 = result["overlap4"]
            self.sg_binder = 1 - self.overlap4 / (3 * self.overlap2**2)
            self.link_overlap = result["link_overlap"]
            self.link_overlap2 = result["link_overlap2"]
            self.link_overlap4 = result["link_overlap4"]
            self.link_overlap_binder = 1 - self.link_overlap4 / (
                3 * self.link_overlap2**2
            )

        if "overlap_histogram" in result:
            self.overlap_histogram = result["overlap_histogram"]

        if "ql_at_q_sum" in result:
            self.ql_at_q_sum = result["ql_at_q_sum"]
            self.ql2_at_q_sum = result["ql2_at_q_sum"]

        if "per_sample_overlap_histogram" in result:
            self.per_sample_overlap_histogram = result["per_sample_overlap_histogram"]

        if "per_sample_ql_at_q_sum" in result:
            self.per_sample_ql_at_q_sum = result["per_sample_ql_at_q_sum"]
            self.per_sample_ql2_at_q_sum = result["per_sample_ql2_at_q_sum"]

        if "fk_csd" in result:
            self.fk_csd = result["fk_csd"]
            mcs = np.empty(self.n_temps)
            for t, h in enumerate(self.fk_csd):
                s = np.arange(len(h))
                sh = s * h
                n_sites = sh.sum()
                mcs[t] = (s * sh).sum() / n_sites if n_sites > 0 else 0.0
            self.mean_cluster_size = mcs

        if "top_cluster_sizes" in result:
            self.top_cluster_sizes = result["top_cluster_sizes"]

        if "mags2_tau" in result:
            self.mags2_tau = result["mags2_tau"]
        if "overlap2_tau" in result:
            self.overlap2_tau = result["overlap2_tau"]

        if "equil_sweeps" in result:
            self._equil_sweeps = result["equil_sweeps"]
            self._equil_energy_avg = result["equil_energy_avg"]
            self._equil_link_overlap_avg = result["equil_link_overlap_avg"]

        if "cluster_snapshots" in result:
            self.cluster_snapshots = result["cluster_snapshots"]

        self.per_disorder = result.get("per_disorder", {})

        return result

    def equilibration_delta(self, j_squared=1.0):
        """Compute equilibration diagnostic Δ(t) = e(t) - J²β z (1 - q_l(t)).

        Δ approaches zero as the system thermalizes (Zhu et al. 2015).
        Note: the Rust energy convention is e = +Σ J s_i s_j / N (no minus
        sign), so the sign here is flipped relative to the Hamiltonian form.

        Args:
            j_squared: Average squared coupling ⟨J²⟩. 1.0 for bimodal and
                Gaussian (unit variance) spin glasses.

        Returns:
            Tuple of (sweeps, delta) where sweeps has shape ``(n_checkpoints,)``
            and delta has shape ``(n_checkpoints, n_temps)``.
        """
        beta = 1.0 / self.temperatures
        delta = self._equil_energy_avg - j_squared * beta * self.n_neighbors * (
            1 - self._equil_link_overlap_avg
        )
        return self._equil_sweeps, delta

    def get_energies(self):
        """Return the mean energies per temperature from the last sample run."""
        return self.energies_avg
