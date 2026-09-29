import numpy as np

from peapods._core import IsingSimulation, XYSimulation

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


def _prepare_temperatures(temperatures, dtype):
    result = np.array(temperatures, dtype=dtype, copy=True, order="C")
    if (
        result.ndim != 1
        or not result.size
        or not np.all(np.isfinite(result) & (result > 0))
    ):
        raise ValueError(
            "temperatures must be a nonempty vector of positive finite values"
        )
    return result


def _prepare_couplings(shape, neighbors, n_disorder, couplings, coupling_seed, dtype):
    if not isinstance(n_disorder, (int, np.integer)) or n_disorder < 1:
        raise ValueError("n_disorder must be a positive integer")
    single_shape = tuple(shape) + (neighbors,)
    if not isinstance(couplings, str):
        result = np.array(couplings, dtype=dtype, copy=True, order="C")
    else:
        realizations = []
        for child in coupling_seed.spawn(n_disorder):
            rng = np.random.default_rng(child)
            match couplings:
                case "ferro":
                    realization = np.ones(single_shape, dtype=dtype)
                case "bimodal":
                    realization = (
                        2 * rng.integers(0, 2, size=single_shape) - 1
                    ).astype(dtype)
                case "gaussian":
                    realization = rng.standard_normal(single_shape).astype(dtype)
                case _:
                    raise ValueError(
                        "couplings must be 'ferro', 'bimodal', 'gaussian', or an array"
                    )
            realizations.append(realization)
        result = realizations[0] if n_disorder == 1 else np.stack(realizations)
    if result.shape != single_shape and not (
        result.ndim == len(single_shape) + 1
        and result.shape[0] > 0
        and result.shape[1:] == single_shape
    ):
        raise ValueError(
            f"couplings must have shape {single_shape} or (n_disorder, {single_shape})"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("couplings must be finite")
    return result


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
        heat_capacity: Heat capacity per spin `N [<e^2> - <e>^2] / T^2`, with the
            thermal variance taken within each disorder realization before the
            disorder average `[...]`, set after [`sample`][peapods.Ising.sample].
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
                >= 2 for overlap statistics and same-temperature overlap moves.
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
        self.temperatures = _prepare_temperatures(temperatures, np.float32)
        self.n_temps = len(temperatures)
        self.n_replicas = n_replicas
        self.n_disorder = n_disorder
        self.seed = seed
        coupling_seed, self._constructor_dynamics_seed = _seed_material(seed)

        coup = _prepare_couplings(
            self.lattice_shape,
            self.n_neighbors,
            n_disorder,
            couplings,
            coupling_seed,
            np.float32,
        )

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
        overlap_cluster_max_temperature=None,
        warmup_ratio=0.25,
        collect_cluster_stats=False,
        autocorrelation_max_lag=None,
        autocorrelation_backend="ring",
        sequential=False,
        equilibration_diagnostic=False,
        snapshot_interval=None,
        collect_physics=False,
        displacements=None,
        block_size=None,
    ):
        """Run Monte Carlo sampling and compute observables.

        After sampling, the following attributes are set on the instance:

        - `binder_cumulant` — Binder cumulant per temperature.
        - `heat_capacity` — Heat capacity per spin and temperature, from the
          disorder-averaged thermal energy variance `energy_variance`.
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
                moves every this many sweeps. Requires `n_replicas >= 2`, except
                for `"rmc"`, which needs at least one replica and two temperatures.
            overlap_cluster_build_mode: Overlap cluster algorithm. `"houdayer"`
                (deterministic, group_size=2), `"houdN"` where N is even >= 2
                (e.g. `"houd4"`, `"houd6"` — isoenergetic balanced-site
                criterion, requires `n_replicas >= N`;
                **experimental for N > 2: does not satisfy detailed
                balance**), `"pairN"` (Houdayer clusters for N/2
                replica pairs swapped jointly, requires `n_replicas >= N`;
                `"pair2"` equals `"houdayer"`), `"jorg"` (stochastic
                FK bonds, group_size=2), `"jorgN"` (Jörg bonds for N/2 pairs
                swapped jointly; `"jorg2"` equals `"jorg"`), `"cmr"`
                (two-phase grey+blue, group_size=2), or `"rmc"`
                (Swendsen-Wang replica Monte Carlo between adjacent
                temperatures; records no cluster statistics). Multiple modes
                can be alternated with `+`, e.g. `"cmr+houdayer"` round-robins
                each overlap update call. See the
                [overlap moves guide](overlap_moves.md) for derivations and
                recommendations.
            overlap_cluster_mode: Cluster type used inside the overlap move.
                `"wolff"` or `"sw"`.
            overlap_cluster_action: `"update"` to perform the move or
                `"observe"` to record the full graph without acting on replicas.
            overlap_cluster_max_temperature: If set, apply overlap moves only at
                temperatures up to this value (for `"rmc"`, only edges whose two
                temperatures both qualify). Same-temperature cluster moves pay off
                mainly well below T_c (Zhu, Ochoa & Katzgraber 2015), so this
                saves their cost where they barely help. Skipped temperatures
                report no overlap cluster statistics.
            warmup_ratio: Fraction of sweeps discarded as warmup before
                collecting statistics. Default 0.25.
            collect_cluster_stats: If `True`, collect FK cluster size
                distribution and top-4 overlap cluster sizes.
            autocorrelation_max_lag: If set, estimate the integrated
                autocorrelation times `mags2_tau` (and `overlap2_tau` with
                `n_replicas >= 2`) in measured sweeps, using lags up to this value
                (capped at a quarter of the measured sweeps). `overlap2_tau` uses
                q^2 averaged over all replica pairs at each temperature, so moves
                that only relabel replicas do not count as decorrelation.
            autocorrelation_backend: `"ring"` for exact bounded-memory
                accumulation or `"fft"` to retain the full measurement history
                and evaluate autocorrelation with an FFT.
            sequential: If `True`, disable inner-loop parallelism over
                replicas/temperatures. Use when outer-level parallelism over
                disorder realizations already saturates all physical cores.
            equilibration_diagnostic: If `True`, record replica-averaged energy
                and link overlap over log-binned windows for
                [`equilibration_delta`][peapods.Ising.equilibration_delta].

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
        if autocorrelation_backend not in {"ring", "fft"}:
            raise ValueError("autocorrelation_backend must be 'ring' or 'fft'")
        if autocorrelation_backend == "fft" and autocorrelation_max_lag is None:
            raise ValueError(
                "autocorrelation_backend='fft' requires autocorrelation_max_lag"
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
            overlap_cluster_max_temperature=overlap_cluster_max_temperature
            if oci
            else None,
            warmup_ratio=warmup_ratio,
            collect_cluster_stats=collect_cluster_stats,
            autocorrelation_max_lag=autocorrelation_max_lag,
            autocorrelation_backend=autocorrelation_backend,
            sequential=sequential,
            equilibration_diagnostic=equilibration_diagnostic,
            snapshot_interval=snapshot_interval if oci else None,
            collect_physics=collect_physics,
            displacements=displacements,
            block_size=block_size,
        )
        self.mags = result["mags"]
        self.mags2 = result["mags2"]
        self.mags4 = result["mags4"]
        self.energies_avg = result["energies"]
        self.energies2_avg = result["energies2"]
        # [<e^2>] - [<e>]^2 would add the disorder variance of <e>.
        self.energy_variance = result["energy_variance"]

        self.binder_cumulant = 1 - self.mags4 / (3 * self.mags2**2)
        self.heat_capacity = self.n_spins * self.energy_variance / self.temperatures**2

        if "physics" in result:
            self.physics = result["physics"]
            self.heat_capacity = self.physics["heat_capacity"]

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
        """Compute the equilibration diagnostic Δ = e - J²β (N_b/N) (1 - q_l).

        For Gaussian couplings of variance J², integrating by parts over the
        disorder gives the equilibrium identity [<e>] = J²β (N_b/N) (1 - [<q_l>]),
        where e = -H/N is the stored energy (interaction sum per spin), q_l the
        link overlap and N_b/N = `n_neighbors` bonds per spin (Katzgraber,
        Palassini & Young, PRB 63, 184422 (2001)). The identity does not hold for
        bimodal couplings, so Δ need not vanish there even in equilibrium.

        Each checkpoint t averages over the last ⌈t/2⌉ sweeps, i.e. the
        log-binned window [2^(k-1), 2^k) at t = 2^k. From random initial
        states both e and q_l rise toward equilibrium, so Δ approaches zero from
        below; the simulation is considered thermalized once Δ agrees with zero
        within error bars for the last few windows (Zhu, Ochoa & Katzgraber,
        PRL 115, 077201 (2015)).

        Args:
            j_squared: Coupling variance J². 1.0 for `couplings="gaussian"`.

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


class XY:
    """Zero-field XY model on a periodic hypercubic lattice, with signed bonds.

    All physical reductions use float64. Extensive observables are normalized
    by the original lattice volume, including vacant sites. Uniform magnetic
    observables do not measure spin-glass order. See ``docs/xy.md`` for moments,
    disorder averaging, update clocks and the finite-size validation recipe.
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
        occupation=None,
    ):
        self.lattice_shape = tuple(lattice_shape)
        if not self.lattice_shape or any(
            not isinstance(extent, (int, np.integer)) or extent < 3
            for extent in self.lattice_shape
        ):
            raise ValueError("XY hypercubic extents must be integers at least three")
        if neighbor_offsets is not None or geometry is not None:
            raise ValueError("XY currently supports canonical hypercubic lattices only")
        if not isinstance(n_replicas, (int, np.integer)) or n_replicas < 1:
            raise ValueError("n_replicas must be a positive integer")
        self.n_dims = len(self.lattice_shape)
        self.n_neighbors = self.n_dims
        self.n_spins = int(np.prod(self.lattice_shape))
        self.n_replicas = int(n_replicas)
        self.temperatures = _prepare_temperatures(temperatures, np.float64)
        self.n_temps = len(self.temperatures)
        self.seed = seed
        coupling_seed, self._constructor_dynamics_seed = _seed_material(seed)
        coup = _prepare_couplings(
            self.lattice_shape,
            self.n_dims,
            n_disorder,
            couplings,
            coupling_seed,
            np.float64,
        )
        self.n_disorder = 1 if coup.ndim == self.n_dims + 1 else coup.shape[0]
        mask_shape = (self.n_disorder,) + self.lattice_shape
        if occupation is None:
            mask = np.ones(mask_shape, dtype=bool)
        else:
            mask = np.asarray(occupation)
            if mask.dtype != np.bool_:
                raise ValueError("occupation must be a Boolean array")
            if mask.shape == self.lattice_shape:
                mask = np.broadcast_to(mask, mask_shape)
            if mask.shape != mask_shape:
                raise ValueError(
                    "occupation must match the lattice, optionally with a disorder axis"
                )
            mask = np.ascontiguousarray(mask)
        batch = coup.reshape(
            (self.n_disorder,) + self.lattice_shape + (self.n_dims,)
        ).copy()
        for axis in range(self.n_dims):
            batch[..., axis] *= mask & np.roll(mask, -1, axis=axis + 1)
        self.couplings = batch[0] if self.n_disorder == 1 else batch
        self.occupation = mask[0].copy() if self.n_disorder == 1 else mask.copy()
        self._sim = XYSimulation(
            list(self.lattice_shape),
            self.couplings,
            self.temperatures,
            self.n_replicas,
            mask,
            self._constructor_dynamics_seed,
        )

    def reset(self, seed=None):
        """Reset spins, random streams and tempering; retain couplings and masks."""
        self._sim.reset(None if seed is None else _dynamics_seed(seed))

    def sample(
        self,
        n_sweeps,
        sweep_mode="metropolis",
        cluster_update_interval=1,
        cluster_mode="sw",
        cluster_updates=1,
        overrelaxation_sweeps=0,
        pt_interval=None,
        pt_schedule="single_random_edge",
        warmup_ratio=0.25,
        displacements=None,
        vortices=False,
        collect_blocks=False,
        block_size=128,
        autocorrelation_max_lag=None,
        autocorrelation_backend="ring",
        sequential=False,
    ):
        """Sample physical moments, with Metropolis plus embedded SW by default.

        ``n_sweeps`` includes warmup. ``sweep_mode="gibbs"`` uses an exact
        heat-bath (von Mises) local pass. Set ``sweep_mode="none"`` for
        cluster-only sampling, or ``cluster_update_interval=None`` for local
        updates only.
        ``cluster_updates`` is a fixed count per scheduled event. Overrelaxation
        and tempering are off by default. Blocks retain sums and counts over
        ``block_size`` measured sweeps; autocorrelation diagnostics use measured
        energy and m², averaged over replicas at each temperature.

        Returns arrays indexed by temperature (and then direction/displacement).
        ``per_disorder`` retains individual moments, derived quantities and work
        counters; optional ``blocks`` and PT diagnostics are nested there.
        ``angle_vortex_density`` is geometric angle winding on intact plaquettes,
        not frustration-adjusted vorticity or chirality.
        """
        result = self._sim.sample(
            n_sweeps,
            sweep_mode=sweep_mode,
            cluster_update_interval=cluster_update_interval,
            cluster_mode=cluster_mode,
            cluster_updates=cluster_updates,
            overrelaxation_sweeps=overrelaxation_sweeps,
            pt_interval=pt_interval,
            pt_schedule=pt_schedule,
            warmup_ratio=warmup_ratio,
            displacements=displacements,
            vortices=vortices,
            block_size=block_size if collect_blocks else None,
            autocorrelation_max_lag=autocorrelation_max_lag,
            autocorrelation_backend=autocorrelation_backend,
            sequential=sequential,
        )
        self.result = result
        for name in (
            "energies",
            "energies2",
            "mags",
            "mags2",
            "mags4",
            "heat_capacity",
            "binder_cumulant",
            "susceptibility",
            "helicity_modulus",
            "structure_factor_0",
            "structure_factor_min",
            "correlation_length",
            "correlation_length_ratio",
        ):
            setattr(self, name, result[name])
        return result
