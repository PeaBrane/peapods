import numpy as np
from numpy.random import rand, randn
from peapods._core import IsingSimulation


GEOMETRIES = {
    "triangular": [[1, 0], [0, 1], [1, -1]],
    "tri": [[1, 0], [0, 1], [1, -1]],
    "fcc": [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, -1, 0], [1, 0, -1], [0, 1, -1]],
    "bcc": [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1]],
}


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

        if isinstance(couplings, np.ndarray):
            coup = couplings.astype(np.float32)
        else:
            if n_disorder > 1:
                shape = (n_disorder,) + self.lattice_shape + (self.n_neighbors,)
            else:
                shape = self.lattice_shape + (self.n_neighbors,)

            match couplings:
                case "ferro":
                    coup = np.ones(shape, dtype=np.float32)
                case "bimodal":
                    coup = (-1 + 2 * rand(*shape).round()).astype(np.float32)
                case "gaussian":
                    coup = randn(*shape).astype(np.float32)
                case _:
                    raise ValueError("Invalid mode for couplings.")

        self.couplings = coup
        self._sim = IsingSimulation(
            list(lattice_shape),
            coup,
            self.temperatures,
            n_replicas,
            neighbor_offsets,
        )

    def reset(self):
        """Reset all spins to a random configuration."""
        self._sim.reset()

    def sample(
        self,
        n_sweeps,
        sweep_mode="metropolis",
        cluster_update_interval=None,
        cluster_mode="sw",
        pt_interval=None,
        overlap_cluster_update_interval=None,
        overlap_cluster_build_mode="houdayer",
        overlap_cluster_mode="wolff",
        warmup_ratio=0.25,
        collect_csd=False,
        overlap_update_mode="swap",
        collect_top_clusters=False,
    ):
        """Run Monte Carlo sampling and compute observables.

        After sampling, the following attributes are set on the instance:

        - `binder_cumulant` — Binder cumulant per temperature.
        - `heat_capacity` — Heat capacity per temperature.
        - `sg_binder` — Spin glass Binder parameter (only with `n_replicas >= 2`).
        - `fk_csd` — FK cluster size distribution (only with `collect_csd=True`).
        - `top_cluster_sizes` — Average relative sizes of the 4 largest overlap
          clusters per temperature, shape `(n_temps, 4)` (only with
          `collect_top_clusters=True`).

        Args:
            n_sweeps: Total number of Monte Carlo sweeps (including warmup).
            sweep_mode: Single-spin update algorithm. `"metropolis"` or `"gibbs"`.
            cluster_update_interval: If set, perform a cluster update every this
                many sweeps.
            cluster_mode: Cluster algorithm. `"sw"` (Swendsen-Wang) or `"wolff"`.
            pt_interval: If set, attempt parallel tempering swaps every this many
                sweeps.
            overlap_cluster_update_interval: If set, attempt overlap cluster
                moves every this many sweeps. Requires `n_replicas >= 2`.
            overlap_cluster_build_mode: Overlap cluster algorithm. `"houdayer"`,
                `"jorg"`, or `"cmr"`.
            overlap_cluster_mode: Cluster type used inside the overlap move.
                `"wolff"` or `"sw"`.
            warmup_ratio: Fraction of sweeps discarded as warmup before
                collecting statistics. Default 0.25.
            collect_csd: If `True`, collect the Fortuin-Kasteleyn cluster size
                distribution.
            overlap_update_mode: How overlap clusters are applied. `"swap"`
                exchanges spins between replicas; `"free"` independently flips
                each replica (requires `overlap_cluster_build_mode="cmr"`).
            collect_top_clusters: If `True`, collect average relative sizes of
                the 4 largest overlap clusters per temperature.

        Returns:
            Raw results dictionary with keys like `"mags"`, `"energies"`, etc.
        """
        oci = overlap_cluster_update_interval
        result = self._sim.sample(
            n_sweeps,
            sweep_mode,
            cluster_update_interval=cluster_update_interval,
            cluster_mode=cluster_mode if cluster_update_interval else None,
            pt_interval=pt_interval,
            overlap_cluster_update_interval=oci,
            overlap_cluster_build_mode=overlap_cluster_build_mode if oci else None,
            overlap_cluster_mode=overlap_cluster_mode if oci else None,
            warmup_ratio=warmup_ratio,
            collect_csd=collect_csd,
            overlap_update_mode=overlap_update_mode if oci else None,
            collect_top_clusters=collect_top_clusters,
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

        return result

    def get_energies(self):
        """Return the mean energies per temperature from the last sample run."""
        return self.energies_avg
