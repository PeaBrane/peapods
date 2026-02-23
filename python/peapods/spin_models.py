import numpy as np
from numpy.random import rand, randn
from peapods._core import IsingSimulation


class Ising:
    def __init__(
        self,
        lattice_shape,
        couplings="ferro",
        temperatures=np.geomspace(0.1, 10, 32),
        n_replicas=1,
        n_disorder=1,
    ):
        self.lattice_shape = tuple(lattice_shape)
        self.n_dims = len(lattice_shape)
        self.temperatures = temperatures.copy().astype(np.float32)
        self.n_temps = len(temperatures)
        self.n_replicas = n_replicas
        self.n_disorder = n_disorder

        if isinstance(couplings, np.ndarray):
            coup = couplings.astype(np.float32)
        else:
            if n_disorder > 1:
                shape = (n_disorder,) + self.lattice_shape + (self.n_dims,)
            else:
                shape = self.lattice_shape + (self.n_dims,)

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
            list(lattice_shape), coup, self.temperatures, n_replicas
        )

    def reset(self):
        self._sim.reset()

    def sample(
        self,
        n_sweeps,
        sweep_mode="metropolis",
        cluster_update_interval=None,
        cluster_mode="sw",
        pt_interval=None,
        houdayer_interval=None,
        houdayer_mode="houdayer",
        warmup_ratio=0.25,
        collect_csd=False,
    ):
        result = self._sim.sample(
            n_sweeps,
            sweep_mode,
            cluster_update_interval=cluster_update_interval,
            cluster_mode=cluster_mode if cluster_update_interval else None,
            pt_interval=pt_interval,
            houdayer_interval=houdayer_interval,
            houdayer_mode=houdayer_mode if houdayer_interval else None,
            warmup_ratio=warmup_ratio,
            collect_csd=collect_csd,
        )
        self.mags = result["mags"]
        self.mags2 = result["mags2"]
        self.mags4 = result["mags4"]
        self.energies_avg = result["energies"]
        self.energies2_avg = result["energies2"]

        self.binder_cumulant = 1 - self.mags4 / (3 * self.mags2**2)
        self.heat_capacity = (
            self.energies2_avg - self.energies_avg**2
        ) / self.temperatures**2

        if "overlap2" in result:
            self.overlap = result["overlap"]
            self.overlap2 = result["overlap2"]
            self.overlap4 = result["overlap4"]
            self.sg_binder = 1 - self.overlap4 / (3 * self.overlap2**2)

        if "fk_csd" in result:
            self.fk_csd = result["fk_csd"]

        return result

    def get_energies(self):
        return self.energies_avg
