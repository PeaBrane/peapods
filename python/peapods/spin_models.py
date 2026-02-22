import numpy as np
from numpy.random import rand, randn
from peapods._core import IsingSimulation


class Ising:
    def __init__(
        self, lattice_shape, couplings="ferro", temperatures=np.geomspace(0.1, 10, 32)
    ):
        self.lattice_shape = tuple(lattice_shape)
        self.n_dims = len(lattice_shape)
        self.temperatures = temperatures.copy().astype(np.float32)
        self.n_temps = len(temperatures)

        match couplings:
            case "ferro":
                coup = np.ones(lattice_shape + (self.n_dims,), dtype=np.float32)
            case "bimodal":
                coup = (-1 + 2 * rand(*lattice_shape, self.n_dims).round()).astype(
                    np.float32
                )
            case "gaussian":
                coup = randn(*lattice_shape, self.n_dims).astype(np.float32)
            case _:
                raise ValueError("Invalid mode for couplings.")

        self.couplings = coup
        self._sim = IsingSimulation(list(lattice_shape), coup, self.temperatures)

    def reset(self):
        self._sim.reset()

    def sample(
        self,
        n_sweeps,
        sweep_mode="metropolis",
        cluster_update_interval=None,
        cluster_mode="sw",
        pt_interval=None,
        warmup_ratio=0.25,
    ):
        result = self._sim.sample(
            n_sweeps,
            sweep_mode,
            cluster_update_interval=cluster_update_interval,
            cluster_mode=cluster_mode if cluster_update_interval else None,
            pt_interval=pt_interval,
            warmup_ratio=warmup_ratio,
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

    def get_energies(self):
        return self.energies_avg


class IsingEnsemble:
    def __init__(self, lattice_shape, n_ensemble=2, **kwargs):
        self.n_ensemble = n_ensemble
        self.ising_ensemble = [
            Ising(lattice_shape, **kwargs) for _ in range(n_ensemble)
        ]

    def sample(self, n_sweeps, **kwargs):
        for ising in self.ising_ensemble:
            ising.sample(n_sweeps, **kwargs)

    def get_energies(self):
        energies_ensemble = np.array(
            [ising.get_energies() for ising in self.ising_ensemble]
        )
        return energies_ensemble.mean(0)
