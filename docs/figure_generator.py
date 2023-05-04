from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent))

import importlib

import numpy as np
from matplotlib import pyplot as plt

import utils
importlib.reload(utils)

import sweeps
importlib.reload(sweeps)

import spin_models
importlib.reload(spin_models)
from spin_models import IsingEnsemble

# set colors to blue
plt.rcParams['axes.edgecolor'] = '#0366d6'
plt.rcParams['xtick.color'] = '#0366d6'
plt.rcParams['ytick.color'] = '#0366d6'
plt.rcParams['text.color'] = '#0366d6'
plt.rcParams['axes.titlecolor'] = '#0366d6'

# ensemble simulation
temperatures = np.geomspace(0.1, 10, 32)
ising = IsingEnsemble(lattice_shape=(32, 32),
                      n_ensemble=32,
                      temperatures=temperatures)
ising.sample(n_sweeps=2**24,
             warmup_ratio=0.1,
             cluster_update_interval=2**3,
             pt_interval=2**3)

# plot csds
csds = ising.get_csds()[17:27:2].T
csds[csds == 0] = np.nan

plt.plot(np.arange(1, 32**2+1), csds)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-7, 1)
plt.legend([f"$T = {temp:.3f}$" for temp in temperatures[17:27:2]])
plt.title("cluster size distributions of a 32 x 32 Ising ferromagnet")
plt.savefig("csd.png", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
