import numpy as np
from matplotlib import pyplot as plt

from peapods import Ising

temperatures = np.geomspace(0.1, 10, 32)
ising = Ising(lattice_shape=(32, 32), temperatures=temperatures)
results = ising.sample(
    n_sweeps=2**12, warmup_ratio=0.25, cluster_update_interval=2**3, pt_interval=2**3
)

plt.plot(temperatures, results["energies"])
plt.xlabel("Temperature")
plt.ylabel("Energy per spin")
plt.title("Energy vs temperature for a 32x32 Ising ferromagnet")
plt.show()
