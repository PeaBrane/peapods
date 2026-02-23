import numpy as np
from matplotlib import pyplot as plt
from peapods.spin_models import Ising

plt.rcParams["axes.edgecolor"] = "#0366d6"
plt.rcParams["xtick.color"] = "#0366d6"
plt.rcParams["ytick.color"] = "#0366d6"
plt.rcParams["text.color"] = "#0366d6"
plt.rcParams["axes.titlecolor"] = "#0366d6"

temperatures = np.geomspace(0.1, 10, 32)
ising = Ising(lattice_shape=(32, 32), temperatures=temperatures)
result = ising.sample(
    n_sweeps=2**16,
    warmup_ratio=0.1,
    cluster_update_interval=1,
    pt_interval=1,
    collect_csd=True,
)

n_spins = 32**2
bins = np.arange(1, n_spins + 2)
temp_indices = range(17, 27, 2)

for t in temp_indices:
    sizes = ising.csd_sizes[t]
    if len(sizes) == 0:
        continue
    counts, _ = np.histogram(sizes, bins=bins)
    pdf = counts / counts.sum()
    pdf[pdf == 0] = np.nan
    plt.plot(np.arange(1, n_spins + 1), pdf, label=f"$T = {temperatures[t]:.3f}$")

plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-7, 1)
plt.legend()
plt.title("cluster size distributions of a 32 x 32 Ising ferromagnet")

out_path = __import__("pathlib").Path(__file__).resolve().parent / "csd.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
print(f"Saved to {out_path}")
