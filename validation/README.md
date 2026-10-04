# Validation

Checks of peapods against published results: bounded scripts that run in CI and
longer reproductions run offline, with their summaries and figures committed.

| Result | Reference | Where |
|---|---|---|
| 3D ±J spin glass: ξ/L and U4 crossings near T_c | Baity-Jesi et al. (Janus), PRB 88, 224416 (2013) | [`spin_glass_3d/`](spin_glass_3d/) |
| 2D XY at β = 1.1199: stiffness, ξ/L and χ at L = 16, 32 | Hasenbusch, J. Phys. A 38, 5869 (2005), Table 1 | [`xy_finite_size.py`](xy_finite_size.py) (CI) |
| Ising Binder crossings at known T_c: square (exact), triangular (exact), simple cubic, BCC, FCC | Onsager; standard series and Monte Carlo values | [`tests/binder_crossings.py`](../tests/binder_crossings.py) (CI) |
| 3D ±J spin glass Binder crossings with each overlap move (Houdayer, Jörg, CMR, replica Monte Carlo) | — | [`tests/spin_glass_crossings.py`](../tests/spin_glass_crossings.py) (CI) |

[`plot/`](plot/) holds visualization scripts: the README banner
([`xy_quench_banner.py`](plot/xy_quench_banner.py)), overlap-cluster snapshots
([`cluster_snapshots.py`](plot/cluster_snapshots.py)) and the CMR blue-cluster size
distribution of the 2D ±J spin glass ([`overlap_csd.py`](plot/overlap_csd.py), after
Pei and Di Ventra, arXiv:2105.01188).
