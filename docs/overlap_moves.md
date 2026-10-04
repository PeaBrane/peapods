# Overlap Moves

Overlap moves act on several replicas that share one set of couplings. Select them with
`overlap_cluster_update_interval` and `overlap_cluster_build_mode`. Join modes with `+`
to alternate them from one overlap call to the next, e.g. `"houd2+rmc"`.

| Mode | Replicas | Move | Detailed balance |
|------|----------|------|------------------|
| `houdayer`, `houd2` | 2 per group, one temperature | Houdayer isoenergetic cluster move | yes |
| `houdN`, N > 2 | N per group | flips all N replicas on balanced sites | **no** (experimental) |
| `pairN` | N per group | Houdayer move for N/2 replica pairs swapped jointly | yes |
| `jorg`, `jorg2` | 2 per group | Jörg stochastic bonds | yes |
| `jorgN` | N per group | Jörg bonds for N/2 replica pairs swapped jointly | yes |
| `cmr` | 2 per group | Chayes–Machta–Redner blue and grey clusters | yes |
| `permN`, 2 ≤ N ≤ 8 | N per group | random replica permutation on clusters where the group disagrees | yes |
| `gpermN`, 2 ≤ N ≤ 8 | N per group | `permN` in the gauge of a random reference site | yes |
| `rmc` | pairs adjacent temperatures within each replica row, ≥ 2 temperatures | Swendsen–Wang replica Monte Carlo | yes |

Same-temperature modes shuffle the replicas of every temperature into groups and update
each group independently. `pair2` is bit-identical to `houd2`, and `jorg2` to `jorg`.

`overlap_cluster_mode` selects the cluster action:

- `"sw"` updates every cluster.
- `"wolff"` updates the cluster of one random seed.

## Symmetry moves at one temperature

Site i carries the vector σ_i = (s_i^1, …, s_i^N) of its replica spins. A bond contributes
E_b(σ_i, σ_j) = −J Σ_r s_i^r s_j^r. Replica permutations and replica sign flips, applied to
both endpoints, leave every bond energy unchanged. Each valid same-temperature move applies
one such involution g on whole clusters. For `pairN` and `jorgN`, g is the product of the
pair swaps (g0 g1)(g2 g3)… of the shuffled group.

### `houd2`, `pairN`: deterministic clusters

Clusters are the connected components of the sites that g moves: q_ab = −1 for Houdayer's
single pair, or any disagreeing pair for `pairN`. Every neighbor of a component is fixed by
g, so applying g on the component conserves the summed energy exactly. The components are
the same before and after the move, so it is rejection-free and reversible.

Two caveats:

- The move conserves the summed energy, so combine it with single-spin updates and
  parallel tempering.
- In three dimensions the negative-overlap sites already percolate. A flip of the dominant
  cluster then mostly relabels replicas, and the union of several pairs percolates even
  more readily.

### `jorg`, `jorgN`, CMR blue bonds: stochastic clusters

The Niedermayer–Kandel–Domany embedding treats "apply g at site i" as an Ising variable. A
bond is activated with probability p = 1 − exp(−β max(0, ΔE)), where
ΔE = E(σ_i, gσ_j) − E(σ_i, σ_j). Clusters are then flipped as in Swendsen–Wang or Wolff.

| g | ΔE | Result |
|---|---|---|
| a single pair swap | 4J s_i^a s_j^a where both sites disagree, else 0 | Jörg's bonds |
| `jorgN` pair product | the same term summed over the pairs disagreeing at both ends | `jorgN` bonds |
| flip of both replicas | 4\|J\| on doubly satisfied bonds, else ≤ 0 | CMR blue bonds |

For `jorgN`, activation happens only when the summed ΔE is positive.

### Why `houdN` with N > 2 is not detailed-balanced

`houdN` negates all replicas on clusters of *balanced* sites (Σ_r s_i^r = 0). That is not
one global symmetry: a neighbor outside the cluster is not fixed by a global flip, so
boundary bonds change energy. For example, take J = 1, σ_i = (+,+,−,−) and
σ_j = (+,+,+,−). Flipping i moves the summed bond energy from −2 to +2, yet the move is
accepted unconditionally.

An exact one-step test on a 2×2 torus misses the summed energy by more than 50 standard
errors. In 3D runs, `houd4` shifts the energy and the Binder ratio far outside the errors
of every valid move. The mode is kept as an experiment; use `pairN` or `permN` for a
valid multi-replica move.

### `permN`: replica permutations

Clusters are the connected components of the sites where the N replicas are not all
equal. Every neighbor of a component has all replicas equal, so any replica permutation
fixes it, and permuting the replicas on a component conserves the summed energy.
`"sw"` draws a uniform permutation for every cluster; `"wolff"` applies a uniform
non-identity permutation to the seed's cluster. Either way the move is a rejection-free
heat bath over the permutations of each cluster. `perm2` is the Houdayer move.

A permutation of a component factors into pair swaps on nested disagreement clusters, so
`permN` reaches no states that pair moves cannot.

### `gpermN`: gauged permutations

Pick a uniformly random reference site with replica pattern P = (s^1, …, s^N) and work
with the gauged spins t^r = P_r s^r. Sites whose pattern is ±P are fixed; clusters are
the components of the other sites, and the move permutes the gauged spins. The probability
of choosing a given gauge, the fraction of sites with pattern ±P, is unchanged by the
move, so detailed balance holds.

For N = 2, the clusters are the sites whose overlap differs from the reference's. That
is the minority overlap sign with probability equal to the majority fraction, an exact
randomized form of the minority-sign rule of Zhu, Ochoa & Katzgraber (2015). In 3D at
L = 8 and T = 0.5–0.8, the largest `gperm2` cluster covers 11–12 % of the lattice, against
48 % for `houd2`.

## Replica Monte Carlo across temperatures (`rmc`)

`rmc` pairs the systems at adjacent temperature slots t and t + 1 within each replica row
(Swendsen & Wang 1986). With τ_i = s_i^a s_i^b held fixed, the pair action is an Ising
model with couplings J(β_a + β_b τ_i τ_j).

- **Move.** Negating both replicas on a connected τ-domain, of either sign, changes only
  the domain walls, whose coupling is J(β_a − β_b). The flip is accepted with the
  Metropolis probability of that change.
- **`"sw"`.** Neighboring domains have opposite τ. All τ = +1 domains are therefore
  decided together, then all τ = −1 domains against the updated walls.
- **`"wolff"`.** Proposes the domain of one random site.
- **Order.** Even edges (0,1), (2,3), … are updated before odd edges.
- **Relation to PT.** Flipping every τ = −1 domain is exactly a parallel-tempering swap,
  so `rmc` generalizes PT.

`rmc` updates every replica row independently and needs at least two temperatures. It
moves configuration content between temperature slots without permuting `system_ids`, so
PT round-trip counters do not see its transport. It records no cluster statistics or
snapshots and is rejected with `overlap_cluster_action="observe"`.

## Which move to use

The measurements below use ±J couplings, 4 replicas, Metropolis plus full-ladder parallel
tempering every sweep, and 32 disorder samples. `validation/overlap_moves.py` reproduces
them. Cost is the CPU time per sweep relative to PT alone.

**Equilibrium.** τ is the integrated autocorrelation time, in sweeps, of the
replica-symmetric Q2 (the mean of q_ab² over all replica pairs), from batch means of
1000 sweeps. The table gives cost × τ relative to PT alone; below 1 the move pays.

| System, temperature | PT alone τ | `houd2` | `gperm2` | `rmc` every 4th sweep | `perm4` |
|---|---|---|---|---|---|
| 2D L=16, 16 temperatures, T=0.30 | 14 | 1.5 | 1.5 | 1.4 | 1.6 |
| 3D L=8, 16 temperatures, T=0.50 | 25 | 2.3 | 1.6 | 2.1 | 1.9 |
| 3D L=8, 16 temperatures, T=0.80 | 15 | 2.4 | 2.4 | 2.1 | 2.5 |

**Equilibration from a random start.** The speedup is the CPU-time ratio at which a move
reaches the energy and Q2 that PT alone reaches after 65 536 sweeps.

| System, temperature | `houd2` or `gperm2`, every 4th sweep | `rmc`, every 4th sweep |
|---|---|---|
| 2D L=32, 24 temperatures, T=0.20 | ≥ 6× for Q2, about 25× for the energy | similar |
| 3D L=12, 24 temperatures, T=0.40 | about 1× | about 1× |
| 3D L=16, 32 temperatures, T=0.40 | about 1.5× | about 1.1× |
| 3D L=16, 32 temperatures, T=0.60 | about 1× | about 1× |

- **At small sizes, PT alone is hard to beat.** No overlap move pays in equilibrium at
  2D L=16 or 3D L=8.
- **Pair moves pay at low temperature on larger lattices.** Apply `gperm2` or `houd2`
  every few sweeps (`overlap_cluster_update_interval=4`) and only below T_c
  (`overlap_cluster_max_temperature`; Zhu, Ochoa & Katzgraber, PRL 115, 077201 (2015)).
  The gain is large in 2D and grows with L in 3D. For `rmc` the cutoff selects edges whose
  two temperatures both qualify.
- **Prefer `gperm2` to `houd2`.** It is never worse, and at 3D L=8, T=0.50 it halves τ.
- **Moves on more than two replicas do not pay.** `pairN`, `jorgN`, `permN` and `gpermN`
  with N > 2 are valid, but their clusters percolate more readily: at 3D L=8 the largest
  `perm4` cluster covers 88 % of the lattice at every temperature.
- **Measure τ on replica-symmetric observables.** A fixed replica pair overstates the gain
  of same-temperature moves, because relabeling replicas decorrelates the pair without
  moving the ensemble. The built-in `overlap2_tau` therefore uses Q2. Its automatic
  window can still undercount under full-ladder PT, where the autocorrelation has a small
  first lag and a long tail. For benchmarks, record `overlap2_all_pairs` every sweep
  (`block_size=1`) and use batch means.

## Statistics

`top_cluster_sizes` averages over the replica groups that publish a graph: n_replicas / N
groups per temperature for group size N. `overlap_cluster_action="observe"` supports only
`houdayer`, `jorg` and `cmr`.

## Validation

Every valid mode passes an exact one-step stationarity test, on tiny tori and in both
cluster modes, with and without statistics. Replicas are drawn exactly from the Boltzmann
law, one move is applied, and state histograms, energies and pairwise overlap histograms
are compared with the product law. The test catches:

- wrong bond probabilities;
- the naive single-pair `jorgN` rule;
- wrong RMC couplings;
- deciding all RMC domains simultaneously;
- the balanced-site `houd4` move;
- dropping the gauge when `gpermN` writes spins back.

## References

- J. Houdayer, Eur. Phys. J. B 22, 479 (2001).
- T. Jörg, Prog. Theor. Phys. Suppl. 157, 349 (2005).
- F. Niedermayer, Phys. Rev. Lett. 61, 2026 (1988).
- D. Kandel and E. Domany, Phys. Rev. B 43, 8539 (1991).
- R. H. Swendsen and J.-S. Wang, Phys. Rev. Lett. 57, 2607 (1986).
- Z. Zhu, A. J. Ochoa and H. G. Katzgraber, Phys. Rev. Lett. 115, 077201 (2015).
- J.-S. Wang and R. H. Swendsen, Prog. Theor. Phys. Suppl. 157, 317 (2005).
