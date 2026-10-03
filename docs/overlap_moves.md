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
of every valid move. The mode is kept as an experiment; use `pairN` for a valid
multi-replica Houdayer move.

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

The benchmark below uses ±J couplings, 4 replicas, and Metropolis plus full-ladder
parallel tempering every sweep, with 16 temperatures.

It predates a fix of 2026-10-03: the full-ladder schedule used to flip the order of its
even and odd passes every event, so consecutive passes could undo each other, and
walkers crossed the ladder 10–40× more slowly than they now do (none at all when every
swap is accepted). The "PT alone" times below are therefore upper bounds.

A first re-measurement after the fix, for 2D L=16 with 16 temperatures from 0.3 to 1.5
and 4 replicas (12 disorder samples, τ of Q2 from batch means), gave τ ≈ 8 sweeps for
PT alone at T=0.30 instead of 236. `rmc` with `"wolff"` shortened τ by about 1.5× at 4.7×
the cost per sweep (every 4th sweep: 1.9× the cost), so PT alone was the cheaper choice
there. The 3D rows have not been re-measured.

- **Observable.** τ is the integrated autocorrelation time of the replica-symmetric
  Q2 = mean over all replica pairs of q_ab², in sweeps (median over 16–32 disorder
  samples).
- **Cost.** Cost is τ × ms/sweep relative to PT alone. The T=0.50 row reuses the
  timings of the 0.8–2.0 ladder.

| System, temperature | PT alone τ | Houdayer family (`houd2`, `jorg`, `pair4`, `jorg4`, `cmr`) τ / cost | `rmc` (`"wolff"`) τ / cost |
|---|---|---|---|
| 2D L=16, T=0.30 | 236 | 83–151 / 0.9–2.3 | 1.0 / 0.015 |
| 2D L=16, T=0.40 | 30 | 11–16 / 0.8–1.6 | 1.0 / 0.12 |
| 3D L=8, T=0.50 | 9.6 | 5.3–8.7 / 1.5–3.7 | 2.2 / 1.2 |
| 3D L=8, T=0.80 | 7.1 | 5.5–7.2 / 2.0–4.1 | 3.2 / 2.3 (every 4th sweep: 3.8 / 1.1) |
| 3D L=8, T=0.90 | 2.1 | 1.7–2.2 / 1.9–3.9 | 1.6 / 3.7 (every 4th sweep: 1.4 / 1.3) |

- **`rmc` with the `"wolff"` cluster mode** is the only move that clearly pays for itself.
  In 2D at low temperature it decorrelates Q2 in about one sweep, where PT alone needs
  hundreds. Applying it every few sweeps (`overlap_cluster_update_interval=4`) keeps most
  of the gain at a fraction of the cost. In 3D at L ≤ 8 it breaks roughly even with PT
  alone.
- **The same-temperature moves** (`houd2`, `jorg`, `cmr`) cost 2–5× a PT sweep. They
  shorten τ by at most about 3× in 2D and by less in 3D, so per unit of CPU they are at
  best on par with PT alone at these sizes. Houdayer (2001) reports much larger gains
  in 2D for L = 100 at T = 0.1 with 32 replicas per temperature, a regime this
  benchmark does not cover.
- **Restrict same-temperature moves to low temperatures** with
  `overlap_cluster_max_temperature` (Zhu, Ochoa & Katzgraber, PRL 115, 077201 (2015)):
  their clusters percolate, and the moves stop helping, well below T_c in 3D, so
  applying them only there removes most of their cost. For `rmc` the cutoff selects
  edges whose two temperatures both qualify.
- **`pairN` and `jorgN`** are valid but no better than `houd2` and `jorg`: their clusters
  are unions over several pairs and percolate even more readily.
- **Measure τ on replica-symmetric observables.** An autocorrelation time measured on
  fixed replica pairs overstates the gain of same-temperature moves, because relabeling
  replicas decorrelates a fixed pair without moving the ensemble. In the 2D run above at
  T=0.30, a fixed-pair estimate averaged 39 sweeps for `houd2` against 240 for Q2, while
  PT alone gave 293 against 555. The built-in `overlap2_tau` therefore uses Q2.

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
- the balanced-site `houd4` move.

## References

- J. Houdayer, Eur. Phys. J. B 22, 479 (2001).
- T. Jörg, Prog. Theor. Phys. Suppl. 157, 349 (2005).
- F. Niedermayer, Phys. Rev. Lett. 61, 2026 (1988).
- D. Kandel and E. Domany, Phys. Rev. B 43, 8539 (1991).
- R. H. Swendsen and J.-S. Wang, Phys. Rev. Lett. 57, 2607 (1986).
- J.-S. Wang and R. H. Swendsen, Prog. Theor. Phys. Suppl. 157, 317 (2005).
