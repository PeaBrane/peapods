# Project: Peapods

- Build Rust: `VIRTUAL_ENV=.venv .venv/bin/maturin develop --release`

## Replica Cluster Moves

### Houdayer ICM (2001)
- Pairs replicas at the **same temperature**, identifies sites where spins disagree (negative overlap)
- Grows a connected component on the negative-overlap subgraph (prob=1, deterministic BFS)
- Exchanges that cluster between the two replicas
- **Isoenergetic**: preserves each replica's energy exactly → acceptance ratio = 1, rejection-free
- Temperature-independent cluster growth — works even at very low T where thermal moves freeze
- Studied 2D ±J EA (T_c = 0), equilibrated 100² down to T = 0.1
- Validated via autocorrelation times of energy, not Binder crossings

### Jorg Move (2004)
- Refinement of Houdayer for 3D where Houdayer clusters can span the system
- Takes each Houdayer cluster and further splits it using SW-style bond activation: open bonds with prob 1 - exp(-4β|J|)
- This is the same bond probability as CMR "blue bonds" — may be equivalent in measure to Redner-Machta-Chayes blue clusters restricted to the negative-overlap subgraph
- Produces smaller sub-clusters → better decorrelation in 3D
- Validated by comparing autocorrelation times at ~0.8 T_c across algorithms

### Zhu et al. (2015) — arxiv:1501.05630
- Claims to extend Houdayer ICM to arbitrary dimension and non-regular topologies (e.g. D-Wave Chimera)
- Algorithmically the same as Houdayer; main content is benchmarks across dimensions
- Proposes N/2 trick: if the negative-overlap cluster covers > N/2 sites, globally flip one replica (Z₂ gauge symmetry) so the overlap region becomes the smaller complement, then exchange that
- **Caveat**: the N/2 trick likely violates detailed balance — conditioning the global flip on cluster size makes the move non-reversible (forward path triggers the flip but reverse path may not, so transition probabilities are asymmetric)
- An unconditional Z₂ flip as a separate MCMC move would be fine, but that's not what the paper does

### General Notes
- All replica cluster moves require ≥ 2 replicas at each temperature
- Standard validation: measure autocorrelation time τ of energy/overlap with vs without ICM
- 2D EA spin glass: T_c = 0, correlation length diverges as exp(2J/T)
- 3D bimodal EA: T_c ≈ 1.102 (Baity-Jesi et al. 2013)
- Gaussian EA in 3D: T_c ≈ 0.95 (same universality class, different non-universal T_c)
