# PeaPods

A Python library for simulating Ising spin systems with modern Monte Carlo methods.
The core simulation loop is written in Rust (via PyO3) for performance, with a thin Python wrapper for ease of use.

<div style="text-align:center">
  <img src="./docs/csd.png" alt="CSD" style="width:70%"/>
</div>

## Features

- Ising ferromagnets and spin glasses on arbitrary-dimensional hypercubic lattices
- Arbitrary, bimodal (±J), or Gaussian coupling distributions
- Multiple replicas with overlap statistics for spin glass order parameters

The following algorithms are currently supported:

- Single-spin flips ([Metropolis](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) and [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling))
- [Swendsen-Wang cluster updates](https://en.wikipedia.org/wiki/Swendsen%E2%80%93Wang_algorithm)
- [Wolff cluster updates](https://en.wikipedia.org/wiki/Wolff_algorithm)
- [Parallel tempering](https://en.wikipedia.org/wiki/Parallel_tempering)
- [Houdayer isoenergetic cluster move](https://arxiv.org/abs/cond-mat/0101116) (replica cluster move for spin glasses)

Planned:

- Cluster updates for frustrated systems (e.g. [KBD algorithm](https://en.wikipedia.org/wiki/KBD_algorithm))
- [Jorg move](https://arxiv.org/abs/cond-mat/0410328) (Houdayer + SW-style bond breaking within clusters)

## Quickstart

```python
import numpy as np
from peapods import Ising

# 2D ferromagnet with cluster updates and parallel tempering
model = Ising((32, 32), temperatures=np.linspace(1.5, 3.0, 32), n_replicas=2)
model.sample(n_sweeps=5000, sweep_mode="metropolis",
             cluster_update_interval=1, pt_interval=1)
print(model.binder_cumulant)

# 3D spin glass with Houdayer ICM
sg = Ising((8, 8, 8), couplings="bimodal",
           temperatures=np.linspace(0.8, 1.4, 24), n_replicas=4)
sg.sample(n_sweeps=10000, sweep_mode="metropolis",
          pt_interval=1, houdayer_interval=1)
print(sg.sg_binder)
```

For a more complete example, check out [example.py](example.py).

## Installation

```bash
pip install peapods
```

Pre-built wheels are available for Linux (x86_64, aarch64), macOS (Intel, Apple Silicon), and Windows (x86_64).

## Building from source

Requires a Rust toolchain and [maturin](https://www.maturin.rs/):

```bash
maturin develop --release
```

## Dependencies

- numpy
- matplotlib (for plotting)
