# PeaPods: A python library for simulating spin systems

The goal of this project is to build a Python library for simulating spin systems with modern efficient Monte-Carlo methods. The implementation aims to balance between performance and simplicity. 

## Description

Currently, this project is at the very early stages, and only supports simulating Ising ferromagnets. Development for other spin classes (Potts, clock, and O(N) models) are planned, including their disordered and quantum counterparts.

The following algoritms are currently supported:

- Single-spin flips
- [Cluster updates (only Wolff)](https://en.wikipedia.org/wiki/Wolff_algorithm)
- [Parallel tempering](https://en.wikipedia.org/wiki/Parallel_tempering)

The following algorithms are planned:

- Cluster updates for frustrated spin systems (e.g. [KBD algorithm](https://en.wikipedia.org/wiki/KBD_algorithm#:~:text=The%20KBD%20algorithm%20is%20an,algorithm%20more%20efficient%20in%20comparison.))
- Replica cluster moves

## Dependencies (numba)

The required dependencies should come with any standard Python installations. However, it is highly recommended that you install Numba for efficient simulation. Simply create a new python environment and do:

`pip install numba`

## Contribution

This is an open-source project, so everyone is welcomed to contribute! 

Please open an issue if you spotted a bug or suggest any feature enhancements. Submit a pull request if appropriate. Alternatively, contact me at yanrpei@gmail.com if you wish to be a contributor to this project.