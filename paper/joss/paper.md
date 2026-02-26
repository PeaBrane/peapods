---
title: "peapods: A Rust-Accelerated Monte Carlo Package for Ising Spin Systems on Arbitrary Lattices"
tags:
  - Python
  - Rust
  - Monte Carlo
  - Ising model
  - spin glass
  - statistical physics
authors:
  - name: Yan Ru Pei
    orcid: 0000-0002-7401-3080
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 25 February 2026
bibliography: paper.bib
---

# Summary

`peapods` is a Python package for Monte Carlo simulation of Ising spin
systems on arbitrary Bravais lattices with arbitrary coupling distributions.
It exposes a high-level Python API and a command-line interface (`peapods
simulate`, `peapods bench`) while delegating all numerically intensive work to
a Rust core linked via PyO3 [@PyO3; @Maturin]. The package supports
Metropolis [@Metropolis1953] and Gibbs single-spin-flip updates,
Swendsen--Wang [@Swendsen1987] and Wolff [@Wolff1989] cluster algorithms,
replica-exchange parallel tempering [@Hukushima1996], and the Houdayer
[@Houdayer2001], Jörg [@Jorg2006], and Chayes--Machta--Redner [@CMR2000]
overlap cluster moves for spin glasses. It is pip-installable from PyPI and
available as a Rust crate on crates.io.

# Statement of need

Spin glasses --- disordered magnetic systems with competing ferromagnetic and
antiferromagnetic interactions --- remain one of the most active areas of
statistical physics. Fundamental questions about replica symmetry breaking,
ultrametricity, and the nature of the low-temperature phase in finite
dimensions are still open and can currently only be probed through large-scale
simulation [@Edwards1975]. Effective simulation of spin glasses requires
combining cluster algorithms with parallel tempering and replica overlap moves,
often on non-cubic lattice geometries with random couplings.

Despite this need, no existing pip-installable package offers this combination
of algorithms. Researchers typically rely on private C or Fortran codes, or on
heavyweight frameworks that are difficult to install and extend. `peapods`
fills this gap by providing a batteries-included, easy-to-install package that
covers the full algorithmic toolkit needed for modern spin-glass research,
while remaining simple enough for pedagogical use on clean ferromagnetic
systems.

# State of the field

Several existing tools address parts of this space. ALPS `spinmc` [@Bauer2011]
provides a mature C++ Monte Carlo application but does not support spin-glass
couplings, requires a heavy build toolchain, and has seen limited maintenance.
`tamc` is a Rust-based Monte Carlo CLI for classical spin systems but provides
no cluster algorithms and no Python API. `SpinMonteCarlo.jl` offers cluster
updates in Julia but lacks parallel tempering, spin-glass support, and Python
interoperability. Numerous pedagogical Python repositories implement
Metropolis-only sampling on square lattices but do not scale to research-grade
simulations. To our knowledge, no existing JOSS paper covers a general-purpose
Ising Monte Carlo package with the combination of cluster, tempering, and
overlap algorithms on arbitrary lattice geometries.

# Software design

The computational core is a Rust crate (`peapods_core`) compiled into a Python
extension module via PyO3 and Maturin. Lattice geometry is specified by
user-provided neighbor offset vectors, enabling simulation on any Bravais
lattice (square, triangular, FCC, BCC, or custom). Couplings are stored in a
forward-only scheme to halve memory usage, and neighbor indices are
precomputed at construction time for fast access during sweeps. Replica-level
parallelism is implemented with Rayon [@Rayon], giving each temperature replica
a disjoint spin slice and an independent `Xoshiro256**` random number
generator.

The Python layer is a thin wrapper (`Ising` class) that handles model
construction, temperature grid setup, and post-processing of observables
(Binder cumulant [@Binder1981], heat capacity, spin-glass order parameter).
A CLI (`peapods simulate`) exposes the full functionality for scriptless
batch usage, and `peapods bench` provides built-in benchmarking.

# Research applications

`peapods` has been validated against exact analytical results for the 2D Ising
model on square [@Onsager1944] and triangular lattices, and against published
Monte Carlo estimates of the spin-glass critical temperature on six
lattice/coupling combinations in two and three dimensions. These validations,
along with a detailed description of the algorithms and their implementation,
are presented in an accompanying preprint [@PeapodsPreprint].

# AI usage disclosure

Claude (Anthropic) assisted with code generation, documentation, and paper
drafting. All algorithm selection, architecture decisions, and validation were
performed by the author.

# References
