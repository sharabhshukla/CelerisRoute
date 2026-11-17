# Changelog

All notable changes to CelerisRoute are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - 2026-11-17

First pre-release. The project was called CelerisPath (`celerispath`) during
development and was renamed to CelerisRoute (`celerisroute`) before this release.

### Added

- `StateSpace`, `Graph` and `Network`: model a directed acyclic graph whose
  vertices are named states, and compile it into an immutable network.
- Exact shortest paths on DAGs with negative costs, including K best paths.
- Exact resource-constrained shortest paths with any number of cumulative,
  upper-bounded resources.
- `PricingEngine` for repeated solves with dual-adjusted edge costs, returning
  master columns and an exact convergence certificate.
- `BatchPricingEngine` for pricing several subproblems on native threads.
- `PricingFleet` for pricing many vessels on one shared network, with
  per-vessel sources, targets, resource limits, fixed terms, cost offsets and
  forbidden edges; `IndexedDuals` and `PortTimeProductDuals` for dual updates
  without a sparse matrix; time limits and cancellation.
- Optional oneTBB scheduler for fleets (`CELERISROUTE_WITH_TBB`), with a
  `std::thread` scheduler that is always available.
- Support for free-threaded CPython.
- Examples, benchmarks with recorded results, and documentation.
