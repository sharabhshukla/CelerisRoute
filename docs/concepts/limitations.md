# Limitations

What the library does not do, so you can find out here rather than in
production.

## Modelling

**DAGs only.** Cyclic graphs are rejected with a witness cycle. Model cycles
away by expanding states in time or phase, as the
[vessel routing example](../examples/vessel-routing.md) does. Undirected
graphs are not supported.

**Cumulative resources only.** Nonnegative increments and an upper bound.
Time windows, replenishment and minimum consumption (`lower > initial`) are
not expressible as resources; other `sense` values raise
`NotImplementedError`. Time windows can usually be modelled in the state
space instead.

**Sizes.** At most \(2^{32} - 2\) vertices and \(2^{32} - 2\) edges per graph.

## Resource-constrained search

**One path per call.** `max_paths > 1` with `algorithm="rcsp"` raises
`NotImplementedError`. K best paths are available for the unconstrained
algorithm.

**Exponential in the worst case.** There is no label limit. Boost keeps every
label of a solve and compares the labels of a vertex pairwise. Expect trouble
beyond a few million labels (see the
[RCSP benchmark](../performance.md#resource-constrained-shortest-path)); two
loosely bounded resources on long paths get there quickly.

**No time limit outside fleets.** Only `PricingFleet.solve` and `price` have a
time limit and cancellation. `network.solve` and `PricingEngine.solve` run to
completion.

## Performance

**Per-solve overhead in Boost.** `dag_shortest_paths` repeats a depth-first
search on each call, and `r_c_shortest_paths` allocates per-vertex bookkeeping
on each call. Pricing engines limit this by pruning the graph to the vertices
between source and target once.

**Fleets parallelise across vessels only.** One vessel is priced by one
thread. There is no parallelism inside a single solve and no NUMA-aware
placement. All vessels of a fleet use the same algorithm and the same network;
structurally different networks need separate fleets.

**DAG fleets solve one problem per vessel**, also when many vessels share a
source or a target and differ in nothing else. Such vessels are not merged
into one single-source computation.

## Project status

**Pre-release.** The version is 0.1.0 and the API may still change. Wheels are
published on TestPyPI, not yet on PyPI.

**Benchmarks come from one machine**, an Apple M4 desktop. The Linux and
Windows builds are tested for correctness on every commit, but their speed has
not been measured.
