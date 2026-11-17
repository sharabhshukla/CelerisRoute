# Conventions and guarantees

This page is the fine print: the sign conventions, what "exact" means, and
what is safe to do from several threads.

## Reduced costs and dual signs

For a minimisation master problem with columns \(x_P \ge 0\), cost \(c_P\) and
constraint coefficients \(a_P\), a pricing engine computes

\[
\bar c_P = c_P - \pi^\top a_P, \qquad
c_P = \text{fixed\_cost} + \sum_{e \in P} c_e, \qquad
a_P = \text{fixed\_coefficients} + \sum_{e \in P} A_{\cdot e}
\]

\(\pi\) must be the usual LP duals of a minimisation problem: the change of
the optimal objective per unit of right-hand side.

| master row | sign of its dual |
|---|---|
| `>=` | \(\pi_k \ge 0\) |
| `<=` | \(\pi_k \le 0\) |
| `=` | free |

* Gurobi's `Pi` has this sign for a minimisation model.
* `scipy.optimize.linprog` accepts only `A_ub x <= b_ub`. If you pass a `>=`
  row as `-a x <= -b`, negate that row's `ineqlin.marginals` before pricing.
* For a maximisation master, minimise the negated objective instead.

For a fleet, with \(B_v\) the optional coefficients of vessel \(v\) and
\(o_v\) its optional cost offsets, the reduced cost of edge \(e\) is

\[
\bar c_{v,e} = c_e + o_{v,e} - \sum_k \pi_k \,(A_{ke} + B_{v,ke})
\]

## The pricing certificate

`PricingResult.best_reduced_cost` is the exact minimum reduced cost over all
feasible source-target paths, whether or not a column is returned.

* If `best_reduced_cost >= -tolerance`, no path prices out, and column
  generation has converged for that subproblem.
* `negative_only=True` (the default) returns only columns with
  `reduced_cost < -tolerance`.
* With one convexity row `= 1` per subproblem,
  \(z_\text{RMP} + \sum_s \min(0, \bar c^{\,*}_s)\) is a valid lower bound on
  the master optimum in every iteration.

`FleetResult.converged` is the same statement for a whole fleet: every vessel
was priced exactly and none has a reduced cost below `-tolerance`. It is false
whenever a vessel was interrupted by a time limit or a cancellation.

!!! warning "Exact only with `dominance_tolerance=0`"
    That is the default. A positive `dominance_tolerance` prunes more labels,
    turns the resource-constrained search into a heuristic, and a fleet then
    never reports `converged`.

## What "exact" covers

**Negative costs.** The DAG algorithm relaxes edges in topological order, so
negative costs are handled without restriction.

**Negative costs with resources.** Labels are processed in the order
(topological rank, cost). On a DAG every label of a vertex therefore exists
before any of them is extended, which keeps the algorithm exact with negative
costs and negative reduced costs.

**Dominance.** A label dominates another at the same vertex when its cost and
every bounded resource are no larger. Of two equal labels one survives.

**Completion bounds.** A new label is discarded at creation when it cannot
reach the target within the bounds: its consumption plus the least
consumption of any path to the target (precomputed backwards, per resource)
exceeds the upper bound. This never removes a feasible path. The test carries
a relative margin of `1e-9` so that rounding cannot make it stricter than the
per-edge feasibility test. These labels are counted in
`statistics.labels_infeasible`.

**Feasibility.** Comparisons are exact floating-point comparisons: a total
equal to `upper` is feasible. If increments are not exactly representable and
paths can hit a bound exactly, scale to integers or add slack to `upper`.

**Determinism.** Ties between equally good paths are broken
deterministically. The same input gives the same path on every run, for every
thread count and every scheduler.

## Threads

| object | from several threads |
|---|---|
| `Network` | immutable, share freely; concurrent `solve` calls use separate workspaces |
| `PricingEngine` | owns its workspace; concurrent calls on one engine are serialised |
| `BatchPricingEngine` | starts at most `min(threads, len(engines))` native threads; each engine is solved by one thread |
| `PricingFleet` | calls that change or price it are serialised; `cancel()` never blocks and may come from any thread |

The GIL is released during every native solve. The extension also declares
itself safe for free-threaded CPython.

## How this is tested

Every solver is compared with an independent pure-Python reference,
brute-force path enumeration, on random graphs. The fleet tests check every
scheduler and cost-update mode against the same enumeration and against one
`PricingEngine` per vessel, and they cover time limits, cancellation from
another thread and concurrent calls on one fleet.
