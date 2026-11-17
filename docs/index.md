---
hide:
  - navigation
  - toc
---

<div class="cr-hero" markdown>

# CelerisRoute

<p class="cr-tagline">
Fast shortest paths on state-space DAGs, built for the pricing step of column
generation. A C++20 core on the Boost Graph Library, with an API you can read.
</p>

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[Examples](examples/index.md){ .md-button }
[API reference](reference/index.md){ .md-button }

</div>

```python
import celerisroute as cr

states = cr.StateSpace(day=range(1, 4), port=["A", "B"])   # vertices are named states
graph = cr.Graph(states=states)
graph.add_resource(cr.Resource("fuel", upper=10))

graph.add_edge((1, "A"), (2, "A"), cost=1, resources={"fuel": 1})
graph.add_edge((1, "A"), (2, "B"), cost=5, resources={"fuel": 6})
graph.add_edge((2, "A"), (3, "B"), cost=9, resources={"fuel": 6})
graph.add_edge((2, "B"), (3, "B"), cost=1, resources={"fuel": 5})

route = graph.compile().solve((1, "A"), (3, "B"))
print(route.cost, route.path, route.resources)
# 10.0 [(1, 'A'), (2, 'A'), (3, 'B')] {'fuel': 7.0}
```

The cheapest route costs 6 but needs 11 units of fuel. CelerisRoute returns the
cheapest route that fits in the tank.

## Why CelerisRoute

<div class="grid cards" markdown>

-   :material-shape-outline:{ .lg .middle } **Model in your own words**

    ---

    A vertex is a state such as `(day, port, cargo)`. You never juggle
    integer ids, and paths come back as states.

    [:octicons-arrow-right-24: Modelling a network](guide/modelling.md)

-   :material-gauge:{ .lg .middle } **Exact solvers in C++**

    ---

    Shortest paths with negative costs, and resource-constrained shortest
    paths with any number of cumulative resources. A million-vertex DAG
    solves in tens of milliseconds.

    [:octicons-arrow-right-24: Performance](performance.md)

-   :material-repeat:{ .lg .middle } **Made for column generation**

    ---

    A pricing engine is built once and re-solved for new duals without
    rebuilding anything. It returns ready-made master columns and an exact
    convergence certificate.

    [:octicons-arrow-right-24: Pricing with duals](guide/pricing.md)

-   :material-ferry:{ .lg .middle } **Whole fleets in one call**

    ---

    Hundreds of vehicles share one network. Each is a small profile on top,
    and all are priced in parallel with the GIL released.

    [:octicons-arrow-right-24: Fleet pricing](guide/fleet.md)

</div>

## What is in the box

| | |
|---|---|
| **Shortest path on a DAG** | Negative costs allowed, exact K-best paths. |
| **Resource-constrained shortest path** | Any number of cumulative resources with upper bounds, exact. |
| **`PricingEngine`** | One subproblem, re-solved for new dual values in microseconds of overhead. |
| **`BatchPricingEngine`** | Many subproblems on native threads; results identical to solving one by one. |
| **`PricingFleet`** | One shared network, per-vehicle sources, limits, costs and forbidden edges. 6 to 12 times less memory than a graph per vehicle in our benchmark. |
| **Named states** | `StateSpace(day=..., port=..., cargo=...)` with stable integer ids underneath. |
| **Typed, tested** | Type hints throughout; every solver is checked against brute-force enumeration. |

## Install

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ celerisroute
```

CelerisRoute is in a pre-release phase and is published on TestPyPI. See
[Installation](getting-started/installation.md) for building from source.

## Where to go next

* New here? The [quick start](getting-started/quickstart.md) takes five minutes.
* Want to see complete programs? The [examples](examples/index.md) go from a
  ten-line shortest path to a fleet planned by column generation.
* Need the fine print? [Conventions and guarantees](concepts/guarantees.md)
  says exactly what "exact" means, and [Limitations](concepts/limitations.md)
  says what the library does not do.
