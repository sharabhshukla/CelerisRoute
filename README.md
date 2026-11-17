<h1 align="center">CelerisRoute</h1>

<p align="center">
  <b>Fast shortest paths on state-space DAGs, built for the pricing step of column generation.</b><br>
  A C++20 core on the Boost Graph Library, with a Python API you can read.
</p>

<p align="center">
  <a href="https://github.com/sharabhshukla/CelerisRoute/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/sharabhshukla/CelerisRoute/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://github.com/sharabhshukla/CelerisRoute/actions/workflows/wheels.yml"><img alt="Wheels" src="https://github.com/sharabhshukla/CelerisRoute/actions/workflows/wheels.yml/badge.svg"></a>
  <a href="https://sharabhshukla.github.io/CelerisRoute/"><img alt="Docs" src="https://github.com/sharabhshukla/CelerisRoute/actions/workflows/docs.yml/badge.svg"></a>
  <img alt="Python 3.11 to 3.15" src="https://img.shields.io/badge/python-3.11%20to%203.15-3776ab">
  <a href="https://github.com/sharabhshukla/CelerisRoute/blob/main/LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-2ea44f"></a>
</p>

<p align="center">
  <a href="https://sharabhshukla.github.io/CelerisRoute/"><b>Documentation</b></a> ·
  <a href="https://sharabhshukla.github.io/CelerisRoute/getting-started/quickstart/">Quick start</a> ·
  <a href="https://sharabhshukla.github.io/CelerisRoute/examples/">Examples</a> ·
  <a href="https://sharabhshukla.github.io/CelerisRoute/reference/">API reference</a>
</p>

---

Routing and scheduling models are often solved by column generation: a master
LP picks routes, and a pricing problem keeps inventing better ones. The pricing
problem is a shortest path on a large acyclic network, usually with a resource
limit, solved thousands of times with slightly different costs.

CelerisRoute does that one job well. You describe the network in the words of
your problem, and it returns exact answers from native code.

```python
import celerisroute as cr

states = cr.StateSpace(day=range(1, 4), port=["A", "B"])   # a vertex is a (day, port) state
graph = cr.Graph(states=states)
graph.add_resource(cr.Resource("fuel", upper=10))           # the tank holds 10

graph.add_edge((1, "A"), (2, "A"), cost=1, resources={"fuel": 1})
graph.add_edge((1, "A"), (2, "B"), cost=5, resources={"fuel": 6})
graph.add_edge((2, "A"), (3, "B"), cost=9, resources={"fuel": 6})
graph.add_edge((2, "B"), (3, "B"), cost=1, resources={"fuel": 5})

network = graph.compile()
route = network.solve((1, "A"), (3, "B"))

print(route.cost)        # 10.0
print(route.path)        # [(1, 'A'), (2, 'A'), (3, 'B')]
print(route.resources)   # {'fuel': 7.0}
```

The cheapest route costs 6 but would burn 11 units of fuel. CelerisRoute returns
the cheapest route that fits in the tank.

## Highlights

- **Model with states, not integers.** `StateSpace(day=..., port=..., cargo=...)`
  gives every vertex a meaning. Paths come back as states.
- **Exact.** Shortest paths with negative costs, K best paths, and
  resource-constrained shortest paths with any number of cumulative resources.
  Every solver is tested against brute-force enumeration.
- **Made for column generation.** A `PricingEngine` is built once and re-solved
  for new duals. It returns ready-made master columns and an exact proof of
  convergence.
- **Whole fleets in one call.** A `PricingFleet` shares one network between
  hundreds of vehicles and prices them on all cores with the GIL released,
  using 6 to 12 times less memory than a graph per vehicle.
- **Easy to install.** Wheels for Linux, macOS and Windows, Python 3.11 to
  3.15 including free-threaded builds. NumPy and SciPy are the only
  dependencies.

## Installation

CelerisRoute is in a pre-release phase, and its wheels are on TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ celerisroute
```

To build from source you need a C++20 compiler and the Boost headers
(`apt install libboost-dev`, `brew install boost`):

```bash
pip install git+https://github.com/sharabhshukla/CelerisRoute
```

See the [installation guide](https://sharabhshukla.github.io/CelerisRoute/getting-started/installation/)
for build options and the optional oneTBB scheduler.

## Pricing with duals

In column generation the same network is solved again and again with edge costs
reduced by the duals of the master problem. Tell the engine once which edges
belong to which master row, then hand it the duals of each iteration.

```python
import numpy as np

# One master row: "arrive at B on day 3". Edges 2 and 3 do that.
A = np.array([[0.0, 0.0, 1.0, 1.0]])
engine = cr.PricingEngine(network, (1, "A"), (3, "B"), dual_matrix=A)

result = engine.solve(np.array([12.0]))        # duals from the master
column = result[0]
print(column.reduced_cost)                     # -2.0: this route improves the master
print(column.original_cost)                    # 10.0: its objective coefficient
print(column.dense_coefficients())             # [1.]: its column in the master matrix

result = engine.solve(np.array([4.0]))
print(len(result), result.best_reduced_cost)   # 0 6.0: nothing prices out, proven
```

## A whole fleet

Twelve vessels, each with its own home port, fuel tank and hire, some barred
from a port. One network, one call per iteration:

```py
fleet = cr.PricingFleet(network, dual_matrix=A)
fleet.add_vessels(
    ids=names,
    source=[(1, home, "empty") for home in homes],
    target=[(30, home, "empty") for home in homes],
    resource_limits={"fuel": tanks},
    fixed_cost=hire,
    forbidden_edges=[mumbai_edges if big else None for big in too_large_for_mumbai],
)

while True:
    duals = solve_master(M, costs)                # your LP solver
    result = fleet.price(duals)                   # all vessels, native threads
    if result.converged:                          # exact stopping test
        break
    M = sp.hstack([M, result.coefficient_matrix])
    costs = np.concatenate([costs, result.original_costs])
```

The complete program is
[`examples/fleet_pricing.py`](https://github.com/sharabhshukla/CelerisRoute/blob/main/examples/fleet_pricing.py).

## Performance

Measured on an Apple M4 desktop; see the
[performance page](https://sharabhshukla.github.io/CelerisRoute/performance/)
for the full tables, the method and the caveats.

| task | size | time |
|---|---|---:|
| Shortest path on a DAG | 100,000 vertices, 800,000 edges | 4.2 ms |
| Shortest path on a DAG | 1,000,000 vertices, 8,000,000 edges | 44 ms |
| Re-price with new duals (DAG) | 100,000 vertices, 794,000 edges, 200 master rows | 5.4 ms |
| Price a fleet of 125 vessels (RCSP, 10 threads) | 22,501 vertices, 87,584 edges | 23 ms |

For that fleet, one shared network needs 124 MB where a graph per vessel needs
975 MB, and it is set up in 30 ms instead of 302 ms.

Resource-constrained search is NP-hard, and its running time follows the number
of labels rather than the size of the graph: the same benchmark has instances
that take 3 ms and instances that take 20 s. The performance page shows both.

## Examples

Plain scripts that read from top to bottom, in
[`examples/`](https://github.com/sharabhshukla/CelerisRoute/tree/main/examples):

| script | what it shows |
|---|---|
| [`shortest_path.py`](https://github.com/sharabhshukla/CelerisRoute/blob/main/examples/shortest_path.py) | states, edges, a negative cost, a cycle |
| [`resource_limits.py`](https://github.com/sharabhshukla/CelerisRoute/blob/main/examples/resource_limits.py) | limits on driving hours and toll legs |
| [`vessel_routing.py`](https://github.com/sharabhshukla/CelerisRoute/blob/main/examples/vessel_routing.py) | a `(day, port, cargo)` network with a fuel tank |
| [`column_generation.py`](https://github.com/sharabhshukla/CelerisRoute/blob/main/examples/column_generation.py) | a full column-generation loop with a SciPy master |
| [`fleet_pricing.py`](https://github.com/sharabhshukla/CelerisRoute/blob/main/examples/fleet_pricing.py) | twelve different vessels priced in one call |

## Project layout

```text
cpp/                C++20 solver core
  include/          public headers
  src/              implementation
  bindings/         nanobind module celerisroute._core
  bench/            native benchmark harness
src/celerisroute/   Python package
tests/              pytest suite, including the examples and the docs
examples/           runnable scripts
benchmarks/         benchmark scripts and recorded results
docs/               documentation site (MkDocs)
```

## Development

```bash
git clone https://github.com/sharabhshukla/CelerisRoute
cd CelerisRoute
pip install -e ".[dev]"
python -m pytest
```

See [CONTRIBUTING.md](https://github.com/sharabhshukla/CelerisRoute/blob/main/CONTRIBUTING.md).

## Good to know

- Graphs must be acyclic. Put something that only moves forward, such as a
  day, into the state. A cycle is reported with the vertices involved.
- Resources are cumulative with an upper bound. Time windows usually belong in
  the state space.
- The API may still change before 1.0.

The [limitations page](https://sharabhshukla.github.io/CelerisRoute/concepts/limitations/)
has the full list.

## License

MIT. See [LICENSE](https://github.com/sharabhshukla/CelerisRoute/blob/main/LICENSE).
