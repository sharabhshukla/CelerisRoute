# Modelling a network

Everything in CelerisPath starts with a directed acyclic graph. This page
shows how to describe one in terms of your problem, not in terms of integer
ids.

## States

A `StateSpace` is the Cartesian product of named dimensions. Each combination
is one vertex.

```python
import celerispath as cp

states = cp.StateSpace(day=range(1, 31), port=["Dubai", "Mumbai", "Singapore"], cargo=["empty", "full"])

print(states)                  # StateSpace(day=30, port=3, cargo=2; size=180)
print(states.dimension_names)  # ('day', 'port', 'cargo')
```

A state is written as a tuple in the order of the dimensions, or by name:

```python
states.id((4, "Mumbai", "full"))                 # 21
states.id(day=4, port="Mumbai", cargo="full")    # 21
states.decode(21)                                # (4, 'Mumbai', 'full')
```

Ids are dense and deterministic: they equal `numpy.ravel_multi_index` with the
last dimension running fastest, so the same model gives the same ids on every
run. A value that is not part of a dimension raises `InvalidStateError` instead
of silently creating a vertex.

!!! tip "Not every state has to be used"
    The state space is a full grid, but a state without edges costs four bytes
    and no solver time. Declare the grid generously and add only the edges
    that make sense.

## Edges

An edge goes from one state to another, has a cost, and optionally consumes
[resources](resources.md).

```python
graph = cp.Graph(states=states)

edge = graph.add_edge((1, "Dubai", "empty"), (2, "Dubai", "full"), cost=50.0)
print(edge)   # 0, the id of the new edge
```

Edge ids count up from zero in insertion order. Keep the ones you care about:
they are how you later tell a pricing engine which edges belong to which master
constraint, or which edges a vessel may not use.

Parallel edges and negative costs are allowed. Cycles are not, see
[below](#cycles).

### Many edges at once

`add_edge` in a Python loop is fine up to a few hundred thousand edges. Beyond
that, build arrays and append them in one call:

```python
import numpy as np

days = np.arange(1, 30)
for port in ["Dubai", "Mumbai", "Singapore"]:
    for cargo in ["empty", "full"]:
        waiting = graph.add_edges(
            source=states.ids(day=days, port=port, cargo=cargo),
            target=states.ids(day=days + 1, port=port, cargo=cargo),
            cost=np.full(len(days), 1.0),
        )
print(waiting)   # range(146, 175): the ids of the last batch
```

`states.ids(...)` encodes whole arrays, broadcasting scalars, and
`add_edges` returns the range of new edge ids.

### Without states

If your vertices are already integers, skip the state space:

```python
plain = cp.Graph(num_vertices=4)
plain.add_edge(0, 1, cost=2.0)
plain.add_edge(1, 3, cost=-1.0)
```

An `int` is always a vertex id, and a tuple or mapping is always a state, so
the two can never be confused.

## Compile, then solve

```python
network = graph.compile()
print(network.num_vertices, network.num_edges, network.is_acyclic)   # 180 175 True
```

`compile()` turns the editable `Graph` into an immutable `Network`: compressed
adjacency, a topological order and resource columns. The graph stays editable
and can be compiled again; a network never changes, so it can be shared between
threads and engines freely.

```python
solution = network.solve((1, "Dubai", "empty"), (5, "Dubai", "full"))

solution.status      # 'optimal', 'unreachable' or 'infeasible'
solution.cost        # 53.0
solution.path        # [(1, 'Dubai', 'empty'), (2, 'Dubai', 'full'), ...] decoded states
solution.edges       # edge ids along the path, a NumPy array
solution.resources   # {} here; totals per resource when there are any
```

A missing path is an answer, not an error: check `solution.status` or the
boolean `solution.found`.

| status | meaning |
|---|---|
| `optimal` | a cheapest path was found |
| `unreachable` | no path leads from source to target |
| `infeasible` | paths exist, but none respects the resource limits |

`cp.shortest_path(network, source, target)` and
`cp.resource_shortest_path(network, source, target)` are the same solvers as
plain functions.

### New costs on the same network

`network.with_costs(costs)` returns a network that shares the topology and
replaces the edge costs, without recompiling:

```python
doubled = network.with_costs(2.0 * network.edge_cost)
print(doubled.solve((1, "Dubai", "empty"), (5, "Dubai", "full")).cost)   # 106.0
```

For the repeated re-costing of column generation, use a
[pricing engine](pricing.md) instead; it does this update natively.

## Cycles

The solvers require an acyclic graph. That is what makes them fast, and what
makes negative costs safe. `compile()` detects cycles, and the network tells
you where:

```python
loop = cp.Graph(num_vertices=3)
loop.add_edge(0, 1, cost=1.0)
loop.add_edge(1, 2, cost=1.0)
loop.add_edge(2, 0, cost=1.0)

cyclic = loop.compile()
print(cyclic.is_acyclic)     # False
print(cyclic.find_cycle())   # [2, 0, 1]: 2 -> 0 -> 1 -> back to 2
```

Solving a cyclic network raises `CyclicGraphError` with the same witness
cycle.

Most routing and scheduling models become acyclic once a state carries
something that only moves forward: a day, a stage, a number of stops made.
That is why the examples use `(day, port, cargo)` rather than just `port`.
