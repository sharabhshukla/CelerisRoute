# Quick start

Five minutes, three ideas: build a network, limit a resource, price with duals.

## 1. A shortest path

Vertices are **states**. Declare what a state is made of, then connect states
with edges.

```python
import celerispath as cp

states = cp.StateSpace(day=range(1, 4), port=["A", "B"])
graph = cp.Graph(states=states)

graph.add_edge((1, "A"), (2, "A"), cost=1)
graph.add_edge((1, "A"), (2, "B"), cost=5)
graph.add_edge((2, "A"), (3, "B"), cost=9)
graph.add_edge((2, "B"), (3, "B"), cost=1)

network = graph.compile()
solution = network.solve((1, "A"), (3, "B"))

print(solution.status)   # optimal
print(solution.cost)     # 6.0
print(solution.path)     # [(1, 'A'), (2, 'B'), (3, 'B')]
```

`compile()` freezes the graph into a compact, immutable `Network`. Solve it as
often as you like, between any two states. Costs may be negative.

## 2. A resource limit

A resource adds up along the path and must stay below its upper bound. Give
each edge its consumption and CelerisPath switches to an exact
resource-constrained search.

```python
graph = cp.Graph(states=states)
graph.add_resource(cp.Resource("fuel", upper=10))

graph.add_edge((1, "A"), (2, "A"), cost=1, resources={"fuel": 1})   # edge 0
graph.add_edge((1, "A"), (2, "B"), cost=5, resources={"fuel": 6})   # edge 1
graph.add_edge((2, "A"), (3, "B"), cost=9, resources={"fuel": 6})   # edge 2
graph.add_edge((2, "B"), (3, "B"), cost=1, resources={"fuel": 5})   # edge 3
network = graph.compile()

best = network.solve((1, "A"), (3, "B"))
print(best.cost, best.resources)   # 10.0 {'fuel': 7.0}
print(best.path)                   # [(1, 'A'), (2, 'A'), (3, 'B')]
```

The route from step 1 would burn 11 units of fuel, so it is no longer allowed.
To see what ignoring the limit would give, say so explicitly:

```python
free = network.solve((1, "A"), (3, "B"), algorithm="dag", unconstrained=True)
print(free.cost, free.resources)   # 6.0 {'fuel': 11.0}
```

## 3. Pricing with duals

In column generation the same network is solved again and again, each time with
edge costs reduced by the duals of a master problem. A `PricingEngine` is built
once for that.

Say the master has one constraint, "arrive at B on day 3", and edges 2 and 3
are the ones that do so:

```python
import numpy as np

A = np.array([[0.0, 0.0, 1.0, 1.0]])    # A[k, e]: coefficient of edge e in master row k
engine = cp.PricingEngine(network, (1, "A"), (3, "B"), dual_matrix=A)

result = engine.solve(np.array([12.0]))   # the duals of the restricted master
column = result[0]
print(column.original_cost, column.reduced_cost)   # 10.0 -2.0
print(column.dense_coefficients())                 # [1.]

result = engine.solve(np.array([4.0]))
print(len(result), result.best_reduced_cost)       # 0 6.0
```

With a dual of 12 the route has reduced cost \(10 - 12 = -2\), so it would
improve the master and comes back as a column. With a dual of 4 nothing prices
out, and `best_reduced_cost = 6.0` is the proof: no route can do better.

## Next steps

* [Modelling a network](../guide/modelling.md) covers states, bulk edge
  insertion and what `compile()` gives you.
* [Examples](../examples/index.md) has complete, runnable programs, up to a
  fleet planned by column generation.
