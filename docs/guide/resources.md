# Resource limits

A shortest path is often not allowed to be just any path: the tank holds only
so much fuel, the driver may work only so many hours. CelerisPath handles such
limits as **resources**.

## Declaring a resource

```python
import celerispath as cp

cities = ["Lisbon", "Porto", "Madrid", "Bordeaux", "Lyon", "Paris", "Zurich", "Berlin"]
graph = cp.Graph(states=cp.StateSpace(city=cities))

graph.add_resource(cp.Resource("hours", upper=34))
graph.add_resource(cp.Resource("tolls", upper=2))
```

A resource is *cumulative*: a path starts at `initial` (0 unless you say
otherwise), every edge adds a nonnegative amount, and the running total must
never exceed `upper`.

Each edge states what it consumes. Resources that are not mentioned consume
nothing.

```python
#  (from, to): (cost, hours, toll legs)
legs = {
    ("Lisbon", "Madrid"): (180, 6, 1),
    ("Lisbon", "Porto"): (90, 3, 0),
    ("Porto", "Madrid"): (120, 6, 0),
    ("Porto", "Bordeaux"): (310, 11, 1),
    ("Madrid", "Bordeaux"): (200, 7, 1),
    ("Madrid", "Lyon"): (330, 12, 1),
    ("Bordeaux", "Lyon"): (150, 6, 0),
    ("Bordeaux", "Paris"): (170, 6, 1),
    ("Lyon", "Zurich"): (-40, 5, 0),
    ("Paris", "Berlin"): (320, 10, 1),
    ("Zurich", "Berlin"): (260, 9, 1),
}
for (origin, destination), (cost, hours, tolls) in legs.items():
    graph.add_edge((origin,), (destination,), cost=cost, resources={"hours": hours, "tolls": tolls})

network = graph.compile()
```

## Solving

Nothing changes in the call. Because the network has bounded resources,
`algorithm="auto"` picks the resource-constrained solver.

```python
best = network.solve(("Lisbon",), ("Berlin",))

print(best.cost)        # 770.0
print(best.resources)   # {'hours': 34.0, 'tolls': 2.0}
print([city for (city,) in best.path])
# ['Lisbon', 'Porto', 'Bordeaux', 'Lyon', 'Zurich', 'Berlin']
```

The result is exact: no cheaper path stays within both limits.

### Ignoring the limits on purpose

It is often useful to know what the limits cost you. Asking for the plain DAG
algorithm on a constrained network requires `unconstrained=True`, so that
bounds are never dropped by accident:

```python
cheapest = network.solve(("Lisbon",), ("Berlin",), algorithm="dag", unconstrained=True)

print(cheapest.cost)        # 730.0
print(cheapest.resources)   # {'hours': 32.0, 'tolls': 3.0}
```

The consumption is still reported, which shows that this route breaks the toll
limit.

### Which algorithm runs

| `algorithm` | what happens |
|---|---|
| `"auto"` (default) | resource-constrained search if some resource has a finite `upper`, otherwise the DAG algorithm |
| `"rcsp"` | always the resource-constrained search |
| `"dag"` | the DAG algorithm; on a constrained network only together with `unconstrained=True` |

## Resources that are only measured

A resource without an upper bound is tracked and reported but costs the solver
nothing. Use it to get totals such as distance or days at sea on every
solution:

```python
graph.add_resource(cp.Resource("distance"))   # no upper bound
```

## How the search works, and what it costs

The resource-constrained solver is a labelling algorithm. A *label* is a
partial path with its cost and resource totals. A label is thrown away when
another label at the same vertex is at least as good in cost and in every
bounded resource, or when it can no longer reach the target within the bounds.

The amount of work follows the number of labels, not the size of the graph:

```python
print(best.statistics.labels_created)     # 8
print(best.statistics.labels_dominated)   # 1
```

Three things keep label counts small:

* **Tight bounds.** A loose bound prunes nothing.
* **Few bounded resources.** Each extra bounded resource makes dominance
  rarer. Leave resources you only want to measure unbounded.
* **Short paths.** Labels multiply along the path.

!!! warning "Exponential in the worst case"
    Resource-constrained shortest path is NP-hard. Two loosely bounded
    resources on paths with dozens of edges can produce millions of labels.
    See the [RCSP benchmark](../performance.md#resource-constrained-shortest-path)
    for what to expect.

!!! tip "Use integers where you can"
    Feasibility is an exact floating-point comparison: a total equal to
    `upper` is feasible. If a path can hit a bound exactly, increments such as
    `0.1` may not add up the way you expect. Scale to integers (minutes
    instead of hours) or leave a little slack in `upper`.

## What resources cannot express

Resources are cumulative with an upper bound. Time windows, refuelling and
minimum consumption are not expressible as resources today. Often they can be
moved into the state instead: a `day` dimension handles time windows by simply
not adding the edges that would violate them. See
[Limitations](../concepts/limitations.md).
