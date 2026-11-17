# Shortest path

The smallest useful program: a truck has to get from Lisbon to Berlin as
cheaply as possible. Cities are the states, road legs are the edges, and one
leg has a negative cost because a customer pays for a pickup on the way.

```bash
python examples/shortest_path.py
```

## The script

```python title="examples/shortest_path.py"
--8<-- "examples/shortest_path.py"
```

Things to notice:

* `StateSpace(city=cities)` has a single dimension, so a state is a
  one-element tuple such as `("Lisbon",)`.
* The negative cost on Lyon to Zurich needs no special treatment.
* A pair of cities without a route gives `status == "unreachable"`, not an
  exception.
* Adding a leg back to Lisbon creates a cycle. The network reports it with the
  cities involved, so a modelling mistake is easy to find.

## Output

```text
Network(num_vertices=8, num_edges=11, acyclic=True, resources=[])
status: optimal
cost:   730.0
route:  Lisbon -> Madrid -> Lyon -> Zurich -> Berlin
Paris to Zurich: unreachable
still acyclic? False
cycle found:   Bordeaux -> Lyon -> Zurich -> Berlin -> Lisbon -> Porto -> Madrid
```

Next: the same trip with a limit on driving hours, in
[Resource limits](resource-limits.md).

[:octicons-mark-github-16: View on GitHub](https://github.com/sharabhshukla/CelerisRoute/blob/main/examples/shortest_path.py){ .md-button }
