# Resource limits

The same trip as in [Shortest path](shortest-path.md), but now the driver may
drive at most 34 hours and the company wants at most 2 toll legs. The cheapest
route overall breaks the toll limit, so the answer changes.

```bash
python examples/resource_limits.py
```

## The script

```python title="examples/resource_limits.py"
--8<-- "examples/resource_limits.py"
```

Things to notice:

* Two lines declare the limits; every edge then says what it consumes.
* `network.solve` is called exactly as before. It picks the
  resource-constrained algorithm because the network has bounded resources.
* `algorithm="dag", unconstrained=True` gives the cheapest route with the
  limits ignored, for comparison. Its `resources` still show what it would
  use.
* With a tighter limit there is no route at all, and the status says
  `infeasible` rather than `unreachable`: routes exist, but none fits.

## Output

```text
cheapest route, limits ignored
  route: Lisbon -> Madrid -> Lyon -> Zurich -> Berlin
  cost:  730.0  uses: {'hours': 32.0, 'tolls': 3.0}
cheapest route within 34 hours and 2 toll legs
  route: Lisbon -> Porto -> Bordeaux -> Lyon -> Zurich -> Berlin
  cost:  770.0  uses: {'hours': 34.0, 'tolls': 2.0}
  labels created by the search: 8
with only 28 hours: infeasible
```

The guide page [Resource limits](../guide/resources.md) explains how the
search works and what makes it expensive.

[:octicons-mark-github-16: View on GitHub](https://github.com/sharabhshukla/CelerisRoute/blob/main/examples/resource_limits.py){ .md-button }
