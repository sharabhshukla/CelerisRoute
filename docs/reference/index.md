# API reference

Everything is imported from the top-level package:

```python
import celerisroute as cr
```

| | |
|---|---|
| [Modelling](modelling.md) | `StateSpace`, `Graph`, `Network`, `Resource` |
| [Solving](solving.md) | `shortest_path`, `resource_shortest_path`, `Solution`, `SolveStatistics`, `Status` |
| [Pricing](pricing.md) | `PricingEngine`, `BatchPricingEngine`, `PricingResult`, `Column`, `PricingStatistics` |
| [Fleet pricing](fleet.md) | `PricingFleet`, `FleetResult`, `FleetColumn`, `FleetStatistics`, `IndexedDuals`, `PortTimeProductDuals` |
| [Errors](errors.md) | `CelerisRouteError` and its subclasses |

The package is fully typed and ships a `py.typed` marker, so editors and type
checkers see the same signatures as this reference.

## Conventions used throughout

* A **vertex** is either an `int` (a vertex id) or a state: a tuple in the
  order of the state-space dimensions, or a mapping from dimension name to
  value.
* **Edge ids** count from zero in insertion order and never change.
* **Paths** are returned three ways: `path` (decoded states), `nodes` (vertex
  ids) and `edges` (edge ids).
* Solving never raises for a missing path. Check `status`.
