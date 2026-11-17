# Examples

Plain scripts, meant to be read from top to bottom. They build on each other.

| script | what it shows | lines |
|---|---|---:|
| [`shortest_path.py`](shortest_path.py) | states, edges, a negative cost, an unreachable pair, a cycle | 50 |
| [`resource_limits.py`](resource_limits.py) | the same trip with limits on driving hours and toll legs | 55 |
| [`vessel_routing.py`](vessel_routing.py) | a `(day, port, cargo)` network with a fuel tank and a printed itinerary | 80 |
| [`column_generation.py`](column_generation.py) | a full column-generation loop: SciPy master, one pricing engine per vessel | 150 |
| [`fleet_pricing.py`](fleet_pricing.py) | twelve different vessels on one shared network, priced in one call | 150 |

```bash
python examples/shortest_path.py
```

They need nothing beyond CelerisPath and its dependencies (NumPy and SciPy).
The [documentation](https://sharabhshukla.github.io/CelerisPath/examples/)
walks through each one with its output.
