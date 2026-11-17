# Vessel routing

A model with real structure. A vessel buys cargo in Dubai and sells it in
Mumbai, Singapore or Rotterdam, where farther ports pay more. What is the most
profitable 30-day voyage, and what changes when the vessel has only one tank
of fuel?

A state is `(day, port, cargo)`. Because every edge moves forward in time, the
network is acyclic by construction, although the vessel can visit the same
port many times.

```bash
python examples/vessel_routing.py
```

## The script

```python title="examples/vessel_routing.py"
--8<-- "examples/vessel_routing.py"
```

Things to notice:

* The network is built by three nested loops that read like the rules of the
  problem: wait, sail, load, sell.
* Profit is modelled as negative cost. The solver minimises, so the cheapest
  path is the most profitable voyage.
* The route comes back as a list of states, and turning it into a readable
  itinerary takes a few lines: compare each state with the next one.
* The fuel limit is a single `Resource`. Nothing else in the model changes.

## Output

```text
Network(num_vertices=240, num_edges=844, acyclic=True, resources=['fuel'])

Best voyage with unlimited fuel: profit 127, 24 days at sea
  day  1  load in Dubai
  day  2  sail Dubai -> Rotterdam (full), arrive on day 14
  day 17  sell in Rotterdam
  day 18  sail Rotterdam -> Dubai (empty), arrive on day 30

Best voyage on one tank (20 days at sea): profit 85, 18 days at sea
  day  6  load in Dubai
  day  7  sail Dubai -> Mumbai (full), arrive on day 10
  day 10  sell in Mumbai
  day 11  sail Mumbai -> Dubai (empty), arrive on day 14
  day 14  load in Dubai
  day 15  sail Dubai -> Mumbai (full), arrive on day 18
  day 18  sell in Mumbai
  day 19  sail Mumbai -> Dubai (empty), arrive on day 22
  day 22  load in Dubai
  day 23  sail Dubai -> Mumbai (full), arrive on day 26
  day 26  sell in Mumbai
  day 27  sail Mumbai -> Dubai (empty), arrive on day 30
```

With unlimited fuel the vessel makes the long, lucrative trip to Rotterdam.
On one tank it cannot get there and back, and three short round trips to
Mumbai become the best plan.

[:octicons-mark-github-16: View on GitHub](https://github.com/sharabhshukla/CelerisPath/blob/main/examples/vessel_routing.py){ .md-button }
