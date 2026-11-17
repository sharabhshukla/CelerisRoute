"""Shortest path with resource limits.

The same truck trip as in shortest_path.py, but now every leg also takes
driving hours and some legs use a toll motorway. The driver may drive at most
34 hours and the company pays for at most two toll legs.

Resources add up along the path and must stay below their upper bound.

Run with:  python examples/resource_limits.py
"""

import celerispath as cp

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
cities = ["Lisbon", "Porto", "Madrid", "Bordeaux", "Lyon", "Paris", "Zurich", "Berlin"]

graph = cp.Graph(states=cp.StateSpace(city=cities))
graph.add_resource(cp.Resource("hours", upper=34))
graph.add_resource(cp.Resource("tolls", upper=2))
for (origin, destination), (cost, hours, tolls) in legs.items():
    graph.add_edge((origin,), (destination,), cost=cost, resources={"hours": hours, "tolls": tolls})
network = graph.compile()

# Ignore the limits on purpose, to see what the cheapest route would need.
cheapest = network.solve(("Lisbon",), ("Berlin",), algorithm="dag", unconstrained=True)
print("cheapest route, limits ignored")
print("  route:", " -> ".join(city for (city,) in cheapest.path))
print("  cost: ", cheapest.cost, " uses:", cheapest.resources)

# The default algorithm respects every resource that has an upper bound.
best = network.solve(("Lisbon",), ("Berlin",))
print("cheapest route within 34 hours and 2 toll legs")
print("  route:", " -> ".join(city for (city,) in best.path))
print("  cost: ", best.cost, " uses:", best.resources)
print("  labels created by the search:", best.statistics.labels_created)

# With a limit nobody can meet, the status says so. No exception to catch.
strict = cp.Graph(states=cp.StateSpace(city=cities))
strict.add_resource(cp.Resource("hours", upper=28))
for (origin, destination), (cost, hours, tolls) in legs.items():
    strict.add_edge((origin,), (destination,), cost=cost, resources={"hours": hours})
print("with only 28 hours:", strict.compile().solve(("Lisbon",), ("Berlin",)).status)
