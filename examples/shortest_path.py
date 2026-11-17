"""Your first shortest path.

A truck has to get from Lisbon to Berlin. Every edge is one leg of the trip
with a cost in euros. The Lyon -> Zurich leg has a negative cost because the
truck carries paid freight on it; negative costs are fine on a DAG.

Run with:  python examples/shortest_path.py
"""

import celerispath as cp

legs = {
    ("Lisbon", "Madrid"): 180,
    ("Lisbon", "Porto"): 90,
    ("Porto", "Madrid"): 120,
    ("Porto", "Bordeaux"): 310,
    ("Madrid", "Bordeaux"): 200,
    ("Madrid", "Lyon"): 330,
    ("Bordeaux", "Lyon"): 150,
    ("Bordeaux", "Paris"): 170,
    ("Lyon", "Zurich"): -40,
    ("Paris", "Berlin"): 320,
    ("Zurich", "Berlin"): 260,
}

# Vertices are named states. Here a state is just a city.
cities = ["Lisbon", "Porto", "Madrid", "Bordeaux", "Lyon", "Paris", "Zurich", "Berlin"]
states = cp.StateSpace(city=cities)

graph = cp.Graph(states=states)
for (origin, destination), cost in legs.items():
    graph.add_edge((origin,), (destination,), cost=cost)

# compile() freezes the graph into a compact network that can be solved many times.
network = graph.compile()
print(network)

solution = network.solve(("Lisbon",), ("Berlin",))
print("status:", solution.status)
print("cost:  ", solution.cost)
print("route: ", " -> ".join(city for (city,) in solution.path))

# Any pair of states can be asked for. Nothing leads from Paris to Zurich.
print("Paris to Zurich:", network.solve(("Paris",), ("Zurich",)).status)

# The solvers need an acyclic graph. Add a leg back to Lisbon and compile() notices.
graph.add_edge(("Berlin",), ("Lisbon",), cost=500)
looped = graph.compile()
print("still acyclic?", looped.is_acyclic)
print("cycle found:  ", " -> ".join(states.decode(v)[0] for v in looped.find_cycle()))
