"""Route a tanker through a time-space network.

The tanker starts empty in Dubai on day 1 and has to be back there, empty, on
day 30. In between it can wait, sail, load in Dubai and sell its cargo in
Mumbai, Singapore or Rotterdam.

A state is (day, port, cargo) and every edge is one thing the vessel can do.
Selling earns money, so that edge has a negative cost, and the cheapest path
through the network is the most profitable voyage plan.

Run with:  python examples/vessel_routing.py
"""

import celerispath as cp

DAYS = range(1, 31)
PORTS = ["Dubai", "Mumbai", "Singapore", "Rotterdam"]
SAILING_DAYS = {
    ("Dubai", "Mumbai"): 3,
    ("Dubai", "Singapore"): 7,
    ("Dubai", "Rotterdam"): 12,
    ("Mumbai", "Singapore"): 5,
    ("Mumbai", "Rotterdam"): 13,
    ("Singapore", "Rotterdam"): 16,
}
FUEL_COST_PER_DAY = 10   # at sea
PORT_FEE_PER_DAY = 1     # waiting in port
LOADING_COST = 50        # buying and loading one cargo in Dubai
SELLING_PRICE = {"Mumbai": 140, "Singapore": 230, "Rotterdam": 420}
FUEL_TANK = 20           # days of sailing the vessel can do on one tank

# ---- model: one state per (day, port, cargo), one edge per possible action ----
states = cp.StateSpace(day=DAYS, port=PORTS, cargo=["empty", "full"])
graph = cp.Graph(states=states)
graph.add_resource(cp.Resource("fuel", upper=FUEL_TANK))

for day in DAYS:
    for port in PORTS:
        for cargo in ["empty", "full"]:
            here = (day, port, cargo)

            if day + 1 in DAYS:  # wait in port for a day
                graph.add_edge(here, (day + 1, port, cargo), cost=PORT_FEE_PER_DAY)

            for other in PORTS:  # sail to another port
                duration = SAILING_DAYS.get((port, other)) or SAILING_DAYS.get((other, port))
                if duration and day + duration in DAYS:
                    graph.add_edge(here, (day + duration, other, cargo),
                                   cost=FUEL_COST_PER_DAY * duration, resources={"fuel": duration})

        if day + 1 in DAYS:
            if port == "Dubai":  # load: takes a day, the vessel becomes full
                graph.add_edge((day, port, "empty"), (day + 1, port, "full"), cost=LOADING_COST)
            else:                # sell: takes a day, the vessel becomes empty
                graph.add_edge((day, port, "full"), (day + 1, port, "empty"), cost=-SELLING_PRICE[port])

network = graph.compile()
print(network)


def print_voyage(path):
    """Print one line per action, skipping the days spent waiting."""
    for (day, port, cargo), (next_day, next_port, next_cargo) in zip(path, path[1:]):
        if next_port != port:
            print(f"  day {day:2d}  sail {port} -> {next_port} ({cargo}), arrive on day {next_day}")
        elif next_cargo != cargo:
            print(f"  day {day:2d}  {'load' if next_cargo == 'full' else 'sell'} in {port}")


start = (1, "Dubai", "empty")
end = (30, "Dubai", "empty")

# ---- solve: with unlimited fuel, then with the real tank ----------------------
dream = network.solve(start, end, algorithm="dag", unconstrained=True)
print(f"\nBest voyage with unlimited fuel: profit {-dream.cost:.0f}, "
      f"{dream.resources['fuel']:.0f} days at sea")
print_voyage(dream.path)

plan = network.solve(start, end)
print(f"\nBest voyage on one tank ({FUEL_TANK} days at sea): profit {-plan.cost:.0f}, "
      f"{plan.resources['fuel']:.0f} days at sea")
print_voyage(plan.path)
