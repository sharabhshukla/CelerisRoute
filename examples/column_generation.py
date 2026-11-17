"""Column generation with path pricing, start to finish.

Five customers each want one cargo delivered inside a time window, and three
tankers can serve them. Skipping a customer is allowed but costs a penalty.
The master problem chooses one route per vessel:

    minimise    sum_r cost[r] x[r]  +  sum_k penalty[k] s[k]
    subject to  sum_r delivers[k, r] x[r] + s[k] >= 1    for every contract k
                - sum_{r of vessel v} x[r]       >= -1   for every vessel v

(The second row says "at most one route per vessel". It is written with >= so
that all rows point the same way.)

Nobody can list all routes, so we start with none. Each round the master
gives a price (dual) to every row, and a shortest path per vessel finds the
route that those prices make most attractive. When no vessel has a route
with negative reduced cost, the master is optimal.

Run with:  python examples/column_generation.py
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

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
FUEL_COST_PER_DAY = 10
PORT_FEE_PER_DAY = 1
LOADING_COST = 20
UNLOADING_COST = 20
FUEL_TANK = 26  # days of sailing

#            port, first day, last day, penalty if nobody delivers
CONTRACTS = [
    ("Mumbai", 4, 8, 300),
    ("Singapore", 8, 14, 400),
    ("Rotterdam", 13, 20, 500),
    ("Mumbai", 16, 22, 300),
    ("Singapore", 20, 27, 400),
]
VESSELS = {"Aurora": "Dubai", "Borealis": "Dubai", "Calypso": "Singapore"}  # name: home port

num_rows = len(CONTRACTS) + len(VESSELS)  # contract rows first, then vessel rows

# ---- the network: the same for every vessel -----------------------------------
states = cp.StateSpace(day=DAYS, port=PORTS, cargo=["empty", "full"])
graph = cp.Graph(states=states)
graph.add_resource(cp.Resource("fuel", upper=FUEL_TANK))
delivery_rows, delivery_edges = [], []  # which edge delivers which contract

for day in DAYS:
    for port in PORTS:
        for cargo in ["empty", "full"]:
            here = (day, port, cargo)
            if day + 1 in DAYS:
                graph.add_edge(here, (day + 1, port, cargo), cost=PORT_FEE_PER_DAY)
            for other in PORTS:
                duration = SAILING_DAYS.get((port, other)) or SAILING_DAYS.get((other, port))
                if duration and day + duration in DAYS:
                    graph.add_edge(here, (day + duration, other, cargo),
                                   cost=FUEL_COST_PER_DAY * duration, resources={"fuel": duration})

        if day + 1 in DAYS:
            if port == "Dubai":
                graph.add_edge((day, port, "empty"), (day + 1, port, "full"), cost=LOADING_COST)
            else:
                edge = graph.add_edge((day, port, "full"), (day + 1, port, "empty"), cost=UNLOADING_COST)
                for k, (customer, first, last, _) in enumerate(CONTRACTS):
                    if customer == port and first <= day <= last:
                        delivery_rows.append(k)
                        delivery_edges.append(edge)

network = graph.compile()

# A[k, e] = 1 if edge e delivers contract k. This is all the pricing needs to
# turn master duals into edge costs.
A = sp.csr_array((np.ones(len(delivery_edges)), (delivery_rows, delivery_edges)),
                 shape=(num_rows, network.num_edges))

# ---- one pricing engine per vessel --------------------------------------------
# Every route of vessel v also has a -1 in that vessel's own row.
engines = []
for v, (name, home) in enumerate(VESSELS.items()):
    engines.append(cp.PricingEngine(
        network, (1, home, "empty"), (30, home, "empty"),
        dual_matrix=A, fixed_coefficients={len(CONTRACTS) + v: -1.0}, name=name,
    ))
pricing = cp.BatchPricingEngine(engines)  # solves all engines in parallel

# ---- the master problem -------------------------------------------------------
# Start with one column per contract: "do not deliver, pay the penalty".
costs = [penalty for *_, penalty in CONTRACTS]
columns = [np.eye(num_rows)[k] for k in range(len(CONTRACTS))]
routes = [None] * len(CONTRACTS)  # routes[j] = (vessel, route) for route columns
rhs = np.array([1.0] * len(CONTRACTS) + [-1.0] * len(VESSELS))

print("round  columns  master cost  new routes")
for round_number in range(1, 100):
    # linprog only knows "<=" rows, so "M x >= rhs" goes in as "-M x <= -rhs"
    # and the duals come back with the opposite sign.
    M = np.column_stack(columns)
    master = linprog(costs, A_ub=-M, b_ub=-rhs)
    duals = -master.ineqlin.marginals

    new_routes = 0
    for name, result in zip(VESSELS, pricing.solve_batch(duals, tolerance=1e-6)):
        for route in result:  # only routes with a negative reduced cost come back
            costs.append(route.original_cost)
            columns.append(route.dense_coefficients())
            routes.append((name, route))
            new_routes += 1

    print(f"{round_number:5d}  {M.shape[1]:7d}  {master.fun:11.2f}  {new_routes:10d}")
    if new_routes == 0:
        break

print(f"\nLP optimum: {master.fun:.2f} with {len(costs) - len(CONTRACTS)} generated routes")

# ---- from the LP to a plan ----------------------------------------------------
# The LP may mix routes fractionally. Solving the same master with whole
# numbers over the routes found so far gives a plan a dispatcher can use.
# (It is the best plan from these routes; proving it best overall takes branch-and-price.)
M = np.column_stack(columns)
plan = milp(costs, constraints=LinearConstraint(M, lb=rhs),
            integrality=np.ones(len(costs)), bounds=Bounds(0, 1))
print(f"Integer plan: {plan.fun:.2f}\n")

for j, chosen in enumerate(plan.x):
    if chosen < 0.5:
        continue
    if routes[j] is None:
        port, first, last, penalty = CONTRACTS[j]
        print(f"Nobody delivers to {port} in days {first}-{last}: penalty {penalty}")
        continue
    name, route = routes[j]
    print(f"{name}: cost {route.original_cost:.0f}, {route.resources['fuel']:.0f} days at sea")
    for (day, port, cargo), (next_day, next_port, next_cargo) in zip(route.path, route.path[1:]):
        if next_port != port:
            print(f"  day {day:2d}  sail {port} -> {next_port} ({cargo}), arrive on day {next_day}")
        elif next_cargo != cargo:
            print(f"  day {day:2d}  {'load' if next_cargo == 'full' else 'deliver'} in {port}")
