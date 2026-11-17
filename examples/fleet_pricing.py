"""Column generation for a whole fleet with one PricingFleet.

column_generation.py builds one pricing engine per vessel. That is fine for
three vessels. This script plans twelve, and they are not alike: each has its
own home port, fuel tank and hire, and the largest ones cannot enter Mumbai.
A PricingFleet keeps the network and the reduced costs once, stores only these
differences per vessel, and prices all vessels in one call on all cores.

The master problem is the same as in column_generation.py:

    minimise    sum_r cost[r] x[r]  +  sum_k penalty[k] s[k]
    subject to  sum_r delivers[k, r] x[r] + s[k] >= 1    for every contract k
                - sum_{r of vessel v} x[r]       >= -1   for every vessel v

Run with:  python examples/fleet_pricing.py
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

# A delivery every three days per port: (port, first day, last day, penalty if nobody delivers)
CONTRACTS = []
for first in range(4, 26, 3):
    CONTRACTS.append(("Mumbai", first, first + 2, 300))
for first in range(8, 26, 3):
    CONTRACTS.append(("Singapore", first, first + 2, 400))
for first in range(13, 28, 3):
    CONTRACTS.append(("Rotterdam", first, first + 2, 500))

# Twelve vessels, each a little different.
names = [f"vessel {v + 1:02d}" for v in range(12)]
homes = ["Dubai", "Mumbai", "Dubai", "Singapore", "Mumbai", "Dubai"] * 2
tanks = np.array([14.0, 20.0, 26.0] * 4)           # days of sailing
hire = np.array([40.0, 55.0, 70.0] * 4)            # paid once if the vessel is used
too_large_for_mumbai = tanks == 26.0

num_contracts, num_vessels = len(CONTRACTS), len(names)
num_rows = num_contracts + num_vessels

# ---- the network: built once, shared by all vessels ---------------------------
states = cp.StateSpace(day=DAYS, port=PORTS, cargo=["empty", "full"])
graph = cp.Graph(states=states)
graph.add_resource(cp.Resource("fuel", upper=tanks.max()))
delivery_rows, delivery_edges = [], []  # which edge delivers which contract
mumbai_edges = []                       # every edge that arrives in Mumbai

for day in DAYS:
    for port in PORTS:
        for cargo in ["empty", "full"]:
            here = (day, port, cargo)
            if day + 1 in DAYS:
                graph.add_edge(here, (day + 1, port, cargo), cost=PORT_FEE_PER_DAY)
            for other in PORTS:
                duration = SAILING_DAYS.get((port, other)) or SAILING_DAYS.get((other, port))
                if duration and day + duration in DAYS:
                    edge = graph.add_edge(here, (day + duration, other, cargo),
                                          cost=FUEL_COST_PER_DAY * duration,
                                          resources={"fuel": duration})
                    if other == "Mumbai":
                        mumbai_edges.append(edge)

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
A = sp.csr_array((np.ones(len(delivery_edges)), (delivery_rows, delivery_edges)),
                 shape=(num_rows, network.num_edges))

# ---- the fleet: one network, twelve small vessel profiles ---------------------
fleet = cp.PricingFleet(network, dual_matrix=A)
fleet.add_vessels(
    ids=names,
    source=[(1, home, "empty") for home in homes],
    target=[(30, home, "empty") for home in homes],
    resource_limits={"fuel": tanks},
    fixed_cost=hire,
    fixed_coefficients=[{num_contracts + v: -1.0} for v in range(num_vessels)],
    forbidden_edges=[mumbai_edges if too_large else None for too_large in too_large_for_mumbai],
)
print(network)
print(fleet)

# ---- column generation --------------------------------------------------------
# Start with one column per contract: "do not deliver, pay the penalty".
costs = np.array([penalty for *_, penalty in CONTRACTS], dtype=float)
M = sp.eye_array(num_rows, num_contracts, format="csc")
routes = [None] * num_contracts
rhs = np.array([1.0] * num_contracts + [-1.0] * num_vessels)

print("\nround  columns  master cost  new routes  pricing")
for round_number in range(1, 100):
    master = linprog(costs, A_ub=-M, b_ub=-rhs)     # "M x >= rhs" in linprog's "<=" form
    duals = -master.ineqlin.marginals

    result = fleet.price(duals, tolerance=1e-6)      # all twelve vessels, one native call
    print(f"{round_number:5d}  {M.shape[1]:7d}  {master.fun:11.2f}  {result.num_columns:10d}"
          f"  {result.statistics.total_time * 1e3:6.2f} ms")
    if result.converged:  # every vessel priced exactly, none has an improving route
        break

    # The result already holds the new master columns as one sparse matrix.
    M = sp.hstack([M, result.coefficient_matrix], format="csc")
    costs = np.concatenate([costs, result.original_costs])
    routes.extend(result.columns)

print(f"\nLP optimum: {master.fun:.2f} with {len(routes) - num_contracts} generated routes")

# ---- from the LP to a plan ----------------------------------------------------
plan = milp(costs, constraints=LinearConstraint(M, lb=rhs),
            integrality=np.ones(len(costs)), bounds=Bounds(0, 1))
print(f"Integer plan from these routes: {plan.fun:.2f}\n")

for j in np.flatnonzero(plan.x > 0.5):
    if routes[j] is None:
        port, first, last, penalty = CONTRACTS[j]
        print(f"nobody delivers to {port} in days {first}-{last}: penalty {penalty}")
        continue
    route = routes[j]
    stops = [f"{port} (day {day})" for (day, port, cargo), (_, _, after) in zip(route.path, route.path[1:])
             if cargo == "full" and after == "empty"]
    print(f"{route.vessel} from {homes[route.vessel_index]:<9s} cost {route.original_cost:4.0f}, "
          f"{route.resources['fuel']:2.0f} of {tanks[route.vessel_index]:.0f} days at sea, "
          f"delivers to " + ", ".join(stops))
