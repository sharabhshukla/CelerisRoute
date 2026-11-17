"""celerisroute: fast shortest paths on state-space DAGs for column generation.

Typical use:

    import celerisroute as cr

    states = cr.StateSpace(day=range(1, 36), port=["A", "B"], inventory=[0.0, 0.5, 1.0])
    graph = cr.Graph(states=states)
    graph.add_resource(cr.Resource(name="fuel", upper=100.0))
    graph.add_edge((1, "A", 0.0), (2, "B", 0.5), cost=120.0, resources={"fuel": 30.0})
    network = graph.compile()

    solution = network.solve((1, "A", 0.0), (2, "B", 0.5))
    solution.path, solution.cost, solution.resources

    pricing = cr.PricingEngine(network, source, target, dual_matrix=A)
    columns = pricing.solve(duals, negative_only=True)

    fleet = cr.PricingFleet(network, dual_matrix=A)      # many subproblems, one network
    fleet.add_vessels(ids=names, source=sources, target=targets)
    result = fleet.price(duals)

The numerical work is done by a C++20 core built on the Boost Graph Library
(``dag_shortest_paths`` and ``r_c_shortest_paths``).
"""

from __future__ import annotations

from .errors import BatchPricingError, CyclicGraphError, InvalidStateError, CelerisRouteError
from .fleet import (
    FleetColumn,
    FleetResult,
    FleetStatistics,
    IndexedDuals,
    PortTimeProductDuals,
    PricingFleet,
    tbb_available,
    tbb_version,
)
from .graph import Graph, Network
from .pricing import BatchPricingEngine, PricingEngine
from .resources import Resource
from .results import (
    Column,
    PricingResult,
    PricingStatistics,
    Solution,
    SolveStatistics,
    Status,
)
from .solvers import resource_shortest_path, shortest_path
from .states import StateSpace

__version__ = "0.1.0"

__all__ = [
    "BatchPricingEngine",
    "BatchPricingError",
    "Column",
    "CyclicGraphError",
    "FleetColumn",
    "FleetResult",
    "FleetStatistics",
    "Graph",
    "IndexedDuals",
    "InvalidStateError",
    "Network",
    "CelerisRouteError",
    "PortTimeProductDuals",
    "PricingEngine",
    "PricingFleet",
    "PricingResult",
    "PricingStatistics",
    "Resource",
    "Solution",
    "SolveStatistics",
    "StateSpace",
    "Status",
    "__version__",
    "resource_shortest_path",
    "shortest_path",
    "tbb_available",
    "tbb_version",
]
