"""Functional entry points for single shortest-path solves."""

from __future__ import annotations

from .graph import Graph, Network, Vertex
from .results import Solution

__all__ = ["resource_shortest_path", "shortest_path"]


def _as_network(network: Network | Graph) -> Network:
    if isinstance(network, Network):
        return network
    if isinstance(network, Graph):
        # Convenient for one-off solves; compile once yourself for repeated ones.
        return network.compile()
    raise TypeError(
        f"expected a celerisroute.Network (or Graph), got {type(network).__name__}"
    )


def shortest_path(
    network: Network | Graph,
    source: Vertex,
    target: Vertex,
    *,
    unconstrained: bool = False,
) -> Solution:
    """Unconstrained minimum-cost path on a DAG (``boost::dag_shortest_paths``).

    Runs in time linear in the part of the graph reachable from ``source``
    and handles negative edge costs. Equivalent to
    ``network.solve(source, target, algorithm="dag")``: if the network has
    resources with finite upper bounds this raises ``ValueError`` unless
    ``unconstrained=True`` states that the bounds are to be ignored.
    """
    return _as_network(network).solve(
        source, target, algorithm="dag", unconstrained=unconstrained
    )


def resource_shortest_path(
    network: Network | Graph,
    source: Vertex,
    target: Vertex,
    *,
    dominance_tolerance: float = 0.0,
) -> Solution:
    """Minimum-cost path that respects every resource bound (``boost::r_c_shortest_paths``).

    Exact for the default ``dominance_tolerance=0``. Equivalent to
    ``network.solve(source, target, algorithm="rcsp")``.
    """
    return _as_network(network).solve(
        source, target, algorithm="rcsp", dominance_tolerance=dominance_tolerance
    )
