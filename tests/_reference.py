"""Independent pure-Python reference implementations used to check the solvers.

Nothing here calls the native core for anything but graph construction, so a
bug in the solvers cannot hide behind itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import celerisroute as cr


@dataclass
class Instance:
    num_vertices: int
    source: np.ndarray
    target: np.ndarray
    cost: np.ndarray
    resources: dict[str, np.ndarray] = field(default_factory=dict)
    upper: dict[str, float] = field(default_factory=dict)

    @property
    def num_edges(self) -> int:
        return int(self.source.shape[0])

    def build(self) -> cr.Network:
        graph = cr.Graph(num_vertices=self.num_vertices)
        for name, limit in self.upper.items():
            graph.add_resource(cr.Resource(name=name, upper=limit))
        graph.add_edges(self.source, self.target, self.cost, resources=self.resources or None)
        return graph.compile()


def random_dag(
    rng: np.random.Generator,
    num_vertices: int,
    num_edges: int,
    *,
    cost_low: int = -4,
    cost_high: int = 9,
    num_resources: int = 0,
    resource_high: int = 6,
    tightness: float = 0.5,
) -> Instance:
    """Random DAG with parallel edges, hidden vertex order and integer-valued data.

    Integer values keep every path sum exact in floating point, so costs can
    be compared with ``==`` and ties are real ties.
    """
    order = rng.permutation(num_vertices)  # order[i]: vertex at topological position i
    low = rng.integers(0, num_vertices - 1, size=num_edges)
    high = rng.integers(low + 1, num_vertices)
    source = order[low].astype(np.int64)
    target = order[high].astype(np.int64)
    cost = rng.integers(cost_low, cost_high + 1, size=num_edges).astype(np.float64)
    resources = {}
    upper = {}
    for r in range(num_resources):
        name = f"r{r}"
        resources[name] = rng.integers(0, resource_high + 1, size=num_edges).astype(np.float64)
        # Roughly `tightness` times what a typical medium-length path consumes.
        upper[name] = float(np.ceil(tightness * resource_high * max(2, num_vertices // 3)))
    return Instance(num_vertices, source, target, cost, resources, upper)


def enumerate_paths(instance: Instance, source: int, target: int) -> list[list[int]]:
    """Every directed source-target path, as a list of edge ids (DFS)."""
    out: dict[int, list[int]] = {}
    for edge, u in enumerate(instance.source.tolist()):
        out.setdefault(u, []).append(edge)
    targets = instance.target.tolist()
    paths: list[list[int]] = []
    stack: list[int] = []

    def visit(u: int) -> None:
        if u == target:
            paths.append(list(stack))
            # A DAG cannot return to the target, but may continue past it.
        for edge in out.get(u, ()):
            stack.append(edge)
            visit(targets[edge])
            stack.pop()

    visit(source)
    return paths


def path_cost(instance: Instance, edges: list[int], cost: np.ndarray | None = None) -> float:
    values = instance.cost if cost is None else cost
    total = 0.0
    for edge in edges:
        total += float(values[edge])
    return total


def path_resource(instance: Instance, edges: list[int], name: str) -> float:
    total = 0.0
    for edge in edges:
        total += float(instance.resources[name][edge])
    return total


def is_feasible(instance: Instance, edges: list[int]) -> bool:
    """Running totals stay within the upper bounds after every edge."""
    for name, limit in instance.upper.items():
        total = 0.0
        for edge in edges:
            total += float(instance.resources[name][edge])
            if total > limit:
                return False
    return True


def best_cost(
    instance: Instance,
    paths: list[list[int]],
    *,
    feasible_only: bool,
    cost: np.ndarray | None = None,
) -> float:
    candidates = [p for p in paths if not feasible_only or is_feasible(instance, p)]
    if not candidates:
        return float("inf")
    return min(path_cost(instance, p, cost) for p in candidates)


def check_path(instance: Instance, edges: list[int], source: int, target: int) -> None:
    """Assert that `edges` is a contiguous source-target walk."""
    at = source
    for edge in edges:
        assert int(instance.source[edge]) == at
        at = int(instance.target[edge])
    assert at == target
