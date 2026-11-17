"""Helpers shared by the benchmark scripts: instances, timing, result files."""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

import celerispath as cp

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class LayeredDag:
    """A random layered DAG with one source (vertex 0) and one sink (last vertex).

    The source has an edge to every vertex of the first layer, every vertex of
    a layer has ``degree`` edges to uniformly random vertices of the next
    layer, and every vertex of the last layer has an edge to the sink. Every
    source-sink path therefore has ``num_layers + 1`` edges, and the number of
    paths grows like ``degree ** num_layers``.
    """

    num_layers: int
    width: int
    degree: int
    edge_source: np.ndarray
    edge_target: np.ndarray
    cost: np.ndarray
    resources: list[np.ndarray] = field(default_factory=list)
    # Edge index ranges whose sources lie in the same layer, in layer order.
    blocks: list[tuple[int, int]] = field(default_factory=list)

    @property
    def num_vertices(self) -> int:
        return self.num_layers * self.width + 2

    @property
    def num_edges(self) -> int:
        return len(self.cost)

    @property
    def source(self) -> int:
        return 0

    @property
    def sink(self) -> int:
        return self.num_vertices - 1

    def graph(self, upper: list[float] | None = None) -> cp.Graph:
        """An uncompiled celerispath graph; ``upper[i]`` bounds resource ``r{i}``."""
        graph = cp.Graph(num_vertices=self.num_vertices)
        for index in range(len(self.resources)):
            bound = float("inf") if upper is None else upper[index]
            graph.add_resource(cp.Resource(name=f"r{index}", upper=bound))
        graph.add_edges(
            self.edge_source,
            self.edge_target,
            self.cost,
            resources={f"r{index}": column for index, column in enumerate(self.resources)},
        )
        return graph


def layered_dag(
    rng: np.random.Generator,
    num_vertices: int,
    width: int,
    degree: int,
    *,
    num_resources: int = 0,
    cost_low: int = -5,
    cost_high: int = 20,
    resource_high: int = 10,
) -> LayeredDag:
    """Generate a layered DAG with about ``num_vertices`` vertices.

    Costs are integers in ``[cost_low, cost_high]`` stored as floats (negative
    costs included), so path costs are exact in floating point. Resource 0 is
    anti-correlated with the cost (cheap edges consume more), which is what
    makes constrained instances hard; further resources are independent.
    """
    num_layers = max(2, round(num_vertices / width))
    degree = min(degree, width)
    first = 1 + np.arange(width, dtype=np.int64)
    sources = [np.zeros(width, dtype=np.int64)]
    targets = [first]
    blocks = [(0, width)]
    offset = width
    for layer in range(num_layers - 1):
        base = 1 + layer * width
        sources.append(np.repeat(base + np.arange(width, dtype=np.int64), degree))
        targets.append(base + width + rng.integers(0, width, size=width * degree, dtype=np.int64))
        blocks.append((offset, offset + width * degree))
        offset += width * degree
    last = 1 + (num_layers - 1) * width + np.arange(width, dtype=np.int64)
    sources.append(last)
    targets.append(np.full(width, num_layers * width + 1, dtype=np.int64))
    blocks.append((offset, offset + width))

    edge_source = np.concatenate(sources)
    edge_target = np.concatenate(targets)
    num_edges = len(edge_source)
    cost = rng.integers(cost_low, cost_high + 1, size=num_edges).astype(np.float64)
    resources = []
    for index in range(num_resources):
        if index == 0:
            scaled = (cost_high - cost) / (cost_high - cost_low) * (resource_high - 1)
            column = np.clip(np.rint(scaled + rng.integers(-2, 3, size=num_edges)), 1, resource_high)
        else:
            column = rng.integers(1, resource_high + 1, size=num_edges)
        resources.append(column.astype(np.float64))
    return LayeredDag(num_layers, width, degree, edge_source, edge_target, cost, resources, blocks)


def reference_cost(instance: LayeredDag, cost: np.ndarray | None = None) -> float:
    """Shortest source-sink cost by a NumPy layer-by-layer DP (independent of celerispath)."""
    cost = instance.cost if cost is None else cost
    distance = np.full(instance.num_vertices, np.inf)
    distance[instance.source] = 0.0
    for start, stop in instance.blocks:
        candidates = distance[instance.edge_source[start:stop]] + cost[start:stop]
        np.minimum.at(distance, instance.edge_target[start:stop], candidates)
    return float(distance[instance.sink])


def reference_constrained_cost(instance: LayeredDag, upper: int) -> float:
    """Exact one-resource constrained optimum for integer resource data.

    Dynamic program over (vertex, resource consumed); independent of celerispath.
    Memory is ``num_vertices * (upper + 1)`` doubles.
    """
    resource = instance.resources[0].astype(np.int64)
    table = np.full((instance.num_vertices, upper + 1), np.inf)
    table[instance.source, 0] = 0.0
    for start, stop in instance.blocks:
        source = instance.edge_source[start:stop]
        target = instance.edge_target[start:stop]
        cost = instance.cost[start:stop]
        delta = resource[start:stop]
        for amount in np.unique(delta):
            if amount > upper:
                continue
            chosen = delta == amount
            candidates = table[source[chosen], : upper + 1 - amount] + cost[chosen, None]
            np.minimum.at(table[:, amount:], target[chosen], candidates)
    return float(table[instance.sink].min())


def timed(function, *arguments, **keywords):
    """Return (seconds, result) of one call."""
    start = time.perf_counter()
    result = function(*arguments, **keywords)
    return time.perf_counter() - start, result


def repeat(function, *, budget: float = 0.5, minimum: int = 3, maximum: int = 50) -> list[float]:
    """Wall-clock seconds of repeated calls: at least `minimum`, then until `budget` is used."""
    samples = [timed(function)[0]]
    while len(samples) < minimum or (sum(samples) < budget and len(samples) < maximum):
        samples.append(timed(function)[0])
    return samples


def summary(samples: list[float]) -> dict:
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "repeats": len(samples),
    }


def environment() -> dict:
    def sysctl(name: str) -> str | None:
        try:
            return subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "gil_enabled": gil,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "celerispath": cp.__version__,
    }


def save(name: str, config: dict, results, output: str | None) -> Path:
    """Write configuration, environment and raw results as JSON."""
    path = Path(output) if output else RESULTS_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"benchmark": name, "config": config, "environment": environment(), "results": results}
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


def format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:7.2f} ms"
    return f"{seconds:7.3f} s "
