"""Resource-constrained shortest paths with one and two resources.

Layered random DAGs in which cheap edges consume more of resource 0. The
bound of every resource is placed between the least possible consumption
(``tightness = 0``) and the consumption of the unconstrained optimum
(``tightness = 1``, the bound stops binding). Label counts are reported next
to the times, because they, not the graph size, drive the cost of RCSP.

One-resource answers are checked against an independent NumPy dynamic
program over (vertex, resource consumed) whenever its table fits in memory.

Run with:  python benchmarks/benchmark_rcsp.py [--quick] [--seed N] [--output FILE]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import celerisroute as cr
from _common import (
    format_seconds, layered_dag, reference_constrained_cost, repeat, save, summary, timed,
)

DEGREE = 4
# (approximate vertices, layer width, resources, tightness). Label counts grow
# quickly with the number of layers (path length) and with loose bounds, far
# more than with the number of vertices, so deep two-resource instances with
# loose bounds are left out: they do not finish in benchmark time.
CASES = (
    (1_000, 100, 1, 0.25), (1_000, 100, 1, 0.75), (1_000, 100, 2, 0.25), (1_000, 100, 2, 0.75),
    (1_000, 33, 1, 0.25), (1_000, 33, 1, 0.75), (1_000, 33, 2, 0.25), (1_000, 33, 2, 0.50),
    (10_000, 1_000, 1, 0.25), (10_000, 1_000, 1, 0.75), (10_000, 1_000, 2, 0.25), (10_000, 1_000, 2, 0.75),
    (10_000, 333, 1, 0.25), (10_000, 333, 1, 0.75), (10_000, 333, 2, 0.25),
    (100_000, 10_000, 1, 0.25), (100_000, 10_000, 1, 0.75),
    (100_000, 10_000, 2, 0.25), (100_000, 10_000, 2, 0.75),
)
QUICK_CASES = tuple(case for case in CASES if case[0] <= 10_000 and case[1] != 333)
REFERENCE_TABLE_LIMIT = 60_000_000  # cells of the validation DP


def resource_bounds(instance, tightness: float) -> list[float]:
    """Integer bounds between the minimum consumption and that of the cheapest path."""
    free = instance.graph().compile()  # resources declared without bounds: only reported
    cheapest = free.solve(instance.source, instance.sink, algorithm="dag")
    bounds = []
    for index, column in enumerate(instance.resources):
        least = free.with_costs(column).solve(instance.source, instance.sink, algorithm="dag").cost
        used = cheapest.resources[f"r{index}"]
        bounds.append(float(math.floor(least + tightness * (used - least))))
    return bounds


def run(case, seed: int) -> dict:
    num_vertices, width, num_resources, tightness = case
    degree = DEGREE
    rng = np.random.default_rng([seed, num_vertices, width, degree, num_resources])
    instance = layered_dag(rng, num_vertices, width, degree, num_resources=num_resources)
    bounds = resource_bounds(instance, tightness)

    build_seconds, graph = timed(instance.graph, bounds)
    compile_seconds, network = timed(graph.compile)
    source, sink = instance.source, instance.sink

    native: list[float] = []

    def solve() -> cr.Solution:
        solution = network.solve(source, sink, algorithm="rcsp")
        native.append(solution.statistics.solve_time)
        return solution

    solution = solve()  # warm-up: fills the label pool
    native.clear()
    facing = repeat(solve, budget=1.0, minimum=3, maximum=30)
    stats = solution.statistics
    unconstrained = network.solve(source, sink, algorithm="dag", unconstrained=True)
    unconstrained_seconds = repeat(
        lambda: network.solve(source, sink, algorithm="dag", unconstrained=True)
    )

    # Checks that hold for every instance.
    if solution.found:
        for index, bound in enumerate(bounds):
            assert solution.resources[f"r{index}"] <= bound
        assert solution.cost >= unconstrained.cost
    validated = False
    if num_resources == 1 and instance.num_vertices * (bounds[0] + 1) <= REFERENCE_TABLE_LIMIT:
        expected = reference_constrained_cost(instance, int(bounds[0]))
        actual = solution.cost if solution.found else math.inf
        if actual != expected:
            raise AssertionError(f"wrong answer: {solution.status} {actual} != {expected}")
        validated = True

    return {
        "num_vertices": instance.num_vertices,
        "num_edges": instance.num_edges,
        "num_layers": instance.num_layers,
        "layer_width": width,
        "out_degree": instance.degree,
        "num_resources": num_resources,
        "tightness": tightness,
        "resource_upper_bounds": bounds,
        "status": str(solution.status),
        "cost": solution.cost if solution.found else None,
        "unconstrained_cost": unconstrained.cost,
        "resources_used": solution.resources if solution.found else None,
        "validated_against_numpy_dp": validated,
        "labels_created": stats.labels_created,
        "labels_extended": stats.labels_extended,
        "labels_dominated": stats.labels_dominated,
        "labels_infeasible": stats.labels_infeasible,
        "max_active_labels": stats.max_active_labels,
        "pareto_labels_at_target": stats.pareto_labels,
        "build_seconds": build_seconds,
        "compile_seconds": compile_seconds,
        "solve_python_facing_seconds": summary(facing),
        "solve_native_seconds": summary(native),
        "unconstrained_dag_solve_seconds": summary(unconstrained_seconds),
        "raw_solve_python_facing_seconds": facing,
        "raw_solve_native_seconds": native,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="only the instances that solve in milliseconds")
    parser.add_argument("--seed", type=int, default=20240607)
    parser.add_argument("--output", help="result file (default: benchmarks/results/rcsp.json)")
    arguments = parser.parse_args()

    cases = QUICK_CASES if arguments.quick else CASES
    print(f"{'vertices':>8} {'edges':>7} {'layers':>6} {'res':>3} {'tight':>5} {'status':>10} | "
          f"{'labels':>10} {'dominated':>10} {'infeasible':>10} {'pareto':>6} | "
          f"{'solve':>10} {'us/label':>8} {'vs dag':>7} {'check':>5}")
    results = []
    for case in cases:
        row = run(case, arguments.seed)
        results.append(row)
        native = row["solve_native_seconds"]["median"]
        dag = row["unconstrained_dag_solve_seconds"]["median"]
        print(f"{row['num_vertices']:8d} {row['num_edges']:7d} {row['num_layers']:6d} "
              f"{case[2]:3d} {case[3]:5.2f} {row['status']:>10} | "
              f"{row['labels_created']:10d} {row['labels_dominated']:10d} "
              f"{row['labels_infeasible']:10d} {row['pareto_labels_at_target']:6d} | "
              f"{format_seconds(native)} {native / max(row['labels_created'], 1) * 1e6:8.3f} "
              f"{native / dag:6.0f}x {'dp' if row['validated_against_numpy_dp'] else '-':>5}",
              flush=True)
    config = {
        "seed": arguments.seed, "quick": arguments.quick, "cases": cases, "out_degree": DEGREE,
        "cost_range": [-5, 20], "resource_range": [1, 10], "algorithm": "rcsp",
        "dominance_tolerance": 0.0,
        "timing": "wall clock; solve = median of repeated network.solve calls after one warm-up",
    }
    path = save("rcsp", config, results, arguments.output)
    print("\n'dp' = cost equals the independent NumPy (vertex, resource) dynamic program")
    print(f"raw results written to {path}")


if __name__ == "__main__":
    main()
