"""Unconstrained DAG shortest paths: construction, compilation and solve time.

Layered random DAGs from about 1 thousand to 1 million vertices, varying the
layer width and the out-degree (density). Every reported cost is checked
against an independent NumPy dynamic program.

Run with:  python benchmarks/benchmark_dag.py [--quick] [--seed N] [--output FILE]
"""

from __future__ import annotations

import argparse

import numpy as np

import celerispath as cp
from _common import format_seconds, layered_dag, reference_cost, repeat, save, summary, timed

SIZES = (1_000, 10_000, 100_000, 1_000_000)
WIDTHS = (10, 100, 1_000)
DEGREES = (2, 8)


def run(num_vertices: int, width: int, degree: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, num_vertices, width, degree])
    instance = layered_dag(rng, num_vertices, width, degree)

    build_seconds, graph = timed(instance.graph)
    compile_seconds, network = timed(graph.compile)
    source, sink = instance.source, instance.sink

    native: list[float] = []

    def solve() -> cp.Solution:
        solution = network.solve(source, sink, algorithm="dag")
        native.append(solution.statistics.solve_time)
        return solution

    solution = solve()  # warm-up: allocates the reusable workspace
    native.clear()
    facing = repeat(solve)
    reference_seconds, expected = timed(reference_cost, instance)
    if solution.status != "optimal" or solution.cost != expected:
        raise AssertionError(f"wrong answer: {solution.status} {solution.cost} != {expected}")

    return {
        "num_vertices": instance.num_vertices,
        "num_edges": instance.num_edges,
        "num_layers": instance.num_layers,
        "layer_width": width,
        "out_degree": instance.degree,
        "path_edges": len(solution.edges),
        "cost": solution.cost,
        "validated_against_numpy_dp": True,
        "build_seconds": build_seconds,
        "compile_seconds": compile_seconds,
        "solve_python_facing_seconds": summary(facing),
        "solve_native_seconds": summary(native),
        "numpy_reference_dp_seconds": reference_seconds,
        "raw_solve_python_facing_seconds": facing,
        "raw_solve_native_seconds": native,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="skip the 1M-vertex instances")
    parser.add_argument("--seed", type=int, default=20240607)
    parser.add_argument("--output", help="result file (default: benchmarks/results/dag.json)")
    arguments = parser.parse_args()

    sizes = SIZES[:-1] if arguments.quick else SIZES
    print(f"{'vertices':>9} {'edges':>9} {'layers':>7} {'width':>5} {'deg':>3} | "
          f"{'build':>10} {'compile':>10} {'solve':>10} {'native':>10} | {'M edges/s':>9}")
    results = []
    for num_vertices in sizes:
        for width in WIDTHS:
            if width * 2 > num_vertices:
                continue
            for degree in DEGREES:
                row = run(num_vertices, width, degree, arguments.seed)
                results.append(row)
                solve = row["solve_python_facing_seconds"]["median"]
                native = row["solve_native_seconds"]["median"]
                print(f"{row['num_vertices']:9d} {row['num_edges']:9d} {row['num_layers']:7d} "
                      f"{width:5d} {row['out_degree']:3d} | {format_seconds(row['build_seconds'])} "
                      f"{format_seconds(row['compile_seconds'])} {format_seconds(solve)} "
                      f"{format_seconds(native)} | {row['num_edges'] / native / 1e6:9.1f}")
    config = {
        "seed": arguments.seed, "quick": arguments.quick, "sizes": sizes, "widths": WIDTHS,
        "degrees": DEGREES, "cost_range": [-5, 20], "algorithm": "dag",
        "timing": "wall clock; solve = median of repeated network.solve calls after one warm-up",
    }
    path = save("dag", config, results, arguments.output)
    print(f"\nall costs match the NumPy reference; raw results written to {path}")


if __name__ == "__main__":
    main()
