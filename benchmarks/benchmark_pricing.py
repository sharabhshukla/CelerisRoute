"""Repeated pricing and batch pricing.

Part 1 times every stage of the column-generation inner loop separately:
graph construction, compilation, engine construction, and then, per solve
with fresh duals, the reduced-cost update, the native solve, the path
reconstruction and the total time seen from Python.

Part 2 compares pricing 10, 50 and 100 engines one after another from Python
with ``BatchPricingEngine.solve_batch`` on 1, 2, 4, 8 and all hardware threads.

Run with:  python benchmarks/benchmark_pricing.py [--quick] [--seed N] [--output FILE]
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np
import scipy.sparse as sp

import celerisroute as cr
from _common import format_seconds, layered_dag, reference_cost, save, summary, timed
from benchmark_rcsp import resource_bounds

NUM_ROWS = 200  # master constraints
NONZEROS_PER_EDGE = 2
DUAL_SCALE = 3.0
TIGHTNESS = 0.25

# (algorithm, approximate vertices, layer width, out-degree, solves)
REPEATED = (
    ("dag", 10_000, 100, 8, 200),
    ("dag", 100_000, 1_000, 8, 100),
    ("dag", 1_000_000, 1_000, 4, 30),
    ("rcsp", 1_000, 100, 4, 100),
    ("rcsp", 10_000, 1_000, 4, 20),
)
# (algorithm, approximate vertices, layer width, out-degree) of every engine in a batch
BATCH = (
    ("dag", 10_000, 100, 8),
    ("rcsp", 1_000, 100, 4),
)
BATCH_SIZES = (10, 50, 100)
BATCH_REPEATS = 7


def dual_matrix(rng: np.random.Generator, num_edges: int) -> sp.csc_array:
    """NUM_ROWS x edges, NONZEROS_PER_EDGE random integer coefficients per edge."""
    rows = rng.integers(0, NUM_ROWS, size=num_edges * NONZEROS_PER_EDGE)
    data = rng.integers(1, 4, size=num_edges * NONZEROS_PER_EDGE).astype(np.float64)
    indptr = np.arange(0, num_edges * NONZEROS_PER_EDGE + 1, NONZEROS_PER_EDGE)
    return sp.csc_array((data, rows, indptr), shape=(NUM_ROWS, num_edges))


def make_engine(algorithm: str, num_vertices: int, width: int, degree: int, rng, timings=None):
    """Build instance -> graph -> network -> engine, optionally recording stage times."""
    constrained = algorithm == "rcsp"
    instance = layered_dag(rng, num_vertices, width, degree, num_resources=int(constrained))
    bounds = resource_bounds(instance, TIGHTNESS) if constrained else None
    matrix = dual_matrix(rng, instance.num_edges)
    build_seconds, graph = timed(instance.graph, bounds)
    compile_seconds, network = timed(graph.compile)
    engine_seconds, engine = timed(
        cr.PricingEngine, network, instance.source, instance.sink,
        algorithm=algorithm, dual_matrix=matrix,
    )
    if timings is not None:
        timings.update(build_seconds=build_seconds, compile_seconds=compile_seconds,
                       engine_construction_seconds=engine_seconds)
    return instance, matrix, engine


def repeated_pricing(case, seed: int) -> dict:
    algorithm, num_vertices, width, degree, solves = case
    rng = np.random.default_rng([seed, num_vertices, width, degree, algorithm == "rcsp"])
    row: dict = {}
    instance, matrix, engine = make_engine(algorithm, num_vertices, width, degree, rng, row)

    engine.solve(np.zeros(NUM_ROWS), negative_only=False)  # warm-up: sizes the workspaces
    stages = {"update": [], "native_solve": [], "reconstruction": [], "native_total": [],
              "python_facing_total": []}
    labels = []
    for index in range(solves):
        duals = rng.uniform(0.0, DUAL_SCALE, size=NUM_ROWS)
        start = time.perf_counter()
        result = engine.solve(duals, max_paths=1, negative_only=False)
        elapsed = time.perf_counter() - start
        stats = result.statistics
        stages["update"].append(stats.update_time)
        stages["native_solve"].append(stats.solve_time)
        stages["reconstruction"].append(stats.reconstruction_time)
        stages["native_total"].append(stats.total_time)
        stages["python_facing_total"].append(elapsed)
        labels.append(stats.labels_created)
        if algorithm == "dag" and index < 3:
            # Independent check: NumPy DP on c - A^T pi.
            expected = reference_cost(instance, instance.cost - matrix.T @ duals)
            if abs(result.best_reduced_cost - expected) > 1e-7 * max(1.0, abs(expected)):
                raise AssertionError(f"wrong reduced cost {result.best_reduced_cost} != {expected}")

    row.update(
        algorithm=algorithm,
        num_vertices=instance.num_vertices,
        num_edges=instance.num_edges,
        num_layers=instance.num_layers,
        layer_width=width,
        active_edges=engine.num_active_edges,
        num_constraints=NUM_ROWS,
        matrix_nonzeros=int(matrix.nnz),
        solves=solves,
        median_labels_created=statistics.median(labels),
        seconds={name: summary(samples) for name, samples in stages.items()},
        raw_seconds=stages,
    )
    if algorithm == "dag":
        k_best = []
        for _ in range(min(solves, 20)):
            duals = rng.uniform(0.0, DUAL_SCALE, size=NUM_ROWS)
            k_best.append(timed(engine.solve, duals, max_paths=10, negative_only=False)[0])
        row["ten_best_python_facing_seconds"] = summary(k_best)
    return row


def batch_pricing(case, seed: int) -> list[dict]:
    algorithm, num_vertices, width, degree = case
    rng = np.random.default_rng([seed, num_vertices, width, degree, 99])
    engines = [
        make_engine(algorithm, num_vertices, width, degree, rng)[2] for _ in range(max(BATCH_SIZES))
    ]
    hardware = os.cpu_count() or 1
    thread_counts = sorted({1, 2, 4, 8, hardware})
    rows = []
    for size in BATCH_SIZES:
        subset = engines[:size]
        batch = cr.BatchPricingEngine(subset)
        serial: list[float] = []
        parallel: dict[int, list[float]] = {threads: [] for threads in thread_counts}
        batch.solve_batch(np.zeros(NUM_ROWS), threads=hardware, negative_only=False)  # warm-up
        for _ in range(BATCH_REPEATS):
            duals = rng.uniform(0.0, DUAL_SCALE, size=NUM_ROWS)
            start = time.perf_counter()
            reference = [engine.solve(duals, negative_only=False) for engine in subset]
            serial.append(time.perf_counter() - start)
            for threads in thread_counts:
                start = time.perf_counter()
                results = batch.solve_batch(duals, max_paths_per_engine=1, threads=threads,
                                            negative_only=False)
                parallel[threads].append(time.perf_counter() - start)
                for expected, actual in zip(reference, results):  # deterministic, same order
                    assert actual.best_reduced_cost == expected.best_reduced_cost
                    assert actual[0].edges.tolist() == expected[0].edges.tolist()
        serial_median = statistics.median(serial)
        rows.append({
            "algorithm": algorithm,
            "engines": size,
            "vertices_per_engine": subset[0].network.num_vertices,
            "edges_per_engine": subset[0].network.num_edges,
            "serial_python_loop_seconds": summary(serial),
            "batch_seconds": {str(t): summary(samples) for t, samples in parallel.items()},
            "speedup_over_serial": {
                str(t): serial_median / statistics.median(samples) for t, samples in parallel.items()
            },
            "raw_serial_seconds": serial,
            "raw_batch_seconds": {str(t): samples for t, samples in parallel.items()},
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="skip the 1M-vertex engine")
    parser.add_argument("--seed", type=int, default=20240607)
    parser.add_argument("--output", help="result file (default: benchmarks/results/pricing.json)")
    arguments = parser.parse_args()

    print("Part 1: repeated pricing on one engine (medians per solve)")
    print(f"{'alg':>4} {'vertices':>8} {'edges':>8} | {'build':>10} {'compile':>10} {'engine':>10} | "
          f"{'update':>10} {'solve':>10} {'rebuild':>10} {'total':>10} {'py total':>10} | {'labels':>8}")
    repeated = []
    for case in REPEATED:
        if arguments.quick and case[1] >= 1_000_000:
            continue
        row = repeated_pricing(case, arguments.seed)
        repeated.append(row)
        seconds = row["seconds"]
        print(f"{row['algorithm']:>4} {row['num_vertices']:8d} {row['num_edges']:8d} | "
              f"{format_seconds(row['build_seconds'])} {format_seconds(row['compile_seconds'])} "
              f"{format_seconds(row['engine_construction_seconds'])} | "
              f"{format_seconds(seconds['update']['median'])} "
              f"{format_seconds(seconds['native_solve']['median'])} "
              f"{format_seconds(seconds['reconstruction']['median'])} "
              f"{format_seconds(seconds['native_total']['median'])} "
              f"{format_seconds(seconds['python_facing_total']['median'])} | "
              f"{row['median_labels_created']:8.0f}", flush=True)

    print("\nPart 2: batch pricing, serial Python loop vs solve_batch (median seconds, speedup)")
    batches = []
    for case in BATCH:
        for row in batch_pricing(case, arguments.seed):
            batches.append(row)
            cells = "  ".join(
                f"{threads:>2}t {format_seconds(row['batch_seconds'][threads]['median'])} "
                f"({row['speedup_over_serial'][threads]:4.1f}x)"
                for threads in row["batch_seconds"]
            )
            print(f"{row['algorithm']:>4} {row['engines']:3d} engines | serial "
                  f"{format_seconds(row['serial_python_loop_seconds']['median'])} | {cells}", flush=True)

    config = {
        "seed": arguments.seed, "quick": arguments.quick, "num_constraints": NUM_ROWS,
        "nonzeros_per_edge": NONZEROS_PER_EDGE, "dual_scale": DUAL_SCALE,
        "rcsp_tightness": TIGHTNESS, "repeated_cases": REPEATED, "batch_cases": BATCH,
        "batch_sizes": BATCH_SIZES, "batch_repeats": BATCH_REPEATS,
        "timing": "wall clock; update/solve/reconstruction are measured inside the native "
                  "engine, python-facing totals around the Python call",
    }
    path = save("pricing", config, {"repeated": repeated, "batch": batches}, arguments.output)
    print(f"\nraw results written to {path}")


if __name__ == "__main__":
    main()
