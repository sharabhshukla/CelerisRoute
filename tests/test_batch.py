import math
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import scipy.sparse as sp

import celerisroute as cr
from _reference import random_dag
from test_pricing import busiest_pair, random_matrix


def snapshot(result: cr.PricingResult):
    """Everything observable about a pricing result except its timings."""
    stats = result.statistics
    return (
        str(result.status),
        result.best_reduced_cost,
        [
            (
                c.edges.tolist(),
                c.original_cost,
                c.reduced_cost,
                c.resources,
                c.coefficient_indices.tolist(),
                c.coefficient_values.tolist(),
            )
            for c in result
        ],
        (stats.algorithm, stats.labels_created, stats.labels_extended, stats.labels_dominated,
         stats.labels_infeasible, stats.pareto_labels),
    )


def make_engines(count: int, num_constraints: int = 6, seed: int = 0, shared_network: bool = False):
    """A mix of dag and rcsp engines, on separate networks or on one shared network."""
    rng = np.random.default_rng(seed)
    engines = []
    shared = None
    for index in range(count):
        constrained = index % 2 == 1
        if shared_network and shared is not None:
            instance, network, pairs = shared
        else:
            instance = random_dag(
                rng, 14, 60, num_resources=2 if constrained or shared_network else 0, tightness=0.6
            )
            network = instance.build()
            source, target, _ = busiest_pair(instance)
            pairs = [(source, target)]
            if shared_network:
                pairs = [(s, t) for s in range(14) for t in range(14) if s != t]
                shared = (instance, network, pairs)
        source, target = pairs[index % len(pairs)] if not shared_network else pairs[int(rng.integers(len(pairs)))]
        matrix = sp.csr_array(random_matrix(rng, num_constraints, instance.num_edges))
        engines.append(
            cr.PricingEngine(
                network, source, target,
                algorithm="auto" if not shared_network else ("rcsp" if constrained else "dag"),
                unconstrained=shared_network and not constrained,
                dual_matrix=matrix, name=f"engine-{index}",
            )
        )
    return engines


@pytest.mark.parametrize("threads", [1, 2, 3, 8, 64])
def test_batch_equals_serial(threads):
    engines = make_engines(24)
    assert {e.algorithm for e in engines} == {"dag", "rcsp"}
    batch = cr.BatchPricingEngine(engines=engines)
    assert len(batch) == 24 and batch.engines == tuple(engines)
    rng = np.random.default_rng(1)
    for _ in range(5):
        duals = rng.integers(-8, 9, size=6) / 2.0
        serial = [snapshot(e.solve(duals, negative_only=False)) for e in engines]
        results = batch.solve_batch(duals, max_paths_per_engine=1, threads=threads, negative_only=False)
        assert isinstance(results, list) and len(results) == len(engines)
        assert [snapshot(r) for r in results] == serial  # results[i] belongs to engines[i]

        serial = [snapshot(e.solve(duals)) for e in engines]
        assert [snapshot(r) for r in batch.solve_batch(duals, threads=threads)] == serial


def test_batch_is_deterministic_across_runs_and_thread_counts():
    engines = make_engines(40, seed=7)
    batch = cr.BatchPricingEngine(engines)
    duals = np.linspace(-3.0, 3.0, 6)
    reference = [snapshot(r) for r in batch.solve_batch(duals, threads=1, negative_only=False)]
    for threads in (None, 2, 4, 8, 8, 8, 16):
        again = batch.solve_batch(duals, threads=threads, negative_only=False)
        assert [snapshot(r) for r in again] == reference


def test_engines_sharing_one_network():
    engines = make_engines(30, seed=3, shared_network=True)
    assert len({id(e.network) for e in engines}) == 1
    batch = cr.BatchPricingEngine(engines)
    rng = np.random.default_rng(2)
    for _ in range(4):
        duals = rng.integers(-8, 9, size=6) / 2.0
        serial = [snapshot(e.solve(duals, negative_only=False)) for e in engines]
        parallel = batch.solve_batch(duals, threads=8, negative_only=False)
        assert [snapshot(r) for r in parallel] == serial


def test_per_engine_dual_vectors():
    engines = make_engines(10, seed=4)
    batch = cr.BatchPricingEngine(engines)
    rng = np.random.default_rng(9)
    duals = rng.integers(-8, 9, size=(10, 6)) / 2.0
    serial = [snapshot(e.solve(row, negative_only=False)) for e, row in zip(engines, duals)]
    as_matrix = batch.solve_batch(duals, threads=4, negative_only=False)
    as_list = batch.solve_batch([row for row in duals], threads=4, negative_only=False)
    assert [snapshot(r) for r in as_matrix] == serial
    assert [snapshot(r) for r in as_list] == serial


def test_engines_with_different_numbers_of_constraints():
    first = make_engines(1, num_constraints=3, seed=5)[0]
    second = make_engines(1, num_constraints=5, seed=6)[0]
    batch = cr.BatchPricingEngine([first, second])
    duals = [np.array([1.0, -2.0, 0.5]), np.array([0.5, 0.0, 3.0, -1.0, 2.0])]
    serial = [snapshot(e.solve(d, negative_only=False)) for e, d in zip((first, second), duals)]
    assert [snapshot(r) for r in batch.solve_batch(duals, threads=2, negative_only=False)] == serial
    with pytest.raises(ValueError, match=r"engines\[1\]"):
        batch.solve_batch(np.zeros(3))  # a shared vector fits only the first engine


def test_multiple_paths_per_engine():
    engines = [e for e in make_engines(16, seed=8) if e.algorithm == "dag"]
    batch = cr.BatchPricingEngine(engines)
    duals = np.array([1.0, -1.0, 2.0, 0.5, -2.5, 3.0])
    serial = [snapshot(e.solve(duals, max_paths=4, negative_only=False)) for e in engines]
    parallel = batch.solve_batch(duals, max_paths_per_engine=4, threads=4, negative_only=False)
    assert [snapshot(r) for r in parallel] == serial
    assert max(len(r) for r in parallel) > 1


def test_multiple_paths_with_an_rcsp_engine_raise_not_implemented():
    engines = make_engines(4, seed=8)
    batch = cr.BatchPricingEngine(engines)
    with pytest.raises(NotImplementedError, match="max_paths > 1"):
        batch.solve_batch(np.zeros(6), max_paths_per_engine=2)


def test_failing_engine_is_identified():
    engines = make_engines(6, seed=10)
    batch = cr.BatchPricingEngine(engines)
    duals = np.zeros((6, 6))
    duals[4, 2] = math.nan
    duals[1, 0] = math.inf
    with pytest.raises(cr.BatchPricingError) as info:
        batch.solve_batch(duals, threads=3)
    error = info.value
    assert sorted(error.errors) == [1, 4]
    assert error.engine_index == 1
    assert "engines[1]" in str(error) and "engines[4]" in str(error)
    assert "engine-4" in str(error)  # the engine's name
    assert "not finite" in error.errors[4]
    assert isinstance(error, RuntimeError) and isinstance(error, cr.CelerisRouteError)
    # The batch and its engines remain usable afterwards.
    assert len(batch.solve_batch(np.zeros(6), negative_only=False)) == 6


def test_invalid_batch_arguments():
    engines = make_engines(3, seed=11)
    batch = cr.BatchPricingEngine(engines)
    with pytest.raises(TypeError, match=r"engines\[1\]"):
        cr.BatchPricingEngine([engines[0], "engine"])
    with pytest.raises(ValueError, match="more than once"):
        cr.BatchPricingEngine([engines[0], engines[0]])
    with pytest.raises(ValueError, match="2 dual vectors for 3 engines"):
        batch.solve_batch(np.zeros((2, 6)))
    with pytest.raises(ValueError, match=r"engines\[0\]"):
        batch.solve_batch(np.zeros(5))
    with pytest.raises(ValueError, match=r"engines\[0\]"):
        batch.solve_batch(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        batch.solve_batch(np.zeros((3, 6, 1)))
    for bad in (0, -2):
        with pytest.raises(ValueError, match="threads"):
            batch.solve_batch(np.zeros(6), threads=bad)
        with pytest.raises(ValueError, match="threads"):
            cr.BatchPricingEngine(engines, threads=bad)
    with pytest.raises(TypeError, match="threads"):
        batch.solve_batch(np.zeros(6), threads=2.5)
    with pytest.raises(ValueError, match="at least 1"):
        batch.solve_batch(np.zeros(6), max_paths_per_engine=0)
    with pytest.raises(ValueError):
        batch.solve_batch(np.zeros(6), tolerance=-1.0)


def test_empty_and_single_engine_batches():
    assert cr.BatchPricingEngine([]).solve_batch(np.zeros(3)) == []
    engine = make_engines(1, seed=12)[0]
    duals = np.ones(6)
    result = cr.BatchPricingEngine([engine], threads=4).solve_batch(duals, negative_only=False)
    assert [snapshot(r) for r in result] == [snapshot(engine.solve(duals, negative_only=False))]


def test_one_engine_shared_by_python_threads():
    """Concurrent solve() calls on a single engine are serialised, not corrupted."""
    engine = make_engines(2, seed=13)[1]
    assert engine.algorithm == "rcsp"
    rng = np.random.default_rng(14)
    duals = rng.integers(-8, 9, size=(64, 6)) / 2.0
    expected = [snapshot(engine.solve(row, negative_only=False)) for row in duals]
    barrier = threading.Barrier(8)

    def work(offset: int):
        barrier.wait()
        return [
            (i, snapshot(engine.solve(duals[i], negative_only=False)))
            for i in range(offset, 64, 8)
        ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        for chunk in pool.map(work, range(8)):
            for index, got in chunk:
                assert got == expected[index]


def test_network_solves_from_python_threads():
    """A compiled network is immutable and can be queried concurrently."""
    rng = np.random.default_rng(15)
    instance = random_dag(rng, 40, 200, num_resources=1, tightness=0.6)
    network = instance.build()
    pairs = [(int(a), int(b)) for a, b in rng.integers(0, 40, size=(200, 2))]
    expected = [(str(s.status), s.cost, s.edges.tolist()) for s in (network.solve(a, b) for a, b in pairs)]

    def work(index: int):
        solution = network.solve(*pairs[index])
        return (str(solution.status), solution.cost, solution.edges.tolist())

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(work, range(len(pairs)))) == expected


def test_batch_statistics_are_per_engine():
    engines = make_engines(4, seed=16)
    results = cr.BatchPricingEngine(engines).solve_batch(np.zeros(6), threads=2, negative_only=False)
    for engine, result in zip(engines, results):
        stats = result.statistics
        assert stats.algorithm == engine.algorithm
        assert stats.solve_time >= 0.0 and stats.total_time >= stats.solve_time
        assert (stats.labels_created > 0) == (engine.algorithm == "rcsp")
