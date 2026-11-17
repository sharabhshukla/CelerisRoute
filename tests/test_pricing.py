import math

import numpy as np
import pytest
import scipy.optimize
import scipy.sparse as sp

import celerispath as cp
from _reference import Instance, check_path, enumerate_paths, is_feasible, random_dag


def random_matrix(rng, num_constraints, num_edges, density=0.35):
    """Integer-valued sparse K x E matrix (negative entries included)."""
    dense = rng.integers(-2, 4, size=(num_constraints, num_edges)).astype(np.float64)
    dense[rng.random(dense.shape) > density] = 0.0
    return dense


def random_duals(rng, num_constraints):
    # Multiples of 0.5: all reduced costs are exact binary fractions.
    return rng.integers(-8, 9, size=num_constraints) / 2.0


def busiest_pair(instance: Instance) -> tuple[int, int, list[list[int]]]:
    """The source-target pair connected by the most paths."""
    best = (0, 0, [])
    for source in range(instance.num_vertices):
        for target in range(instance.num_vertices):
            if source == target:
                continue
            paths = enumerate_paths(instance, source, target)
            if len(paths) > len(best[2]):
                best = (source, target, paths)
    return best


def reference_reduced_cost(instance, matrix, duals, edges):
    """c(P) - pi . a_P with a_P accumulated edge by edge, independently of the library."""
    cost = sum(float(instance.cost[e]) for e in edges)
    coefficients = np.zeros(matrix.shape[0])
    for e in edges:
        coefficients += matrix[:, e]
    return cost - float(duals @ coefficients), cost, coefficients


@pytest.fixture
def small():
    """0 -> {1, 2} -> 3 with one constraint row per intermediate vertex."""
    graph = cp.Graph(num_vertices=4)
    graph.add_edge(0, 1, 10.0)  # e0
    graph.add_edge(1, 3, 10.0)  # e1
    graph.add_edge(0, 2, 12.0)  # e2
    graph.add_edge(2, 3, 12.0)  # e3
    network = graph.compile()
    matrix = sp.csr_array(np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]))
    return network, matrix


def test_columns_carry_original_and_reduced_costs(small):
    network, matrix = small
    pricing = cp.PricingEngine(graph=network, source=0, target=3, algorithm="dag", dual_matrix=matrix)
    assert pricing.num_constraints == 2
    assert pricing.algorithm == "dag"
    assert (pricing.source, pricing.target) == (0, 3)
    assert pricing.network is network

    columns = pricing.solve(np.array([0.0, 30.0]), max_paths=5, negative_only=True)
    assert len(columns) == 1 and columns.status == "optimal"
    column = columns[0]
    assert column.path == [0, 2, 3]
    assert column.edges.tolist() == [2, 3]
    assert column.nodes.tolist() == [0, 2, 3]
    assert column.original_cost == 24.0  # the graph's own costs
    assert column.reduced_cost == -6.0  # 24 - 30 * 1
    assert column.resources == {}
    assert column.coefficients.shape == (2, 1)
    assert column.coefficients.toarray().ravel().tolist() == [0.0, 1.0]
    assert column.dense_coefficients().tolist() == [0.0, 1.0]
    assert column.coefficient_indices.tolist() == [1]
    assert column.coefficient_values.tolist() == [1.0]
    assert columns.best_reduced_cost == -6.0


def test_negative_only_filter_and_certificate(small):
    network, matrix = small
    pricing = cp.PricingEngine(network, 0, 3, dual_matrix=matrix)

    nothing = pricing.solve([0.0, 0.0])
    assert len(nothing) == 0 and not nothing
    assert nothing.status == "optimal"  # a path exists; it just does not price out
    assert nothing.best_reduced_cost == 20.0  # certificate: min reduced cost >= 0

    everything = pricing.solve([0.0, 0.0], max_paths=5, negative_only=False)
    assert [c.reduced_cost for c in everything] == [20.0, 24.0]

    # Exactly zero reduced cost is not an improving column.
    assert len(pricing.solve([20.0, 0.0])) == 0
    assert pricing.solve([20.0, 0.0]).best_reduced_cost == 0.0
    # The tolerance decides what counts as negative.
    slightly = pricing.solve([20.0 + 1e-7, 0.0], tolerance=1e-6)
    assert len(slightly) == 0 and slightly.best_reduced_cost == pytest.approx(-1e-7)
    assert len(pricing.solve([20.0 + 1e-7, 0.0], tolerance=1e-9)) == 1
    assert len(pricing.solve([20.0 + 1e-7, 0.0], tolerance=0.0)) == 1


def test_repeated_dual_updates_do_not_leak_state(small):
    network, matrix = small
    pricing = cp.PricingEngine(network, 0, 3, dual_matrix=matrix)
    sequence = [[0.0, 0.0], [50.0, 0.0], [0.0, 50.0], [-5.0, 3.0], [50.0, 0.0], [0.0, 0.0]]
    # (-5, 3): via 1 costs 20 + 5 = 25, via 2 costs 24 - 3 = 21.
    expected = [20.0, -30.0, -26.0, 21.0, -30.0, 20.0]
    for duals, best in zip(sequence, expected):
        assert pricing.solve(duals, negative_only=False)[0].reduced_cost == best
    # The network's own costs are untouched by pricing.
    assert network.edge_cost.tolist() == [10.0, 10.0, 12.0, 12.0]
    assert network.solve(0, 3).cost == 20.0


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo", "dense", "coo_duplicates", "lil"])
def test_dual_matrix_formats_agree(fmt):
    rng = np.random.default_rng(3)
    instance = random_dag(rng, 9, 24)
    source, target, _ = busiest_pair(instance)
    network = instance.build()
    dense = random_matrix(rng, 6, instance.num_edges)
    if fmt == "dense":
        matrix = dense
    elif fmt == "coo_duplicates":
        # Each entry split into two halves stored separately: duplicates must be summed.
        rows, cols = np.nonzero(dense)
        values = dense[rows, cols] / 2.0
        matrix = sp.coo_array(
            (np.concatenate([values, values]), (np.concatenate([rows, rows]), np.concatenate([cols, cols]))),
            shape=dense.shape,
        )
    elif fmt == "lil":
        matrix = sp.lil_array(dense)
    else:
        matrix = getattr(sp, f"{fmt}_array")(dense)
    pricing = cp.PricingEngine(network, source, target, dual_matrix=matrix)
    assert pricing.num_constraints == 6
    for _ in range(5):
        duals = random_duals(rng, 6)
        column = pricing.solve(duals, negative_only=False)[0]
        reduced, cost, coefficients = reference_reduced_cost(instance, dense, duals, column.edges.tolist())
        assert column.reduced_cost == reduced
        assert column.original_cost == cost
        assert column.dense_coefficients().tolist() == coefficients.tolist()


def test_dual_matrix_is_copied():
    graph = cp.Graph(num_vertices=2)
    graph.add_edge(0, 1, 5.0)
    matrix = sp.csc_array(np.array([[1.0]]))
    pricing = cp.PricingEngine(graph.compile(), 0, 1, dual_matrix=matrix)
    matrix.data[0] = 100.0  # later edits by the caller must not change the engine
    assert pricing.solve([2.0], negative_only=False)[0].reduced_cost == 3.0
    assert pricing.reduced_costs([2.0]).tolist() == [3.0]


@pytest.mark.parametrize("seed", range(30))
def test_dag_pricing_against_enumeration(seed):
    rng = np.random.default_rng(seed)
    num_vertices = int(rng.integers(4, 10))
    instance = random_dag(rng, num_vertices, int(rng.integers(num_vertices, 3 * num_vertices)))
    source, target, paths = busiest_pair(instance)
    network = instance.build()
    num_constraints = int(rng.integers(1, 8))
    dense = random_matrix(rng, num_constraints, instance.num_edges)
    pricing = cp.PricingEngine(network, source, target, algorithm="dag", dual_matrix=sp.csr_array(dense))

    for _ in range(6):  # repeated dual updates on one engine
        duals = random_duals(rng, num_constraints)
        reference = sorted(reference_reduced_cost(instance, dense, duals, p)[0] for p in paths)
        edge_reduced = pricing.reduced_costs(duals)
        assert edge_reduced.tolist() == (instance.cost - dense.T @ duals).tolist()

        k = int(rng.integers(1, len(paths) + 3))
        result = pricing.solve(duals, max_paths=k, negative_only=False)
        assert result.status == "optimal"
        assert result.best_reduced_cost == reference[0]
        # Exactly the k cheapest, in order, each a distinct genuine path.
        assert [c.reduced_cost for c in result] == reference[:k]
        assert len({tuple(c.edges.tolist()) for c in result}) == len(result) == min(k, len(paths))
        for column in result:
            edges = column.edges.tolist()
            check_path(instance, edges, source, target)
            reduced, cost, coefficients = reference_reduced_cost(instance, dense, duals, edges)
            assert column.reduced_cost == reduced
            assert column.original_cost == cost
            assert column.dense_coefficients().tolist() == coefficients.tolist()
            assert column.coefficients.toarray().ravel().tolist() == coefficients.tolist()
            assert column.reduced_cost == sum(edge_reduced[e] for e in edges)
            assert column.coefficient_indices.tolist() == sorted(column.coefficient_indices.tolist())

        negative = pricing.solve(duals, max_paths=k, negative_only=True)
        assert [c.reduced_cost for c in negative] == [r for r in reference[:k] if r < -1e-9]
        assert negative.best_reduced_cost == reference[0]


@pytest.mark.parametrize("num_resources", [1, 2])
@pytest.mark.parametrize("seed", range(20))
def test_rcsp_pricing_against_enumeration(seed, num_resources):
    rng = np.random.default_rng(500 * num_resources + seed)
    num_vertices = int(rng.integers(4, 10))
    instance = random_dag(
        rng, num_vertices, int(rng.integers(num_vertices, 3 * num_vertices)),
        num_resources=num_resources, tightness=float(rng.choice([0.3, 0.6])),
    )
    source, target, paths = busiest_pair(instance)
    feasible = [p for p in paths if is_feasible(instance, p)]
    network = instance.build()
    num_constraints = int(rng.integers(1, 8))
    dense = random_matrix(rng, num_constraints, instance.num_edges)
    pricing = cp.PricingEngine(network, source, target, algorithm="rcsp", dual_matrix=sp.csr_array(dense))
    assert pricing.algorithm == "rcsp"

    for _ in range(6):
        duals = random_duals(rng, num_constraints)
        result = pricing.solve(duals, negative_only=False)
        if not feasible:
            assert result.status == "infeasible" and len(result) == 0
            assert result.best_reduced_cost == math.inf
            continue
        reference = min(reference_reduced_cost(instance, dense, duals, p)[0] for p in feasible)
        assert result.status == "optimal" and len(result) == 1
        column = result[0]
        edges = column.edges.tolist()
        check_path(instance, edges, source, target)
        assert is_feasible(instance, edges)
        reduced, cost, coefficients = reference_reduced_cost(instance, dense, duals, edges)
        assert column.reduced_cost == reduced == reference == result.best_reduced_cost
        assert column.original_cost == cost
        assert column.dense_coefficients().tolist() == coefficients.tolist()
        for name, limit in instance.upper.items():
            assert column.resources[name] == sum(float(instance.resources[name][e]) for e in edges)
            assert column.resources[name] <= limit
        negative = pricing.solve(duals)
        assert len(negative) == (1 if reference < -1e-9 else 0)
        assert negative.best_reduced_cost == reference
        assert result.statistics.labels_created >= 1


def test_duals_change_which_path_is_resource_optimal():
    graph = cp.Graph(num_vertices=4)
    graph.add_resource(cp.Resource("fuel", upper=10.0))
    graph.add_edge(0, 1, 1.0, {"fuel": 6.0})
    graph.add_edge(1, 3, 1.0, {"fuel": 6.0})  # infeasible route, visits row 0
    graph.add_edge(0, 2, 5.0, {"fuel": 2.0})
    graph.add_edge(2, 3, 5.0, {"fuel": 2.0})  # feasible route, visits row 1
    matrix = sp.csr_array(np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]))
    pricing = cp.PricingEngine(graph.compile(), 0, 3, algorithm="auto", dual_matrix=matrix)
    assert pricing.algorithm == "rcsp"
    # However attractive row 0 becomes, its route stays infeasible.
    result = pricing.solve([1000.0, 0.0])
    assert len(result) == 0 and result.best_reduced_cost == 10.0
    result = pricing.solve([1000.0, 25.0])
    assert result[0].path == [0, 2, 3]
    assert result[0].reduced_cost == -15.0 and result[0].original_cost == 10.0
    assert result[0].resources == {"fuel": 4.0}
    stats = result.statistics
    assert stats.algorithm == "rcsp" and stats.labels_infeasible >= 1
    assert stats.update_time >= 0 and stats.solve_time >= 0 and stats.reconstruction_time >= 0
    assert stats.total_time >= stats.solve_time


def test_parallel_and_duplicate_edges_are_distinct_columns():
    graph = cp.Graph(num_vertices=2)
    graph.add_edge(0, 1, 5.0)
    graph.add_edge(0, 1, 5.0)  # exact duplicate
    graph.add_edge(0, 1, 7.0)
    matrix = sp.csr_array(np.array([[1.0, 0.0, 4.0]]))
    pricing = cp.PricingEngine(graph.compile(), 0, 1, dual_matrix=matrix)
    result = pricing.solve([1.0], max_paths=10, negative_only=False)
    assert [(c.edges.tolist(), c.reduced_cost) for c in result] == [([2], 3.0), ([0], 4.0), ([1], 5.0)]
    assert [c.dense_coefficients().tolist() for c in result] == [[4.0], [1.0], [0.0]]


def test_zero_cost_edges_and_ties():
    graph = cp.Graph(num_vertices=4)
    for source, target in ((0, 1), (1, 3), (0, 2), (2, 3)):
        graph.add_edge(source, target, 0.0)
    pricing = cp.PricingEngine(graph.compile(), 0, 3, dual_matrix=sp.csr_array((1, 4)))
    result = pricing.solve([3.0], max_paths=4, negative_only=False)
    assert [c.reduced_cost for c in result] == [0.0, 0.0]
    assert sorted(c.edges.tolist() for c in result) == [[0, 1], [2, 3]]
    assert all(c.coefficient_indices.size == 0 for c in result)
    assert len(pricing.solve([3.0], max_paths=4)) == 0


def test_fixed_cost_and_fixed_coefficients():
    graph = cp.Graph(num_vertices=3)
    graph.add_edge(0, 1, 4.0)
    graph.add_edge(1, 2, 6.0)
    network = graph.compile()
    # Row 0: covering constraint touched by edge 0. Row 1: convexity row of this vehicle.
    matrix = sp.csr_array(np.array([[1.0, 0.0], [0.0, 0.0]]))
    for fixed in ([0.0, 1.0], {1: 1.0}, sp.csc_array(np.array([[0.0], [1.0]])), np.array([0.0, 1.0])):
        pricing = cp.PricingEngine(
            network, 0, 2, dual_matrix=matrix, fixed_cost=100.0, fixed_coefficients=fixed
        )
        assert pricing.fixed_cost == 100.0
        column = pricing.solve([7.0, 90.0], negative_only=False)[0]
        assert column.original_cost == 110.0  # 100 + 4 + 6
        assert column.dense_coefficients().tolist() == [1.0, 1.0]
        assert column.reduced_cost == 110.0 - 7.0 - 90.0
        assert column.reduced_cost == column.original_cost - np.dot([7.0, 90.0], column.dense_coefficients())
        # The fixed terms take part in the sign test.
        assert len(pricing.solve([7.0, 90.0])) == 0
        assert len(pricing.solve([7.0, 110.0])) == 1
    # Fixed coefficients on a row the path also touches are summed.
    pricing = cp.PricingEngine(network, 0, 2, dual_matrix=matrix, fixed_coefficients={0: 2.0})
    column = pricing.solve([1.0, 0.0], negative_only=False)[0]
    assert column.dense_coefficients().tolist() == [3.0, 0.0] and column.reduced_cost == 7.0


def test_engine_without_dual_matrix():
    graph = cp.Graph(num_vertices=2)
    graph.add_edge(0, 1, 4.0)
    network = graph.compile()
    plain = cp.PricingEngine(network, 0, 1)
    assert plain.num_constraints == 0
    assert plain.solve([], negative_only=False)[0].reduced_cost == 4.0
    convexity = cp.PricingEngine(network, 0, 1, fixed_coefficients=[1.0])
    assert convexity.num_constraints == 1
    assert convexity.solve([10.0])[0].reduced_cost == -6.0


def test_unreachable_and_infeasible_pricing_problems():
    graph = cp.Graph(num_vertices=4)
    graph.add_resource(cp.Resource("fuel", upper=1.0))
    graph.add_edge(0, 1, 1.0, {"fuel": 2.0})
    graph.add_edge(2, 3, 1.0)
    network = graph.compile()
    matrix = sp.csr_array((3, 2))

    unreachable = cp.PricingEngine(network, 0, 3, dual_matrix=matrix)
    assert not unreachable.is_target_reachable
    assert unreachable.num_active_edges == 0
    result = unreachable.solve(np.ones(3), negative_only=False)
    assert result.status == "unreachable" and len(result) == 0
    assert result.best_reduced_cost == math.inf

    infeasible = cp.PricingEngine(network, 0, 1, dual_matrix=matrix)
    assert infeasible.is_target_reachable
    result = infeasible.solve(np.ones(3), negative_only=False)
    assert result.status == "infeasible" and len(result) == 0


def test_engine_prunes_to_the_relevant_subgraph():
    # A long chain 0..9 plus a side branch 3 -> 10 -> 11 that never returns.
    graph = cp.Graph(num_vertices=12)
    for v in range(9):
        graph.add_edge(v, v + 1, 1.0)
    graph.add_edge(3, 10, 1.0)
    graph.add_edge(10, 11, 1.0)
    network = graph.compile()
    pricing = cp.PricingEngine(network, 2, 6)
    assert pricing.num_active_vertices == 5  # 2, 3, 4, 5, 6
    assert pricing.num_active_edges == 4
    assert pricing.solve([], negative_only=False)[0].path == [2, 3, 4, 5, 6]
    assert "active_edges=4/11" in repr(pricing)


def test_source_equals_target():
    graph = cp.Graph(num_vertices=2)
    graph.add_edge(0, 1, 1.0)
    pricing = cp.PricingEngine(graph.compile(), 0, 0, fixed_cost=3.0, fixed_coefficients=[1.0])
    assert len(pricing.solve([5.0], negative_only=False)) == 1
    column = pricing.solve([5.0])[0]
    assert column.edges.size == 0 and column.path == [0]
    assert column.original_cost == 3.0 and column.reduced_cost == -2.0


def test_state_space_columns_decode_paths():
    states = cp.StateSpace(day=range(1, 4), port=["A", "B"])
    graph = cp.Graph(states=states)
    visit = graph.add_edge((1, "A"), (2, "B"), 5.0)
    graph.add_edge((2, "B"), (3, "A"), 5.0)
    graph.add_edge((1, "A"), (3, "A"), 9.0)
    network = graph.compile()
    matrix = sp.coo_array(([1.0], ([0], [visit])), shape=(1, network.num_edges))
    pricing = cp.PricingEngine(network, (1, "A"), (3, "A"), dual_matrix=matrix, name="vessel-1")
    assert pricing.name == "vessel-1" and "vessel-1" in repr(pricing)
    column = pricing.solve([4.0], negative_only=False)[0]
    assert column.path == [(1, "A"), (2, "B"), (3, "A")]
    assert column.reduced_cost == 6.0


# ------------------------------------------------------------ unsupported


def test_multiple_paths_with_rcsp_raise_not_implemented():
    graph = cp.Graph(num_vertices=3)
    graph.add_resource(cp.Resource("fuel", upper=5.0))
    graph.add_edge(0, 1, 1.0, {"fuel": 1.0})
    graph.add_edge(1, 2, 1.0, {"fuel": 1.0})
    network = graph.compile()
    pricing = cp.PricingEngine(network, 0, 2, algorithm="rcsp")
    with pytest.raises(NotImplementedError, match="max_paths > 1"):
        pricing.solve([], max_paths=2)
    assert len(pricing.solve([], max_paths=1, negative_only=False)) == 1
    # The unconstrained engine on the same network supports it.
    relaxed = cp.PricingEngine(network, 0, 2, algorithm="dag", unconstrained=True)
    assert len(relaxed.solve([], max_paths=2, negative_only=False)) == 1
    with pytest.raises(ValueError, match="unconstrained=True"):
        cp.PricingEngine(network, 0, 2, algorithm="dag")


# ------------------------------------------------------------ invalid input


def test_invalid_engine_construction(small):
    network, matrix = small
    graph = cp.Graph(num_vertices=2)
    with pytest.raises(TypeError, match="compile"):
        cp.PricingEngine(graph, 0, 1)
    with pytest.raises(TypeError):
        cp.PricingEngine("network", 0, 1)
    with pytest.raises(ValueError, match="one column per edge"):
        cp.PricingEngine(network, 0, 3, dual_matrix=sp.csr_array((2, 5)))
    with pytest.raises(ValueError, match="one column per edge"):
        cp.PricingEngine(network, 0, 3, dual_matrix=matrix.T)
    with pytest.raises((TypeError, ValueError)):
        cp.PricingEngine(network, 0, 3, dual_matrix=np.zeros(4))
    with pytest.raises(TypeError):
        cp.PricingEngine(network, 0, 3, dual_matrix="matrix")
    for bad in (np.nan, np.inf):
        dense = np.zeros((2, 4))
        dense[1, 2] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            cp.PricingEngine(network, 0, 3, dual_matrix=dense)
    with pytest.raises(cp.InvalidStateError):
        cp.PricingEngine(network, 0, 9, dual_matrix=matrix)
    with pytest.raises(ValueError, match="algorithm"):
        cp.PricingEngine(network, 0, 3, algorithm="astar", dual_matrix=matrix)
    with pytest.raises(ValueError):
        cp.PricingEngine(network, 0, 3, dual_matrix=matrix, fixed_cost=math.inf)
    with pytest.raises(TypeError):
        cp.PricingEngine(network, 0, 3, dual_matrix=matrix, fixed_cost="free")
    with pytest.raises(ValueError, match="length 3"):
        cp.PricingEngine(network, 0, 3, dual_matrix=matrix, fixed_coefficients=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="outside"):
        cp.PricingEngine(network, 0, 3, dual_matrix=matrix, fixed_coefficients={2: 1.0})
    with pytest.raises(ValueError, match="finite"):
        cp.PricingEngine(network, 0, 3, dual_matrix=matrix, fixed_coefficients=[1.0, math.nan])
    with pytest.raises(ValueError):
        cp.PricingEngine(network, 0, 3, dual_matrix=matrix, dominance_tolerance=-1.0)


def test_invalid_solve_arguments(small):
    network, matrix = small
    pricing = cp.PricingEngine(network, 0, 3, dual_matrix=matrix)
    with pytest.raises(ValueError, match=r"expected \(2,\)"):
        pricing.solve([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"expected \(2,\)"):
        pricing.solve([1.0])
    with pytest.raises(ValueError, match=r"expected \(2,\)"):
        pricing.solve(np.zeros((2, 1)))
    with pytest.raises(TypeError):
        pricing.solve(["a", "b"])
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="not finite"):
            pricing.solve([0.0, bad])
    with pytest.raises(ValueError, match="at least 1"):
        pricing.solve([0.0, 0.0], max_paths=0)
    with pytest.raises(TypeError):
        pricing.solve([0.0, 0.0], max_paths=1.5)
    with pytest.raises(ValueError):
        pricing.solve([0.0, 0.0], tolerance=-1e-9)
    with pytest.raises(ValueError):
        pricing.solve([0.0, 0.0], tolerance=math.nan)
    # A failed call leaves the engine usable.
    assert pricing.solve([50.0, 0.0])[0].reduced_cost == -30.0


# -------------------------------------------------------- column generation


def routing_instance(seed: int, num_tasks: int = 7, capacity: float = 3.0):
    """Tasks in a fixed order; a route is an increasing subsequence within capacity."""
    rng = np.random.default_rng(seed)
    source, sink = 0, num_tasks + 1
    sources, targets, costs, load = [], [], [], []
    for task in range(1, num_tasks + 1):
        sources += [source, task]
        targets += [task, sink]
        costs += [float(rng.integers(5, 15)), float(rng.integers(5, 15))]
        load += [1.0, 0.0]
        for later in range(task + 1, num_tasks + 1):
            sources.append(task)
            targets.append(later)
            costs.append(float(rng.integers(1, 8)))
            load.append(1.0)
    instance = Instance(
        num_tasks + 2,
        np.array(sources),
        np.array(targets),
        np.array(costs),
        {"load": np.array(load)},
        {"load": capacity},
    )
    # Row k is covered by every edge that enters task k + 1.
    entering = [(t - 1, e) for e, t in enumerate(targets) if t != sink]
    rows, cols = zip(*entering)
    matrix = sp.csr_array((np.ones(len(rows)), (rows, cols)), shape=(num_tasks, len(sources)))
    return instance, matrix, source, sink


def solve_covering_lp(costs, columns, num_rows):
    """min c.x  s.t.  A x >= 1, x >= 0. Returns (objective, duals of the >= rows)."""
    a = np.column_stack(columns)
    result = scipy.optimize.linprog(
        np.asarray(costs), A_ub=-a, b_ub=-np.ones(num_rows), bounds=(0, None), method="highs"
    )
    assert result.status == 0, result.message
    # linprog reports sensitivities of "<=" rows (nonpositive); for the
    # equivalent ">=" rows the duals are their negatives (nonnegative).
    return result.fun, -result.ineqlin.marginals


@pytest.mark.parametrize("seed", range(6))
def test_column_generation_reaches_the_lp_optimum(seed):
    instance, matrix, source, sink = routing_instance(seed)
    num_rows = matrix.shape[0]
    network = instance.build()
    pricing = cp.PricingEngine(network, source, sink, algorithm="rcsp", dual_matrix=matrix)

    # Reference: the LP over every feasible route.
    dense = matrix.toarray()
    all_paths = [p for p in enumerate_paths(instance, source, sink) if is_feasible(instance, p)]
    all_costs = [sum(float(instance.cost[e]) for e in p) for p in all_paths]
    all_columns = [dense[:, p].sum(axis=1) for p in all_paths]
    expected, _ = solve_covering_lp(all_costs, all_columns, num_rows)

    # Restricted master, started from expensive artificial columns.
    big_m = 1e4
    costs = [big_m] * num_rows
    columns = [row for row in np.eye(num_rows)]
    objective = math.inf
    for _ in range(200):
        objective, duals = solve_covering_lp(costs, columns, num_rows)
        priced = pricing.solve(duals, tolerance=1e-7)
        if not priced:
            assert priced.best_reduced_cost >= -1e-7  # the certificate
            break
        column = priced[0]
        assert column.reduced_cost == pytest.approx(
            column.original_cost - duals @ column.dense_coefficients(), abs=1e-9
        )
        costs.append(column.original_cost)
        columns.append(column.dense_coefficients())
    else:
        pytest.fail("column generation did not terminate")
    assert objective == pytest.approx(expected, rel=1e-7, abs=1e-7)
    generated = len(columns) - num_rows
    assert 0 < generated < len(all_paths)  # some columns, but far fewer than enumeration
