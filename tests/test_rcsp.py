import math

import numpy as np
import pytest

import celerisroute as cr
from _reference import (
    best_cost,
    check_path,
    enumerate_paths,
    is_feasible,
    path_cost,
    path_resource,
    random_dag,
)


def diamond(upper: float, **resource_options) -> cr.Network:
    """Two routes 0 -> 3: cheap but thirsty (via 1), dear but frugal (via 2)."""
    graph = cr.Graph(num_vertices=4)
    graph.add_resource(cr.Resource("fuel", upper=upper, **resource_options))
    graph.add_edge(0, 1, 1.0, {"fuel": 6.0})
    graph.add_edge(1, 3, 1.0, {"fuel": 6.0})
    graph.add_edge(0, 2, 5.0, {"fuel": 2.0})
    graph.add_edge(2, 3, 5.0, {"fuel": 2.0})
    return graph.compile()


def test_one_resource_changes_the_optimal_path():
    loose = diamond(upper=100.0).solve(0, 3)
    assert loose.statistics.algorithm == "rcsp"  # chosen by "auto"
    assert (loose.cost, loose.path, loose.resources) == (2.0, [0, 1, 3], {"fuel": 12.0})
    tight = diamond(upper=10.0).solve(0, 3)
    assert (tight.cost, tight.path, tight.resources) == (10.0, [0, 2, 3], {"fuel": 4.0})
    assert tight.edges.tolist() == [2, 3]


def test_bounds_hold_at_exact_limits():
    # 12 units needed: feasible at upper == 12 exactly, infeasible just below.
    assert diamond(upper=12.0).solve(0, 3).cost == 2.0
    assert diamond(upper=math.nextafter(12.0, 0.0)).solve(0, 3).cost == 10.0
    # Same with a nonzero initial level.
    assert diamond(upper=15.0, initial=3.0).solve(0, 3).resources == {"fuel": 15.0}
    assert diamond(upper=15.0, initial=3.5).solve(0, 3).cost == 10.0
    # Exact limit on the frugal route too.
    assert diamond(upper=4.0).solve(0, 3).status == "optimal"


def test_infeasible_resource_restrictions():
    solution = diamond(upper=3.0).solve(0, 3)
    assert solution.status == "infeasible" and solution.status is cr.Status.INFEASIBLE
    assert not solution
    assert solution.cost == float("inf")
    assert solution.edges.size == 0 and solution.path == [] and solution.resources == {}
    assert solution.statistics.labels_infeasible >= 1
    # "infeasible" (paths exist, all violate a bound) differs from "unreachable".
    assert diamond(upper=3.0).solve(3, 0).status == "unreachable"


def test_bound_is_checked_along_the_path_not_only_at_the_end():
    # With nonnegative increments the running total is monotone, so the check at
    # the end implies the check everywhere; make sure an early violation is
    # caught even when the target is never reached through it.
    graph = cr.Graph(num_vertices=3)
    graph.add_resource(cr.Resource("fuel", upper=5.0))
    graph.add_edge(0, 1, 1.0, {"fuel": 6.0})
    graph.add_edge(1, 2, 1.0, {"fuel": 0.0})
    assert graph.compile().solve(0, 2).status == "infeasible"


def test_two_resources():
    graph = cr.Graph(num_vertices=5)
    graph.add_resource(cr.Resource("fuel", upper=10.0))
    graph.add_resource(cr.Resource("time", upper=8.0))
    # Three routes 0 -> 4 with (cost, fuel, time):
    graph.add_edge(0, 1, 1.0, {"fuel": 6.0, "time": 1.0})
    graph.add_edge(1, 4, 1.0, {"fuel": 6.0, "time": 1.0})  # (2, 12, 2): too much fuel
    graph.add_edge(0, 2, 2.0, {"fuel": 1.0, "time": 5.0})
    graph.add_edge(2, 4, 2.0, {"fuel": 1.0, "time": 5.0})  # (4, 2, 10): too slow
    graph.add_edge(0, 3, 4.0, {"fuel": 4.0, "time": 3.0})
    graph.add_edge(3, 4, 4.0, {"fuel": 4.0, "time": 3.0})  # (8, 8, 6): feasible
    network = graph.compile()
    assert network.tracked_resources == ("fuel", "time")
    solution = cr.resource_shortest_path(network, 0, 4)
    assert solution.cost == 8.0
    assert solution.path == [0, 3, 4]
    assert solution.resources == {"fuel": 8.0, "time": 6.0}


def test_equal_labels_keep_exactly_one():
    # Two routes with identical cost and consumption: each dominates the other
    # in the weak sense. One must survive, and only one.
    graph = cr.Graph(num_vertices=5)
    graph.add_resource(cr.Resource("fuel", upper=10.0))
    graph.add_edge(0, 1, 2.0, {"fuel": 3.0})
    graph.add_edge(1, 3, 2.0, {"fuel": 3.0})
    graph.add_edge(0, 2, 2.0, {"fuel": 3.0})
    graph.add_edge(2, 3, 2.0, {"fuel": 3.0})
    graph.add_edge(3, 4, 1.0, {"fuel": 1.0})
    network = graph.compile()
    solution = network.solve(0, 4)
    assert solution.cost == 5.0 and solution.resources == {"fuel": 7.0}
    assert solution.edges.tolist() in ([0, 1, 4], [2, 3, 4])
    stats = solution.statistics
    assert stats.labels_dominated == 1  # the duplicate at vertex 3
    assert stats.pareto_labels == 1
    # Deterministic choice between the two.
    assert all(network.solve(0, 4).edges.tolist() == solution.edges.tolist() for _ in range(5))


def test_equal_parallel_edges():
    graph = cr.Graph(num_vertices=2)
    graph.add_resource(cr.Resource("fuel", upper=10.0))
    graph.add_edge(0, 1, 2.0, {"fuel": 3.0})
    graph.add_edge(0, 1, 2.0, {"fuel": 3.0})
    solution = graph.compile().solve(0, 1)
    assert solution.cost == 2.0 and solution.edges.tolist() in ([0], [1])
    assert solution.statistics.pareto_labels == 1


def test_multiple_nondominated_labels():
    # k parallel trade-offs between consecutive vertices: every combination that
    # is not dominated has to be carried forward.
    graph = cr.Graph(num_vertices=4)
    graph.add_resource(cr.Resource("fuel", upper=9.0))
    for a in range(3):
        graph.add_edge(a, a + 1, 1.0, {"fuel": 4.0})  # cheap, thirsty
        graph.add_edge(a, a + 1, 3.0, {"fuel": 2.0})  # dear, frugal
    network = graph.compile()
    solution = network.solve(0, 3)
    # Feasible mixes: at most one thirsty leg (4+2+2 = 8 <= 9); two would need 10.
    assert solution.cost == 7.0 and solution.resources == {"fuel": 8.0}
    stats = solution.statistics
    # Target front: (cost 7, fuel 8) and (cost 9, fuel 6).
    assert stats.pareto_labels == 2
    assert stats.labels_dominated >= 1 and stats.labels_infeasible >= 1
    assert stats.labels_created >= stats.labels_extended >= stats.pareto_labels
    assert stats.max_active_labels >= 2
    assert stats.solve_time >= 0.0


def test_cheaper_but_thirstier_label_is_not_pruned():
    # At vertex 1 the cheap label has used more fuel. Pruning it by cost alone
    # (or the dear one by fuel alone) would lose the optimum for one of the bounds.
    def solve(upper):
        graph = cr.Graph(num_vertices=3)
        graph.add_resource(cr.Resource("fuel", upper=upper))
        graph.add_edge(0, 1, 1.0, {"fuel": 5.0})
        graph.add_edge(0, 1, 4.0, {"fuel": 1.0})
        graph.add_edge(1, 2, 1.0, {"fuel": 5.0})
        return graph.compile().solve(0, 2)

    assert solve(10.0).cost == 2.0
    assert solve(9.0).cost == 5.0
    assert solve(5.0).status == "infeasible"


def test_negative_costs_with_resources():
    graph = cr.Graph(num_vertices=4)
    graph.add_resource(cr.Resource("fuel", upper=5.0))
    graph.add_edge(0, 1, 2.0, {"fuel": 1.0})
    graph.add_edge(1, 3, -10.0, {"fuel": 5.0})  # tempting but infeasible after edge 0
    graph.add_edge(0, 2, 3.0, {"fuel": 2.0})
    graph.add_edge(2, 3, -4.0, {"fuel": 3.0})
    solution = graph.compile().solve(0, 3)
    assert solution.cost == -1.0 and solution.path == [0, 2, 3]


def test_zero_increment_edges_and_unused_resources():
    graph = cr.Graph(num_vertices=3)
    graph.add_resource(cr.Resource("fuel", upper=1.0))
    graph.add_resource(cr.Resource("crew", upper=2.0, initial=2.0))  # never consumed
    graph.add_edge(0, 1, 1.0)  # no increments at all
    graph.add_edge(1, 2, 1.0, {"fuel": 1.0})
    solution = graph.compile().solve(0, 2)
    assert solution.cost == 2.0
    assert solution.resources == {"fuel": 1.0, "crew": 2.0}


def test_source_equals_target_with_resources():
    solution = diamond(upper=10.0, initial=4.0).solve(2, 2)
    assert solution.status == "optimal" and solution.cost == 0.0
    assert solution.resources == {"fuel": 4.0}


def test_many_resources_use_the_general_code_path():
    names = [f"r{i}" for i in range(6)]
    graph = cr.Graph(num_vertices=3)
    for name in names:
        graph.add_resource(cr.Resource(name, upper=3.0))
    graph.add_edge(0, 1, 1.0, {name: 1.0 for name in names})
    graph.add_edge(1, 2, 1.0, {name: 2.0 for name in names})
    graph.add_edge(0, 2, 9.0, {"r5": 3.0})
    graph.add_edge(1, 2, 0.5, {"r5": 2.5})  # breaks r5 only
    solution = graph.compile().solve(0, 2)
    assert solution.cost == 2.0
    assert solution.resources == {name: 3.0 for name in names}


def test_repeated_solves_reuse_the_network():
    network = diamond(upper=10.0)
    reference = network.solve(0, 3)
    for _ in range(50):
        solution = network.solve(0, 3)
        assert solution.cost == reference.cost
        assert solution.edges.tolist() == reference.edges.tolist()
        assert solution.statistics.labels_created == reference.statistics.labels_created


def test_dominance_tolerance_is_a_documented_heuristic():
    network = diamond(upper=100.0)
    assert network.solve(0, 3, dominance_tolerance=0.0).cost == 2.0
    assert network.solve(0, 3, dominance_tolerance=1e-9).cost == 2.0
    assert cr.resource_shortest_path(network, 0, 3, dominance_tolerance=1e-9).cost == 2.0


# ------------------------------------------------------------ declarations


def test_resource_validation():
    with pytest.raises(NotImplementedError, match="cumulative"):
        cr.Resource("fuel", upper=10.0, sense="replenishable")
    with pytest.raises(ValueError):
        cr.Resource("fuel", lower=5.0, upper=1.0)
    with pytest.raises(ValueError):
        cr.Resource("fuel", upper=5.0, initial=6.0)
    with pytest.raises(ValueError):
        cr.Resource("fuel", lower=2.0, initial=0.0)  # would need a minimum-consumption rule
    with pytest.raises(ValueError):
        cr.Resource("fuel", upper=math.nan)
    with pytest.raises(ValueError):
        cr.Resource("fuel", initial=math.inf)
    with pytest.raises(TypeError):
        cr.Resource("", upper=1.0)
    with pytest.raises(TypeError):
        cr.Resource("fuel", upper="full")
    resource = cr.Resource(name="fuel", lower=0.0, upper=100.0, initial=0.0, sense="cumulative")
    assert resource.is_constraining
    assert not cr.Resource("distance").is_constraining


def test_resource_declaration_errors():
    graph = cr.Graph(num_vertices=3)
    graph.add_resource(cr.Resource("fuel", upper=10.0))
    with pytest.raises(ValueError, match="already declared"):
        graph.add_resource(cr.Resource("fuel", upper=20.0))
    with pytest.raises(TypeError):
        graph.add_resource("fuel")
    with pytest.raises(ValueError, match=">= 0"):
        graph.add_edge(0, 1, 1.0, {"fuel": -1.0})
    with pytest.raises(ValueError, match=">= 0"):
        graph.add_edges([0, 1], [1, 2], 1.0, resources={"fuel": np.array([1.0, -0.5])})
    assert graph.num_edges == 0


def test_undeclared_resource_is_rejected_at_compile():
    graph = cr.Graph(num_vertices=3)
    graph.add_resource(cr.Resource("fuel", upper=10.0))
    graph.add_edge(0, 1, 1.0, {"feul": 3.0})  # typo
    with pytest.raises(ValueError, match="feul"):
        graph.compile()


def test_negative_increment_declared_late_is_rejected_at_compile():
    graph = cr.Graph(num_vertices=3)
    graph.add_edge(0, 1, 1.0, {"fuel": -3.0})  # accepted: not yet known to be cumulative
    graph.add_resource(cr.Resource("fuel", upper=10.0))
    with pytest.raises(ValueError, match="fuel"):
        graph.compile()


# -------------------------------------------------------------- exhaustive


@pytest.mark.parametrize("num_resources", [1, 2, 3])
@pytest.mark.parametrize("seed", range(25))
def test_random_instances_against_exhaustive_enumeration(seed, num_resources):
    rng = np.random.default_rng(1000 * num_resources + seed)
    num_vertices = int(rng.integers(3, 10))
    num_edges = int(rng.integers(num_vertices, 3 * num_vertices))
    instance = random_dag(
        rng,
        num_vertices,
        num_edges,
        num_resources=num_resources,
        tightness=float(rng.choice([0.25, 0.5, 0.9])),
    )
    network = instance.build()
    outcomes = set()
    for source in range(num_vertices):
        for target in range(num_vertices):
            paths = enumerate_paths(instance, source, target)
            feasible = [p for p in paths if is_feasible(instance, p)]
            solution = network.solve(source, target)
            assert solution.statistics.algorithm == "rcsp"
            outcomes.add(str(solution.status))
            if not paths:
                assert solution.status == "unreachable"
                continue
            if not feasible:
                assert solution.status == "infeasible"
                continue
            expected = best_cost(instance, paths, feasible_only=True)
            assert solution.status == "optimal"
            assert solution.cost == expected
            edges = solution.edges.tolist()
            check_path(instance, edges, source, target)
            assert is_feasible(instance, edges)
            assert path_cost(instance, edges) == expected
            for name in instance.upper:
                assert solution.resources[name] == path_resource(instance, edges, name)
                assert solution.resources[name] <= instance.upper[name]
            # The unconstrained optimum is a lower bound.
            relaxed = network.solve(source, target, algorithm="dag", unconstrained=True)
            assert relaxed.cost <= solution.cost
    assert "optimal" in outcomes


def test_exhaustive_suite_exercises_binding_constraints():
    """Guard against a vacuous random suite: bounds must actually bind sometimes."""
    binding = infeasible = 0
    for seed in range(25):
        rng = np.random.default_rng(1000 + seed)
        num_vertices = int(rng.integers(3, 10))
        num_edges = int(rng.integers(num_vertices, 3 * num_vertices))
        instance = random_dag(
            rng, num_vertices, num_edges, num_resources=1,
            tightness=float(rng.choice([0.25, 0.5, 0.9])),
        )
        network = instance.build()
        for source in range(num_vertices):
            for target in range(num_vertices):
                constrained = network.solve(source, target)
                relaxed = network.solve(source, target, algorithm="dag", unconstrained=True)
                if constrained.status == "infeasible":
                    infeasible += 1
                elif constrained.found and constrained.cost > relaxed.cost:
                    binding += 1
    assert binding >= 10
    assert infeasible >= 1


def test_completion_bound_keeps_paths_exactly_at_the_limit():
    """Labels are pruned by 'consumption so far + least consumption to go', a sum

    taken in a different order than the path's own accumulation. Rounding must
    never make that test stricter than the exact per-edge feasibility test.
    """
    for count in (7, 60, 500):
        increments = [0.1 * (1 + index % 7) for index in range(count)]
        accumulated = 0.0
        for value in increments:
            accumulated += value  # exactly how the solver accumulates
        for upper, expected in ((accumulated, "optimal"), (math.nextafter(accumulated, 0.0), "infeasible")):
            graph = cr.Graph(num_vertices=count + 1)
            graph.add_resource(cr.Resource("r", upper=upper))
            for index, value in enumerate(increments):
                graph.add_edge(index, index + 1, 1.0, resources={"r": value})
            solution = graph.compile().solve(0, count)
            assert solution.status == expected
            if solution:
                assert solution.resources["r"] == accumulated


def test_labels_that_cannot_be_completed_are_not_created():
    # 0 -> 1 -> 2 is the only route to the target; 0 also fans out into dead
    # ends (3..9) and into a detour (10) that is too expensive to complete.
    graph = cr.Graph(num_vertices=11)
    graph.add_resource(cr.Resource("r", upper=10.0))
    graph.add_edge(0, 1, 1.0, resources={"r": 4.0})
    graph.add_edge(1, 2, 1.0, resources={"r": 4.0})
    for dead_end in range(3, 10):
        graph.add_edge(0, dead_end, -5.0, resources={"r": 1.0})
    graph.add_edge(0, 10, -9.0, resources={"r": 3.0})
    graph.add_edge(10, 2, -9.0, resources={"r": 8.0})  # 3 + 8 > 10
    solution = graph.compile().solve(0, 2)
    assert solution.status == "optimal" and solution.cost == 2.0
    stats = solution.statistics
    assert stats.labels_created == 3  # at 0, 1 and 2 only
    assert stats.labels_infeasible == 8  # seven dead ends and the detour
