import numpy as np
import pytest

import celerispath as cp
from _reference import best_cost, check_path, enumerate_paths, path_cost, random_dag


def build(num_vertices, edges):
    graph = cp.Graph(num_vertices=num_vertices)
    for source, target, cost in edges:
        graph.add_edge(source, target, cost)
    return graph.compile()


def test_single_edge_path():
    network = build(2, [(0, 1, 4.5)])
    solution = network.solve(0, 1, algorithm="dag")
    assert solution.status == "optimal" and solution.status is cp.Status.OPTIMAL
    assert solution.found and bool(solution)
    assert solution.cost == 4.5
    assert solution.edges.tolist() == [0]
    assert solution.nodes.tolist() == [0, 1]
    assert solution.path == [0, 1]
    assert solution.resources == {}
    assert solution.statistics.algorithm == "dag"
    assert solution.statistics.solve_time >= 0.0


def test_source_equals_target():
    network = build(3, [(0, 1, 1.0), (1, 2, 1.0)])
    solution = network.solve(1, 1)
    assert solution.status == "optimal"
    assert solution.cost == 0.0
    assert solution.edges.size == 0
    assert solution.path == [1]


def test_multiple_competing_paths():
    #     1
    #   /   \
    # 0 - 2 - 4      0-3-4 is cheapest
    #   \   /
    #     3
    network = build(
        5,
        [(0, 1, 2.0), (1, 4, 9.0), (0, 2, 5.0), (2, 4, 5.0), (0, 3, 6.0), (3, 4, 1.0), (0, 4, 8.0)],
    )
    solution = network.solve(0, 4)
    assert solution.cost == 7.0
    assert solution.edges.tolist() == [4, 5]
    assert solution.path == [0, 3, 4]


def test_negative_cost_edges():
    # The greedy first step (0 -> 1, cost 1) is wrong; the negative edge wins.
    network = build(4, [(0, 1, 1.0), (1, 3, 1.0), (0, 2, 10.0), (2, 3, -12.0)])
    solution = network.solve(0, 3)
    assert solution.cost == -2.0
    assert solution.path == [0, 2, 3]
    all_negative = build(3, [(0, 1, -1.0), (1, 2, -2.5), (0, 2, -3.0)])
    assert all_negative.solve(0, 2).cost == -3.5


def test_unreachable_target():
    network = build(4, [(0, 1, 1.0), (2, 3, 1.0)])
    solution = network.solve(0, 3)
    assert solution.status == "unreachable" and solution.status is cp.Status.UNREACHABLE
    assert not solution.found and not solution
    assert solution.cost == float("inf")
    assert solution.edges.size == 0 and solution.nodes.size == 0
    assert solution.path == []
    assert solution.resources == {}
    # Reachability is directional.
    assert network.solve(1, 0).status == "unreachable"


def test_parallel_edges_pick_the_cheapest():
    network = build(3, [(0, 1, 5.0), (0, 1, 2.0), (0, 1, 3.0), (1, 2, 4.0), (1, 2, 1.0)])
    solution = network.solve(0, 2)
    assert solution.cost == 3.0
    assert solution.edges.tolist() == [1, 4]  # the edge ids, not just the vertices


def test_zero_cost_and_equal_cost_paths():
    network = build(4, [(0, 1, 0.0), (1, 3, 0.0), (0, 2, 0.0), (2, 3, 0.0), (0, 3, 0.0)])
    solution = network.solve(0, 3)
    assert solution.cost == 0.0
    edges = solution.edges.tolist()
    assert edges in ([0, 1], [2, 3], [4])
    # Ties are broken deterministically.
    assert all(network.solve(0, 3).edges.tolist() == edges for _ in range(5))


def test_duplicate_edges():
    network = build(2, [(0, 1, 2.0), (0, 1, 2.0)])
    solution = network.solve(0, 1)
    assert solution.cost == 2.0 and solution.edges.tolist() in ([0], [1])


def test_path_through_state_space():
    states = cp.StateSpace(day=range(1, 5), port=["A", "B"])
    graph = cp.Graph(states=states)
    for day in range(1, 4):
        graph.add_edge((day, "A"), (day + 1, "A"), 1.0)  # wait at A
        graph.add_edge((day, "A"), (day + 1, "B"), 0.25)  # sail A -> B
        graph.add_edge((day, "B"), (day + 1, "B"), 1.0)  # wait at B
    network = graph.compile()
    solution = cp.shortest_path(network, (1, "A"), (4, "B"))
    assert solution.cost == 2.25
    assert solution.path == [(1, "A"), (2, "B"), (3, "B"), (4, "B")]
    assert solution.nodes.tolist() == [states.id(s) for s in solution.path]


def test_repeated_solves_on_a_compiled_network_are_independent():
    rng = np.random.default_rng(5)
    instance = random_dag(rng, 30, 120)
    network = instance.build()
    pairs = [(int(a), int(b)) for a, b in rng.integers(0, 30, size=(40, 2))]
    first = [(network.solve(a, b).status, network.solve(a, b).cost) for a, b in pairs]
    # Same queries, shuffled and repeated: results must not depend on history.
    for index in rng.permutation(len(pairs)).tolist() * 2:
        a, b = pairs[index]
        solution = network.solve(a, b)
        assert (solution.status, solution.cost) == first[index]


def test_shortest_path_function_reuses_the_network_and_accepts_graphs():
    graph = cp.Graph(num_vertices=3)
    graph.add_edge(0, 1, 1.0)
    graph.add_edge(1, 2, 1.0)
    network = graph.compile()
    assert cp.shortest_path(network, 0, 2).cost == 2.0
    assert cp.shortest_path(graph, 0, 2).cost == 2.0  # compiled on the fly


def test_dag_refuses_to_silently_ignore_resource_bounds():
    graph = cp.Graph(num_vertices=3)
    graph.add_resource(cp.Resource("fuel", upper=5.0))
    graph.add_edge(0, 1, 1.0, {"fuel": 4.0})
    graph.add_edge(1, 2, 1.0, {"fuel": 4.0})
    graph.add_edge(0, 2, 9.0, {"fuel": 1.0})
    network = graph.compile()
    with pytest.raises(ValueError, match="unconstrained=True"):
        network.solve(0, 2, algorithm="dag")
    with pytest.raises(ValueError, match="unconstrained=True"):
        cp.shortest_path(network, 0, 2)
    with pytest.raises(ValueError, match="contradicts"):
        network.solve(0, 2, algorithm="rcsp", unconstrained=True)
    relaxed = network.solve(0, 2, algorithm="dag", unconstrained=True)
    assert relaxed.cost == 2.0
    assert relaxed.resources == {"fuel": 8.0}  # reported, although above the bound
    assert cp.shortest_path(network, 0, 2, unconstrained=True).cost == 2.0
    assert network.solve(0, 2, unconstrained=True).statistics.algorithm == "dag"
    assert network.solve(0, 2).cost == 9.0  # "auto" respects the bound


def test_unbounded_resource_is_reported_by_the_dag_solver():
    graph = cp.Graph(num_vertices=3)
    graph.add_resource(cp.Resource("distance", initial=2.0))
    graph.add_edge(0, 1, 1.0, {"distance": 10.0})
    graph.add_edge(1, 2, 1.0, {"distance": 5.5})
    solution = graph.compile().solve(0, 2)
    assert solution.statistics.algorithm == "dag"
    assert solution.resources == {"distance": 17.5}


@pytest.mark.parametrize("seed", range(40))
def test_random_dags_against_exhaustive_enumeration(seed):
    rng = np.random.default_rng(seed)
    num_vertices = int(rng.integers(2, 11))
    num_edges = int(rng.integers(1, 3 * num_vertices))
    instance = random_dag(rng, num_vertices, num_edges)
    network = instance.build()
    assert network.is_acyclic
    for source in range(num_vertices):
        for target in range(num_vertices):
            paths = enumerate_paths(instance, source, target)
            expected = best_cost(instance, paths, feasible_only=False)
            for solution in (
                network.solve(source, target, algorithm="dag"),
                network.solve(source, target, algorithm="rcsp"),
                cp.shortest_path(network, source, target),
            ):
                if not paths:
                    assert solution.status == "unreachable"
                    continue
                assert solution.status == "optimal"
                assert solution.cost == expected
                edges = solution.edges.tolist()
                check_path(instance, edges, source, target)
                assert path_cost(instance, edges) == expected


def test_layered_graph_built_in_bulk():
    # 200 layers of 50 vertices, complete bipartite between consecutive layers.
    rng = np.random.default_rng(11)
    layers, width = 200, 50
    num_vertices = layers * width + 2
    source_vertex, target_vertex = num_vertices - 2, num_vertices - 1
    left = np.repeat(np.arange(width), width)
    right = np.tile(np.arange(width), width)
    sources, targets = [], []
    for layer in range(layers - 1):
        sources.append(layer * width + left)
        targets.append((layer + 1) * width + right)
    sources.append(np.full(width, source_vertex))
    targets.append(np.arange(width))
    sources.append((layers - 1) * width + np.arange(width))
    targets.append(np.full(width, target_vertex))
    source = np.concatenate(sources)
    target = np.concatenate(targets)
    cost = rng.uniform(-1.0, 1.0, size=source.shape[0])

    graph = cp.Graph(num_vertices=num_vertices)
    graph.add_edges(source, target, cost)
    solution = graph.compile().solve(source_vertex, target_vertex)

    # Reference: plain dynamic programme over the layers.
    best = np.full(num_vertices, np.inf)
    best[source_vertex] = 0.0
    for edge in np.argsort(target, kind="stable"):  # targets ascend with the layers
        if target[edge] == target_vertex:
            continue
        best[target[edge]] = min(best[target[edge]], best[source[edge]] + cost[edge])
    final = np.flatnonzero(target == target_vertex)
    expected = float(np.min(best[source[final]] + cost[final]))
    assert solution.cost == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert solution.edges.size == layers + 1
    assert float(np.sum(cost[solution.edges])) == pytest.approx(solution.cost, abs=1e-9)
