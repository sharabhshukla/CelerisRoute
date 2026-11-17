import math

import numpy as np
import pytest

import celerisroute as cr


@pytest.fixture
def states() -> cr.StateSpace:
    return cr.StateSpace(day=range(1, 6), port=["A", "B"])


def test_add_edge_returns_stable_consecutive_ids(states):
    graph = cr.Graph(states=states, directed=True)
    assert graph.num_vertices == 10 and graph.num_edges == 0
    first = graph.add_edge((1, "A"), (2, "B"), cost=3.0)
    second = graph.add_edge(source=(2, "B"), target=(3, "A"), cost=-1.5)
    parallel = graph.add_edge((1, "A"), (2, "B"), cost=7.0)
    assert (first, second, parallel) == (0, 1, 2)
    network = graph.compile()
    # Edge ids survive compilation, whatever order the adjacency is stored in.
    assert network.edge_source.tolist() == [0, 3, 0]
    assert network.edge_target.tolist() == [3, 4, 3]
    assert network.edge_cost.tolist() == [3.0, -1.5, 7.0]


def test_integer_and_state_endpoints_are_interchangeable(states):
    graph = cr.Graph(states=states)
    graph.add_edge(states.id((1, "A")), (2, "B"), 1.0)
    graph.add_edge({"day": 2, "port": "B"}, states.id((3, "A")), 1.0)
    network = graph.compile()
    solution = network.solve((1, "A"), states.id((3, "A")))
    assert solution.path == [(1, "A"), (2, "B"), (3, "A")]


def test_graph_without_state_space():
    graph = cr.Graph(num_vertices=4)
    graph.add_edge(0, 1, 1.0)
    graph.add_edge(1, 3, 1.0)
    solution = graph.compile().solve(0, 3)
    assert solution.path == [0, 1, 3]
    with pytest.raises(TypeError, match="StateSpace"):
        graph.add_edge((0,), 1, 1.0)


def test_bulk_edges_match_single_edges(states):
    rng = np.random.default_rng(1)
    source = rng.integers(0, 5, size=50)
    target = rng.integers(5, 10, size=50)
    cost = rng.normal(size=50)
    fuel = rng.uniform(0, 3, size=50)

    bulk = cr.Graph(states=states)
    bulk.add_resource(cr.Resource("fuel", upper=10.0))
    ids = bulk.add_edges(source=source, target=target, cost=cost, resources={"fuel": fuel})
    assert ids == range(0, 50)
    assert bulk.add_edges(source[:3], target[:3], cost[:3]) == range(50, 53)

    single = cr.Graph(states=states)
    single.add_resource(cr.Resource("fuel", upper=10.0))
    for s, t, c, f in zip(source.tolist(), target.tolist(), cost.tolist(), fuel.tolist()):
        single.add_edge(s, t, c, resources={"fuel": f})
    for s, t, c in zip(source[:3].tolist(), target[:3].tolist(), cost[:3].tolist()):
        single.add_edge(s, t, c)

    a, b = bulk.compile(), single.compile()
    assert a.edge_source.tolist() == b.edge_source.tolist()
    assert a.edge_target.tolist() == b.edge_target.tolist()
    assert a.edge_cost.tolist() == b.edge_cost.tolist()
    assert a.edge_resource("fuel").tolist() == b.edge_resource("fuel").tolist()
    # Edges added without the resource carry a zero increment.
    assert a.edge_resource("fuel")[50:].tolist() == [0.0, 0.0, 0.0]


def test_bulk_edges_accept_other_layouts_and_scalars():
    graph = cr.Graph(num_vertices=6)
    graph.add_resource(cr.Resource("time", upper=100.0))
    source = np.arange(10, dtype=np.int32)[::2]  # non-contiguous view, narrower dtype
    target = [1, 3, 5, 5, 5]
    with pytest.raises(cr.InvalidStateError):
        graph.add_edges(source, target, 1.0)  # ids 6 and 8 do not exist
    assert graph.num_edges == 0  # nothing was added
    ids = graph.add_edges(source[:3], target[:3], cost=2, resources={"time": 1.5})
    assert list(ids) == [0, 1, 2]
    network = graph.compile()
    assert network.edge_cost.tolist() == [2.0, 2.0, 2.0]
    assert network.edge_resource("time").tolist() == [1.5, 1.5, 1.5]


def test_compile_reports_structure(states):
    graph = cr.Graph(states=states)
    graph.add_resource(cr.Resource("fuel", upper=5.0))
    graph.add_resource(cr.Resource("distance"))  # reported, not constraining
    graph.add_edge((1, "A"), (2, "A"), 1.0, {"fuel": 1.0, "distance": 10.0})
    graph.add_edge((2, "A"), (3, "A"), 1.0, {"fuel": 1.0, "distance": 12.0})
    network = graph.compile()
    assert network.is_acyclic
    assert network.num_vertices == 10 and network.num_edges == 2
    assert [r.name for r in network.resources] == ["fuel", "distance"]
    assert network.tracked_resources == ("fuel",)
    assert network.find_cycle() is None
    assert network.states is states
    assert network.decode(network.node_id((2, "A"))) == (2, "A")

    order = network.topological_order.tolist()
    assert sorted(order) == list(range(10))
    position = {v: i for i, v in enumerate(order)}
    for s, t in zip(network.edge_source.tolist(), network.edge_target.tolist()):
        assert position[s] < position[t]

    solution = network.solve((1, "A"), (3, "A"))
    assert solution.resources == {"fuel": 2.0, "distance": 22.0}


def test_network_is_a_snapshot(states):
    graph = cr.Graph(states=states)
    graph.add_edge((1, "A"), (2, "A"), 5.0)
    network = graph.compile()
    graph.add_edge((1, "A"), (2, "A"), 1.0)  # later edits do not leak into the snapshot
    assert network.num_edges == 1
    assert network.solve((1, "A"), (2, "A")).cost == 5.0
    assert graph.compile().solve((1, "A"), (2, "A")).cost == 1.0


def test_exposed_arrays_are_read_only(states):
    graph = cr.Graph(states=states)
    graph.add_edge((1, "A"), (2, "A"), 5.0)
    network = graph.compile()
    for array in (network.edge_source, network.edge_target, network.edge_cost):
        with pytest.raises(ValueError):
            array[0] = 0
    solution = network.solve((1, "A"), (2, "A"))
    with pytest.raises(ValueError):
        solution.edges[0] = 3


def test_with_costs_shares_topology(states):
    graph = cr.Graph(states=states)
    graph.add_edge((1, "A"), (2, "A"), 5.0)
    graph.add_edge((1, "A"), (2, "A"), 6.0)
    network = graph.compile()
    flipped = network.with_costs([9.0, 2.0])
    assert network.solve((1, "A"), (2, "A")).edges.tolist() == [0]
    assert flipped.solve((1, "A"), (2, "A")).edges.tolist() == [1]
    with pytest.raises(ValueError):
        network.with_costs([1.0])
    with pytest.raises(ValueError):
        network.with_costs([1.0, math.nan])


# ------------------------------------------------------------------ cycles


def test_cyclic_graph_is_rejected_by_the_solvers():
    graph = cr.Graph(num_vertices=5)
    graph.add_edge(0, 1, 1.0)
    graph.add_edge(1, 2, 1.0)
    graph.add_edge(2, 3, 1.0)
    graph.add_edge(3, 1, 1.0)  # closes 1 -> 2 -> 3 -> 1
    graph.add_edge(3, 4, 1.0)
    network = graph.compile()
    assert not network.is_acyclic

    cycle = network.find_cycle()
    assert sorted(cycle) == [1, 2, 3]
    edges = set(zip(network.edge_source.tolist(), network.edge_target.tolist()))
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        assert (a, b) in edges

    for algorithm in ("auto", "dag", "rcsp"):
        with pytest.raises(cr.CyclicGraphError, match="cycle") as info:
            network.solve(0, 4, algorithm=algorithm)
        assert sorted(info.value.cycle) == [1, 2, 3]
    with pytest.raises(cr.CyclicGraphError):
        cr.shortest_path(network, 0, 4)
    with pytest.raises(cr.CyclicGraphError):
        cr.resource_shortest_path(network, 0, 4)
    with pytest.raises(cr.CyclicGraphError):
        cr.PricingEngine(network, 0, 4)
    with pytest.raises(cr.CyclicGraphError):
        network.topological_order
    assert issubclass(cr.CyclicGraphError, ValueError)


def test_self_loop_is_a_cycle():
    graph = cr.Graph(num_vertices=2)
    graph.add_edge(0, 0, 1.0)
    graph.add_edge(0, 1, 1.0)
    network = graph.compile()
    assert network.find_cycle() == [0]
    with pytest.raises(cr.CyclicGraphError):
        network.solve(0, 1)


def test_cycle_error_names_states(states):
    graph = cr.Graph(states=states)
    graph.add_edge((1, "A"), (1, "B"), 0.0)
    graph.add_edge((1, "B"), (1, "A"), 0.0)
    with pytest.raises(cr.CyclicGraphError, match=r"\(1, 'A'\)"):
        graph.compile().solve((1, "A"), (1, "B"))


# ------------------------------------------------------------ invalid input


def test_invalid_graph_construction(states):
    with pytest.raises(ValueError):
        cr.Graph()
    with pytest.raises(NotImplementedError, match="undirected"):
        cr.Graph(states=states, directed=False)
    with pytest.raises(TypeError):
        cr.Graph(states=[1, 2, 3])
    with pytest.raises(ValueError):
        cr.Graph(num_vertices=0)
    with pytest.raises(ValueError):
        cr.Graph(states=states, num_vertices=3)
    with pytest.raises(TypeError):
        cr.Graph(num_vertices=2.5)


def test_invalid_single_edges(states):
    graph = cr.Graph(states=states)
    with pytest.raises(cr.InvalidStateError):
        graph.add_edge((1, "A"), (9, "A"), 1.0)
    with pytest.raises(cr.InvalidStateError):
        graph.add_edge(0, 10, 1.0)
    with pytest.raises(cr.InvalidStateError):
        graph.add_edge(-1, 0, 1.0)
    with pytest.raises(TypeError):
        graph.add_edge(0, 1, "cheap")
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            graph.add_edge(0, 1, bad)
    with pytest.raises(ValueError, match="finite"):
        graph.add_edge(0, 1, 1.0, resources={"fuel": math.nan})
    with pytest.raises(TypeError):
        graph.add_edge(0, 1, 1.0, resources={"fuel": "lots"})
    with pytest.raises(TypeError):
        graph.add_edge(0, 1, 1.0, resources=[("fuel", 1.0)])
    assert graph.num_edges == 0


def test_invalid_bulk_edges():
    graph = cr.Graph(num_vertices=4)
    ok = np.array([0, 1])
    with pytest.raises(ValueError, match="same edges"):
        graph.add_edges(ok, np.array([1, 2, 3]), 1.0)
    with pytest.raises(ValueError, match="one per edge"):
        graph.add_edges(ok, ok + 1, np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="one per edge"):
        graph.add_edges(ok, ok + 1, 1.0, resources={"fuel": np.ones(5)})
    with pytest.raises(ValueError, match="1-D"):
        graph.add_edges(np.zeros((2, 2), dtype=np.int64), ok, 1.0)
    with pytest.raises(TypeError, match="integer"):
        graph.add_edges(np.array([0.0, 1.0]), ok + 1, 1.0)
    with pytest.raises(TypeError, match="numeric"):
        graph.add_edges(ok, ok + 1, np.array(["a", "b"]))
    with pytest.raises(ValueError, match="finite"):
        graph.add_edges(ok, ok + 1, np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="finite"):
        graph.add_edges(ok, ok + 1, np.array([1.0, np.inf]))
    with pytest.raises(ValueError, match="finite"):
        graph.add_edges(ok, ok + 1, 1.0, resources={"fuel": np.array([1.0, np.nan])})
    with pytest.raises(cr.InvalidStateError):
        graph.add_edges(ok, np.array([1, 4]), 1.0)
    with pytest.raises(cr.InvalidStateError):
        graph.add_edges(np.array([-1, 0]), ok + 1, 1.0)
    assert graph.num_edges == 0
    assert graph.add_edges([], [], []) == range(0, 0)


def test_solve_rejects_bad_arguments(states):
    graph = cr.Graph(states=states)
    graph.add_edge((1, "A"), (2, "A"), 1.0)
    network = graph.compile()
    with pytest.raises(ValueError, match="algorithm"):
        network.solve((1, "A"), (2, "A"), algorithm="dijkstra")
    with pytest.raises(cr.InvalidStateError):
        network.solve((1, "A"), (7, "A"))
    with pytest.raises(cr.InvalidStateError):
        network.solve(0, 99)
    with pytest.raises(ValueError, match="dominance_tolerance"):
        network.solve(0, 2, algorithm="rcsp", dominance_tolerance=-1.0)
    with pytest.raises(TypeError):
        cr.shortest_path("not a network", 0, 1)
