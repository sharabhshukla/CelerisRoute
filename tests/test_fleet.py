import math
import threading
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp

import celerispath as cp
from _reference import Instance, check_path, enumerate_paths, random_dag
from test_pricing import random_duals, random_matrix

SCHEDULERS = ["threads"] + (["tbb"] if cp.tbb_available() else [])
COST_UPDATES = ["shared_sparse", "indexed", "per_vessel"]


# ------------------------------------------------------------------ reference


@dataclass
class Case:
    """A random fleet together with everything needed to price it by hand."""

    instance: Instance
    network: cp.Network
    matrix: np.ndarray  # shared coefficients, dense (K, E)
    duals: np.ndarray
    source: np.ndarray
    target: np.ndarray
    limits: dict[str, np.ndarray]
    fixed_cost: np.ndarray
    fixed: list[dict[int, float]]
    offsets: list[dict[int, float]]
    forbidden: list[list[int]]
    coefficients: list[np.ndarray | None]  # vessel-specific (K, E), dense

    @property
    def num_vessels(self) -> int:
        return int(self.source.shape[0])

    def fleet(self, algorithm="rcsp", **options) -> cp.PricingFleet:
        if algorithm == "dag" and self.instance.upper:
            options.setdefault("unconstrained", True)  # ignore the bounds on purpose
        fleet = cp.PricingFleet(
            self.network, sp.csr_array(self.matrix), algorithm=algorithm, **options
        )
        self.add_to(fleet, algorithm)
        return fleet

    def add_to(self, fleet, algorithm="rcsp", vessels=None) -> range:
        chosen = range(self.num_vessels) if vessels is None else vessels
        pick = lambda values: [values[v] for v in chosen]  # noqa: E731
        return fleet.add_vessels(
            ids=[f"v{v}" for v in chosen],
            source=self.source[list(chosen)],
            target=self.target[list(chosen)],
            resource_limits=(
                {name: values[list(chosen)] for name, values in self.limits.items()}
                if algorithm == "rcsp" and self.limits else None
            ),
            fixed_cost=self.fixed_cost[list(chosen)],
            fixed_coefficients=pick(self.fixed),
            cost_offsets=pick(self.offsets),
            forbidden_edges=pick(self.forbidden),
            edge_coefficients=[
                None if block is None else sp.csc_array(block) for block in pick(self.coefficients)
            ],
        )

    # Everything below prices a vessel by enumerating its paths in Python.

    def paths(self, v: int) -> list[list[int]]:
        """Paths of the vessel's own network: forbidden edges removed."""
        removed = set(self.forbidden[v])
        removed.update(e for e, offset in self.offsets[v].items() if offset == math.inf)
        return [
            path
            for path in enumerate_paths(self.instance, int(self.source[v]), int(self.target[v]))
            if removed.isdisjoint(path)
        ]

    def feasible(self, v: int, path: list[int]) -> bool:
        for name, limits in self.limits.items():
            total = 0.0
            for edge in path:
                total += float(self.instance.resources[name][edge])
                if total > limits[v]:
                    return False
        return True

    def evaluate(self, v: int, path: list[int]):
        """(original cost, reduced cost, coefficient vector) of a path for vessel v."""
        cost = float(self.fixed_cost[v])
        coefficients = np.zeros(self.matrix.shape[0])
        for row, value in self.fixed[v].items():
            coefficients[row] += value
        for edge in path:
            cost += float(self.instance.cost[edge]) + self.offsets[v].get(edge, 0.0)
            coefficients += self.matrix[:, edge]
            if self.coefficients[v] is not None:
                coefficients += self.coefficients[v][:, edge]
        return cost, cost - float(self.duals @ coefficients), coefficients

    def expected(self, v: int, algorithm: str = "rcsp"):
        """(status, sorted reduced costs of all admissible paths)."""
        paths = self.paths(v)
        if not paths:
            return "unreachable", []
        if algorithm == "rcsp":
            paths = [path for path in paths if self.feasible(v, path)]
            if not paths:
                return "infeasible", []
        return "optimal", sorted(self.evaluate(v, path)[1] for path in paths)


def make_case(seed: int, num_vessels: int = 24, num_resources: int = 2, overlays: bool = True) -> Case:
    rng = np.random.default_rng(seed)
    instance = random_dag(rng, 13, 54, num_resources=num_resources, tightness=0.9)
    network = instance.build()
    num_edges, num_rows = instance.num_edges, 7
    matrix = random_matrix(rng, num_rows, num_edges)
    duals = random_duals(rng, num_rows)
    # Sources early and targets late in the hidden order give most vessels
    # paths; the rest exercise the unreachable branch.
    rank = np.argsort(network.topological_order)
    source = rng.integers(0, 13, size=num_vessels)
    target = rng.integers(0, 13, size=num_vessels)
    swap = rank[source] > rank[target]
    source[swap], target[swap] = target[swap], source[swap]
    limits = {}
    for name, upper in instance.upper.items():
        # Between "nothing fits" and "the graph's own bound".
        limits[name] = np.floor(rng.uniform(0.15, 1.0, size=num_vessels) * upper)
    fixed_cost = rng.integers(-3, 4, size=num_vessels).astype(np.float64)
    fixed, offsets, forbidden, coefficients = [], [], [], []
    for _ in range(num_vessels):
        fixed.append({int(rng.integers(num_rows)): float(rng.integers(1, 3))})
        if not overlays:
            offsets.append({})
            forbidden.append([])
            coefficients.append(None)
            continue
        touched = rng.choice(num_edges, size=int(rng.integers(0, 9)), replace=False)
        offsets.append({int(e): float(rng.integers(-5, 6)) for e in touched})
        forbidden.append(rng.choice(num_edges, size=int(rng.integers(0, 5)), replace=False).tolist())
        if rng.random() < 0.6:
            block = random_matrix(rng, num_rows, num_edges, density=0.08)
            coefficients.append(block)
        else:
            coefficients.append(None)
    return Case(instance, network, matrix, duals, source.astype(np.int64),
                target.astype(np.int64), limits, fixed_cost, fixed, offsets, forbidden,
                coefficients)


def check_against_reference(case: Case, result: cp.FleetResult, algorithm: str = "rcsp",
                            max_paths: int = 1, negative_only: bool = False) -> None:
    assert result.vessel_ids == tuple(f"v{v}" for v in range(case.num_vessels))
    assert result.all_priced_exactly
    seen = 0
    for v in range(case.num_vessels):
        status, reduced = case.expected(v, algorithm)
        assert result.statuses[v] == status, v
        columns = result.columns_of(v)
        seen += len(columns)
        if status != "optimal":
            assert result.best_reduced_costs[v] == math.inf and not columns
            continue
        assert result.best_reduced_costs[v] == reduced[0], v
        wanted = reduced[:max_paths]
        if negative_only:
            wanted = [value for value in wanted if value < -result.tolerance]
        assert [column.reduced_cost for column in columns] == wanted, v
        for column in columns:
            edges = column.edges.tolist()
            assert column.vessel == f"v{v}" and column.vessel_index == v
            check_path(case.instance, edges, int(case.source[v]), int(case.target[v]))
            assert edges in case.paths(v)
            if algorithm == "rcsp":
                assert case.feasible(v, edges)
            cost, reduced_cost, coefficients = case.evaluate(v, edges)
            assert column.original_cost == cost
            assert column.reduced_cost == reduced_cost
            np.testing.assert_array_equal(column.dense_coefficients(), coefficients)
            for name in case.instance.resources:
                assert column.resources[name] == sum(
                    float(case.instance.resources[name][e]) for e in edges
                )
    assert seen == result.num_columns == len(result.columns)


def snapshot(result: cp.FleetResult):
    """Everything observable about a fleet result except its timings."""
    stats = result.statistics
    return (
        result.vessel_ids,
        result.status_codes.tolist(),
        result.best_reduced_costs.tolist(),
        result.column_ptr.tolist(),
        result.column_vessel.tolist(),
        result.original_costs.tolist(),
        result.reduced_costs.tolist(),
        result.resources.tolist(),
        result.edge_ptr.tolist(),
        result.edges.tolist(),
        result.coefficient_ptr.tolist(),
        result.coefficient_indices.tolist(),
        result.coefficient_values.tolist(),
        (stats.labels_created, stats.labels_extended, stats.labels_dominated,
         stats.labels_infeasible, stats.max_active_labels, stats.pareto_labels),
    )


# --------------------------------------------------------- exactness of pricing


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("cost_update", COST_UPDATES)
def test_rcsp_fleet_against_enumeration(seed, cost_update):
    case = make_case(seed)
    fleet = case.fleet(cost_update=cost_update, threads=3)
    result = fleet.price(case.duals, negative_only=False)
    check_against_reference(case, result)
    statuses = {str(status) for status in result.statuses}
    assert "optimal" in statuses


def test_random_cases_cover_every_status_and_negative_reduced_costs():
    statuses, negative = set(), 0
    for seed in range(8):
        case = make_case(seed)
        result = case.fleet().price(case.duals, negative_only=False)
        statuses.update(str(status) for status in result.statuses)
        negative += int((result.best_reduced_costs < 0).sum())
    assert statuses == {"optimal", "unreachable", "infeasible"}
    assert negative > 10


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("cost_update", COST_UPDATES)
@pytest.mark.parametrize("max_paths", [1, 4])
def test_dag_fleet_against_enumeration(seed, cost_update, max_paths):
    case = make_case(100 + seed, num_resources=0)
    fleet = case.fleet(algorithm="dag", cost_update=cost_update, threads=3)
    result = fleet.price(case.duals, max_paths, negative_only=False)
    check_against_reference(case, result, "dag", max_paths)
    result = fleet.price(case.duals, max_paths, negative_only=True)
    check_against_reference(case, result, "dag", max_paths, negative_only=True)


@pytest.mark.parametrize("seed", range(4))
def test_dag_fleet_ignores_resources(seed):
    case = make_case(200 + seed, num_resources=2)
    case.limits = {}
    fleet = case.fleet(algorithm="dag")
    check_against_reference(case, fleet.price(case.duals, negative_only=False), "dag")


@pytest.mark.parametrize("algorithm", ["dag", "rcsp"])
def test_fleet_agrees_with_one_pricing_engine_per_vessel(algorithm):
    case = make_case(7, num_vessels=30, overlays=False)
    case.limits = {}  # engines use the graph's bounds
    fleet = case.fleet(algorithm=algorithm)
    result = fleet.price(case.duals, negative_only=False)
    for v in range(case.num_vessels):
        engine = cp.PricingEngine(
            case.network, int(case.source[v]), int(case.target[v]),
            dual_matrix=sp.csr_array(case.matrix), algorithm=algorithm,
            unconstrained=algorithm == "dag",
            fixed_cost=float(case.fixed_cost[v]), fixed_coefficients=case.fixed[v],
        )
        single = engine.solve(case.duals, negative_only=False)
        assert result.statuses[v] == single.status
        assert result.best_reduced_costs[v] == single.best_reduced_cost
        if single:
            column = result.columns_of(v)[0]
            assert column.original_cost == single[0].original_cost
            np.testing.assert_array_equal(column.dense_coefficients(), single[0].dense_coefficients())


def test_negative_only_keeps_the_certificate():
    case = make_case(3)
    fleet = case.fleet()
    everything = fleet.price(case.duals, negative_only=False)
    improving = fleet.price(case.duals, negative_only=True)
    check_against_reference(case, improving, negative_only=True)
    np.testing.assert_array_equal(everything.best_reduced_costs, improving.best_reduced_costs)
    assert 0 < improving.num_columns < everything.num_columns
    assert (improving.reduced_costs < 0).all()
    assert improving.best_reduced_cost == everything.best_reduced_cost < 0
    assert not improving.converged

    # Duals of zero and costs shifted to be positive: nothing improves, and
    # the empty result is a certificate because every vessel was priced.
    quiet = cp.PricingFleet(case.network, num_constraints=0)
    quiet.add_vessels(source=case.source, target=case.target,
                      fixed_cost=np.full(case.num_vessels, 1000.0))
    result = quiet.price([])
    assert result.num_columns == 0 and result.columns == ()
    assert result.all_priced_exactly and result.converged
    assert result.best_reduced_cost > 0


def test_tolerance_decides_what_counts_as_negative():
    graph = cp.Graph(num_vertices=2)
    graph.add_edge(0, 1, 1.0)
    fleet = cp.PricingFleet(graph.compile(), sp.csr_array(np.array([[1.0]])))
    fleet.add_vessels(source=0, target=1)
    assert fleet.price([1.5]).num_columns == 1
    result = fleet.price([1.5], tolerance=0.75)
    assert result.num_columns == 0 and result.converged
    assert result.best_reduced_cost == -0.5
    assert not fleet.price([1.5], tolerance=0.25).converged


# ------------------------------------------------- determinism and schedulers


@pytest.mark.parametrize("algorithm", ["dag", "rcsp"])
def test_parallel_pricing_equals_serial_pricing(algorithm):
    case = make_case(11, num_vessels=90, num_resources=2)
    paths = 3 if algorithm == "dag" else 1
    serial = case.fleet(algorithm=algorithm, threads=1, scheduler="threads")
    reference = snapshot(serial.price(case.duals, paths, negative_only=False))
    assert serial.threads == 1
    for scheduler in SCHEDULERS:
        partitioners = ["auto", "simple", "static"] if scheduler == "tbb" else ["auto"]
        for partitioner in partitioners:
            for threads in (2, 5, 16):
                fleet = case.fleet(algorithm=algorithm, threads=threads, scheduler=scheduler,
                                   partitioner=partitioner)
                assert fleet.scheduler == scheduler and fleet.threads == threads
                for _ in range(3):
                    result = fleet.price(case.duals, paths, negative_only=False)
                    assert snapshot(result) == reference
                    assert result.statistics.threads == threads


@pytest.mark.parametrize("cost_update", COST_UPDATES)
def test_cost_update_modes_are_numerically_identical(cost_update):
    case = make_case(12, num_vessels=40)
    reference = snapshot(case.fleet(cost_update="shared_sparse", threads=1)
                         .price(case.duals, negative_only=False))
    fleet = case.fleet(cost_update=cost_update, threads=4)
    assert fleet.cost_update == cost_update
    assert snapshot(fleet.price(case.duals, negative_only=False)) == reference


def test_repeated_dual_updates_do_not_leak_state():
    case = make_case(5, num_vessels=30)
    rng = np.random.default_rng(1)
    for cost_update in COST_UPDATES:
        fleet = case.fleet(cost_update=cost_update, threads=4)
        first = snapshot(fleet.price(case.duals, negative_only=False))
        for _ in range(4):
            fleet.price(random_duals(rng, 7), negative_only=False)
        assert snapshot(fleet.price(case.duals, negative_only=False)) == first


def test_update_then_solve_equals_price():
    case = make_case(6)
    fleet = case.fleet()
    with pytest.raises(ValueError, match="update_duals"):
        fleet.solve()
    fused = fleet.price(case.duals, negative_only=False)
    fleet.update_duals(np.zeros(7))
    fleet.update_duals(case.duals)
    split = fleet.solve(negative_only=False)
    assert snapshot(split) == snapshot(fused)
    assert snapshot(fleet.solve(negative_only=False)) == snapshot(fused)  # duals persist
    assert split.statistics.update_time == 0.0
    assert fused.statistics.total_time >= fused.statistics.solve_time > 0.0


def test_results_follow_registration_order_and_ids():
    case = make_case(8, num_vessels=40)
    fleet = case.fleet(threads=6)
    result = fleet.price(case.duals, negative_only=False)
    assert result.vessel_ids == fleet.vessel_ids
    assert fleet.index_of("v17") == 17
    np.testing.assert_array_equal(np.diff(result.column_vessel) >= 0, True)
    for v in range(case.num_vessels):
        assert [column.vessel_index for column in result.columns_of(v)] == [v] * len(
            result.columns_of(v)
        )
    with pytest.raises(KeyError):
        fleet.index_of("nobody")
    with pytest.raises(IndexError):
        result.columns_of(case.num_vessels)


def test_vessels_can_be_added_between_solves():
    case = make_case(9, num_vessels=30)
    whole = snapshot(case.fleet().price(case.duals, negative_only=False))
    fleet = cp.PricingFleet(case.network, sp.csr_array(case.matrix), algorithm="rcsp", threads=3)
    assert case.add_to(fleet, vessels=range(0, 12)) == range(0, 12)
    assert not fleet.is_prepared
    assert fleet.prepare() is fleet and fleet.is_prepared
    partial = fleet.price(case.duals, negative_only=False)
    assert partial.num_vessels == 12
    assert case.add_to(fleet, vessels=range(12, 30)) == range(12, 30)
    assert not fleet.is_prepared and len(fleet) == 30
    assert snapshot(fleet.solve(negative_only=False)) == whole  # duals survive the addition
    assert fleet.is_prepared


def test_an_empty_fleet_prices_to_an_empty_result():
    case = make_case(0)
    fleet = cp.PricingFleet(case.network, sp.csr_array(case.matrix))
    result = fleet.price(case.duals)
    assert result.num_vessels == 0 and result.num_columns == 0
    assert result.best_reduced_cost == math.inf and result.converged
    assert result.coefficient_matrix.shape == (7, 0)


# ------------------------------------------------------------ vessel profiles


def chain(num_edges: int = 4, fuel: float | None = None) -> cp.Network:
    """0 -> 1 -> ... with a parallel, dearer edge between each pair."""
    graph = cp.Graph(num_vertices=num_edges + 1)
    if fuel is not None:
        graph.add_resource(cp.Resource("fuel", upper=fuel))
    for v in range(num_edges):
        resources = None if fuel is None else {"fuel": 2.0}
        cheap = None if fuel is None else {"fuel": 1.0}
        graph.add_edge(v, v + 1, 1.0, resources)   # edge 2v: cheap, thirsty
        graph.add_edge(v, v + 1, 5.0, cheap)       # edge 2v + 1: dear, frugal
    return graph.compile()


def test_cost_offsets_masks_and_forbidden_edges():
    network = chain()
    fleet = cp.PricingFleet(network, num_constraints=0)
    mask = np.ones((5, network.num_edges), dtype=bool)
    mask[3, [0, 2]] = False
    mask[4, [6, 7]] = False
    fleet.add_vessels(
        ids=["plain", "offset", "forbidden", "masked", "cut"],
        source=0, target=4,
        cost_offsets=[None, {0: 10.0, 1: -1.0}, None, None, None],
        forbidden_edges=[None, None, [4], None, None],
        edge_mask=mask,
    )
    result = fleet.price([], negative_only=False)
    by_id = {column.vessel: column for column in result.columns}
    assert by_id["plain"].edges.tolist() == [0, 2, 4, 6] and by_id["plain"].original_cost == 4.0
    assert by_id["offset"].edges.tolist() == [1, 2, 4, 6] and by_id["offset"].original_cost == 7.0
    assert by_id["forbidden"].edges.tolist() == [0, 2, 5, 6]
    assert by_id["masked"].edges.tolist() == [1, 3, 4, 6] and by_id["masked"].original_cost == 12.0
    assert "cut" not in by_id
    assert [str(status) for status in result.statuses] == ["optimal"] * 4 + ["unreachable"]
    np.testing.assert_array_equal(fleet.reachable, [True, True, True, True, False])
    # The overlays of one vessel never leak into the next solve of another.
    assert snapshot(fleet.price([], negative_only=False)) == snapshot(result)


def test_offset_formats_agree_and_duplicates_add_up():
    network = chain()
    dense = np.zeros((2, network.num_edges))
    dense[0, 0], dense[1, 2] = 7.0, math.inf
    results = []
    for offsets in (dense, sp.csr_array(dense), sp.coo_array(dense), [{0: 7.0}, {2: math.inf}],
                    [dense[0], dense[1]]):
        fleet = cp.PricingFleet(network, num_constraints=0)
        fleet.add_vessels(source=0, target=4, cost_offsets=offsets)
        results.append(snapshot(fleet.price([], negative_only=False)))
    assert all(result == results[0] for result in results)
    assert results[0][5] == [8.0, 8.0]  # original costs: 0 -> 1 by edge 1, and 1 -> 2 by edge 3

    fleet = cp.PricingFleet(network, num_constraints=0)
    twice = sp.coo_array(([1.5, 1.5], ([0, 0], [0, 0])), shape=(1, network.num_edges))
    fleet.add_vessels(source=0, target=4, cost_offsets=twice, forbidden_edges=[[1, 1]])
    assert fleet.price([], negative_only=False).original_costs.tolist() == [7.0]


def test_per_vessel_resource_limits():
    network = chain(fuel=8.0)  # all cheap edges need 8 fuel, all dear ones 4
    fleet = cp.PricingFleet(network, num_constraints=0)
    assert fleet.algorithm == "rcsp"
    fleet.add_vessels(source=0, target=4,
                      resource_limits={"fuel": [8.0, 6.0, 4.0, 3.0, math.inf, 100.0]})
    fleet.add_vessels(source=0, target=4)                         # the graph's bound
    fleet.add_vessels(source=[0, 0], target=[4, 4], resource_limits=np.array([[5.0], [7.0]]))
    result = fleet.price([], negative_only=False)
    assert result.original_costs.tolist() == [4.0, 12.0, 20.0, 4.0, 4.0, 4.0, 16.0, 8.0]
    assert [str(status) for status in result.statuses] == (
        ["optimal"] * 3 + ["infeasible"] + ["optimal"] * 5
    )
    assert result.best_reduced_costs[3] == math.inf
    fuel = result.resources[:, 0].tolist()
    assert fuel == [8.0, 6.0, 4.0, 8.0, 8.0, 8.0, 5.0, 7.0]
    assert result.columns[1].resources == {"fuel": 6.0}
    np.testing.assert_array_equal(fleet.reachable, True)  # reachability ignores resources


def test_vessel_specific_coefficients_and_fixed_parts():
    network = chain(2)
    shared = sp.csr_array(np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]]))
    extra = sp.csr_array(np.array([[0, 0, 0, 0], [0, 0, 2.0, 0], [0, 3.0, 0, 0]]))
    for cost_update in COST_UPDATES:
        fleet = cp.PricingFleet(network, shared, cost_update=cost_update)
        fleet.add_vessels(
            ids=["shared", "own"], source=0, target=2,
            edge_coefficients={1: extra},
            fixed_cost=[0.0, 2.5],
            fixed_coefficients=sp.csr_array(np.array([[0, 0, 0], [0, 0, 1.0]])),
        )
        # Row 2 only matters to "own": edge 1 gets 3 * 4 = 12 cheaper for it.
        result = fleet.price([1.0, 1.0, 4.0], negative_only=False)
        shared_column, own = result.columns
        assert shared_column.edges.tolist() == [0, 2]
        assert shared_column.reduced_cost == 2.0 - 2.0
        assert shared_column.dense_coefficients().tolist() == [1.0, 1.0, 0.0]
        assert own.edges.tolist() == [1, 2]
        assert own.dense_coefficients().tolist() == [0.0, 3.0, 4.0]
        assert own.original_cost == 6.0 + 2.5
        assert own.reduced_cost == 8.5 - (3.0 + 16.0)
        matrix = result.coefficient_matrix
        assert matrix.shape == (3, 2)
        np.testing.assert_array_equal(matrix.toarray(), [[1.0, 0.0], [1.0, 3.0], [0.0, 4.0]])


def test_cancelling_coefficients_are_dropped_from_columns():
    network = chain(1)
    fleet = cp.PricingFleet(network, sp.csr_array(np.array([[1.0, 0.0], [1.0, 0.0]])))
    fleet.add_vessels(source=0, target=1,
                      edge_coefficients=[sp.csr_array(np.array([[-1.0, 0.0], [0.0, 0.0]]))])
    column = fleet.price([100.0, 1.0], negative_only=False).columns[0]
    assert column.edges.tolist() == [0]
    assert column.coefficient_indices.tolist() == [1]
    assert column.reduced_cost == 0.0


def test_sources_and_targets_accept_states_ids_and_scalars():
    graph = cp.Graph(states=cp.StateSpace(day=range(3), port=["A", "B", "C"]))
    graph.add_edge((0, "A"), (1, "B"), 1.0)
    graph.add_edge((1, "B"), (2, "C"), 1.0)
    graph.add_edge((0, "A"), (2, "C"), 5.0)
    network = graph.compile()
    a, b, c = (network.node_id(state) for state in [(0, "A"), (1, "B"), (2, "C")])
    fleet = cp.PricingFleet(network, num_constraints=0)
    fleet.add_vessels(source=[(0, "A"), (1, "B")], target=[(2, "C"), (2, "C")])
    fleet.add_vessels(source=a, target=np.array([b, c]))
    fleet.add_vessels(source=a, target=a)  # a vessel that stays where it is
    result = fleet.price([], negative_only=False)
    assert [column.path for column in result.columns] == [
        [(0, "A"), (1, "B"), (2, "C")],
        [(1, "B"), (2, "C")],
        [(0, "A"), (1, "B")],
        [(0, "A"), (1, "B"), (2, "C")],
        [(0, "A")],
    ]
    assert result.columns[-1].edges.size == 0 and result.original_costs[-1] == 0.0
    assert result.vessel_ids == (0, 1, 2, 3, 4)


# --------------------------------------------------------------- dual mappings


def test_indexed_duals_match_the_sparse_matrix():
    case = make_case(13, num_vessels=20)
    rng = np.random.default_rng(0)
    num_edges = case.instance.num_edges
    row = rng.integers(-1, 7, size=(num_edges, 3))
    coefficient = rng.integers(-2, 3, size=(num_edges, 3)).astype(np.float64)
    mapping = cp.IndexedDuals(row, coefficient, num_constraints=7)
    assert (mapping.num_edges, mapping.width, mapping.num_constraints) == (num_edges, 3, 7)
    dense = np.zeros((7, num_edges))
    for edge in range(num_edges):
        for k in range(3):
            if row[edge, k] >= 0:
                dense[row[edge, k], edge] += coefficient[edge, k]
    np.testing.assert_array_equal(mapping.matrix().toarray(), dense)
    case.matrix = dense

    reference = snapshot(case.fleet(threads=1).price(case.duals, negative_only=False))
    for cost_update in (None, "auto", *COST_UPDATES):
        fleet = cp.PricingFleet(case.network, dual_mapping=mapping, algorithm="rcsp",
                                cost_update=cost_update)
        expected = "indexed" if cost_update in (None, "auto") else cost_update
        assert fleet.cost_update == expected
        case.add_to(fleet)
        result = fleet.price(case.duals, negative_only=False)
        assert snapshot(result) == reference
        np.testing.assert_array_equal(fleet.dual_matrix.toarray(), dense)
    check_against_reference(case, result)


def test_port_time_product_duals():
    # Two ports, three days, two products; every edge discharges at one
    # (port, day, product), edge 3 is ballast.
    graph = cp.Graph(num_vertices=4)
    for u, v in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        graph.add_edge(u, v, 10.0)
    network = graph.compile()
    mapping = cp.PortTimeProductDuals(
        port=[0, 1, 1, -1], day=[0, 1, 2, 0], product=[1, 0, 1, 0],
        quantity=[5.0, 2.0, 1.0, 9.0], shape=(2, 3, 2), extra_constraints=1,
    )
    assert mapping.shape == (2, 3, 2) and mapping.num_constraints == 13
    assert mapping.row.ravel().tolist() == [1, 8, 11, -1]
    assert mapping.row_of(1, 2, 1) == 11
    with pytest.raises(IndexError):
        mapping.row_of(2, 0, 0)

    duals = np.zeros((2, 3, 2))
    duals[0, 0, 1] = 1.0   # edge 0: 5 * 1
    duals[1, 1, 0] = 4.0   # edge 1: 2 * 4
    duals[1, 2, 1] = 3.0   # edge 2: 1 * 3
    flat = mapping.flatten(duals, extra=[6.0])
    assert flat.shape == (13,) and flat[12] == 6.0
    with pytest.raises(ValueError, match="extra"):
        mapping.flatten(duals)
    with pytest.raises(ValueError, match="shape"):
        mapping.flatten(np.zeros((3, 2, 2)), extra=[0.0])

    fleet = cp.PricingFleet(network, dual_mapping=mapping)
    assert fleet.cost_update == "indexed" and fleet.num_constraints == 13
    fleet.add_vessels(ids=["a", "b"], source=0, target=3,
                      fixed_coefficients=[{12: 1.0}, None])
    result = fleet.price(flat, negative_only=False)
    a, b = result.columns
    # Via vertex 1: 20 - 5 - 3 = 12; via vertex 2: 20 - 8 = 12 -> a tie on
    # edges, so compare values only.
    assert b.reduced_cost == 12.0 and a.reduced_cost == 12.0 - 6.0
    assert a.dense_coefficients()[12] == 1.0

    plain = cp.PortTimeProductDuals(port=[0, 1, 1, -1], day=[0, 1, 2, 0], product=[1, 0, 1, 0],
                                    quantity=[5.0, 2.0, 1.0, 9.0])
    assert plain.shape == (2, 3, 2) and plain.extra_constraints == 0
    fleet = cp.PricingFleet(network, dual_mapping=plain, cost_update="shared_sparse")
    fleet.add_vessels(source=0, target=3)
    assert fleet.price(duals, negative_only=False).reduced_costs.tolist() == [12.0]  # 3-d duals
    fleet.update_duals(duals.ravel())
    assert fleet.solve(negative_only=False).reduced_costs.tolist() == [12.0]


def test_indexed_duals_validation():
    with pytest.raises(TypeError, match="integer"):
        cp.IndexedDuals([0.5, 1.0])
    with pytest.raises(ValueError, match="num_constraints"):
        cp.IndexedDuals([0, 4], num_constraints=4)
    with pytest.raises(ValueError, match="does not match"):
        cp.IndexedDuals([0, 1, 2], [1.0, 2.0])
    with pytest.raises(ValueError, match="NaN"):
        cp.IndexedDuals([0, 1], [1.0, math.nan])
    assert cp.IndexedDuals([0, -1], [1.0, math.nan]).num_constraints == 1  # unused pair
    with pytest.raises(ValueError, match="same shape"):
        cp.PortTimeProductDuals([0, 1], [0], [0, 0])
    with pytest.raises(ValueError, match="outside the shape"):
        cp.PortTimeProductDuals([0, 2], [0, 0], [0, 0], shape=(2, 1, 1))
    with pytest.raises(ValueError, match="extra_constraints"):
        cp.PortTimeProductDuals([0], [0], [0], extra_constraints=-1)


# ------------------------------------- interruption, cancellation and failures


def heavy_fleet(num_vessels: int = 400, threads: int = 4, scheduler: str = "auto"):
    """A fleet whose solve takes long enough to be interrupted half-way."""
    rng = np.random.default_rng(0)
    layers, width = 10, 8  # about a millisecond of labelling per vessel
    num_vertices = layers * width + 2
    source, target = [], []
    for j in range(width):
        source.append(0)
        target.append(1 + j)
    for layer in range(layers - 1):
        for a in range(width):
            for b in rng.choice(width, size=4, replace=False):
                source.append(1 + layer * width + a)
                target.append(1 + (layer + 1) * width + int(b))
    for j in range(width):
        source.append(1 + (layers - 1) * width + j)
        target.append(num_vertices - 1)
    num_edges = len(source)
    graph = cp.Graph(num_vertices=num_vertices)
    graph.add_resource(cp.Resource("a", upper=27.5))
    graph.add_resource(cp.Resource("b", upper=27.5))
    graph.add_edges(np.array(source), np.array(target), rng.uniform(1, 9, num_edges),
                    resources={"a": rng.uniform(0, 5, num_edges), "b": rng.uniform(0, 5, num_edges)})
    fleet = cp.PricingFleet(graph.compile(), num_constraints=0, threads=threads,
                            scheduler=scheduler)
    fleet.add_vessels(source=np.zeros(num_vessels, dtype=np.int64), target=num_vertices - 1)
    return fleet


def test_a_time_limit_of_zero_interrupts_every_vessel():
    case = make_case(2)
    for scheduler in SCHEDULERS:
        fleet = case.fleet(scheduler=scheduler)
        result = fleet.price(case.duals, negative_only=False, time_limit=0)
        assert result.timed_out and not result.cancelled
        assert result.num_interrupted == case.num_vessels == result.interrupted.size
        assert all(status == "interrupted" for status in result.statuses)
        assert np.isnan(result.best_reduced_costs).all()
        assert result.num_columns == 0
        # No column, and yet no proof of anything.
        assert not result.all_priced_exactly and not result.converged
        assert math.isnan(result.best_reduced_cost)
        assert "not priced" in repr(result)
        # The fleet is as good as new.
        check_against_reference(case, fleet.price(case.duals, negative_only=False, time_limit=60))
        check_against_reference(case, fleet.solve(negative_only=False))


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_an_expiring_time_limit_keeps_what_was_priced(scheduler):
    fleet = heavy_fleet(scheduler=scheduler)
    complete = fleet.price([], negative_only=False)
    assert complete.all_priced_exactly and complete.num_columns == len(fleet)
    limit = complete.statistics.solve_time / 4
    result = fleet.price([], negative_only=False, time_limit=limit)
    if result.num_interrupted == 0:  # the machine got faster in between
        pytest.skip("the solve finished within a quarter of its previous time")
    assert result.timed_out and not result.all_priced_exactly and not result.converged
    done = result.status_codes != 3
    assert result.num_interrupted == int((~done).sum())
    assert result.num_columns == int(done.sum())
    np.testing.assert_array_equal(result.best_reduced_costs[done],
                                  complete.best_reduced_costs[done])
    assert np.isnan(result.best_reduced_costs[~done]).all()
    for column in result.columns:
        reference = complete.columns_of(column.vessel_index)[0]
        assert column.edges.tolist() == reference.edges.tolist()
    assert snapshot(fleet.price([], negative_only=False)) == snapshot(complete)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_cancel_from_another_thread(scheduler):
    fleet = heavy_fleet(num_vessels=1500, scheduler=scheduler).prepare()
    fleet.cancel()  # nothing is running: no effect on the next solve
    complete = fleet.price([], negative_only=False)
    assert not complete.cancelled and complete.all_priced_exactly

    results = []
    worker = threading.Thread(
        target=lambda: results.append(fleet.price([], negative_only=False))
    )
    worker.start()
    while worker.is_alive():
        fleet.cancel()
    worker.join()
    (result,) = results
    assert result.cancelled and not result.timed_out
    assert 0 < result.num_interrupted <= len(fleet)
    assert not result.all_priced_exactly and not result.converged
    done = result.status_codes != 3
    np.testing.assert_array_equal(result.best_reduced_costs[done],
                                  complete.best_reduced_costs[done])
    # A cancellation does not outlive the solve it interrupted.
    again = fleet.price([], negative_only=False)
    assert not again.cancelled and snapshot(again) == snapshot(complete)


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_a_failing_vessel_is_reported_and_the_fleet_survives(scheduler):
    case = make_case(4)
    fleet = case.fleet(scheduler=scheduler, threads=4)
    expected = snapshot(fleet.price(case.duals, negative_only=False))
    fleet._core.debug_fail_vessel(5)
    with pytest.raises(cp.BatchPricingError, match="1 of 24 vessels failed: vessel 'v5'") as info:
        fleet.price(case.duals, negative_only=False)
    assert list(info.value.errors) == [5] and info.value.errors[5]
    with pytest.raises(cp.BatchPricingError):
        fleet.solve()
    fleet._core.debug_fail_vessel(-1)
    assert snapshot(fleet.price(case.duals, negative_only=False)) == expected


def test_a_heuristic_fleet_never_claims_convergence():
    case = make_case(3)
    fleet = case.fleet(dominance_tolerance=0.5)
    result = fleet.price(np.zeros(7), negative_only=True, tolerance=1e6)
    assert result.num_columns == 0 and result.num_interrupted == 0
    assert not result.all_priced_exactly and not result.converged


def test_concurrent_calls_on_one_fleet_are_serialised():
    case = make_case(10, num_vessels=40)
    fleet = case.fleet(threads=3)
    rng = np.random.default_rng(0)
    duals = [random_duals(rng, 7) for _ in range(6)]
    expected = [snapshot(fleet.price(pi, negative_only=False)) for pi in duals]
    outcomes: list = [None] * len(duals)

    def work(index: int) -> None:
        for _ in range(5):
            outcomes[index] = snapshot(fleet.price(duals[index], negative_only=False))

    workers = [threading.Thread(target=work, args=(index,)) for index in range(len(duals))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert outcomes == expected


# ------------------------------------------------------------------ validation


def test_invalid_fleet_construction():
    case = make_case(0)
    network, matrix = case.network, sp.csr_array(case.matrix)
    graph = cp.Graph(num_vertices=2)
    graph.add_edge(0, 1, 1.0)
    with pytest.raises(TypeError, match="compile"):
        cp.PricingFleet(graph)
    with pytest.raises(TypeError):
        cp.PricingFleet("network")
    with pytest.raises(ValueError, match="not both"):
        cp.PricingFleet(network, matrix, dual_mapping=cp.IndexedDuals(np.zeros(54, dtype=int)))
    with pytest.raises(ValueError, match="edges"):
        cp.PricingFleet(network, dual_mapping=cp.IndexedDuals([0, 1]))
    with pytest.raises(TypeError, match="IndexedDuals"):
        cp.PricingFleet(network, dual_mapping=matrix)
    with pytest.raises(ValueError):
        cp.PricingFleet(network, sp.csr_array((3, 5)))
    with pytest.raises(ValueError, match="num_constraints"):
        cp.PricingFleet(network, matrix, num_constraints=3)
    for options in ({"scheduler": "openmp"}, {"partitioner": "affinity"},
                    {"cost_update": "dense"}, {"threads": 0}, {"threads": "many"},
                    {"dominance_tolerance": -1.0}, {"bound_cache_bytes": -1},
                    {"algorithm": "dijkstra"}):
        with pytest.raises(ValueError):
            cp.PricingFleet(network, matrix, **options)
    with pytest.raises(TypeError):
        cp.PricingFleet(network, matrix, threads=2.5)
    if not cp.tbb_available():
        with pytest.raises(RuntimeError, match="oneTBB"):
            cp.PricingFleet(network, matrix, scheduler="tbb")
        assert cp.tbb_version() is None
        assert cp.PricingFleet(network, matrix).scheduler == "threads"
    else:
        assert cp.tbb_version()
        assert cp.PricingFleet(network, matrix).scheduler == "tbb"
    fleet = cp.PricingFleet(network, matrix, threads=None)
    assert fleet.threads >= 1 and fleet.num_constraints == 7
    assert "PricingFleet(vessels=0" in repr(fleet)


def test_cyclic_networks_are_rejected():
    graph = cp.Graph(num_vertices=2)
    graph.add_edge(0, 1, 1.0)
    graph.add_edge(1, 0, 1.0)
    with pytest.raises(cp.CyclicGraphError):
        cp.PricingFleet(graph.compile(), num_constraints=0)


def test_invalid_vessels_leave_the_fleet_unchanged():
    case = make_case(1, num_vessels=6)
    fleet = case.fleet()
    expected = snapshot(fleet.price(case.duals, negative_only=False))
    num_edges = case.instance.num_edges
    two = {"source": [0, 1], "target": [2, 3]}
    bad = [
        ({"source": [0, 1], "target": [2, 3, 4]}, ValueError),
        ({"source": [0, 99], "target": [2, 3]}, cp.InvalidStateError),
        ({"source": np.array([0, -1]), "target": [2, 3]}, cp.InvalidStateError),
        ({"source": 0}, TypeError),
        ({**two, "ids": ["v0", "new"]}, ValueError),            # v0 exists
        ({**two, "ids": ["same", "same"]}, ValueError),
        ({**two, "ids": ["only"]}, ValueError),
        ({**two, "ids": [["unhashable"], "x"]}, TypeError),
        ({**two, "resource_limits": {"nope": 1.0}}, ValueError),
        ({**two, "resource_limits": {"r0": [1.0, 2.0, 3.0]}}, ValueError),
        ({**two, "resource_limits": {"r0": math.nan}}, ValueError),
        ({**two, "resource_limits": np.ones((2, 5))}, ValueError),
        ({**two, "fixed_cost": [1.0, math.inf]}, ValueError),
        ({**two, "fixed_cost": [1.0, 2.0, 3.0]}, ValueError),
        ({**two, "fixed_coefficients": [{7: 1.0}, None]}, ValueError),
        ({**two, "fixed_coefficients": [{0: math.nan}, None]}, ValueError),
        ({**two, "fixed_coefficients": np.ones((2, 3))}, ValueError),
        ({**two, "cost_offsets": [{num_edges: 1.0}, None]}, ValueError),
        ({**two, "cost_offsets": [{0: math.nan}, None]}, ValueError),
        ({**two, "cost_offsets": [{0: -math.inf}, None]}, ValueError),
        ({**two, "cost_offsets": [None]}, ValueError),
        ({**two, "forbidden_edges": [[0], [num_edges]]}, ValueError),
        ({**two, "forbidden_edges": [[0.5], None]}, TypeError),
        ({**two, "forbidden_edges": [[0]]}, ValueError),
        ({**two, "edge_mask": np.ones((2, 3), dtype=bool)}, ValueError),
        ({**two, "edge_mask": np.ones((2, num_edges))}, ValueError),
        ({**two, "edge_coefficients": [sp.csr_array((7, 3)), None]}, ValueError),
        ({**two, "edge_coefficients": {5: sp.csr_array((7, num_edges))}}, ValueError),
        ({**two, "edge_coefficients": [None]}, ValueError),
    ]
    for arguments, error in bad:
        with pytest.raises(error):
            fleet.add_vessels(**arguments)
    assert len(fleet) == 6 and fleet.vessel_ids == tuple(f"v{v}" for v in range(6))
    assert snapshot(fleet.price(case.duals, negative_only=False)) == expected

    dag = case.fleet(algorithm="dag")
    with pytest.raises(ValueError, match="rcsp"):
        dag.add_vessels(source=0, target=1, resource_limits={"r0": 1.0})


def test_untracked_resources_cannot_be_limited_per_vessel():
    graph = cp.Graph(num_vertices=2)
    graph.add_resource(cp.Resource("fuel", upper=5.0))
    graph.add_resource(cp.Resource("distance"))
    graph.add_edge(0, 1, 1.0, {"fuel": 1.0, "distance": 9.0})
    fleet = cp.PricingFleet(graph.compile(), num_constraints=0)
    with pytest.raises(ValueError, match="finite upper"):
        fleet.add_vessels(source=0, target=1, resource_limits={"distance": 3.0})
    fleet.add_vessels(source=0, target=1, resource_limits={"fuel": 2.0})
    column = fleet.price([], negative_only=False).columns[0]
    assert column.resources == {"fuel": 1.0, "distance": 9.0}


def test_invalid_solve_arguments():
    case = make_case(0)
    fleet = case.fleet()
    for duals in (np.zeros(6), np.zeros((7, 1)), [math.nan] * 7, [math.inf] * 7):
        with pytest.raises(ValueError):
            fleet.price(duals)
        with pytest.raises(ValueError):
            fleet.update_duals(duals)
    with pytest.raises(TypeError):
        fleet.price("duals")
    with pytest.raises(NotImplementedError):
        fleet.price(case.duals, 2)
    for arguments in ({"max_paths_per_vessel": 0}, {"tolerance": -1.0}, {"time_limit": -1.0},
                      {"time_limit": math.nan}):
        with pytest.raises(ValueError):
            fleet.price(case.duals, **arguments)
    with pytest.raises(TypeError):
        fleet.price(case.duals, time_limit="soon")
    check_against_reference(case, fleet.price(case.duals, negative_only=False))


# ---------------------------------------------------------------------- memory


def test_results_are_read_only_views():
    case = make_case(0)
    result = case.fleet().price(case.duals, negative_only=False)
    for name in ("status_codes", "best_reduced_costs", "reduced_costs", "edges",
                 "coefficient_values", "resources"):
        with pytest.raises(ValueError):
            getattr(result, name)[...] = 0
    assert result.resources.shape == (result.num_columns, 2)
    assert result.vessel_seconds.shape == (case.num_vessels,)
    assert result.statistics.vessel_time == pytest.approx(result.vessel_seconds.sum())
    assert result.statistics.algorithm == "rcsp"
    assert result.statistics.labels_created > 0
    assert "FleetColumn(vessel=" in repr(result.columns[0])
    assert not hasattr(result, "__len__")  # an empty result must not read as "converged"


def test_memory_grows_with_overlays_not_with_vessels():
    case = make_case(0)
    num_edges = case.instance.num_edges
    fleet = cp.PricingFleet(case.network, sp.csr_array(case.matrix), threads=2).prepare()
    fleet.update_duals(case.duals)
    empty = fleet.memory
    assert set(empty) == {"shared_graph", "shared_pricing", "vessels", "bound_cache",
                          "workspaces", "results", "total"}
    assert empty["total"] == sum(value for key, value in empty.items() if key != "total")
    assert empty["shared_graph"] > 0 and empty["shared_pricing"] > 0

    fleet.add_vessels(source=np.zeros(1000, dtype=np.int64), target=1)
    fleet.price(case.duals)
    plain = fleet.memory
    assert plain["shared_graph"] == empty["shared_graph"]
    assert plain["shared_pricing"] == empty["shared_pricing"]
    per_vessel = (plain["vessels"] - empty["vessels"]) / 1000
    assert 0 < per_vessel < num_edges * 8  # far below one cost array per vessel

    fleet.add_vessels(source=np.zeros(1000, dtype=np.int64), target=1,
                      forbidden_edges=[[0, 1, 2]] * 1000)
    assert fleet.memory["vessels"] - plain["vessels"] > plain["vessels"] - empty["vessels"]
