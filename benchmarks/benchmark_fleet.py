"""Fleet pricing: one complete column-generation pricing iteration, end to end.

The instance is a maritime pricing problem: ``V`` vessels on a time-expanded
network of ``P`` ports and ``T`` days (125 vessels, 125 ports and 30- and
60-day horizons by default). A vertex is ``(day, port, cargo, phase)`` with
cargo "empty" or one of two products; arcs are port operations (idle, load,
discharge), waiting and sailing between neighbouring ports. Sailing burns
fuel, a cumulative resource. The master has one row per (port, day, product)
plus one convexity row per vessel.

Every vessel has its own start port, bunker capacity, three ports it may not
enter, discounted fees at a few ports, a charter cost and a convexity row.

Two designs solve exactly the same subproblems and are checked against each
other:

* ``fleet``             one shared network, one ``PricingFleet``; vessels are
                        profiles with sparse overlays.
* ``graph_per_vessel``  one network and one ``PricingEngine`` per vessel
                        (each with its own edges, costs, bounds and dual
                        matrix), solved together by ``BatchPricingEngine``.

Within the fleet design, ``cost_update`` compares shared reduced-cost
preprocessing (``shared_sparse``, ``indexed``) with per-vessel preprocessing
(``per_vessel``), and ``scheduler`` compares oneTBB with plain threads.

What is timed is the whole iteration as Python sees it: the dual update,
scheduling, the shortest-path or RCSP solves, route reconstruction, and the
creation of the Python-facing result (the ``price`` column), plus turning the
result into column objects and a sparse coefficient matrix (``columns``).
Every configuration runs in a fresh process, so its peak resident memory is
its own.

Run with:  python benchmarks/benchmark_fleet.py [--quick] [--seed N] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

import celerisroute as cr
from _common import format_seconds, save

NUM_PRODUCTS = 2
NEIGHBOURS = 5          # nearest ports each port is linked to (links are symmetric)
DAYS_PER_UNIT = 25.0    # sailing days per unit of distance on the unit square
MAX_SAILING_DAYS = 6
LOAD_SHARE = 0.4        # share of ports that load; the others discharge

SAILING_COST = 10.0     # per day; 25 % more when laden
WAITING_COST = 1.0
PORT_FEE = 5.0
FORBIDDEN_PORTS = 3     # ports a vessel may not enter
PREFERRED_PORTS = 5     # ports where a vessel pays a lower fee
FEE_DISCOUNT = 2.0
FUEL_SHARE = (0.25, 0.40)  # bunker capacity in sailing days, as a share of the horizon

MAX_VESSELS = 1000      # vessel data is generated for this many, so fleets are nested


# ------------------------------------------------------------------- instance


@dataclass
class Instance:
    num_ports: int
    num_days: int
    num_vessels: int
    num_vertices: int
    sink: int
    edge_source: np.ndarray
    edge_target: np.ndarray
    cost: np.ndarray
    fuel: np.ndarray
    entered_port: np.ndarray  # per edge: port a sailing arc arrives at, else -1
    op_port: np.ndarray       # per edge: port of a load/discharge arc, else -1
    op_day: np.ndarray
    op_product: np.ndarray
    is_load_port: np.ndarray
    # Vessels.
    start: np.ndarray
    fuel_limit: np.ndarray
    forbidden_ports: np.ndarray  # vessels x FORBIDDEN_PORTS
    preferred_ports: np.ndarray  # vessels x PREFERRED_PORTS
    charter: np.ndarray

    @property
    def num_edges(self) -> int:
        return int(self.cost.shape[0])

    @property
    def num_constraints(self) -> int:
        return self.num_ports * self.num_days * NUM_PRODUCTS + self.num_vessels

    def mapping(self) -> cr.PortTimeProductDuals:
        return cr.PortTimeProductDuals(
            self.op_port, self.op_day, self.op_product, 1.0,
            shape=(self.num_ports, self.num_days, NUM_PRODUCTS),
            extra_constraints=self.num_vessels,
        )

    def convexity_row(self, vessel: int) -> int:
        return self.num_ports * self.num_days * NUM_PRODUCTS + vessel

    def forbidden_edges(self, vessel: int) -> np.ndarray:
        return np.flatnonzero(np.isin(self.entered_port, self.forbidden_ports[vessel]))

    def discounted_edges(self, vessel: int) -> np.ndarray:
        return np.flatnonzero(np.isin(self.op_port, self.preferred_ports[vessel]))

    def duals(self, seed: int, iteration: int) -> np.ndarray:
        """Dual values of one column-generation iteration (minimisation master).

        Demand rows at discharge ports reward deliveries, supply rows at load
        ports charge for cargo, and every vessel has a convexity dual. A new
        random subset of the rows is active in every iteration.
        """
        rng = np.random.default_rng([seed, self.num_ports, self.num_days, 1000 + iteration])
        shape = (self.num_ports, self.num_days, NUM_PRODUCTS)
        reward = np.where(rng.random(shape) < 0.15, rng.uniform(20.0, 120.0, shape), 0.0)
        charge = np.where(rng.random(shape) < 0.10, -rng.uniform(0.0, 20.0, shape), 0.0)
        block = np.where(self.is_load_port[:, None, None], charge, reward)
        convexity = rng.uniform(0.0, 40.0, MAX_VESSELS)[: self.num_vessels]
        return np.concatenate([block.ravel(), convexity])


def build_instance(num_ports: int, num_days: int, num_vessels: int, seed: int) -> Instance:
    if num_vessels > MAX_VESSELS:
        raise ValueError(f"at most {MAX_VESSELS} vessels")
    rng = np.random.default_rng([seed, num_ports, num_days])
    ports, days, loads = num_ports, num_days, NUM_PRODUCTS + 1

    def vertex(day, port, load, phase):
        return ((day * ports + port) * loads + load) * 2 + phase

    sink = days * ports * loads * 2

    position = rng.random((ports, 2))
    distance = np.hypot(*(position[:, None, :] - position[None, :, :]).transpose(2, 0, 1))
    nearest = np.argsort(distance, axis=1)[:, 1 : NEIGHBOURS + 1]
    linked = np.zeros((ports, ports), dtype=bool)
    linked[np.arange(ports)[:, None], nearest] = True
    linked |= linked.T
    origin, destination = np.nonzero(linked)
    sailing = np.clip(np.ceil(distance[origin, destination] * DAYS_PER_UNIT), 1, MAX_SAILING_DAYS)
    sailing = sailing.astype(np.int64)

    is_load_port = rng.random(ports) < LOAD_SHARE
    port_product = rng.integers(0, NUM_PRODUCTS, size=ports)

    families = []  # (source, target, cost, fuel, entered port, op port, op day, op product)

    def add(source, target, cost, fuel=0.0, entered=-1, port=-1, day=-1, product=-1):
        count = np.size(source)
        full = lambda value, dtype: np.broadcast_to(np.asarray(value, dtype=dtype), (count,))  # noqa: E731
        families.append((
            np.asarray(source, dtype=np.int64).ravel(), np.asarray(target, dtype=np.int64).ravel(),
            full(cost, np.float64), full(fuel, np.float64), full(entered, np.int64),
            full(port, np.int64), full(day, np.int64), full(product, np.int64),
        ))

    day, port, load = (a.ravel() for a in np.meshgrid(
        np.arange(days), np.arange(ports), np.arange(loads), indexing="ij"))
    # Port operations: pre -> post on the same day.
    add(vertex(day, port, load, 0), vertex(day, port, load, 1), 0.0)
    day2, port2 = (a.ravel() for a in np.meshgrid(np.arange(days), np.arange(ports), indexing="ij"))
    cargo = port_product[port2] + 1
    loading = is_load_port[port2]
    add(vertex(day2[loading], port2[loading], 0, 0),
        vertex(day2[loading], port2[loading], cargo[loading], 1), PORT_FEE,
        port=port2[loading], day=day2[loading], product=port_product[port2[loading]])
    add(vertex(day2[~loading], port2[~loading], cargo[~loading], 0),
        vertex(day2[~loading], port2[~loading], 0, 1), PORT_FEE,
        port=port2[~loading], day=day2[~loading], product=port_product[port2[~loading]])
    # Waiting: post -> pre of the next day.
    stay = day + 1 < days
    add(vertex(day[stay], port[stay], load[stay], 1),
        vertex(day[stay] + 1, port[stay], load[stay], 0), WAITING_COST)
    # Sailing: post -> pre at a neighbouring port, some days later.
    link, depart, hold = (a.ravel() for a in np.meshgrid(
        np.arange(origin.shape[0]), np.arange(days), np.arange(loads), indexing="ij"))
    arrive = depart + sailing[link]
    in_time = arrive < days
    link, depart, hold, arrive = link[in_time], depart[in_time], hold[in_time], arrive[in_time]
    add(vertex(depart, origin[link], hold, 1), vertex(arrive, destination[link], hold, 0),
        sailing[link] * SAILING_COST * np.where(hold > 0, 1.25, 1.0),
        fuel=sailing[link], entered=destination[link])
    # A route ends empty, in any port, on the last evening.
    add(vertex(days - 1, np.arange(ports), 0, 1), np.full(ports, sink), 0.0)

    columns = [np.concatenate([family[k] for family in families]) for k in range(8)]

    vessel_rng = np.random.default_rng([seed, num_ports, num_days, 7])
    home = vessel_rng.integers(0, ports, size=MAX_VESSELS)
    share = vessel_rng.uniform(*FUEL_SHARE, size=MAX_VESSELS)
    special = np.array([
        vessel_rng.permutation(ports)[: FORBIDDEN_PORTS + PREFERRED_PORTS + 1]
        for _ in range(MAX_VESSELS)
    ])
    for v in range(MAX_VESSELS):  # a vessel may always enter its home port
        row = special[v][special[v] != home[v]]
        special[v, : FORBIDDEN_PORTS + PREFERRED_PORTS] = row[: FORBIDDEN_PORTS + PREFERRED_PORTS]
    charter = vessel_rng.uniform(10.0, 60.0, size=MAX_VESSELS)
    n = num_vessels
    return Instance(
        num_ports, num_days, n, sink + 1, sink, *columns[:5], *columns[5:], is_load_port,
        start=vertex(0, home[:n], 0, 0).astype(np.int64),
        fuel_limit=np.floor(share[:n] * days),
        forbidden_ports=special[:n, :FORBIDDEN_PORTS],
        preferred_ports=special[:n, FORBIDDEN_PORTS : FORBIDDEN_PORTS + PREFERRED_PORTS],
        charter=charter[:n],
    )


# -------------------------------------------------------------------- designs


def shared_network(instance: Instance, algorithm: str) -> cr.Network:
    graph = cr.Graph(num_vertices=instance.num_vertices)
    # The loosest bunker capacity bounds the shared graph; vessels tighten it.
    upper = float(instance.fuel_limit.max()) if algorithm == "rcsp" else float("inf")
    graph.add_resource(cr.Resource("fuel", upper=upper))
    graph.add_edges(instance.edge_source, instance.edge_target, instance.cost,
                    resources={"fuel": instance.fuel})
    return graph.compile()


def build_fleet(instance: Instance, config: dict) -> cr.PricingFleet:
    network = shared_network(instance, config["algorithm"])
    fleet = cr.PricingFleet(
        network, dual_mapping=instance.mapping(), algorithm=config["algorithm"],
        threads=config["threads"], scheduler=config["scheduler"],
        cost_update=config["cost_update"], partitioner=config.get("partitioner", "auto"),
    )
    n = instance.num_vessels
    discounted = [instance.discounted_edges(v) for v in range(n)]
    offsets = sp.csr_array(
        (
            np.full(sum(map(len, discounted)), -FEE_DISCOUNT),
            np.concatenate(discounted),
            np.concatenate([[0], np.cumsum([len(edges) for edges in discounted])]),
        ),
        shape=(n, instance.num_edges),
    )
    convexity = sp.csr_array(
        (np.ones(n), (np.arange(n), [instance.convexity_row(v) for v in range(n)])),
        shape=(n, instance.num_constraints),
    )
    fleet.add_vessels(
        source=instance.start, target=instance.sink,
        resource_limits={"fuel": instance.fuel_limit} if config["algorithm"] == "rcsp" else None,
        fixed_cost=instance.charter, fixed_coefficients=convexity, cost_offsets=offsets,
        forbidden_edges=[instance.forbidden_edges(v) for v in range(n)],
    )
    return fleet.prepare()


def build_engines(instance: Instance, config: dict) -> cr.BatchPricingEngine:
    """The same subproblems as one self-contained network and engine per vessel."""
    matrix = instance.mapping().matrix()
    engines = []
    for v in range(instance.num_vessels):
        keep = np.ones(instance.num_edges, dtype=bool)
        keep[instance.forbidden_edges(v)] = False
        cost = instance.cost.copy()
        cost[instance.discounted_edges(v)] -= FEE_DISCOUNT
        graph = cr.Graph(num_vertices=instance.num_vertices)
        upper = float(instance.fuel_limit[v]) if config["algorithm"] == "rcsp" else float("inf")
        graph.add_resource(cr.Resource("fuel", upper=upper))
        graph.add_edges(instance.edge_source[keep], instance.edge_target[keep], cost[keep],
                        resources={"fuel": instance.fuel[keep]})
        engines.append(cr.PricingEngine(
            graph.compile(), int(instance.start[v]), instance.sink,
            algorithm=config["algorithm"], dual_matrix=matrix[:, keep],
            fixed_cost=float(instance.charter[v]),
            fixed_coefficients={instance.convexity_row(v): 1.0},
        ))
    return cr.BatchPricingEngine(engines, threads=config["threads"])


# --------------------------------------------------------------------- worker


def peak_rss() -> int:
    """Peak resident set size of this process in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def run_worker(config: dict) -> dict:
    start = time.perf_counter()
    instance = build_instance(config["ports"], config["days"], config["vessels"], config["seed"])
    row: dict = {
        "num_vertices": instance.num_vertices,
        "num_edges": instance.num_edges,
        "num_constraints": instance.num_constraints,
        "instance_seconds": time.perf_counter() - start,
        "peak_rss_after_instance": peak_rss(),
    }
    design = config["design"]
    if design == "instance_only":
        row["peak_rss"] = peak_rss()
        return row

    start = time.perf_counter()
    solver = build_fleet(instance, config) if design == "fleet" else build_engines(instance, config)
    row["setup_seconds"] = time.perf_counter() - start
    if design == "fleet":
        row["scheduler"] = solver.scheduler
        row["overlay_entries"] = int(sum(
            len(instance.forbidden_edges(v)) + len(instance.discounted_edges(v))
            for v in range(instance.num_vessels)
        ))

    iterations = []
    for iteration in range(config["iterations"] + 1):  # iteration 0 warms the workspaces up
        duals = instance.duals(config["seed"], iteration)
        begin = time.perf_counter()
        if design == "fleet":
            result = solver.price(duals)
            priced = time.perf_counter()
            columns = result.columns
            matrix = result.coefficient_matrix
            done = time.perf_counter()
            stats = result.statistics
            entry = {
                "update": stats.update_time, "solve": stats.solve_time,
                "assemble": stats.assemble_time, "vessel_seconds": stats.vessel_time,
                "labels_created": stats.labels_created,
                "best": result.best_reduced_costs.tolist(),
                "status": result.status_codes.tolist(),
                "original": [
                    result.original_costs[result.column_ptr[v]]
                    if result.column_ptr[v + 1] > result.column_ptr[v] else None
                    for v in range(instance.num_vessels)
                ],
            }
            if not result.all_priced_exactly:
                raise AssertionError("the fleet was not priced exactly")
        else:
            results = solver.solve_batch(duals, threads=config["threads"])
            priced = time.perf_counter()
            columns = [column for result in results for column in result]
            matrix = sp.hstack([column.coefficients for column in columns], format="csc") \
                if columns else sp.csc_array((instance.num_constraints, 0))
            done = time.perf_counter()
            status = {"optimal": 0, "unreachable": 1, "infeasible": 2}
            entry = {
                "update": sum(r.statistics.update_time for r in results),
                "vessel_seconds": sum(r.statistics.total_time for r in results),
                "labels_created": sum(r.statistics.labels_created for r in results),
                "best": [r.best_reduced_cost for r in results],
                "status": [status[str(r.status)] for r in results],
                "original": [r[0].original_cost if len(r) else None for r in results],
            }
        entry.update(price=priced - begin, columns=done - priced, total=done - begin,
                     num_columns=len(columns), nonzeros=int(matrix.nnz))
        iterations.append(entry)
    if design == "fleet":
        row["native_bytes"] = solver.memory  # after solving: workspaces are in use
    row["warmup"] = iterations[0]
    row["iterations"] = iterations[1:]
    row["peak_rss"] = peak_rss()
    return row


# ----------------------------------------------------------------- experiment


def launch(config: dict) -> dict:
    """Run one configuration in a fresh interpreter and return its measurements."""
    completed = subprocess.run(
        [sys.executable, __file__, "--worker", json.dumps(config)],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"worker failed for {config}:\n{completed.stderr}")
    return {"config": config, **json.loads(completed.stdout.splitlines()[-1])}


def median(row: dict, key: str) -> float:
    return float(np.median([entry[key] for entry in row["iterations"]]))


def label(config: dict) -> str:
    if config["design"] != "fleet":
        return "graph per vessel (BatchPricingEngine)"
    return f"fleet  {config['scheduler']:7s} {config['cost_update']}"


def check_equivalent(rows: list[dict]) -> None:
    """All designs, schedulers and thread counts must price identically."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        c = row["config"]
        if c["design"] == "instance_only":
            continue
        key = (c["ports"], c["days"], c["algorithm"], c["seed"])
        groups.setdefault(key, []).append(row)
    for key, members in groups.items():
        widest = max(members, key=lambda row: row["config"]["vessels"])
        for row in members:
            n = row["config"]["vessels"]
            same_design = row["config"]["design"] == widest["config"]["design"]
            for mine, theirs in zip(row["iterations"], widest["iterations"]):
                if mine["status"] != theirs["status"][:n]:
                    raise AssertionError(f"statuses differ for {row['config']}")
                a = np.array(mine["best"])
                b = np.array(theirs["best"][:n])
                # Fleets are bit-identical among themselves; the per-vessel
                # graphs add the same numbers in another edge order.
                equal = np.array_equal(a, b) if same_design else np.allclose(a, b, rtol=1e-9, atol=1e-7)
                if not equal:
                    raise AssertionError(
                        f"reduced costs differ for {row['config']} (max "
                        f"{np.nanmax(np.abs(a - b))}) in group {key}"
                    )


def print_table(title: str, rows: list[dict], extra: str = "threads") -> None:
    print(f"\n{title}")
    print(f"  {'design':40s} {extra:>7s} {'price':>10s} {'+columns':>10s} {'total':>10s} "
          f"{'vessels/s':>10s} {'setup':>10s} {'peak RSS':>10s}")
    for row in rows:
        c = row["config"]
        total = median(row, "total")
        print(f"  {label(c):40s} {c[extra]:7d} {format_seconds(median(row, 'price'))} "
              f"{format_seconds(median(row, 'columns'))} {format_seconds(total)} "
              f"{c['vessels'] / total:10.0f} {format_seconds(row['setup_seconds'])} "
              f"{row['peak_rss'] / 2**20:8.0f} MB")


def print_stages(title: str, rows: list[dict]) -> None:
    """Where one fleet iteration spends its time, by stage (medians)."""
    print(f"\n{title}")
    print(f"  {'design':40s} {'dual update':>11s} {'solve':>10s} {'assemble':>10s} "
          f"{'Python':>10s} {'columns':>10s}")
    for row in rows:
        if row["config"]["design"] != "fleet":
            continue
        overhead = float(np.median([
            entry["price"] - entry["update"] - entry["solve"] - entry["assemble"]
            for entry in row["iterations"]
        ]))
        print(f"  {label(row['config']):40s}  {format_seconds(median(row, 'update'))} "
              f"{format_seconds(median(row, 'solve'))} {format_seconds(median(row, 'assemble'))} "
              f"{format_seconds(overhead)} {format_seconds(median(row, 'columns'))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="small instance, few repeats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="JSON file (default benchmarks/results/fleet.json)")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        print(json.dumps(run_worker(json.loads(arguments.worker))))
        return

    hardware = os.cpu_count() or 1
    if arguments.quick:
        ports, vessels, horizons, iterations = 30, 30, (10, 20), 3
        thread_counts = sorted({1, 2, hardware})
        vessel_counts = (10, 30, 60)
        per_vessel_limit = 60
    else:
        ports, vessels, horizons, iterations = 125, 125, (30, 60), 5
        thread_counts = sorted({1, 2, 4, 8, hardware})
        vessel_counts = (25, 125, 500)
        per_vessel_limit = 125  # graph-per-vessel beyond this needs too much memory
    scheduler = "tbb" if cr.tbb_available() else "threads"
    base = {"ports": ports, "seed": arguments.seed, "iterations": iterations,
            "scheduler": scheduler, "cost_update": "indexed"}
    load = os.getloadavg()[0]
    print(f"celerisroute {cr.__version__}, oneTBB {cr.tbb_version()}, {hardware} hardware threads, "
          f"load average {load:.2f}")
    if load > 2.0:
        print("  warning: the machine is busy; timings will be noisy")

    rows: list[dict] = []

    def run(**overrides) -> dict:
        config = {**base, "vessels": vessels, "threads": hardware, "design": "fleet",
                  "algorithm": "rcsp", **overrides}
        if config["design"] != "fleet":
            config.pop("scheduler"), config.pop("cost_update")
        for row in rows:  # identical configurations are measured once
            if row["config"] == config:
                return row
        row = launch(config)
        rows.append(row)
        return row

    for days in horizons:
        baseline = run(days=days, design="instance_only")
        print(f"\n=== {vessels} vessels, {ports} ports, {days} days: "
              f"{baseline['num_vertices']:,} vertices, {baseline['num_edges']:,} edges, "
              f"{baseline['num_constraints']:,} master rows; instance alone peaks at "
              f"{baseline['peak_rss'] / 2**20:.0f} MB ===")
        for algorithm in ("rcsp", "dag"):
            table = [
                run(days=days, algorithm=algorithm, cost_update="indexed"),
                run(days=days, algorithm=algorithm, cost_update="shared_sparse"),
                run(days=days, algorithm=algorithm, cost_update="per_vessel"),
            ]
            if scheduler == "tbb":
                table.append(run(days=days, algorithm=algorithm, scheduler="threads"))
            table.append(run(days=days, algorithm=algorithm, design="graph_per_vessel"))
            print_table(f"{algorithm}, {hardware} threads (medians of {iterations} iterations)",
                        table)
            print_stages(f"{algorithm}, {hardware} threads, stages of one fleet iteration", table)

        table = []
        for threads in thread_counts:
            table.append(run(days=days, threads=threads))
            table.append(run(days=days, threads=threads, design="graph_per_vessel"))
        print_table(f"rcsp, {vessels} vessels, by thread count", table)

        table = []
        for count in vessel_counts:
            table.append(run(days=days, vessels=count))
            if count <= per_vessel_limit:
                table.append(run(days=days, vessels=count, design="graph_per_vessel"))
        print_table("rcsp, by vessel count", table, extra="vessels")

    check_equivalent(rows)
    for row in rows:  # compared above; the result file keeps a summary per iteration
        for entry in [row.get("warmup"), *row.get("iterations", [])]:
            if entry is None:
                continue
            best = np.array(entry.pop("best"))
            status = np.array(entry.pop("status"))
            del entry["original"]
            entry["num_optimal"] = int((status == 0).sum())
            entry["min_best_reduced_cost"] = float(best.min())
            entry["sum_best_reduced_cost"] = float(best[np.isfinite(best)].sum())
    print("\nAll designs, schedulers, cost updates and thread counts returned the same "
          "statuses and reduced costs.")
    config = {"ports": ports, "vessels": vessels, "horizons": list(horizons),
              "iterations": iterations, "thread_counts": thread_counts,
              "vessel_counts": list(vessel_counts), "seed": arguments.seed,
              "tbb": cr.tbb_version(), "load_average_at_start": load,
              "load_average_at_end": os.getloadavg()[0]}
    path = save("fleet", config, rows, arguments.output)
    print(f"Raw results: {path}")


if __name__ == "__main__":
    main()
