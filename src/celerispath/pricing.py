"""Column-generation pricing engines.

Sign convention
---------------
The master problem is taken to be a *minimisation* LP whose rows are indexed
``k = 0 .. K-1`` and whose path columns have objective coefficient ``c_P``
(the path's cost) and row coefficients ``a_P``. With ``pi`` the dual values
of the rows, the reduced cost of a path is:

    rc(P) = c_P - pi @ a_P

and a column can improve the master exactly when ``rc(P) < 0``. The
coefficients are path-additive, ``a_P = sum over edges e of P of A[:, e]``,
for a sparse ``K x num_edges`` matrix ``A`` (``dual_matrix``), so the reduced
cost is itself a path length under the edge costs:

    reduced_cost[e] = cost[e] - sum_k pi[k] * A[k, e]

Pass duals exactly as an LP solver reports them for a minimisation problem
(``Constr.Pi`` in Gurobi, ``row_dual`` in HiGHS, ``ineqlin.marginals`` /
``eqlin.marginals`` in SciPy): nonnegative for ``>=`` rows, nonpositive for
``<=`` rows, free for ``==`` rows. For a maximisation master, negate costs
and duals or price with ``-rc``.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse

from . import _core
from .errors import BatchPricingError
from .graph import _ALGORITHM_CODE, Graph, Network, Vertex
from .results import Column, PricingResult, PricingStatistics, decode_status

__all__ = ["BatchPricingEngine", "PricingEngine"]

_EMPTY_INDEX = np.empty(0, dtype=np.int64)
_EMPTY_VALUE = np.empty(0, dtype=np.float64)


def _check_max_paths(max_paths: object, name: str = "max_paths") -> int:
    if isinstance(max_paths, bool) or not isinstance(max_paths, Integral):
        raise TypeError(f"{name} must be a positive integer, got {max_paths!r}")
    if max_paths < 1:
        raise ValueError(f"{name} must be at least 1, got {max_paths}")
    return int(max_paths)


def _check_tolerance(tolerance: object) -> float:
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError(f"tolerance must be a real number, got {tolerance!r}")
    tolerance = float(tolerance)
    if not tolerance >= 0.0 or math.isinf(tolerance):
        raise ValueError("tolerance must be a finite number >= 0")
    return tolerance


def _multiple_paths_unsupported() -> NotImplementedError:
    return NotImplementedError(
        "max_paths > 1 is not implemented for algorithm='rcsp': the Boost RCSP backend returns "
        "the Pareto front of nondominated paths, which does not contain the 2nd, 3rd, ... "
        "cheapest feasible paths in general, so an exact K-best answer cannot be derived from "
        "it. Use max_paths=1 (exact), or an engine with algorithm='dag' (exact K best paths) "
        "if the pricing problem has no binding resources."
    )


class PricingEngine:
    """Repeatedly prices paths between a fixed source and target.

    The engine is built once per pricing subproblem. Construction prunes the
    network to the part that lies on some source-target path, lays that part
    out in topological order, and stores the matching columns of
    ``dual_matrix`` alongside it. Each ``solve`` then only rewrites one
    array of reduced edge costs and runs the solver: no graph, matrix or
    workspace is rebuilt, and no per-edge Python code runs.

    Parameters
    ----------
    graph:
        A compiled ``Network`` (from ``Graph.compile``).
    source, target:
        Vertex ids or states.
    algorithm:
        ``"dag"``, ``"rcsp"`` or ``"auto"``; see ``Network.solve``.
    dual_matrix:
        ``A``, of shape ``(num_constraints, graph.num_edges)``: ``A[k, e]`` is
        the coefficient of edge ``e`` in master constraint ``k``. Any SciPy
        sparse format or a dense 2-D array; converted once. ``None`` means no
        edge contributes to any constraint (``num_constraints`` is then taken
        from ``fixed_coefficients``, or 0).
    fixed_cost:
        A route-level cost added to every column's ``original_cost``
        (e.g. a vehicle's fixed charge).
    fixed_coefficients:
        Route-level master coefficients added to every column's
        ``coefficients`` (e.g. a ``1`` in the vehicle's convexity row): a
        dense vector of length ``num_constraints``, a sparse vector, or a
        ``{row: value}`` mapping. Its dual contribution
        ``-duals @ fixed_coefficients`` is included in every reduced cost.
    unconstrained:
        Allow ``algorithm="dag"`` on a network with constraining resources.
    dominance_tolerance:
        See ``Network.solve``. Keep 0 for exact pricing.
    name:
        Optional label, used in error messages of batch solves.

    Notes
    -----
    An engine may be shared between threads; concurrent calls to
    ``solve`` on the *same* engine are serialised. To price several
    subproblems in parallel use ``BatchPricingEngine``.
    """

    __slots__ = (
        "_core", "_network", "_source", "_target", "_algorithm", "_num_constraints",
        "_matrix", "_fixed_cost", "_fixed", "name",
    )

    def __init__(
        self,
        graph: Network,
        source: Vertex,
        target: Vertex,
        algorithm: str = "auto",
        dual_matrix: Any = None,
        *,
        fixed_cost: float = 0.0,
        fixed_coefficients: Any = None,
        unconstrained: bool = False,
        dominance_tolerance: float = 0.0,
        name: str | None = None,
    ) -> None:
        if isinstance(graph, Graph):
            raise TypeError(
                "PricingEngine needs a compiled network: pass graph=my_graph.compile()"
            )
        if not isinstance(graph, Network):
            raise TypeError(f"graph must be a celerispath.Network, got {type(graph).__name__}")
        self._network = graph
        self._algorithm = graph._resolve_algorithm(algorithm, unconstrained)
        graph._require_acyclic()
        self._source = graph.node_id(source)
        self._target = graph.node_id(target)
        self.name = name

        if isinstance(fixed_cost, bool) or not isinstance(fixed_cost, Real):
            raise TypeError(f"fixed_cost must be a real number, got {fixed_cost!r}")
        self._fixed_cost = float(fixed_cost)
        if not math.isfinite(self._fixed_cost):
            raise ValueError("fixed_cost must be finite")
        tolerance = float(dominance_tolerance)
        if not tolerance >= 0.0 or math.isinf(tolerance):
            raise ValueError("dominance_tolerance must be a finite number >= 0")

        num_edges = graph.num_edges
        if dual_matrix is None:
            rows = _fixed_length(fixed_coefficients)
            matrix = scipy.sparse.csc_array((rows, num_edges), dtype=np.float64)
        else:
            matrix = _as_dual_matrix(dual_matrix, num_edges)
        self._matrix = matrix
        self._num_constraints = int(matrix.shape[0])
        fixed_index, fixed_value = _fixed_arrays(fixed_coefficients, self._num_constraints)
        self._fixed = (fixed_index, fixed_value)

        self._core = _core.PricingEngine(
            graph._core,
            self._source,
            self._target,
            _ALGORITHM_CODE[self._algorithm],
            self._num_constraints,
            np.ascontiguousarray(matrix.indptr, dtype=np.int64),
            np.ascontiguousarray(matrix.indices, dtype=np.int64),
            np.ascontiguousarray(matrix.data, dtype=np.float64),
            self._fixed_cost,
            fixed_index,
            fixed_value,
            tolerance,
        )

    # ------------------------------------------------------------ description

    @property
    def network(self) -> Network:
        return self._network

    @property
    def source(self) -> int:
        """Source vertex id."""
        return self._source

    @property
    def target(self) -> int:
        """Target vertex id."""
        return self._target

    @property
    def algorithm(self) -> str:
        """The resolved algorithm, ``"dag"`` or ``"rcsp"``."""
        return self._algorithm

    @property
    def num_constraints(self) -> int:
        """Number of master constraints, i.e. the expected length of ``duals``."""
        return self._num_constraints

    @property
    def dual_matrix(self) -> scipy.sparse.csc_array:
        """The coefficient matrix ``A`` in CSC form. Treat as read-only."""
        return self._matrix

    @property
    def fixed_cost(self) -> float:
        return self._fixed_cost

    @property
    def is_target_reachable(self) -> bool:
        """Whether any path (ignoring resources) leads from source to target."""
        return self._core.target_reachable

    @property
    def num_active_vertices(self) -> int:
        """Vertices on some source-target path: the size of the pruned subgraph."""
        return self._core.active_vertices

    @property
    def num_active_edges(self) -> int:
        """Edges on some source-target path."""
        return self._core.active_edges

    def __repr__(self) -> str:
        label = f"name={self.name!r}, " if self.name is not None else ""
        return (
            f"PricingEngine({label}source={self._source}, target={self._target}, "
            f"algorithm={self._algorithm!r}, num_constraints={self._num_constraints}, "
            f"active_edges={self.num_active_edges}/{self._network.num_edges})"
        )

    # ---------------------------------------------------------------- pricing

    def reduced_costs(self, duals: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Reduced cost of every edge of the network, ``cost - A.T @ duals``.

        A NumPy reference implementation for inspection and testing; the
        native solve computes the same quantity on its pruned subgraph.
        """
        pi = self._check_duals(duals)
        return self._network.edge_cost - self._matrix.T @ pi

    def _check_duals(self, duals: object) -> np.ndarray:
        try:
            pi = np.ascontiguousarray(duals, dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError("duals must be a 1-D array of real numbers") from None
        if pi.ndim != 1 or pi.shape[0] != self._num_constraints:
            raise ValueError(
                f"duals has shape {pi.shape}, expected ({self._num_constraints},): one dual "
                "value per master constraint (row of dual_matrix)"
            )
        return pi

    def _wrap(
        self, raw: tuple[Any, ...], start: float, total_time: float | None = None
    ) -> PricingResult:
        status, best, raw_columns, timings, labels = raw
        network = self._network
        columns = tuple(
            Column(
                network, self._source, self._num_constraints, edges, original, reduced,
                network._resource_dict(resources), index, value,
            )
            for edges, original, reduced, resources, index, value in raw_columns
        )
        if total_time is None:
            total_time = time.perf_counter() - start
        statistics = PricingStatistics(
            self._algorithm, timings[0], timings[1], timings[2], total_time, *labels[1:]
        )
        return PricingResult(columns, decode_status(status), best, statistics)

    def solve(
        self,
        duals: npt.ArrayLike,
        max_paths: int = 1,
        negative_only: bool = True,
        *,
        tolerance: float = 1e-9,
    ) -> PricingResult:
        """Price paths for the given dual values.

        Parameters
        ----------
        duals:
            One value per master constraint.
        max_paths:
            Maximum number of columns to return. With ``algorithm="dag"`` the
            result holds the ``max_paths`` cheapest *distinct* paths by
            reduced cost, exactly (paths differing only in which parallel
            edge they use are distinct). With ``algorithm="rcsp"`` only
            ``max_paths=1`` is supported; larger values raise
            ``NotImplementedError``.
        negative_only:
            Return only columns with ``reduced_cost < -tolerance``. With
            ``False`` the best paths are returned regardless of sign.
        tolerance:
            Threshold for ``negative_only``. Use the optimality tolerance of
            your LP solver (typically 1e-6 to 1e-9 times the cost scale).

        Returns
        -------
        PricingResult
            A sequence of ``Column`` objects in nondecreasing order of
            reduced cost. When it is empty and ``status`` is not
            ``"optimal"``, no feasible path exists at all; when it is empty
            and ``best_reduced_cost >= -tolerance``, the current duals admit
            no improving column: this is the termination certificate of
            column generation (exact for ``dominance_tolerance == 0``).
        """
        pi = self._check_duals(duals)
        max_paths = _check_max_paths(max_paths)
        if max_paths > 1 and self._algorithm == "rcsp":
            raise _multiple_paths_unsupported()
        tolerance = _check_tolerance(tolerance)
        start = time.perf_counter()
        raw = self._core.solve(pi, max_paths, bool(negative_only), tolerance)
        return self._wrap(raw, start)


def _as_dual_matrix(dual_matrix: Any, num_edges: int) -> scipy.sparse.csc_array:
    """A private, canonical CSC copy of a ``(num_constraints, num_edges)`` matrix."""
    try:
        matrix = scipy.sparse.csc_array(dual_matrix, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "dual_matrix must be a SciPy sparse matrix or a 2-D array of shape "
            f"(num_constraints, num_edges): {error}"
        ) from None
    if matrix.ndim != 2 or matrix.shape[1] != num_edges:
        raise ValueError(
            f"dual_matrix has shape {matrix.shape}, expected (num_constraints, "
            f"{num_edges}): one column per edge of the network"
        )
    matrix.sum_duplicates()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all():
        raise ValueError("dual_matrix contains NaN or infinite coefficients")
    return matrix


def _fixed_length(fixed: object) -> int:
    """Number of constraints implied by fixed coefficients when there is no matrix."""
    if fixed is None:
        return 0
    if isinstance(fixed, Mapping):
        return max((int(k) for k in fixed), default=-1) + 1
    if scipy.sparse.issparse(fixed):
        return int(max(fixed.shape))
    return int(np.asarray(fixed).size)


def _fixed_arrays(fixed: object, num_constraints: int) -> tuple[np.ndarray, np.ndarray]:
    if fixed is None:
        return _EMPTY_INDEX, _EMPTY_VALUE
    if isinstance(fixed, Mapping):
        rows = sorted(fixed)
        index = np.array(rows, dtype=np.int64)
        value = np.array([fixed[k] for k in rows], dtype=np.float64)
    else:
        if scipy.sparse.issparse(fixed):
            if 1 not in fixed.shape and fixed.ndim != 1:
                raise ValueError(
                    f"fixed_coefficients must be a vector, got a sparse matrix of shape {fixed.shape}"
                )
            dense = np.asarray(fixed.todense(), dtype=np.float64).ravel()
        else:
            dense = np.asarray(fixed, dtype=np.float64)
            if dense.ndim != 1:
                raise ValueError(
                    f"fixed_coefficients must be a vector, got an array of shape {dense.shape}"
                )
        if dense.shape[0] != num_constraints:
            raise ValueError(
                f"fixed_coefficients has length {dense.shape[0]}, expected {num_constraints} "
                "(one entry per master constraint)"
            )
        index = np.flatnonzero(dense).astype(np.int64)
        value = dense[index]
    if index.size and (index.min() < 0 or index.max() >= num_constraints):
        raise ValueError(
            f"fixed_coefficients refers to a constraint outside [0, {num_constraints})"
        )
    if not np.isfinite(value).all():
        raise ValueError("fixed_coefficients must be finite")
    return np.ascontiguousarray(index), np.ascontiguousarray(value)


class BatchPricingEngine:
    """Solves many pricing engines in one call on native threads.


        batch = BatchPricingEngine(engines=[engine_v1, engine_v2, engine_v3])
        results = batch.solve_batch(duals, max_paths_per_engine=1, threads=8)
        # results[i] is the PricingResult of engines[i]

    The whole batch runs inside the native core with the GIL released; Python
    is re-entered once, to wrap the results. Each engine owns its workspace
    (reduced costs, labels, scratch memory), so engines never share mutable
    state, and ``results[i]`` always belongs to ``engines[i]`` whatever the
    thread schedule: the output is identical to solving the engines one by
    one.

    Parameters
    ----------
    engines:
        The ``PricingEngine`` objects to solve. They may be built on the
        same or on different networks.
    threads:
        Default thread count for ``solve_batch``; ``None`` uses
        ``min(len(engines), os.cpu_count())``.
    """

    __slots__ = ("_engines", "_threads")

    def __init__(self, engines: Sequence[PricingEngine], threads: int | None = None) -> None:
        engines = list(engines)
        for position, engine in enumerate(engines):
            if not isinstance(engine, PricingEngine):
                raise TypeError(
                    f"engines[{position}] is a {type(engine).__name__}, expected a PricingEngine"
                )
        if len({id(engine) for engine in engines}) != len(engines):
            raise ValueError(
                "the same PricingEngine object appears more than once in engines; build one "
                "engine per subproblem"
            )
        self._engines: tuple[PricingEngine, ...] = tuple(engines)
        self._threads = _check_threads(threads)

    @property
    def engines(self) -> tuple[PricingEngine, ...]:
        return self._engines

    def __len__(self) -> int:
        return len(self._engines)

    def _split_duals(self, duals: object) -> list[np.ndarray]:
        """One validated dual vector per engine, or a single shared one."""
        count = len(self._engines)
        try:
            array = np.asarray(duals, dtype=np.float64)
        except (TypeError, ValueError):
            array = None  # ragged: engines with different numbers of constraints
        if array is not None and array.ndim == 1:
            # One vector shared by every engine.
            for position, engine in enumerate(self._engines):
                if engine.num_constraints != array.shape[0]:
                    raise ValueError(
                        f"engines[{position}]: the shared duals vector has length "
                        f"{array.shape[0]}, but this engine has {engine.num_constraints} "
                        "constraints; pass one dual vector per engine instead"
                    )
            return [np.ascontiguousarray(array)]
        rows: Sequence[Any]
        if array is not None and array.ndim == 2:
            rows = array
        elif array is None and isinstance(duals, Sequence):
            rows = duals
        else:
            raise ValueError(
                "duals must be one 1-D vector shared by all engines, or one vector per engine "
                "(a 2-D array or a list of 1-D arrays)"
            )
        if len(rows) != count:
            raise ValueError(
                f"got {len(rows)} dual vectors for {count} engines; pass one per engine, or a "
                "single 1-D vector to share"
            )
        vectors = []
        for position, (engine, row) in enumerate(zip(self._engines, rows)):
            try:
                vectors.append(engine._check_duals(row))
            except (TypeError, ValueError) as error:
                raise type(error)(f"engines[{position}]: {error}") from None
        return vectors

    def solve_batch(
        self,
        duals: npt.ArrayLike | Sequence[npt.ArrayLike],
        max_paths_per_engine: int = 1,
        threads: int | None = None,
        negative_only: bool = True,
        *,
        tolerance: float = 1e-9,
    ) -> list[PricingResult]:
        """Solve every engine once.

        Parameters
        ----------
        duals:
            A 1-D vector shared by all engines (they must then agree on the
            number of constraints), or one vector per engine: a 2-D array
            with one row per engine or a list of 1-D arrays.
        max_paths_per_engine, negative_only, tolerance:
            As ``max_paths``, ``negative_only`` and ``tolerance`` of
            ``PricingEngine.solve``, applied to every engine.
        threads:
            Number of native threads. ``None`` (default) uses the value given
            at construction, or else ``min(len(engines), os.cpu_count())``.
            Never more threads than engines are started, and each engine is
            solved single-threaded, so the machine is not oversubscribed
            unless you ask for more threads than it has cores.

        Returns
        -------
        list[PricingResult]
            ``results[i]`` belongs to ``engines[i]``.

        Raises
        ------
        BatchPricingError
            If any engine fails. The message names the failing engines; the
            other engines still ran to completion.
        """
        max_paths = _check_max_paths(max_paths_per_engine, "max_paths_per_engine")
        tolerance = _check_tolerance(tolerance)
        threads = self._threads if threads is None else _check_threads(threads)
        if not self._engines:
            return []
        if max_paths > 1:
            for engine in self._engines:
                if engine.algorithm == "rcsp":
                    raise _multiple_paths_unsupported()
        vectors = self._split_duals(duals)
        if threads is None:
            threads = min(len(self._engines), os.cpu_count() or 1)

        start = time.perf_counter()
        raw_results, errors = _core.solve_batch(
            [engine._core for engine in self._engines],
            vectors,
            max_paths,
            bool(negative_only),
            tolerance,
            threads,
        )
        failed = {i: message for i, message in enumerate(errors) if message}
        if failed:
            details = "; ".join(
                f"engines[{i}]"
                + (f" ({self._engines[i].name!r})" if self._engines[i].name is not None else "")
                + f": {message}"
                for i, message in failed.items()
            )
            raise BatchPricingError(
                f"{len(failed)} of {len(self._engines)} pricing engines failed: {details}", failed
            )
        elapsed = time.perf_counter() - start
        return [
            engine._wrap(raw, start, elapsed) for engine, raw in zip(self._engines, raw_results)
        ]


def _check_threads(threads: object) -> int | None:
    if threads is None:
        return None
    if isinstance(threads, bool) or not isinstance(threads, Integral):
        raise TypeError(f"threads must be a positive integer or None, got {threads!r}")
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    return int(threads)
