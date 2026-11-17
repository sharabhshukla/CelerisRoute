"""Fleet pricing: many pricing subproblems ("vessels") on one shared network.

A ``celerisroute.PricingEngine`` owns a pruned copy of the network for
its own source and target. A ``PricingFleet`` is the opposite trade: it
keeps one network, one coefficient mapping and one array of reduced edge
costs in native memory, shares them between all vessels, and describes each
vessel by a small profile (source, target, resource limits, and sparse
overlays of forbidden edges, cost offsets and vessel-specific coefficients).

One column-generation iteration is then two native steps, or one fused call:

* the duals are folded into the shared reduced costs **once**, whatever the
  number of vessels, and
* every vessel is priced on a pool of native worker threads (a oneTBB
  work-stealing arena when the library was built with oneTBB), with the GIL
  released for the whole batch and no Python code in the loop.

With ``A`` the shared coefficient matrix, ``B_v`` the optional coefficients
of vessel ``v`` and ``offset_v`` its optional cost offsets, the reduced cost
of edge ``e`` for vessel ``v`` is (sign convention of
``celerisroute.pricing``):

    reduced_v[e] = cost[e] + offset_v[e] - sum_k pi[k] * (A[k, e] + B_v[k, e])

and a column additionally carries the vessel's ``fixed_cost`` and
``fixed_coefficients``.
"""

from __future__ import annotations

import math
import operator
import os
import threading
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse

from . import _core
from .errors import BatchPricingError, InvalidStateError
from .graph import _ALGORITHM_CODE, Graph, Network
from .pricing import (
    _as_dual_matrix,
    _check_max_paths,
    _check_threads,
    _check_tolerance,
    _multiple_paths_unsupported,
)
from .results import Column, Status, _freeze, decode_status

__all__ = [
    "FleetColumn",
    "FleetResult",
    "FleetStatistics",
    "IndexedDuals",
    "PortTimeProductDuals",
    "PricingFleet",
    "tbb_available",
    "tbb_version",
]

_SCHEDULER_CODE = {"threads": 0, "tbb": 1}
_PARTITIONER_CODE = {"auto": 0, "simple": 1, "static": 2}
_COST_UPDATES = ("shared_sparse", "indexed", "per_vessel")
_INTERRUPTED = 3  # native code of Status.INTERRUPTED

_EMPTY_INDEX = np.empty(0, dtype=np.int64)
_EMPTY_VALUE = np.empty(0, dtype=np.float64)


def tbb_available() -> bool:
    """Whether this build of celerisroute can schedule fleets with oneTBB."""
    return bool(_core.tbb_available())


def tbb_version() -> str | None:
    """Version of the oneTBB runtime in use, or ``None`` without oneTBB."""
    return _core.tbb_version() or None


# --------------------------------------------------------------------- duals


def _index_columns(values: object, what: str) -> np.ndarray:
    """``(num_edges, width)`` int64 array from per-edge integer input."""
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{what} must be an integer array, got dtype {array.dtype}")
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(
            f"{what} must have shape (num_edges,) or (num_edges, width), got {array.shape}"
        )
    return np.ascontiguousarray(array, dtype=np.int64)


class IndexedDuals:
    """Shared dual coefficients given as a fixed number of rows per edge.

    The general way to say how edges enter the master constraints is a sparse
    matrix (``dual_matrix``). Often the structure is simpler: every edge
    touches at most ``width`` constraints, known by index. Then no sparse
    structure is needed at all, and the reduced-cost refresh becomes a
    straight gather over two flat arrays.

    Parameters
    ----------
    row:
        Integer array of shape ``(num_edges,)`` or ``(num_edges, width)``:
        the master constraint(s) each edge contributes to. A negative entry
        means "no contribution".
    coefficient:
        The matching coefficients: a scalar, one value per edge, or an array
        shaped like ``row``.
    num_constraints:
        Number of master constraints (length of the dual vector). Defaults
        to ``row.max() + 1``.
    """

    __slots__ = ("row", "coefficient", "num_constraints")

    def __init__(
        self,
        row: npt.ArrayLike,
        coefficient: npt.ArrayLike = 1.0,
        *,
        num_constraints: int | None = None,
    ) -> None:
        rows = _index_columns(row, "row")
        rows = np.where(rows < 0, -1, rows)
        try:
            values = np.asarray(coefficient, dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError("coefficient must be numeric") from None
        if values.ndim == 1:
            values = values[:, None]
        try:
            values = np.array(np.broadcast_to(values, rows.shape), dtype=np.float64, order="C")
        except ValueError:
            raise ValueError(
                f"coefficient of shape {np.shape(coefficient)} does not match row, which "
                f"describes {rows.shape[0]} edges with {rows.shape[1]} entries each"
            ) from None
        if not np.isfinite(values[rows >= 0]).all():
            raise ValueError("coefficient contains NaN or infinite values")
        largest = int(rows.max()) if rows.size else -1
        if num_constraints is None:
            num_constraints = largest + 1
        else:
            if isinstance(num_constraints, bool) or not isinstance(num_constraints, Integral):
                raise TypeError(f"num_constraints must be an integer, got {num_constraints!r}")
            num_constraints = int(num_constraints)
            if num_constraints <= largest:
                raise ValueError(
                    f"row refers to constraint {largest}, but num_constraints is {num_constraints}"
                )
        self.row = _freeze(np.ascontiguousarray(rows))
        self.coefficient = _freeze(values)
        self.num_constraints = num_constraints

    @property
    def num_edges(self) -> int:
        return self.row.shape[0]

    @property
    def width(self) -> int:
        """Coefficient slots per edge."""
        return self.row.shape[1]

    def matrix(self) -> scipy.sparse.csc_array:
        """The same coefficients as a sparse ``(num_constraints, num_edges)`` matrix."""
        used = self.row >= 0
        edges = np.nonzero(used)[0]
        matrix = scipy.sparse.coo_array(
            (self.coefficient[used], (self.row[used], edges)),
            shape=(self.num_constraints, self.num_edges),
        ).tocsc()
        matrix.sum_duplicates()
        matrix.sort_indices()
        return matrix

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_edges={self.num_edges}, width={self.width}, "
            f"num_constraints={self.num_constraints})"
        )


class PortTimeProductDuals(IndexedDuals):
    """Dual mapping for constraints indexed by (port, day, product).

    Models the common maritime master problem whose rows are inventory or
    demand balances, one per port, day and product: an edge that loads or
    discharges ``quantity`` units of ``product`` at ``port`` on ``day`` has
    that coefficient in row ``(port, day, product)``.

    Rows are numbered in C order, ``row = (port * num_days + day) * num_products
    + product``, so a dual array of shape ``(num_ports, num_days, num_products)``
    can be passed to ``PricingFleet.update_duals`` as it is.

    Parameters
    ----------
    port, day, product:
        Zero-based integer indices per edge, each of shape ``(num_edges,)``
        or ``(num_edges, width)`` when an edge touches several rows. An edge
        with any negative index contributes nothing.
    quantity:
        Coefficient per edge (scalar, per edge, or shaped like the indices).
    shape:
        ``(num_ports, num_days, num_products)``. Defaults to the largest
        index of each kind plus one.
    extra_constraints:
        Number of further master rows that follow the port-day-product block
        (convexity rows, for instance). Edges do not contribute to them
        through this mapping; vessels can, via ``fixed_coefficients`` and
        ``edge_coefficients``.
    """

    __slots__ = ("shape", "extra_constraints")

    def __init__(
        self,
        port: npt.ArrayLike,
        day: npt.ArrayLike,
        product: npt.ArrayLike,
        quantity: npt.ArrayLike = 1.0,
        *,
        shape: tuple[int, int, int] | None = None,
        extra_constraints: int = 0,
    ) -> None:
        indices = [
            _index_columns(values, name)
            for name, values in (("port", port), ("day", day), ("product", product))
        ]
        if not indices[0].shape == indices[1].shape == indices[2].shape:
            raise ValueError(
                "port, day and product must have the same shape, got "
                f"{[index.shape for index in indices]}"
            )
        unused = (indices[0] < 0) | (indices[1] < 0) | (indices[2] < 0)
        largest = [int(np.where(unused, -1, index).max(initial=-1)) for index in indices]
        if shape is None:
            shape = (largest[0] + 1, largest[1] + 1, largest[2] + 1)
        else:
            shape = tuple(operator.index(extent) for extent in shape)  # type: ignore[assignment]
            if len(shape) != 3 or min(shape) < 0:
                raise ValueError("shape must be (num_ports, num_days, num_products)")
            for name, extent, seen in zip(("port", "day", "product"), shape, largest):
                if seen >= extent:
                    raise ValueError(f"{name} index {seen} is outside the shape {shape}")
        if isinstance(extra_constraints, bool) or not isinstance(extra_constraints, Integral):
            raise TypeError("extra_constraints must be an integer")
        if extra_constraints < 0:
            raise ValueError("extra_constraints must be >= 0")
        row = (indices[0] * shape[1] + indices[1]) * shape[2] + indices[2]
        row[unused] = -1
        super().__init__(
            row,
            quantity,
            num_constraints=shape[0] * shape[1] * shape[2] + int(extra_constraints),
        )
        self.shape = shape
        self.extra_constraints = int(extra_constraints)

    def row_of(self, port: int, day: int, product: int) -> int:
        """Master row of one (port, day, product) triple."""
        for name, index, extent in zip(("port", "day", "product"), (port, day, product), self.shape):
            if not 0 <= index < extent:
                raise IndexError(f"{name} index {index} is outside [0, {extent})")
        return (port * self.shape[1] + day) * self.shape[2] + product

    def flatten(
        self, duals: npt.ArrayLike, extra: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Dual vector of length ``num_constraints`` from a ``shape``-d array.

        ``extra`` supplies the duals of the ``extra_constraints`` trailing
        rows; it may be omitted only when there are none.
        """
        block = np.asarray(duals, dtype=np.float64)
        size = self.shape[0] * self.shape[1] * self.shape[2]
        if block.shape != self.shape and block.shape != (size,):
            raise ValueError(
                f"duals has shape {block.shape}, expected {self.shape} "
                "(num_ports, num_days, num_products)"
            )
        tail = _EMPTY_VALUE if extra is None else np.asarray(extra, dtype=np.float64).ravel()
        if tail.shape[0] != self.extra_constraints:
            raise ValueError(
                f"expected {self.extra_constraints} dual values for the extra constraints, "
                f"got {tail.shape[0]}"
            )
        return np.concatenate([block.ravel(), tail])


def _matrix_to_indexed(matrix: scipy.sparse.csc_array) -> tuple[np.ndarray, np.ndarray]:
    """Pads the columns of a CSC matrix to equal length: ``(row, value)`` per edge."""
    num_edges = matrix.shape[1]
    indptr = np.asarray(matrix.indptr, dtype=np.int64)
    counts = np.diff(indptr)
    width = max(1, int(counts.max(initial=0)))
    row = np.full((num_edges, width), -1, dtype=np.int64)
    value = np.zeros((num_edges, width), dtype=np.float64)
    edge = np.repeat(np.arange(num_edges), counts)
    position = np.arange(matrix.nnz) - np.repeat(indptr[:-1], counts)
    row[edge, position] = matrix.indices
    value[edge, position] = matrix.data
    return row, value


# ------------------------------------------------------------ vessel helpers


def _length(values: object) -> int | None:
    """Number of vessels a per-vessel argument describes; None for a scalar."""
    if isinstance(values, Integral) and not isinstance(values, bool):
        return None
    try:
        return len(values)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(
            "expected one vertex per vessel (an integer array of vertex ids or a list of "
            f"states), or a single vertex id shared by all vessels; got {values!r}"
        ) from None


def _batch_size(values: object) -> int | None:
    """Leading length of a per-vessel argument; None if it has none."""
    if scipy.sparse.issparse(values):
        return int(values.shape[0])  # type: ignore[attr-defined]
    if values is None or isinstance(values, (str, bytes, Mapping)):
        return None
    try:
        return len(values)  # type: ignore[arg-type]
    except TypeError:
        return None


def _vertex_ids(network: Network, values: object, count: int, what: str) -> np.ndarray:
    if isinstance(values, Integral) and not isinstance(values, bool):
        return np.full(count, network.node_id(int(values)), dtype=np.int64)
    if isinstance(values, np.ndarray) and values.dtype.kind in "iu":
        if values.ndim != 1:
            raise ValueError(f"{what} must be 1-D, got shape {values.shape}")
        ids = np.ascontiguousarray(values, dtype=np.int64)
        limit = network.num_vertices
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= limit):
            position = int(np.flatnonzero((ids < 0) | (ids >= limit))[0])
            raise InvalidStateError(
                f"{what}[{position}] = {int(ids[position])} is outside [0, {limit}): not a "
                "vertex id of this network"
            )
    else:
        if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Iterable):
            raise TypeError(f"{what} must list one vertex id or state per vessel")
        ids = np.array([network.node_id(vertex) for vertex in values], dtype=np.int64)
    if ids.shape[0] != count:
        raise ValueError(f"{what} describes {ids.shape[0]} vessels, expected {count}")
    return ids


def _per_vessel_values(values: object, count: int, what: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(f"{what} must be a real number or one per vessel") from None
    if array.ndim == 0:
        return np.full(count, float(array), dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"{what} has shape {array.shape}, expected ({count},): one per vessel")
    return np.ascontiguousarray(array)


def _vessel_triplets(
    values: object, count: int, width: int, what: str, column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(vessel, column, value)`` entries of a sparse per-vessel quantity.

    Accepts a ``(count, width)`` matrix (sparse or dense), or a sequence with
    one entry per vessel: ``None``, a ``{column: value}`` mapping, or a vector
    of length ``width``.
    """
    if values is None:
        return _EMPTY_INDEX, _EMPTY_INDEX, _EMPTY_VALUE
    if scipy.sparse.issparse(values) or isinstance(values, np.ndarray):
        try:
            matrix = scipy.sparse.csr_array(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{what} is not a numeric matrix: {error}") from None
        if matrix.ndim != 2 or matrix.shape != (count, width):
            raise ValueError(
                f"{what} has shape {matrix.shape}, expected ({count}, {width}): one row per "
                f"vessel and one column per {column}"
            )
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        vessel = np.repeat(np.arange(count, dtype=np.int64), np.diff(matrix.indptr))
        return vessel, matrix.indices.astype(np.int64), matrix.data.astype(np.float64)
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise TypeError(
            f"{what} must be a ({count}, {width}) matrix or a sequence with one entry per vessel"
        )
    if len(values) != count:
        raise ValueError(f"{what} has {len(values)} entries, expected {count} (one per vessel)")
    vessels: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for position, item in enumerate(values):
        if item is None:
            continue
        if isinstance(item, Mapping):
            index = np.array([operator.index(key) for key in item], dtype=np.int64)
            value = np.array(list(item.values()), dtype=np.float64)
        else:
            if scipy.sparse.issparse(item):
                dense = np.asarray(item.todense(), dtype=np.float64).ravel()
            else:
                dense = np.asarray(item, dtype=np.float64)
            if dense.shape != (width,):
                raise ValueError(
                    f"{what}[{position}] has shape {dense.shape}, expected ({width},): one "
                    f"entry per {column}"
                )
            index = np.flatnonzero(dense).astype(np.int64)
            value = dense[index]
        if index.size and (int(index.min()) < 0 or int(index.max()) >= width):
            raise ValueError(
                f"{what}[{position}] refers to a {column} outside [0, {width})"
            )
        vessels.append(np.full(index.shape[0], position, dtype=np.int64))
        columns.append(index)
        data.append(value)
    if not vessels:
        return _EMPTY_INDEX, _EMPTY_INDEX, _EMPTY_VALUE
    return np.concatenate(vessels), np.concatenate(columns), np.concatenate(data)


def _group_by_vessel(
    vessel: np.ndarray, count: int, *arrays: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Row pointers per vessel, followed by ``arrays`` sorted to match."""
    order = np.argsort(vessel, kind="stable")
    ptr = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(vessel, minlength=count), out=ptr[1:])
    return (ptr, *(np.ascontiguousarray(array[order]) for array in arrays))


def _check_time_limit(time_limit: object) -> float:
    if time_limit is None:
        return math.inf
    if isinstance(time_limit, bool) or not isinstance(time_limit, Real):
        raise TypeError(f"time_limit must be a number of seconds or None, got {time_limit!r}")
    time_limit = float(time_limit)
    if not time_limit >= 0.0:
        raise ValueError("time_limit must be >= 0 seconds")
    return time_limit


# ------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class FleetStatistics:
    """Timings (seconds) and label counters of one fleet pricing call."""

    algorithm: str
    scheduler: str
    threads: int
    """Worker slots that were available to the call."""
    update_time: float
    """Folding the duals into the shared reduced costs (0 for a bare ``solve``)."""
    solve_time: float
    """Wall time of the parallel pricing phase, including column assembly per vessel."""
    assemble_time: float
    """Merging the per-vessel outcomes into the flat result arrays."""
    total_time: float
    """Wall time of the whole call as seen from Python."""
    vessel_time: float
    """Sum of the per-vessel solve times; ``vessel_time / solve_time`` is the
    realised parallel speed-up."""
    labels_created: int = 0
    labels_extended: int = 0
    labels_dominated: int = 0
    labels_infeasible: int = 0
    max_active_labels: int = 0
    pareto_labels: int = 0


class FleetColumn(Column):
    """A ``Column`` that knows which vessel it belongs to.

    ``vessel`` is the id the vessel was registered with and ``vessel_index``
    its position in the fleet.
    """

    __slots__ = ("vessel", "vessel_index")

    def __init__(self, vessel: Hashable, vessel_index: int, *args: Any) -> None:
        super().__init__(*args)
        self.vessel = vessel
        self.vessel_index = vessel_index

    def __repr__(self) -> str:
        return (
            f"FleetColumn(vessel={self.vessel!r}, reduced_cost={self.reduced_cost!r}, "
            f"original_cost={self.original_cost!r}, edges={self.edges.size}, "
            f"nonzeros={self.coefficient_indices.size})"
        )


class FleetResult:
    """Outcome of pricing a whole fleet once.

    Everything is held in a handful of flat, read-only NumPy arrays that the
    native core filled; ``FleetColumn`` objects are only created when
    ``columns`` is first read.

    Per vessel (indexed by position in the fleet):

    vessel_ids:
        The ids the vessels were registered with.
    status_codes, statuses:
        Outcome per vessel as ``int8`` codes and as ``Status`` members.
        ``"optimal"`` means a feasible path exists (even if its reduced cost
        was not negative); ``"interrupted"`` means the vessel was not priced.
    best_reduced_costs:
        Minimum reduced cost over the vessel's feasible paths: ``inf`` when
        there is none, ``nan`` when the vessel was not priced.
    vessel_seconds:
        Native solve time per vessel.
    column_ptr:
        Vessel ``i`` owns columns ``column_ptr[i]:column_ptr[i + 1]``.

    Per column (vessels in registration order, cheapest first within a vessel):

    column_vessel, original_costs, reduced_costs:
        Vessel position, objective coefficient and reduced cost.
    resources:
        ``(num_columns, num_resources)`` final resource values, columns in
        the order of ``network.resources``.
    edge_ptr, edges:
        Column ``c`` uses edges ``edges[edge_ptr[c]:edge_ptr[c + 1]]``.
    coefficient_ptr, coefficient_indices, coefficient_values:
        Master coefficients of all columns in compressed-column form.

    Warnings
    --------
    An empty result is **not** by itself a proof that column generation
    has converged. Use ``converged``, which also requires that every
    vessel was priced to completion with an exact algorithm.
    """

    __slots__ = (
        "vessel_ids", "status_codes", "best_reduced_costs", "vessel_seconds", "column_ptr",
        "column_vessel", "original_costs", "reduced_costs", "resources", "edge_ptr", "edges",
        "coefficient_ptr", "coefficient_indices", "coefficient_values", "cancelled",
        "timed_out", "num_interrupted", "tolerance", "statistics",
        "_network", "_sources", "_num_constraints", "_exact", "_columns", "_statuses",
    )

    def __init__(
        self,
        network: Network,
        num_constraints: int,
        exact: bool,
        vessel_ids: tuple[Hashable, ...],
        sources: np.ndarray,
        raw: Mapping[str, Any],
        tolerance: float,
        statistics: FleetStatistics,
    ) -> None:
        self._network = network
        self._num_constraints = num_constraints
        self._exact = exact
        self._sources = sources
        self._columns: tuple[FleetColumn, ...] | None = None
        self._statuses: tuple[Status, ...] | None = None
        self.vessel_ids = vessel_ids
        self.status_codes = _freeze(raw["status"])
        self.best_reduced_costs = _freeze(raw["best_reduced_cost"])
        self.vessel_seconds = _freeze(raw["vessel_seconds"])
        self.column_ptr = _freeze(raw["column_ptr"])
        self.column_vessel = _freeze(raw["column_vessel"])
        self.original_costs = _freeze(raw["original_cost"])
        self.reduced_costs = _freeze(raw["reduced_cost"])
        self.resources = _freeze(
            raw["resources"].reshape(self.reduced_costs.shape[0], len(network._columns))
        )
        self.edge_ptr = _freeze(raw["edge_ptr"])
        self.edges = _freeze(raw["edges"])
        self.coefficient_ptr = _freeze(raw["coef_ptr"])
        self.coefficient_indices = _freeze(raw["coef_row"])
        self.coefficient_values = _freeze(raw["coef_value"])
        self.cancelled: bool = raw["cancelled"]
        self.timed_out: bool = raw["timed_out"]
        self.num_interrupted: int = raw["num_interrupted"]
        self.tolerance = tolerance
        self.statistics = statistics

    # ------------------------------------------------------------ per vessel

    @property
    def num_vessels(self) -> int:
        return self.status_codes.shape[0]

    @property
    def statuses(self) -> tuple[Status, ...]:
        """``Status`` of every vessel, before the ``negative_only`` filter."""
        if self._statuses is None:
            self._statuses = tuple(decode_status(code) for code in self.status_codes.tolist())
        return self._statuses

    @property
    def interrupted(self) -> npt.NDArray[np.int64]:
        """Positions of the vessels that were not priced."""
        return np.flatnonzero(self.status_codes == _INTERRUPTED)

    @property
    def all_priced_exactly(self) -> bool:
        """Whether every vessel was priced to completion by an exact algorithm.

        ``False`` after a cancellation or an expired time limit, and always
        for a fleet with ``dominance_tolerance > 0`` (a heuristic). Only when
        this is ``True`` do the reduced costs certify anything about vessels
        that returned no column.
        """
        return self._exact and self.num_interrupted == 0

    @property
    def best_reduced_cost(self) -> float:
        """Minimum of ``best_reduced_costs`` over the fleet.

        ``nan`` if any vessel was not priced (the minimum is then unknown),
        ``inf`` if no vessel has a feasible path.
        """
        if self.num_interrupted:
            return math.nan
        return float(self.best_reduced_costs.min(initial=math.inf))

    @property
    def converged(self) -> bool:
        """Whether this result proves that no improving column exists.

        True exactly when ``all_priced_exactly`` holds and no vessel has
        a path with reduced cost below ``-tolerance``: the termination
        certificate of column generation for the duals that were priced.
        """
        return self.all_priced_exactly and not self.best_reduced_cost < -self.tolerance

    # ---------------------------------------------------------------- columns

    @property
    def num_columns(self) -> int:
        return self.reduced_costs.shape[0]

    @property
    def columns(self) -> tuple[FleetColumn, ...]:
        """All returned columns as objects (created on first access)."""
        if self._columns is None:
            network = self._network
            names = network._columns
            edge_ptr = self.edge_ptr.tolist()
            coef_ptr = self.coefficient_ptr.tolist()
            vessels = self.column_vessel.tolist()
            sources = self._sources
            ids = self.vessel_ids
            columns = []
            for c, (vessel, original, reduced, resources) in enumerate(
                zip(vessels, self.original_costs.tolist(), self.reduced_costs.tolist(),
                    self.resources.tolist())
            ):
                columns.append(
                    FleetColumn(
                        ids[vessel], vessel, network, int(sources[vessel]),
                        self._num_constraints, self.edges[edge_ptr[c]:edge_ptr[c + 1]],
                        original, reduced, dict(zip(names, resources)),
                        self.coefficient_indices[coef_ptr[c]:coef_ptr[c + 1]],
                        self.coefficient_values[coef_ptr[c]:coef_ptr[c + 1]],
                    )
                )
            self._columns = tuple(columns)
        return self._columns

    def columns_of(self, vessel_index: int) -> tuple[FleetColumn, ...]:
        """The columns of the vessel at ``vessel_index``, cheapest first."""
        index = operator.index(vessel_index)
        if not 0 <= index < self.num_vessels:
            raise IndexError(f"vessel index {index} is outside [0, {self.num_vessels})")
        return self.columns[int(self.column_ptr[index]):int(self.column_ptr[index + 1])]

    @property
    def coefficient_matrix(self) -> scipy.sparse.csc_array:
        """Master coefficients of all columns: sparse ``(num_constraints, num_columns)``.

        Column ``c`` of the matrix belongs to ``columns[c]``; its objective
        coefficient is ``original_costs[c]``. Ready to be appended to a
        master problem in one call.
        """
        return scipy.sparse.csc_array(
            (
                self.coefficient_values.copy(),
                self.coefficient_indices.copy(),
                self.coefficient_ptr.copy(),
            ),
            shape=(self._num_constraints, self.num_columns),
        )

    def __repr__(self) -> str:
        if self.num_interrupted:
            best = f"unknown, {self.num_interrupted} vessels not priced"
        else:
            best = repr(self.best_reduced_cost)
        return (
            f"FleetResult(vessels={self.num_vessels}, columns={self.num_columns}, "
            f"best_reduced_cost={best}, converged={self.converged})"
        )


# --------------------------------------------------------------------- fleet


class PricingFleet:
    """Prices many vessels on one shared network, in parallel, in one call.


        fleet = cr.PricingFleet(network, dual_matrix=A, algorithm="rcsp", threads="auto")
        fleet.add_vessels(ids=names, source=sources, target=targets,
                          resource_limits={"fuel": fuel_capacity})
        fleet.prepare()

        while True:
            result = fleet.price(master_duals)        # one native call
            if result.converged:
                break
            add_to_master(result.coefficient_matrix, result.original_costs)

    Parameters
    ----------
    graph:
        A compiled ``Network``. It is shared, never copied.
    dual_matrix:
        Shared coefficients ``A`` of shape ``(num_constraints, num_edges)``,
        as for ``celerisroute.PricingEngine``.
    dual_mapping:
        Alternative to ``dual_matrix``: an ``IndexedDuals`` (for example
        ``PortTimeProductDuals``) giving a fixed number of constraint
        rows per edge.
    algorithm:
        ``"dag"``, ``"rcsp"`` or ``"auto"``; see ``Network.solve``. One
        algorithm serves the whole fleet.
    threads:
        Number of worker threads, or ``"auto"`` for one per CPU. The calling
        thread is one of them.
    scheduler:
        ``"tbb"``: a oneTBB task arena whose workers persist between calls
        and steal work from each other, so a few expensive vessels do not
        leave the other threads idle. ``"threads"``: plain native threads
        started per call that claim vessels from a shared counter.
        ``"auto"`` (default) is ``"tbb"`` when available.
    cost_update:
        How duals become reduced costs. ``"shared_sparse"``: one shared
        array, refreshed from the sparse matrix. ``"indexed"``: one shared
        array, refreshed by a flat gather over ``width`` rows per edge.
        ``"per_vessel"``: every vessel recomputes its own full array, the
        design the shared modes are benchmarked against. ``"auto"``
        (default) is ``"indexed"`` with a ``dual_mapping`` and
        ``"shared_sparse"`` otherwise; either input is converted to the
        other form if asked.
    partitioner:
        oneTBB range partitioner: ``"auto"`` (default), ``"simple"`` (one
        task per vessel) or ``"static"`` (even shares, no stealing).
    unconstrained, dominance_tolerance:
        As for ``celerisroute.PricingEngine``. A positive
        ``dominance_tolerance`` makes pricing heuristic, and results then
        never report ``FleetResult.converged``.
    num_constraints:
        Number of master constraints when neither ``dual_matrix`` nor
        ``dual_mapping`` is given.
    bound_cache_bytes:
        Memory budget for the resource completion bounds that are cached per
        distinct target vertex (``rcsp`` only). Vessels beyond the budget
        recompute their bounds on every solve.

    Notes
    -----
    A fleet is safe to use from several threads; calls that change or price
    it are serialised. ``cancel`` may be called from any thread while a
    solve is running.
    """

    def __init__(
        self,
        graph: Network,
        dual_matrix: Any = None,
        *,
        dual_mapping: IndexedDuals | None = None,
        algorithm: str = "auto",
        threads: int | str | None = "auto",
        scheduler: str = "auto",
        cost_update: str | None = "auto",
        partitioner: str = "auto",
        unconstrained: bool = False,
        dominance_tolerance: float = 0.0,
        num_constraints: int | None = None,
        bound_cache_bytes: int = 1 << 30,
    ) -> None:
        if isinstance(graph, Graph):
            raise TypeError("PricingFleet needs a compiled network: pass graph=my_graph.compile()")
        if not isinstance(graph, Network):
            raise TypeError(f"graph must be a celerisroute.Network, got {type(graph).__name__}")
        self._network = graph
        self._algorithm = graph._resolve_algorithm(algorithm, unconstrained)
        graph._require_acyclic()

        if scheduler == "auto":
            scheduler = "tbb" if tbb_available() else "threads"
        if scheduler not in _SCHEDULER_CODE:
            raise ValueError(
                f"unknown scheduler {scheduler!r}; choose 'auto', 'tbb' or 'threads'"
            )
        if scheduler == "tbb" and not tbb_available():
            raise RuntimeError(
                "scheduler='tbb' was requested, but this build of celerisroute has no oneTBB "
                "support. Install oneTBB and rebuild, or use scheduler='threads'"
            )
        if partitioner not in _PARTITIONER_CODE:
            raise ValueError(
                f"unknown partitioner {partitioner!r}; choose one of {list(_PARTITIONER_CODE)}"
            )
        if threads is None or threads == "auto":
            cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
            threads = int(cpus)
        else:
            if isinstance(threads, str):
                raise ValueError(f"threads must be a positive integer or 'auto', got {threads!r}")
            threads = _check_threads(threads)
        tolerance = float(dominance_tolerance)
        if not tolerance >= 0.0 or math.isinf(tolerance):
            raise ValueError("dominance_tolerance must be a finite number >= 0")
        if isinstance(bound_cache_bytes, bool) or not isinstance(bound_cache_bytes, Integral):
            raise TypeError("bound_cache_bytes must be an integer")
        if bound_cache_bytes < 0:
            raise ValueError("bound_cache_bytes must be >= 0")

        num_edges = graph.num_edges
        if dual_matrix is not None and dual_mapping is not None:
            raise ValueError("pass either dual_matrix or dual_mapping, not both")
        if cost_update is None or cost_update == "auto":
            cost_update = "indexed" if dual_mapping is not None else "shared_sparse"
        if cost_update not in _COST_UPDATES:
            raise ValueError(
                f"unknown cost_update {cost_update!r}; choose 'auto' or one of "
                f"{list(_COST_UPDATES)}"
            )
        matrix: scipy.sparse.csc_array | None = None
        if dual_mapping is not None:
            if not isinstance(dual_mapping, IndexedDuals):
                raise TypeError(
                    "dual_mapping must be an IndexedDuals (e.g. PortTimeProductDuals), got "
                    f"{type(dual_mapping).__name__}"
                )
            if dual_mapping.num_edges != num_edges:
                raise ValueError(
                    f"dual_mapping describes {dual_mapping.num_edges} edges, but the network "
                    f"has {num_edges}"
                )
            rows = dual_mapping.num_constraints
        elif dual_matrix is not None:
            matrix = _as_dual_matrix(dual_matrix, num_edges)
            rows = int(matrix.shape[0])
        else:
            rows = 0 if num_constraints is None else operator.index(num_constraints)
            if rows < 0:
                raise ValueError("num_constraints must be >= 0")
            matrix = scipy.sparse.csc_array((rows, num_edges), dtype=np.float64)
        if num_constraints is not None and operator.index(num_constraints) != rows:
            raise ValueError(
                f"num_constraints={num_constraints} contradicts the dual coefficients, which "
                f"have {rows} rows"
            )

        indexed = cost_update == "indexed" or (
            cost_update == "per_vessel" and dual_mapping is not None
        )
        indptr, indices, data = _EMPTY_INDEX, _EMPTY_INDEX, _EMPTY_VALUE
        width, row, value = 0, _EMPTY_INDEX, _EMPTY_VALUE
        if indexed:
            if dual_mapping is not None:
                pairs = (dual_mapping.row, dual_mapping.coefficient)
            else:
                assert matrix is not None
                pairs = _matrix_to_indexed(matrix)
            width = pairs[0].shape[1]
            row = np.ascontiguousarray(pairs[0], dtype=np.int64).ravel()
            value = np.ascontiguousarray(pairs[1], dtype=np.float64).ravel()
        else:
            if matrix is None:
                assert dual_mapping is not None
                matrix = dual_mapping.matrix()
            indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int64)
            indices = np.ascontiguousarray(matrix.indices, dtype=np.int64)
            data = np.ascontiguousarray(matrix.data, dtype=np.float64)

        self._num_constraints = rows
        self._matrix = matrix
        self._mapping = dual_mapping
        self._scheduler = scheduler
        self._partitioner = partitioner
        self._cost_update = cost_update
        self._exact = tolerance == 0.0
        self._lock = threading.Lock()
        self._ids: list[Hashable] = []
        self._index: dict[Hashable, int] = {}
        self._sources = np.empty(0, dtype=np.int64)
        self._core = _core.PricingFleet(
            graph._core, _ALGORITHM_CODE[self._algorithm], rows, indptr, indices, data,
            width, row, value, _SCHEDULER_CODE[scheduler], _PARTITIONER_CODE[partitioner],
            threads, cost_update == "per_vessel", tolerance, int(bound_cache_bytes),
        )

    # ------------------------------------------------------------ description

    @property
    def network(self) -> Network:
        return self._network

    @property
    def algorithm(self) -> str:
        """The resolved algorithm, ``"dag"`` or ``"rcsp"``."""
        return self._algorithm

    @property
    def scheduler(self) -> str:
        """The resolved scheduler, ``"tbb"`` or ``"threads"``."""
        return self._scheduler

    @property
    def cost_update(self) -> str:
        return self._cost_update

    @property
    def threads(self) -> int:
        """Number of worker slots, the calling thread included."""
        return self._core.threads

    @property
    def num_constraints(self) -> int:
        """Number of master constraints, i.e. the expected length of ``duals``."""
        return self._num_constraints

    @property
    def num_vessels(self) -> int:
        return self._core.num_vessels

    def __len__(self) -> int:
        return self.num_vessels

    @property
    def vessel_ids(self) -> tuple[Hashable, ...]:
        with self._lock:
            return tuple(self._ids)

    def index_of(self, vessel: Hashable) -> int:
        """Position of the vessel registered under the id ``vessel``."""
        with self._lock:
            try:
                return self._index[vessel]
            except KeyError:
                raise KeyError(f"no vessel with id {vessel!r} in this fleet") from None

    @property
    def dual_matrix(self) -> scipy.sparse.csc_array:
        """The shared coefficients ``A`` in CSC form. Treat as read-only."""
        if self._matrix is None:
            assert self._mapping is not None
            self._matrix = self._mapping.matrix()
        return self._matrix

    @property
    def is_prepared(self) -> bool:
        """Whether ``prepare`` covers every vessel added so far."""
        return self._core.prepared

    @property
    def reachable(self) -> npt.NDArray[np.bool_]:
        """Per vessel: whether its target can be reached at all (ignoring
        resources) in the vessel's own network. Prepares the fleet if needed."""
        self._core.prepare()
        return self._core.reachable().astype(bool)

    @property
    def memory(self) -> dict[str, int]:
        """Bytes of native memory held by the fleet, by purpose.

        ``shared_graph`` belongs to the network and is counted once however
        many fleets or engines use it.
        """
        names = ("shared_graph", "shared_pricing", "vessels", "bound_cache", "workspaces",
                 "results")
        usage = dict(zip(names, self._core.memory()))
        usage["total"] = sum(usage.values())
        return usage

    def __repr__(self) -> str:
        return (
            f"PricingFleet(vessels={self.num_vessels}, algorithm={self._algorithm!r}, "
            f"scheduler={self._scheduler!r}, threads={self.threads}, "
            f"cost_update={self._cost_update!r}, num_constraints={self._num_constraints})"
        )

    # ---------------------------------------------------------------- vessels

    def _limits(self, resource_limits: object, count: int) -> np.ndarray:
        if resource_limits is None:
            return _EMPTY_VALUE
        tracked = self._network.tracked_resources
        if self._algorithm != "rcsp":
            raise ValueError(
                "resource_limits only apply to algorithm='rcsp'; this fleet uses "
                f"{self._algorithm!r}, which ignores resource bounds"
            )
        upper = {resource.name: resource.upper for resource in self._network.resources}
        if isinstance(resource_limits, Mapping):
            limits = np.empty((count, len(tracked)), dtype=np.float64)
            for column, name in enumerate(tracked):
                limits[:, column] = upper[name]
            for name, values in resource_limits.items():
                if name not in tracked:
                    known = name in upper
                    raise ValueError(
                        f"resource {name!r} "
                        + ("has no finite upper bound in the graph, so it is not tracked by "
                           "the solver; declare it with a finite upper (the loosest limit of "
                           "any vessel) to limit it per vessel"
                           if known else f"is not declared; tracked resources: {list(tracked)}")
                    )
                limits[:, tracked.index(name)] = _per_vessel_values(
                    values, count, f"resource_limits[{name!r}]"
                )
        else:
            limits = np.array(resource_limits, dtype=np.float64)
            if limits.shape != (count, len(tracked)):
                raise ValueError(
                    f"resource_limits has shape {limits.shape}, expected ({count}, "
                    f"{len(tracked)}): one row per vessel and one column per tracked resource "
                    f"{list(tracked)}; or pass a mapping from resource name to limits"
                )
        if np.isnan(limits).any() or (limits == -math.inf).any():
            raise ValueError("resource limits must be numbers (inf for no limit)")
        return np.ascontiguousarray(limits).ravel()

    def add_vessels(
        self,
        ids: Sequence[Hashable] | None = None,
        source: Any = None,
        target: Any = None,
        *,
        resource_limits: Mapping[str, npt.ArrayLike] | npt.ArrayLike | None = None,
        fixed_cost: npt.ArrayLike = 0.0,
        fixed_coefficients: Any = None,
        cost_offsets: Any = None,
        forbidden_edges: Sequence[npt.ArrayLike | None] | None = None,
        edge_mask: npt.ArrayLike | None = None,
        edge_coefficients: Sequence[Any] | Mapping[int, Any] | None = None,
    ) -> range:
        """Register vessels and return their positions in the fleet.

        Vessels are registered once and live in native memory from then on;
        nothing about them is rebuilt in later pricing iterations. May be
        called repeatedly, also between solves. Either the whole batch is
        added or, on error, none of it.

        Parameters
        ----------
        ids:
            One hashable id per vessel, unique within the fleet. Defaults to
            the vessels' positions.
        source, target:
            One vertex per vessel: an integer array of vertex ids or a list
            of states. A single integer id is shared by all vessels of the
            batch, whose size then follows from ``ids`` or from the first
            per-vessel argument (one vessel if there is none).
        resource_limits:
            Upper bounds per vessel for the tracked resources (those with a
            finite ``upper`` in the graph), replacing the graph's bound:
            ``{name: scalar or one value per vessel}``, or a
            ``(num_vessels, num_tracked)`` array in the order of
            ``network.tracked_resources``. ``rcsp`` only.
        fixed_cost:
            Route-level cost per vessel (scalar or one per vessel), part of
            every column's ``original_cost``.
        fixed_coefficients:
            Route-level master coefficients per vessel, e.g. a 1 in the
            vessel's convexity row: a ``(num_vessels, num_constraints)``
            matrix (sparse recommended) or one ``{row: value}`` mapping per
            vessel.
        cost_offsets:
            Vessel-specific additions to edge costs: a
            ``(num_vessels, num_edges)`` matrix (sparse recommended) or one
            ``{edge: offset}`` mapping per vessel. An offset of ``inf``
            forbids the edge.
        forbidden_edges:
            Per vessel, the edge ids it must not use (or ``None``).
        edge_mask:
            Boolean ``(num_vessels, num_edges)`` array, ``True`` where a
            vessel may use an edge. Equivalent to ``forbidden_edges``; prefer
            that when few edges are excluded.
        edge_coefficients:
            Vessel-specific master coefficients ``B_v`` that are added to the
            shared ones: per vessel a ``(num_constraints, num_edges)`` matrix
            (sparse recommended) or ``None``; or a mapping from position in
            the batch to such a matrix.

        Notes
        -----
        Only what differs from the shared network is stored: a vessel
        without overlays costs a few dozen bytes.
        """
        network = self._network
        num_edges = network.num_edges
        rows = self._num_constraints
        if source is None or target is None:
            raise TypeError("add_vessels requires source and target")
        if ids is not None:
            ids = list(ids)
        count = next(
            (n for n in (None if ids is None else len(ids), _length(source), _length(target))
             if n is not None),
            None,
        )
        if count is None:
            # Shared source and target: the first per-vessel argument decides.
            candidates = [fixed_cost, forbidden_edges, edge_mask, fixed_coefficients,
                          cost_offsets]
            if isinstance(resource_limits, Mapping):
                candidates.extend(resource_limits.values())
            else:
                candidates.append(resource_limits)
            if not isinstance(edge_coefficients, Mapping):
                candidates.append(edge_coefficients)
            count = next(filter(None, map(_batch_size, candidates)), 1)
        sources = _vertex_ids(network, source, count, "source")
        targets = _vertex_ids(network, target, count, "target")
        fixed_costs = _per_vessel_values(fixed_cost, count, "fixed_cost")
        if not np.isfinite(fixed_costs).all():
            raise ValueError("fixed_cost must be finite")
        limits = self._limits(resource_limits, count)

        vessel, fixed_row, fixed_value = _vessel_triplets(
            fixed_coefficients, count, rows, "fixed_coefficients", "master constraint"
        )
        fixed_ptr, fixed_row, fixed_value = _group_by_vessel(vessel, count, fixed_row, fixed_value)

        parts = [_vessel_triplets(cost_offsets, count, num_edges, "cost_offsets", "edge")]
        if forbidden_edges is not None:
            if isinstance(forbidden_edges, (str, bytes, Mapping)) or len(forbidden_edges) != count:
                raise ValueError(
                    f"forbidden_edges must have one entry per vessel ({count}): an array of "
                    "edge ids or None"
                )
            for position, edges in enumerate(forbidden_edges):
                if edges is None:
                    continue
                edges = np.asarray(edges)
                if edges.size and edges.dtype.kind not in "iu":
                    raise TypeError(f"forbidden_edges[{position}] must hold integer edge ids")
                edges = edges.astype(np.int64).ravel()
                if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= num_edges):
                    raise ValueError(
                        f"forbidden_edges[{position}] refers to an edge outside [0, {num_edges})"
                    )
                parts.append((
                    np.full(edges.shape[0], position, dtype=np.int64),
                    edges,
                    np.full(edges.shape[0], math.inf),
                ))
        if edge_mask is not None:
            mask = np.asarray(edge_mask)
            if mask.dtype != np.bool_ or mask.shape != (count, num_edges):
                raise ValueError(
                    f"edge_mask must be a boolean array of shape ({count}, {num_edges}), True "
                    f"where a vessel may use an edge; got dtype {mask.dtype}, shape {mask.shape}"
                )
            masked_vessel, masked_edge = np.nonzero(~mask)
            parts.append((
                masked_vessel.astype(np.int64),
                masked_edge.astype(np.int64),
                np.full(masked_edge.shape[0], math.inf),
            ))
        offset_ptr, offset_edge, offset_value = _group_by_vessel(
            np.concatenate([part[0] for part in parts]),
            count,
            np.concatenate([part[1] for part in parts]),
            np.concatenate([part[2] for part in parts]),
        )

        coef_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        if edge_coefficients is not None:
            if isinstance(edge_coefficients, Mapping):
                items = list(edge_coefficients.items())
            elif isinstance(edge_coefficients, Sequence) and len(edge_coefficients) == count:
                items = list(enumerate(edge_coefficients))
            else:
                raise ValueError(
                    f"edge_coefficients must have one entry per vessel ({count}), or map "
                    "positions in the batch to matrices"
                )
            for position, item in items:
                if item is None:
                    continue
                position = operator.index(position)
                if not 0 <= position < count:
                    raise ValueError(f"edge_coefficients: no vessel at position {position}")
                try:
                    block = scipy.sparse.csc_array(item, dtype=np.float64)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"edge_coefficients[{position}] is not a numeric matrix: {error}"
                    ) from None
                if block.ndim != 2 or block.shape != (rows, num_edges):
                    raise ValueError(
                        f"edge_coefficients[{position}] has shape {block.shape}, expected "
                        f"({rows}, {num_edges}): (num_constraints, num_edges)"
                    )
                block.sum_duplicates()
                block.eliminate_zeros()
                coef_parts.append((
                    np.full(block.nnz, position, dtype=np.int64),
                    np.repeat(np.arange(num_edges, dtype=np.int64), np.diff(block.indptr)),
                    block.indices.astype(np.int64),
                    block.data.astype(np.float64),
                ))
        if coef_parts:
            coef_ptr, coef_edge, coef_row, coef_value = _group_by_vessel(
                np.concatenate([part[0] for part in coef_parts]),
                count,
                *(np.concatenate([part[k] for part in coef_parts]) for k in (1, 2, 3)),
            )
        else:
            coef_ptr = np.zeros(count + 1, dtype=np.int64)
            coef_edge, coef_row, coef_value = _EMPTY_INDEX, _EMPTY_INDEX, _EMPTY_VALUE

        with self._lock:
            first = len(self._ids)
            if ids is None:
                ids = list(range(first, first + count))
            elif len(ids) != count:
                raise ValueError(f"got {len(ids)} ids for {count} vessels")
            seen: set[Hashable] = set()
            for vessel_id in ids:
                try:
                    duplicate = vessel_id in self._index or vessel_id in seen
                except TypeError:
                    raise TypeError(f"vessel id {vessel_id!r} is not hashable") from None
                if duplicate:
                    raise ValueError(f"vessel id {vessel_id!r} is already used in this fleet")
                seen.add(vessel_id)
            native_first = self._core.add_vessels(
                sources, targets, fixed_costs, limits, fixed_ptr, fixed_row, fixed_value,
                offset_ptr, offset_edge, offset_value, coef_ptr, coef_edge, coef_row, coef_value,
            )
            assert native_first == first
            self._ids.extend(ids)
            for position, vessel_id in enumerate(ids, start=first):
                self._index[vessel_id] = position
            self._sources = np.concatenate([self._sources, sources])
        return range(first, first + count)

    def prepare(self) -> PricingFleet:
        """Build everything that does not depend on the duals.

        Starts the worker pool, allocates one reusable workspace per worker,
        determines which vessels can reach their target, and computes the
        resource completion bounds (once per distinct target vertex).
        Optional: the first solve after a change prepares implicitly; calling
        it yourself keeps that cost out of the first pricing iteration.
        Returns the fleet.
        """
        self._core.prepare()
        return self

    # ---------------------------------------------------------------- pricing

    def _check_duals(self, duals: object) -> np.ndarray:
        mapping = self._mapping
        if isinstance(mapping, PortTimeProductDuals) and np.ndim(duals) == 3:
            duals = mapping.flatten(duals)  # type: ignore[arg-type]
        try:
            pi = np.ascontiguousarray(duals, dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError("duals must be a 1-D array of real numbers") from None
        if pi.ndim != 1 or pi.shape[0] != self._num_constraints:
            raise ValueError(
                f"duals has shape {pi.shape}, expected ({self._num_constraints},): one dual "
                "value per master constraint"
            )
        if not np.isfinite(pi).all():
            raise ValueError("duals must be finite")
        return pi

    def _solve_arguments(
        self, max_paths: object, negative_only: object, tolerance: object, time_limit: object
    ) -> tuple[int, bool, float, float]:
        max_paths = _check_max_paths(max_paths, "max_paths_per_vessel")
        if max_paths > 1 and self._algorithm == "rcsp":
            raise _multiple_paths_unsupported()
        return max_paths, bool(negative_only), _check_tolerance(tolerance), _check_time_limit(
            time_limit
        )

    def _wrap(self, raw: dict[str, Any], start: float, tolerance: float) -> FleetResult:
        count = raw["status"].shape[0]
        with self._lock:
            ids = tuple(self._ids[:count])
            sources = self._sources[:count]
        errors = raw["errors"]
        if errors is not None:
            failed = {index: message for index, message in enumerate(errors) if message}
            details = "; ".join(
                f"vessel {ids[index]!r}: {message}" for index, message in failed.items()
            )
            raise BatchPricingError(
                f"{len(failed)} of {count} vessels failed: {details}", failed
            )
        update_time, solve_time, assemble_time = raw["timings"]
        labels = raw["labels"]
        statistics = FleetStatistics(
            self._algorithm, self._scheduler, raw["threads"], update_time, solve_time,
            assemble_time, time.perf_counter() - start, labels[0], *labels[1:],
        )
        return FleetResult(
            self._network, self._num_constraints, self._exact, ids, sources, raw, tolerance,
            statistics,
        )

    def update_duals(self, duals: npt.ArrayLike) -> None:
        """Set the dual values for the next ``solve``.

        Called once per column-generation iteration. In the shared modes this
        refreshes the single reduced-cost array all vessels read; its cost
        does not depend on the number of vessels.

        ``duals`` holds one value per master constraint, exactly as an LP
        solver reports them for a minimisation master. With a
        ``PortTimeProductDuals`` mapping without extra constraints, an
        array of shape ``(num_ports, num_days, num_products)`` is accepted too.
        """
        self._core.update_duals(self._check_duals(duals))

    def solve(
        self,
        max_paths_per_vessel: int = 1,
        negative_only: bool = True,
        *,
        tolerance: float = 1e-9,
        time_limit: float | None = None,
    ) -> FleetResult:
        """Price every vessel with the duals of the last ``update_duals``.

        The whole fleet is priced inside one native call with the GIL
        released. Results are stored per vessel, so the outcome does not
        depend on the thread schedule: it equals pricing the vessels one
        after another.

        Parameters
        ----------
        max_paths_per_vessel:
            Maximum number of columns per vessel. ``"dag"`` returns the
            cheapest distinct paths exactly; ``"rcsp"`` supports only 1.
        negative_only:
            Return only columns with ``reduced_cost < -tolerance``.
        tolerance:
            Threshold for ``negative_only`` and for
            ``FleetResult.converged``.
        time_limit:
            Seconds after which vessels that are still unpriced are given
            up (``None``: no limit). They are reported as
            ``"interrupted"``, and the result does not claim convergence.

        Raises
        ------
        BatchPricingError
            If pricing a vessel raised an error. ``errors`` maps the
            positions of the failed vessels to messages; the fleet remains
            usable.
        """
        arguments = self._solve_arguments(
            max_paths_per_vessel, negative_only, tolerance, time_limit
        )
        start = time.perf_counter()
        raw = self._core.solve(*arguments)
        return self._wrap(raw, start, arguments[2])

    def price(
        self,
        duals: npt.ArrayLike,
        max_paths_per_vessel: int = 1,
        negative_only: bool = True,
        *,
        tolerance: float = 1e-9,
        time_limit: float | None = None,
    ) -> FleetResult:
        """``update_duals`` and ``solve`` fused into one native call.

        The preferred entry point of a column-generation loop: Python is
        left once and re-entered once per iteration.
        """
        arguments = self._solve_arguments(
            max_paths_per_vessel, negative_only, tolerance, time_limit
        )
        pi = self._check_duals(duals)
        start = time.perf_counter()
        raw = self._core.price(pi, *arguments)
        return self._wrap(raw, start, arguments[2])

    def cancel(self) -> None:
        """Ask a running ``solve`` / ``price`` to stop.

        Safe to call from another thread (or a signal handler). Vessels that
        were already priced keep their results; the others are reported as
        ``"interrupted"`` and ``FleetResult.cancelled`` is set. Has no
        effect when no solve is running.
        """
        self._core.cancel()
