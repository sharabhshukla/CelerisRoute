"""Result objects returned by the solvers and the pricing engines."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import scipy.sparse

    from .graph import Network

__all__ = [
    "Column",
    "PricingResult",
    "PricingStatistics",
    "Solution",
    "SolveStatistics",
    "Status",
]


class Status(StrEnum):
    """Outcome of a solve. Members compare equal to their string values."""

    OPTIMAL = "optimal"
    """A minimum-cost path (feasible for all resources, if any) was found."""
    UNREACHABLE = "unreachable"
    """The graph has no directed path from the source to the target."""
    INFEASIBLE = "infeasible"
    """Paths exist, but every one of them violates a resource bound."""
    INTERRUPTED = "interrupted"
    """The subproblem was not solved to completion (cancelled, out of time, or
    never started), so nothing is known about it. Only fleet solves report it."""


_STATUS = (Status.OPTIMAL, Status.UNREACHABLE, Status.INFEASIBLE, Status.INTERRUPTED)


@dataclass(frozen=True, slots=True)
class SolveStatistics:
    """Counters of one solve. Label fields are zero for the ``"dag"`` algorithm."""

    algorithm: str
    solve_time: float
    """Seconds spent inside the native solver (excluding Python overhead)."""
    labels_created: int = 0
    """Feasible labels generated, including the initial label at the source."""
    labels_extended: int = 0
    """Labels that survived dominance and were extended along their out-edges."""
    labels_dominated: int = 0
    """Labels discarded because another label at the same vertex dominated them."""
    labels_infeasible: int = 0
    """Extensions rejected because a resource bound was exceeded, or because the
    label could not have reached the target within the bounds."""
    max_active_labels: int = 0
    """Largest number of labels waiting to be processed at any time."""
    pareto_labels: int = 0
    """Nondominated labels at the target (the size of its Pareto front)."""


@dataclass(frozen=True, slots=True)
class PricingStatistics:
    """Timings (seconds) and label counters of one pricing call."""

    algorithm: str
    update_time: float
    """Recomputing reduced edge costs from the duals."""
    solve_time: float
    """The shortest-path algorithm itself."""
    reconstruction_time: float
    """Turning native paths into columns (edges, costs, coefficients)."""
    total_time: float
    """Wall time of the whole ``solve`` call as seen from Python."""
    labels_created: int = 0
    labels_extended: int = 0
    labels_dominated: int = 0
    labels_infeasible: int = 0
    max_active_labels: int = 0
    pareto_labels: int = 0


def _freeze(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class _PathMixin:
    """Lazy ``nodes`` and ``path`` views shared by solutions and columns."""

    __slots__ = ()

    _network: Network
    _source: int
    _nodes: np.ndarray | None
    _path: list[Any] | None
    edges: np.ndarray

    def _has_path(self) -> bool:
        return True

    @property
    def nodes(self) -> npt.NDArray[np.int64]:
        """Vertex ids along the path, source first (empty when there is no path)."""
        if self._nodes is None:
            if not self._has_path():
                nodes = np.empty(0, dtype=np.int64)
            elif self.edges.size == 0:
                nodes = np.array([self._source], dtype=np.int64)
            else:
                network = self._network
                nodes = np.empty(self.edges.size + 1, dtype=np.int64)
                nodes[0] = network.edge_source[self.edges[0]]
                nodes[1:] = network.edge_target[self.edges]
            self._nodes = _freeze(nodes)
        return self._nodes

    @property
    def path(self) -> list[Any]:
        """States along the path, source first.

        Decoded state tuples when the graph was built on a
        ``celerisroute.StateSpace``, plain vertex ids otherwise. Decoding
        happens on first access.
        """
        if self._path is None:
            states = self._network.states
            nodes = self.nodes
            self._path = nodes.tolist() if states is None else states.decode_many(nodes)
        return list(self._path)


class Solution(_PathMixin):
    """Result of a single shortest-path solve.

    Attributes
    ----------
    status:
        ``Status``; also usable as a string (``solution.status == "optimal"``).
    cost:
        Total cost of the path, ``inf`` when no path was found.
    edges:
        Edge ids of the path in travel order (read-only ``int64`` array).
    nodes, path:
        Vertex ids / decoded states along the path.
    resources:
        Final value of every declared resource, ``initial`` plus the
        increments of the path's edges. Empty when no path was found.
    statistics:
        ``SolveStatistics``.

    A solution is truthy exactly when a path was found.
    """

    __slots__ = (
        "status", "cost", "edges", "resources", "statistics",
        "_network", "_source", "_nodes", "_path",
    )

    def __init__(
        self,
        network: Network,
        source: int,
        status: Status,
        cost: float,
        edges: np.ndarray,
        resources: dict[str, float],
        statistics: SolveStatistics,
    ) -> None:
        self._network = network
        self._source = source
        self._nodes = None
        self._path = None
        self.status = status
        self.cost = cost
        self.edges = _freeze(edges)
        self.resources = resources
        self.statistics = statistics

    def _has_path(self) -> bool:
        return self.status is Status.OPTIMAL

    @property
    def found(self) -> bool:
        """True when a path was found (``status == "optimal"``)."""
        return self.status is Status.OPTIMAL

    def __bool__(self) -> bool:
        return self.status is Status.OPTIMAL

    def __repr__(self) -> str:
        if not self.found:
            return f"Solution(status={self.status.value!r})"
        return (
            f"Solution(status='optimal', cost={self.cost!r}, edges={self.edges.size}, "
            f"resources={self.resources!r})"
        )


class Column(_PathMixin):
    """A path returned by a pricing engine, ready to enter a master problem.

    Attributes
    ----------
    edges, nodes, path:
        The path, as for ``Solution``.
    original_cost:
        Cost of the path under the graph's own edge costs, plus the engine's
        ``fixed_cost``. This is the objective coefficient of the column.
    reduced_cost:
        ``original_cost - duals @ coefficients``.
    resources:
        Final value of every declared resource along the path.
    coefficients:
        The column's master-constraint coefficients ``a_P = sum of A[:, e]``
        over the path's edges (plus the engine's ``fixed_coefficients``), as
        a SciPy sparse column of shape ``(num_constraints, 1)``.
    coefficient_indices, coefficient_values:
        The same coefficients as parallel arrays: ascending constraint rows
        and their nonzero values.
    """

    __slots__ = (
        "edges", "original_cost", "reduced_cost", "resources",
        "coefficient_indices", "coefficient_values",
        "_network", "_source", "_nodes", "_path", "_num_constraints",
    )

    def __init__(
        self,
        network: Network,
        source: int,
        num_constraints: int,
        edges: np.ndarray,
        original_cost: float,
        reduced_cost: float,
        resources: dict[str, float],
        coefficient_indices: np.ndarray,
        coefficient_values: np.ndarray,
    ) -> None:
        self._network = network
        self._source = source
        self._nodes = None
        self._path = None
        self._num_constraints = num_constraints
        self.edges = _freeze(edges)
        self.original_cost = original_cost
        self.reduced_cost = reduced_cost
        self.resources = resources
        self.coefficient_indices = _freeze(coefficient_indices)
        self.coefficient_values = _freeze(coefficient_values)

    @property
    def coefficients(self) -> scipy.sparse.csc_array:
        """Sparse ``(num_constraints, 1)`` column of master coefficients."""
        import scipy.sparse

        indptr = np.array([0, self.coefficient_indices.size], dtype=np.int64)
        return scipy.sparse.csc_array(
            (self.coefficient_values.copy(), self.coefficient_indices.copy(), indptr),
            shape=(self._num_constraints, 1),
        )

    def dense_coefficients(self) -> npt.NDArray[np.float64]:
        """Coefficients as a dense vector of length ``num_constraints``."""
        dense = np.zeros(self._num_constraints, dtype=np.float64)
        dense[self.coefficient_indices] = self.coefficient_values
        return dense

    def __repr__(self) -> str:
        return (
            f"Column(reduced_cost={self.reduced_cost!r}, original_cost={self.original_cost!r}, "
            f"edges={self.edges.size}, nonzeros={self.coefficient_indices.size})"
        )


class PricingResult(Sequence[Column]):
    """Columns found by one pricing call, cheapest reduced cost first.

    Behaves like an immutable list of ``Column`` objects (it is empty,
    and therefore falsy, when no column qualified) and additionally carries
    information about the pricing problem itself:

    status:
        ``Status`` of the subproblem *before* the ``negative_only``
        filter: ``"optimal"`` means a feasible path exists, even if its
        reduced cost was not negative.
    best_reduced_cost:
        The minimum reduced cost over all feasible paths (``inf`` if there
        are none). With an exact algorithm this is the pricing certificate:
        ``best_reduced_cost >= -tolerance`` proves that no improving column
        exists for the given duals.
    statistics:
        ``PricingStatistics``.
    """

    __slots__ = ("columns", "status", "best_reduced_cost", "statistics")

    def __init__(
        self,
        columns: tuple[Column, ...],
        status: Status,
        best_reduced_cost: float,
        statistics: PricingStatistics,
    ) -> None:
        self.columns = columns
        self.status = status
        self.best_reduced_cost = best_reduced_cost
        self.statistics = statistics

    @overload
    def __getitem__(self, index: int) -> Column: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[Column, ...]: ...
    def __getitem__(self, index: int | slice) -> Column | tuple[Column, ...]:
        return self.columns[index]

    def __len__(self) -> int:
        return len(self.columns)

    def __iter__(self) -> Iterator[Column]:
        return iter(self.columns)

    def __repr__(self) -> str:
        best = "inf" if math.isinf(self.best_reduced_cost) else repr(self.best_reduced_cost)
        return (
            f"PricingResult(columns={len(self.columns)}, status={self.status.value!r}, "
            f"best_reduced_cost={best})"
        )


def decode_status(code: int) -> Status:
    return _STATUS[code]

