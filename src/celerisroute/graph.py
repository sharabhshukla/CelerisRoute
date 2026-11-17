"""Graph construction (``Graph``) and the compiled, immutable ``Network``."""

from __future__ import annotations

import math
import operator
from collections.abc import Hashable, Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _core
from .errors import CyclicGraphError, InvalidStateError
from .resources import Resource
from .results import Solution, SolveStatistics, decode_status
from .states import StateSpace

__all__ = ["Graph", "Network"]

Vertex = int | tuple[Hashable, ...] | Mapping[str, Hashable]

_ALGORITHMS = ("auto", "dag", "rcsp")
_ALGORITHM_CODE = {"dag": 0, "rcsp": 1}
# The native core uses 32-bit vertex and edge ids.
_MAX_VERTICES = 2**32 - 2


def _as_index_array(values: object, what: str) -> np.ndarray:
    """1-D C-contiguous int64 array from integer input; no silent float casts."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{what} must be a 1-D array, got shape {array.shape}")
    if array.dtype.kind not in "iu":
        if array.size == 0:
            return np.empty(0, dtype=np.int64)
        raise TypeError(
            f"{what} must be an integer array of vertex ids, got dtype {array.dtype}; "
            "translate states to ids first with StateSpace.ids(...)"
        )
    if array.dtype.kind == "u" and array.size and int(array.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{what} contains a vertex id that is out of range")
    return np.ascontiguousarray(array, dtype=np.int64)


def _as_value_array(values: object, length: int, what: str) -> np.ndarray:
    """1-D C-contiguous float64 array of ``length``; scalars are broadcast."""
    array = np.asarray(values)
    if array.dtype.kind not in "iufb":
        raise TypeError(f"{what} must be numeric, got dtype {array.dtype}")
    if array.ndim == 0:
        return np.full(length, float(array), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{what} must be a scalar or a 1-D array, got shape {array.shape}")
    if array.shape[0] != length:
        raise ValueError(f"{what} has {array.shape[0]} entries, expected {length} (one per edge)")
    return np.ascontiguousarray(array, dtype=np.float64)


class Graph:
    """A directed graph under construction.

    Vertices are the states of a ``StateSpace`` (``Graph(states=...)``)
    or plain integers ``0 .. num_vertices-1`` (``Graph(num_vertices=n)``).
    Edges are added one at a time with ``add_edge`` or in bulk from
    arrays with ``add_edges``, and receive consecutive ids starting at 0
    in insertion order. Parallel edges and negative costs are allowed.

    Call ``compile`` to obtain a ``Network``, the immutable form the
    solvers run on. The graph itself stays editable; a network is a snapshot
    and is never affected by later edits.

    Wherever a vertex is expected, an ``int`` is always a vertex id and a
    tuple (or mapping) is always a state. For a one-dimensional state space
    with integer values this means the state ``7`` must be written ``(7,)``.
    """

    def __init__(
        self,
        states: StateSpace | None = None,
        *,
        num_vertices: int | None = None,
        directed: bool = True,
    ) -> None:
        if not directed:
            raise NotImplementedError(
                "undirected graphs are not supported: the solvers require a directed acyclic "
                "graph. Model an undirected link as two opposite directed edges between "
                "time- or phase-expanded states."
            )
        if states is not None and not isinstance(states, StateSpace):
            raise TypeError(f"states must be a StateSpace, got {type(states).__name__}")
        if states is None and num_vertices is None:
            raise ValueError("pass either states=StateSpace(...) or num_vertices=<int>")
        if num_vertices is not None:
            if isinstance(num_vertices, bool) or not isinstance(num_vertices, Integral):
                raise TypeError(f"num_vertices must be an integer, got {num_vertices!r}")
            num_vertices = int(num_vertices)
            if states is not None and num_vertices != states.size:
                raise ValueError(
                    f"num_vertices={num_vertices} contradicts the state space, which has "
                    f"{states.size} states; omit num_vertices"
                )
        size = states.size if states is not None else num_vertices
        assert size is not None
        if size < 1:
            raise ValueError("a graph needs at least one vertex")
        if size > _MAX_VERTICES:
            raise ValueError(
                f"{size} vertices exceed the supported maximum of {_MAX_VERTICES} "
                "(vertex ids are 32-bit)"
            )
        self._states = states
        self._builder = _core.GraphBuilder(size)
        self._resources: dict[str, Resource] = {}

    # ------------------------------------------------------------ description

    @property
    def states(self) -> StateSpace | None:
        """The state space the vertices are drawn from, if any."""
        return self._states

    @property
    def directed(self) -> bool:
        """Always ``True``."""
        return True

    @property
    def num_vertices(self) -> int:
        return self._builder.num_vertices

    @property
    def num_edges(self) -> int:
        return self._builder.num_edges

    @property
    def resources(self) -> tuple[Resource, ...]:
        """Resources declared so far, in declaration order."""
        return tuple(self._resources.values())

    def __repr__(self) -> str:
        return (
            f"Graph(num_vertices={self.num_vertices}, num_edges={self.num_edges}, "
            f"resources={list(self._resources)})"
        )

    # --------------------------------------------------------------- vertices

    def node_id(self, vertex: Vertex) -> int:
        """Vertex id of a state tuple/mapping; integer ids are range-checked."""
        return _resolve_vertex(self._states, self.num_vertices, vertex)

    # ---------------------------------------------------------------- editing

    def add_resource(self, resource: Resource) -> Resource:
        """Declare a resource. See ``Resource`` for the supported semantics."""
        if not isinstance(resource, Resource):
            raise TypeError(
                f"add_resource expects a celerisroute.Resource, got {type(resource).__name__}"
            )
        if resource.name in self._resources:
            raise ValueError(f"resource {resource.name!r} is already declared")
        self._resources[resource.name] = resource
        return resource

    def add_edge(
        self,
        source: Vertex,
        target: Vertex,
        cost: float,
        resources: Mapping[str, float] | None = None,
    ) -> int:
        """Add one edge and return its id.

        ``resources`` maps resource names to this edge's increment; resources
        not mentioned get an increment of zero.
        """
        s = self.node_id(source)
        t = self.node_id(target)
        if isinstance(cost, bool) or not isinstance(cost, Real):
            raise TypeError(f"cost must be a real number, got {cost!r}")
        names: list[str] = []
        values: list[float] = []
        if resources:
            if not isinstance(resources, Mapping):
                raise TypeError("resources must be a mapping from resource name to increment")
            for name, value in resources.items():
                if not isinstance(name, str) or not name:
                    raise TypeError(f"resource names must be nonempty strings, got {name!r}")
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(f"resource {name!r}: increment must be a real number, got {value!r}")
                value = float(value)
                if name in self._resources and value < 0.0:
                    raise ValueError(_negative_increment_message(name, value))
                names.append(name)
                values.append(value)
        return self._builder.add_edge(s, t, float(cost), names, values)

    def add_edges(
        self,
        source: npt.ArrayLike,
        target: npt.ArrayLike,
        cost: npt.ArrayLike,
        resources: Mapping[str, npt.ArrayLike] | None = None,
    ) -> range:
        """Add many edges from arrays and return their ids as a ``range``.

        ``source`` and ``target`` are integer arrays of vertex ids (use
        ``StateSpace.ids`` to translate states in bulk); ``cost`` and each
        resource are arrays of the same length, or scalars applied to every
        edge. The arrays are validated and copied natively in one pass, with
        no per-edge Python work. Either all edges are added or, on error, none.
        """
        s = _as_index_array(source, "source")
        t = _as_index_array(target, "target")
        if s.shape[0] != t.shape[0]:
            raise ValueError(
                f"source has {s.shape[0]} entries but target has {t.shape[0]}; "
                "they must describe the same edges"
            )
        limit = self.num_vertices
        for what, ids in (("source", s), ("target", t)):
            if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= limit):
                position = int(np.flatnonzero((ids < 0) | (ids >= limit))[0])
                raise InvalidStateError(
                    f"{what}[{position}] = {int(ids[position])} is outside [0, {limit}): "
                    "not a vertex id of this graph"
                )
        c = _as_value_array(cost, s.shape[0], "cost")
        names: list[str] = []
        columns: list[np.ndarray] = []
        if resources:
            if not isinstance(resources, Mapping):
                raise TypeError("resources must be a mapping from resource name to an array")
            for name, values in resources.items():
                if not isinstance(name, str) or not name:
                    raise TypeError(f"resource names must be nonempty strings, got {name!r}")
                column = _as_value_array(values, s.shape[0], f"resource {name!r}")
                if name in self._resources and column.size and column.min() < 0.0:
                    raise ValueError(_negative_increment_message(name, float(column.min())))
                names.append(name)
                columns.append(column)
        first = self._builder.add_edges(s, t, c, names, columns)
        return range(first, first + s.shape[0])

    # ---------------------------------------------------------------- compile

    def compile(self) -> Network:
        """Validate the graph and freeze it into a ``Network``.

        Compilation sorts the edges into a contiguous adjacency structure,
        checks every declared resource (nonnegative increments, consistent
        bounds), and determines whether the graph is acyclic, caching a
        topological order if it is. A cyclic graph still compiles (so it can
        be inspected with ``Network.find_cycle``), but cannot be solved.
        """
        undeclared = [n for n in self._builder.resource_names() if n not in self._resources]
        if undeclared:
            raise ValueError(
                f"edges use the resource(s) {undeclared}, which were never declared. Declare "
                "each with graph.add_resource(celerisroute.Resource(name, upper=...)); a resource "
                "with the default upper=inf is only accumulated and reported"
            )
        declared = list(self._resources.values())
        core = self._builder.compile(
            [r.name for r in declared],
            [r.lower for r in declared],
            [r.upper for r in declared],
            [r.initial for r in declared],
        )
        return Network(core, self._states, tuple(declared))


def _negative_increment_message(name: str, value: float) -> str:
    return (
        f"resource {name!r} is cumulative, so increments must be >= 0; got {value}. "
        "Resources that can decrease along a path are not supported"
    )


def _resolve_vertex(states: StateSpace | None, num_vertices: int, vertex: object) -> int:
    if isinstance(vertex, Integral) and not isinstance(vertex, bool):
        v = int(vertex)
        if not 0 <= v < num_vertices:
            raise InvalidStateError(f"vertex id {v} is outside [0, {num_vertices})")
        return v
    if states is None:
        raise TypeError(
            f"this graph has no StateSpace, so vertices must be integer ids; got {vertex!r}"
        )
    return states.id(vertex)  # type: ignore[arg-type]


class Network:
    """A compiled, immutable graph: the object the solvers run on.

    Obtained from ``Graph.compile``. Topology, costs and resources are
    fixed; all methods are safe to call from several threads at once.
    """

    __slots__ = ("_core", "_states", "_declared", "_columns", "_tracked", "_arrays", "_cycle")

    def __init__(
        self, core: Any, states: StateSpace | None, declared: tuple[Resource, ...]
    ) -> None:
        # Not part of the public API: use Graph.compile().
        self._core = core
        self._states = states
        self._declared = declared
        info = core.resource_info()
        self._columns: tuple[str, ...] = tuple(row[0] for row in info)
        self._tracked: tuple[str, ...] = tuple(row[0] for row in info if row[5])
        self._arrays: dict[str, np.ndarray] = {}
        self._cycle: list[int] | None = None

    # ------------------------------------------------------------ description

    @property
    def states(self) -> StateSpace | None:
        return self._states

    @property
    def num_vertices(self) -> int:
        return self._core.num_vertices

    @property
    def num_edges(self) -> int:
        return self._core.num_edges

    @property
    def is_acyclic(self) -> bool:
        """Whether the graph is a DAG. Only acyclic networks can be solved."""
        return self._core.is_acyclic

    @property
    def resources(self) -> tuple[Resource, ...]:
        """Declared resources."""
        return self._declared

    @property
    def tracked_resources(self) -> tuple[str, ...]:
        """Names of the resources that constrain paths (finite upper bound)."""
        return self._tracked

    def _array(self, key: str) -> np.ndarray:
        array = self._arrays.get(key)
        if array is None:
            array = getattr(self._core, key)()
            array.setflags(write=False)
            self._arrays[key] = array
        return array

    @property
    def edge_source(self) -> npt.NDArray[np.int64]:
        """Source vertex id of every edge, indexed by edge id (read-only)."""
        return self._array("edge_source")

    @property
    def edge_target(self) -> npt.NDArray[np.int64]:
        """Target vertex id of every edge, indexed by edge id (read-only)."""
        return self._array("edge_target")

    @property
    def edge_cost(self) -> npt.NDArray[np.float64]:
        """Cost of every edge, indexed by edge id (read-only)."""
        return self._array("edge_cost")

    @property
    def topological_order(self) -> npt.NDArray[np.int64]:
        """Vertex ids in a topological order (read-only).

        Deterministic: among the vertices that are ready, the one that became
        ready first (ties by smaller id) comes first.
        """
        self._require_acyclic()
        return self._array("topological_order")

    def edge_resource(self, name: str) -> npt.NDArray[np.float64]:
        """Increment of resource ``name`` on every edge (read-only copy)."""
        try:
            column = self._columns.index(name)
        except ValueError:
            raise KeyError(f"unknown resource {name!r}; declared: {list(self._columns)}") from None
        array = self._core.edge_resource(column)
        array.setflags(write=False)
        return array

    def find_cycle(self) -> list[int] | None:
        """One directed cycle as vertex ids ``[v0, ..., vk]`` (``vk -> v0`` closes it), or ``None``."""
        if self.is_acyclic:
            return None
        if self._cycle is None:
            self._cycle = self._core.cycle().tolist()
        return list(self._cycle)

    def node_id(self, vertex: Vertex) -> int:
        """Vertex id of a state tuple/mapping; integer ids are range-checked."""
        return _resolve_vertex(self._states, self.num_vertices, vertex)

    def decode(self, vertex_id: int) -> Any:
        """State of a vertex id (the id itself when the graph has no state space)."""
        v = _resolve_vertex(None, self.num_vertices, operator.index(vertex_id))
        return v if self._states is None else self._states.decode(v)

    def __repr__(self) -> str:
        return (
            f"Network(num_vertices={self.num_vertices}, num_edges={self.num_edges}, "
            f"acyclic={self.is_acyclic}, resources={list(self._columns)})"
        )

    # ------------------------------------------------------------- derivation

    def with_costs(self, cost: npt.ArrayLike) -> Network:
        """A network with the same topology and resources but new edge costs.

        The topology is shared, not copied. For repeated dual updates prefer
        ``celerisroute.PricingEngine``, which updates costs in place.
        """
        values = _as_value_array(cost, self.num_edges, "cost")
        return Network(self._core.with_costs(values), self._states, self._declared)

    # ---------------------------------------------------------------- solving

    def _require_acyclic(self) -> None:
        if not self.is_acyclic:
            cycle = self.find_cycle() or []
            shown = cycle[:8]
            if self._states is not None:
                rendered = " -> ".join(repr(self._states.decode(v)) for v in shown)
            else:
                rendered = " -> ".join(str(v) for v in shown)
            if len(cycle) > len(shown):
                rendered += " -> ..."
            raise CyclicGraphError(
                "the graph contains a directed cycle, but the solvers require an acyclic graph. "
                f"A cycle of length {len(cycle)}: {rendered} -> (back to start). Break cycles by "
                "expanding states with a monotone component such as time or phase.",
                cycle,
            )

    def _resolve_algorithm(self, algorithm: str, unconstrained: bool) -> str:
        """Shared by solve() and the pricing engine: returns "dag" or "rcsp"."""
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"unknown algorithm {algorithm!r}; choose one of {list(_ALGORITHMS)}")
        constrained = bool(self._tracked)
        if algorithm == "auto":
            return "rcsp" if constrained and not unconstrained else "dag"
        if algorithm == "rcsp":
            if unconstrained:
                raise ValueError(
                    "unconstrained=True contradicts algorithm='rcsp'; use algorithm='dag' to "
                    "ignore the resource bounds"
                )
            return "rcsp"
        if constrained and not unconstrained:
            raise ValueError(
                f"algorithm='dag' ignores resource bounds, but this graph constrains "
                f"{list(self._tracked)}. Use algorithm='rcsp' (or 'auto') to respect them, or "
                "pass unconstrained=True to ignore them deliberately"
            )
        return "dag"

    def _resource_dict(self, values: np.ndarray) -> dict[str, float]:
        return dict(zip(self._columns, values.tolist()))

    def solve(
        self,
        source: Vertex,
        target: Vertex,
        algorithm: str = "auto",
        *,
        unconstrained: bool = False,
        dominance_tolerance: float = 0.0,
    ) -> Solution:
        """Minimum-cost path from ``source`` to ``target``.

        Parameters
        ----------
        source, target:
            Vertex ids or states.
        algorithm:
            ``"dag"``: unconstrained shortest path (``boost::dag_shortest_paths``).
            ``"rcsp"``: resource-constrained shortest path
            (``boost::r_c_shortest_paths``); exact, but its work grows with the
            number of nondominated labels.
            ``"auto"`` (default): ``"rcsp"`` if any resource has a finite upper
            bound, else ``"dag"``.
        unconstrained:
            Explicitly ignore resource bounds. Required to run ``"dag"`` on a
            graph with constraining resources; the reported ``resources`` may
            then exceed their bounds.
        dominance_tolerance:
            Slack for RCSP dominance tests. The default 0 is exact. A positive
            value prunes more labels but can discard the optimal path; it
            turns the algorithm into a heuristic.

        Returns
        -------
        Solution
            With ``status`` ``"optimal"``, ``"unreachable"`` (no path at all)
            or ``"infeasible"`` (paths exist but all violate a resource bound).

        Raises
        ------
        CyclicGraphError
            If the graph has a directed cycle.
        """
        name = self._resolve_algorithm(algorithm, unconstrained)
        self._require_acyclic()
        s = self.node_id(source)
        t = self.node_id(target)
        tolerance = float(dominance_tolerance)
        if not tolerance >= 0.0 or math.isinf(tolerance):
            raise ValueError("dominance_tolerance must be a finite number >= 0")
        status, cost, edges, resources, stats = self._core.solve(
            s, t, _ALGORITHM_CODE[name], tolerance
        )
        status = decode_status(status)
        return Solution(
            self,
            s,
            status,
            cost,
            edges,
            self._resource_dict(resources) if resources.size else {},
            SolveStatistics(name, *stats),
        )
