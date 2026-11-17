"""Named, multidimensional state spaces with deterministic integer ids.

A ``StateSpace`` is the Cartesian product of named dimensions, e.g.
``day x port x inventory``. Every state is addressed by a dense integer id
(its mixed-radix position, last dimension varying fastest), which is what
the graph and the native solvers work with. The product itself is never
materialised: memory use is proportional to the sum of the dimension sizes.
"""

from __future__ import annotations

import itertools
import math
import operator
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .errors import InvalidStateError

__all__ = ["StateSpace"]

State = tuple[Hashable, ...]

_INT64_MAX = np.iinfo(np.int64).max


def _preview(values: Sequence[Any], limit: int = 6) -> str:
    shown = ", ".join(repr(v) for v in values[:limit])
    if len(values) > limit:
        shown += f", ... ({len(values)} values)"
    return shown


class _Dimension:
    """One axis of a state space: an ordered set of distinct hashable values."""

    __slots__ = ("name", "values", "size", "_lookup", "_range", "_sorted")

    def __init__(self, name: str, values: Iterable[Hashable]) -> None:
        self.name = name
        self._range: range | None = None
        self._lookup: dict[Hashable, int] = {}
        self._sorted: tuple[np.ndarray, np.ndarray] | None | bool = False  # False: not built yet

        if isinstance(values, (str, bytes)):
            raise TypeError(
                f"dimension {name!r} must be an iterable of values, got the single string "
                f"{values!r}; wrap it in a list to make it one value"
            )
        if isinstance(values, range):
            # Ranges are kept symbolic: lookups are arithmetic, not hashed.
            self._range = values
            self.values: Sequence[Hashable] = values
        else:
            if isinstance(values, np.ndarray):
                if values.ndim != 1:
                    raise ValueError(f"dimension {name!r}: an array of values must be 1-D")
                values = values.tolist()  # plain Python scalars decode more naturally
            try:
                self.values = tuple(values)
            except TypeError:
                raise TypeError(
                    f"dimension {name!r} must be an iterable of values, got "
                    f"{type(values).__name__}"
                ) from None
            for position, value in enumerate(self.values):
                try:
                    known = self._lookup.get(value)
                except TypeError:
                    raise TypeError(
                        f"dimension {name!r}: value {value!r} at position {position} is not "
                        "hashable; use tuples instead of lists for compound values"
                    ) from None
                if value != value:  # NaN never compares equal, so it could not be looked up
                    raise ValueError(f"dimension {name!r}: NaN cannot be used as a state value")
                if known is not None:
                    raise ValueError(
                        f"dimension {name!r} contains a duplicate: {value!r} at position "
                        f"{position} equals {self.values[known]!r} at position {known}"
                    )
                self._lookup[value] = position
        self.size = len(self.values)
        if self.size == 0:
            raise ValueError(f"dimension {name!r} is empty; every dimension needs a value")

    def _invalid(self, value: object) -> InvalidStateError:
        return InvalidStateError(
            f"{value!r} is not a value of dimension {self.name!r} "
            f"(valid: {_preview(self.values)})"
        )

    def index(self, value: Hashable) -> int:
        """Position of ``value`` within this dimension."""
        r = self._range
        if r is not None:
            if type(value) is int:
                position, remainder = divmod(value - r.start, r.step)
                if remainder == 0 and 0 <= position < self.size:
                    return position
                raise self._invalid(value)
            try:
                return r.index(value)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                raise self._invalid(value) from None
        try:
            return self._lookup[value]
        except (KeyError, TypeError):
            raise self._invalid(value) from None

    def __contains__(self, value: object) -> bool:
        try:
            self.index(value)  # type: ignore[arg-type]
        except InvalidStateError:
            return False
        return True

    def _sorted_numeric(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(sorted values, their positions) when every value is a real number."""
        if self._sorted is False:
            self._sorted = None
            if all(type(v) in (int, float) for v in self.values):
                try:
                    as_float = np.array(self.values, dtype=np.float64)
                except OverflowError:
                    as_float = None
                # Only use the float view when it represents every value exactly.
                if as_float is not None and all(
                    f == v for f, v in zip(as_float.tolist(), self.values)
                ):
                    order = np.argsort(as_float, kind="stable")
                    self._sorted = (as_float[order], order.astype(np.int64))
        return self._sorted  # type: ignore[return-value]

    def indices(self, column: object) -> np.ndarray:
        """Vectorised ``index``. A single member value yields a 0-d array."""
        if not isinstance(column, (np.ndarray, list)) and column in self:
            return np.asarray(self.index(column), dtype=np.int64)  # type: ignore[arg-type]

        if isinstance(column, np.ndarray) and column.dtype.kind in "iuf":
            r = self._range
            if r is not None and column.dtype.kind in "iu":
                if column.dtype.kind == "u" and column.size and int(column.max()) > _INT64_MAX:
                    raise self._invalid(int(column.max()))
                position, remainder = np.divmod(column.astype(np.int64) - r.start, r.step)
                bad = (remainder != 0) | (position < 0) | (position >= self.size)
                if bad.any():
                    raise self._invalid(column[bad].flat[0].item())
                return position
            lookup = self._sorted_numeric() if r is None else None
            if lookup is not None:
                sorted_values, positions = lookup
                at = np.minimum(np.searchsorted(sorted_values, column), self.size - 1)
                bad = sorted_values[at] != column
                if bad.any():
                    raise self._invalid(column[bad].flat[0].item())
                return positions[at]

        if isinstance(column, np.ndarray):
            shape, flat = column.shape, column.ravel().tolist()
        else:
            try:
                flat = list(column)  # type: ignore[call-overload]
            except TypeError:
                raise self._invalid(column) from None
            shape = (len(flat),)
        result = np.fromiter((self.index(v) for v in flat), dtype=np.int64, count=len(flat))
        return result.reshape(shape)

    def take(self, positions: np.ndarray) -> list[Hashable]:
        r = self._range
        if r is not None:
            return (r.start + r.step * positions).tolist()
        values = self.values
        return [values[i] for i in positions.tolist()]


class StateSpace:
    """Cartesian product of named dimensions with dense integer state ids.

    Dimensions are given as keyword arguments (or as one mapping, for names
    that are not valid identifiers) and keep their insertion order:

        states = StateSpace(
            day=range(1, 36),
            port=["Ras Tanura", "Fujairah", "Dubai", "Doha", "Jubail"],
            inventory=[0.0, 0.25, 0.5, 0.75, 1.0],
        )
        len(states)                                   # 875
        states.id(day=4, port="Dubai", inventory=0.25)
        states.id((4, "Dubai", 0.25))                 # same id
        states.decode(states.id((4, "Dubai", 0.25)))  # (4, 'Dubai', 0.25)

    Values may be any hashable objects (ints, floats, strings, tuples, ...)
    and must be distinct within a dimension under ``==``; note that ``1``,
    ``1.0`` and ``True`` are equal in Python and therefore count as
    duplicates. NaN is rejected.

    The id of a state is its mixed-radix position with the *last* dimension
    varying fastest, identical to ``numpy.ravel_multi_index(indices, shape)``.
    Ids depend only on the order of dimensions and of the values within each
    dimension, so they are stable across runs and processes.
    """

    __slots__ = ("_dimensions", "_by_name", "_strides", "_size")

    def __init__(
        self,
        dimensions: Mapping[str, Iterable[Hashable]] | Iterable[tuple[str, Iterable[Hashable]]] | None = None,
        /,
        **named: Iterable[Hashable],
    ) -> None:
        items: list[tuple[str, Iterable[Hashable]]] = []
        if dimensions is not None:
            items.extend(dimensions.items() if isinstance(dimensions, Mapping) else dimensions)
        items.extend(named.items())
        if not items:
            raise ValueError("a StateSpace needs at least one dimension, e.g. StateSpace(day=range(5))")

        self._dimensions: list[_Dimension] = []
        self._by_name: dict[str, int] = {}
        for name, values in items:
            if not isinstance(name, str) or not name:
                raise TypeError(f"dimension names must be nonempty strings, got {name!r}")
            if name in self._by_name:
                raise ValueError(f"dimension {name!r} is defined more than once")
            self._by_name[name] = len(self._dimensions)
            self._dimensions.append(_Dimension(name, values))

        strides = [1] * len(self._dimensions)
        for i in range(len(self._dimensions) - 2, -1, -1):
            strides[i] = strides[i + 1] * self._dimensions[i + 1].size
        self._strides: tuple[int, ...] = tuple(strides)
        self._size: int = math.prod(d.size for d in self._dimensions)

    # ------------------------------------------------------------ description

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """Dimension names in definition order."""
        return tuple(d.name for d in self._dimensions)

    @property
    def dimensions(self) -> dict[str, Sequence[Hashable]]:
        """Values of every dimension (tuples, or ranges where ranges were given)."""
        return {d.name: d.values for d in self._dimensions}

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of values per dimension."""
        return tuple(d.size for d in self._dimensions)

    @property
    def size(self) -> int:
        """Total number of states (the product of ``shape``)."""
        return self._size

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        dims = ", ".join(f"{d.name}={d.size}" for d in self._dimensions)
        return f"StateSpace({dims}; size={self._size})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateSpace):
            return NotImplemented
        return self.dimension_names == other.dimension_names and all(
            a.size == b.size and all(x == y for x, y in zip(a.values, b.values))
            for a, b in zip(self._dimensions, other._dimensions)
        )

    __hash__ = None  # type: ignore[assignment]

    # ----------------------------------------------------------------- encode

    def _components(self, state: object, named: Mapping[str, Hashable]) -> Sequence[Hashable]:
        if state is not None and named:
            raise TypeError("pass a state either as one tuple or as keyword arguments, not both")
        if state is None and not named:
            raise TypeError(
                "no state given; pass a tuple like states.id((...)) or one keyword per "
                f"dimension ({', '.join(self.dimension_names)})"
            )
        if isinstance(state, Mapping):
            named, state = state, None
        if state is None:
            if len(named) != len(self._dimensions) or any(n not in self._by_name for n in named):
                missing = [n for n in self.dimension_names if n not in named]
                unknown = [n for n in named if n not in self._by_name]
                details = []
                if missing:
                    details.append(f"missing {missing}")
                if unknown:
                    details.append(f"unknown {unknown}")
                raise InvalidStateError(
                    f"a state needs exactly the dimensions {list(self.dimension_names)}: "
                    + ", ".join(details)
                )
            return [named[d.name] for d in self._dimensions]
        if not isinstance(state, (tuple, list)):
            raise TypeError(
                "a state must be a tuple with one component per dimension "
                f"({', '.join(self.dimension_names)}), got {type(state).__name__}"
            )
        if len(state) != len(self._dimensions):
            raise InvalidStateError(
                f"state {state!r} has {len(state)} components, expected {len(self._dimensions)} "
                f"({', '.join(self.dimension_names)})"
            )
        return state

    def id(self, state: State | Mapping[str, Hashable] | None = None, /, **named: Hashable) -> int:
        """Integer id of one state.

        Accepts a tuple in dimension order, a mapping, or keyword arguments.
        Raises ``celerispath.InvalidStateError`` for a state outside the
        space.
        """
        components = self._components(state, named)
        total = 0
        for dimension, stride, value in zip(self._dimensions, self._strides, components):
            total += dimension.index(value) * stride
        return total

    def ids(self, *columns: object, **named: object) -> npt.NDArray[np.int64]:
        """Ids of many states at once, one array (or list) of values per dimension.

        Columns are given in dimension order or by name; a single value is
        broadcast against the others. Integer arrays on ``range`` dimensions
        and numeric arrays on numeric dimensions are translated without a
        Python-level loop:

            states.ids(day=np.arange(1, 35), port="Dubai", inventory=0.5)
        """
        if columns and named:
            raise TypeError("pass the columns either positionally or by name, not both")
        if named:
            if len(named) != len(self._dimensions) or any(n not in self._by_name for n in named):
                raise InvalidStateError(
                    f"ids() needs exactly the dimensions {list(self.dimension_names)}, "
                    f"got {list(named)}"
                )
            columns = tuple(named[d.name] for d in self._dimensions)
        if len(columns) != len(self._dimensions):
            raise InvalidStateError(
                f"ids() needs one column per dimension ({', '.join(self.dimension_names)}), "
                f"got {len(columns)}"
            )
        if self._size > _INT64_MAX:
            raise OverflowError("this state space is too large for 64-bit ids")
        total: np.ndarray | int = 0
        try:
            for dimension, stride, column in zip(self._dimensions, self._strides, columns):
                total = total + dimension.indices(column) * stride
        except ValueError as error:
            if isinstance(error, InvalidStateError):
                raise
            raise ValueError(f"the columns passed to ids() cannot be broadcast together: {error}") from None
        return np.asarray(total, dtype=np.int64)

    def __contains__(self, state: object) -> bool:
        try:
            self.id(state)  # type: ignore[arg-type]
        except (InvalidStateError, TypeError):
            return False
        return True

    # ----------------------------------------------------------------- decode

    def decode(self, state_id: int) -> State:
        """State (tuple of dimension values) with the given id."""
        try:
            remainder = operator.index(state_id)
        except TypeError:
            raise TypeError(f"a state id must be an integer, got {type(state_id).__name__}") from None
        if not 0 <= remainder < self._size:
            raise InvalidStateError(f"state id {remainder} is outside [0, {self._size})")
        state = []
        for dimension, stride in zip(self._dimensions, self._strides):
            position, remainder = divmod(remainder, stride)
            state.append(dimension.values[position])
        return tuple(state)

    def decode_many(self, state_ids: npt.ArrayLike) -> list[State]:
        """States for an array of ids, in the same order."""
        ids = np.asarray(state_ids)
        if ids.size == 0:
            return []
        if ids.dtype.kind not in "iu":
            raise TypeError(f"state ids must be integers, got dtype {ids.dtype}")
        ids = ids.ravel()
        if self._size > _INT64_MAX:
            return [self.decode(i) for i in ids.tolist()]
        low, high = int(ids.min()), int(ids.max())
        if low < 0 or high >= self._size:
            raise InvalidStateError(
                f"state id {low if low < 0 else high} is outside [0, {self._size})"
            )
        positions = np.unravel_index(ids.astype(np.int64), self.shape)
        columns = [d.take(p) for d, p in zip(self._dimensions, positions)]
        return list(zip(*columns))

    def __iter__(self) -> Iterator[State]:
        """Iterate over all states in id order (lazily)."""
        return itertools.product(*(d.values for d in self._dimensions))
