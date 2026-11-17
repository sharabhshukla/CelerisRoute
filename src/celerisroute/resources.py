"""Resource declarations for resource-constrained shortest paths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

__all__ = ["Resource"]

_SUPPORTED_SENSES = ("cumulative",)


@dataclass(frozen=True, slots=True)
class Resource:
    """A quantity accumulated along a path and kept within ``[lower, upper]``.

    Version 1 supports one kind of resource, ``sense="cumulative"``:

    * a path starts with ``initial`` units;
    * traversing an edge adds that edge's increment (``resources={name: x}``
      in ``Graph.add_edge``), which must be **nonnegative**;
    * a path is feasible only if the running total stays ``<= upper`` after
      every edge.

    Because increments are nonnegative and ``lower <= initial`` is required,
    the lower bound can never be violated; it is accepted for symmetry and
    validated, but a *minimum consumption* requirement (``lower > initial``)
    is not expressible in this version.

    A resource with ``upper=inf`` is accumulated and reported on solutions
    but does not constrain anything and adds no work to the solver.

    Comparisons are exact: a path whose total is ``upper`` is feasible, one
    whose floating-point total exceeds it by one ulp is not. If increments
    are not exactly representable (e.g. multiples of 0.1) and paths may hit
    the limit exactly, add a small slack to ``upper`` or scale to integers.

    Other senses (replenishable, time windows, ...) raise
    ``NotImplementedError``.
    """

    name: str
    lower: float = 0.0
    upper: float = math.inf
    initial: float = 0.0
    sense: str = "cumulative"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"a resource name must be a nonempty string, got {self.name!r}")
        if not isinstance(self.sense, str):
            raise TypeError(f"sense must be a string, got {self.sense!r}")
        if self.sense not in _SUPPORTED_SENSES:
            raise NotImplementedError(
                f"resource {self.name!r}: sense={self.sense!r} is not supported; this version "
                f"implements only {list(_SUPPORTED_SENSES)} (additive, nonnegative increments, "
                "upper-bounded)"
            )
        for field in ("lower", "upper", "initial"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    f"resource {self.name!r}: {field} must be a real number, got {value!r}"
                )
            value = float(value)
            if math.isnan(value):
                raise ValueError(f"resource {self.name!r}: {field} must not be NaN")
            object.__setattr__(self, field, value)
        if math.isinf(self.initial):
            raise ValueError(f"resource {self.name!r}: initial must be finite")
        if self.lower > self.upper:
            raise ValueError(
                f"resource {self.name!r}: lower ({self.lower}) exceeds upper ({self.upper})"
            )
        if not self.lower <= self.initial <= self.upper:
            raise ValueError(
                f"resource {self.name!r}: initial ({self.initial}) must lie within "
                f"[lower, upper] = [{self.lower}, {self.upper}]"
            )

    @property
    def is_constraining(self) -> bool:
        """True when the resource has a finite upper bound, i.e. limits paths."""
        return math.isfinite(self.upper)
