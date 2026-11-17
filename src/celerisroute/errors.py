"""Exception types raised by celerisroute.

Each one also derives from the built-in exception a caller would naturally
catch (``ValueError``, ``RuntimeError``), so ``except ValueError`` keeps
working while ``except celerisroute.CelerisRouteError`` catches everything specific
to this library.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["BatchPricingError", "CyclicGraphError", "InvalidStateError", "CelerisRouteError"]


class CelerisRouteError(Exception):
    """Base class of all celerisroute-specific errors."""


class InvalidStateError(CelerisRouteError, ValueError):
    """A state (or vertex id) does not belong to the state space or graph."""


class CyclicGraphError(CelerisRouteError, ValueError):
    """A solver was asked to run on a graph that contains a directed cycle.

    ``cycle`` holds one witness cycle as a list of vertex ids
    ``[v0, v1, ..., vk]`` with edges ``v0 -> v1 -> ... -> vk -> v0``.
    """

    def __init__(self, message: str, cycle: list[int] | None = None) -> None:
        super().__init__(message)
        self.cycle: list[int] = list(cycle or [])


class BatchPricingError(CelerisRouteError, RuntimeError):
    """One or more engines of a batch failed.

    ``errors`` maps the index of every failing engine (its position in the
    ``engines`` list) to its error message; ``engine_index`` is the first one.
    """

    def __init__(self, message: str, errors: Mapping[int, str]) -> None:
        super().__init__(message)
        self.errors: dict[int, str] = dict(errors)
        self.engine_index: int = min(self.errors) if self.errors else -1
