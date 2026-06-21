"""Resolve the worker-pool size for the parallel pipeline stages."""

from __future__ import annotations

import os
from typing import Optional

__all__ = ["ENV_MAX_WORKERS", "resolve_worker_count"]

ENV_MAX_WORKERS = "FTMW_MAX_WORKERS"


def resolve_worker_count(
    explicit: Optional[int] = None, *, override: Optional[int] = None
) -> int:
    """Resolve the parallel worker count for a pipeline stage.

    Precedence: ``override`` (an internal test pin) > ``explicit`` (a
    user-supplied ``jobs`` / ``--jobs`` value) > the ``FTMW_MAX_WORKERS``
    environment variable > the default ``cpu_count() - 2``. The result is
    clamped to at least 1; a malformed environment value is ignored.
    """
    if override is not None:
        return max(1, int(override))
    if explicit is not None:
        return max(1, int(explicit))
    env = os.environ.get(ENV_MAX_WORKERS)
    if env is not None and env.strip():
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, (os.cpu_count() or 2) - 2)
