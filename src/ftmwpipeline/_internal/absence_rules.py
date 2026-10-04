"""The rules that turn stored "no value" encodings into :class:`~ftmwpipeline.contract.Absent`.

Normative spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Missing values. Storage keeps
its encodings (``None``, ``nan``, ``-1``, ``""``); conversion happens once, at
the contract boundary, through these helpers, so every surface that exposes the
same quantity (a ``FinalPeak`` field and its ``fit_peaks`` column, for example)
reports the same status.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

from ..contract import Absent

__all__ = [
    "clock_lattice_or_absent",
    "float_or_absent",
    "int_or_absent",
    "knockout_absence",
]


def float_or_absent(
    value: Any, *, when_none: Absent = Absent.UNDEFINED
) -> Union[float, Absent]:
    """A stored float as a contract value.

    ``None`` becomes *when_none* (``UNDEFINED`` unless the caller knows the
    quantity was never computed). A non-finite value is ``UNDEFINED``: it was
    computed and has no finite value.
    """
    if value is None:
        return when_none
    f = float(value)
    return f if math.isfinite(f) else Absent.UNDEFINED


def int_or_absent(
    value: Any, *, sentinel: Optional[int] = -1, absent: Absent = Absent.NOT_RUN
) -> Union[int, Absent]:
    """A stored integer as a contract value: ``None`` or *sentinel* is *absent*."""
    if value is None or (sentinel is not None and int(value) == sentinel):
        return absent
    return int(value)


def knockout_absence(supported: Any, delta_chi2: Any) -> Optional[Absent]:
    """``Absent.NOT_RUN`` when the knockout test never ran for a line, else ``None``.

    The test did not run when there is no knockout record (``supported`` is
    ``None``), the stored tri-state is negative (``-1``), or the stored
    ``delta_chi2`` is ``nan`` -- the loader's own rule
    (:mod:`ftmwpipeline.io.fitting_serialization`). When it ran, each knockout
    field is converted on its own with :func:`float_or_absent` (a refit that did
    not converge leaves ``nan`` or ``inf``: ``UNDEFINED``).
    """
    if supported is None:
        return Absent.NOT_RUN
    if not isinstance(supported, bool) and int(supported) < 0:
        return Absent.NOT_RUN
    if delta_chi2 is not None and math.isnan(float(delta_chi2)):
        return Absent.NOT_RUN
    return None


def clock_lattice_or_absent(value: Any, *, declared: bool) -> Union[str, Absent]:
    """A line's stored clock-lattice identity as a contract value.

    A non-empty identity is present. Otherwise the lattice test never ran when
    the fit recorded no clock declaration (``declared`` false: ``NOT_RUN``),
    and the line is off-lattice when it did (``UNDEFINED``). Storage writes
    ``None`` or ``""`` for both cases; *declared* -- whether the fit's
    recorded ``spur.clocks`` is non-empty -- tells them apart.
    """
    if value is not None and str(value) != "":
        return str(value)
    return Absent.UNDEFINED if declared else Absent.NOT_RUN
