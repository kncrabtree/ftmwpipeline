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

import numpy as np

from ..contract import Absent

__all__ = [
    "clock_lattice_or_absent",
    "degenerate_edge_coherence",
    "degenerate_f_test",
    "degenerate_internal_snr",
    "degenerate_orth_evidence",
    "degenerate_stage3_snr",
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


# ---------------------------------------------------------------------------
# Degenerate statistics (spec: "Degenerate statistics are UNDEFINED")
# ---------------------------------------------------------------------------
# Earlier writers stored some undefined statistics as ordinary numbers. Each
# predicate below recognizes one such stored row from what the file holds, so
# a file written before the writers switched to ``nan`` reads ``UNDEFINED`` as
# well. They take scalars or equal-length arrays and return a bool (array); a
# row they flag is ``UNDEFINED`` and its value reads as ``nan``. Rows written
# since already hold ``nan``, which the ordinary non-finite rule covers.


def degenerate_f_test(
    f_statistic: Any, p_value: Any, chi2_before: Any, chi2_after: Any
) -> Any:
    """An add-loop F-test that was stored as ``F = 0, p = 1`` but never had a value.

    The F-test returned ``(1.0, 0.0)`` both for a genuine non-improvement
    (``chi2_after >= chi2_before``) and for a degenerate test (no residual
    degrees of freedom, a non-positive ``chi2_after``, no added parameter).
    A stored ``F = 0, p = 1`` is therefore degenerate unless it could be a
    genuine non-improvement: ``chi2_after`` not below ``chi2_before`` and
    positive. (A degenerate test that also showed no improvement cannot be
    told apart: the degrees of freedom are not stored. Those rows keep their
    numbers.) Callers exclude rows where the test never ran at all (a
    separation reject).
    """
    f = np.asarray(f_statistic, dtype=float)
    p = np.asarray(p_value, dtype=float)
    before = np.asarray(chi2_before, dtype=float)
    after = np.asarray(chi2_after, dtype=float)
    with np.errstate(invalid="ignore"):
        maybe_no_improvement = (before - after <= 0.0) & (after > 0.0)
    return (f == 0.0) & (p == 1.0) & ~maybe_no_improvement


def degenerate_edge_coherence(value: Any) -> Any:
    """A residual edge coherence stored as exactly ``0.0``.

    ``S_coh = |sum z| / (sigma sqrt(M))`` is exactly zero only when the
    complex sum of the residual band is exactly zero, which a real residual
    never is; earlier writers stored ``0.0`` for an empty residual or a band
    with no positive noise. (The residual itself is not stored, so the
    stored value is the only witness.)
    """
    return np.asarray(value, dtype=float) == 0.0


def degenerate_orth_evidence(value: Any, support_bins: Any) -> Any:
    """A doublet ``orth_evidence_delta_chi2`` of ``0.0`` from a test that never ran.

    ``support_bins`` is ``0`` exactly when the weak partner's template had no
    usable support (or the computation failed), and earlier writers stored
    ``0.0`` evidence there.
    """
    v = np.asarray(value, dtype=float)
    return (v == 0.0) & (np.asarray(support_bins) == 0)


def degenerate_stage3_snr(noise_std_local: Any) -> Any:
    """A Stage 3 peak whose stored local noise is not positive: its SNR is undefined.

    Earlier writers stored ``0.0`` for that SNR. A ``nan`` noise (not
    recorded) does not by itself make the SNR undefined.
    """
    sd = np.asarray(noise_std_local, dtype=float)
    with np.errstate(invalid="ignore"):
        return sd <= 0.0


def degenerate_internal_snr(internal_snr: Any, internal_frequency: Any) -> Any:
    """A Stage 3 internal-pass SNR that was computed but has no value.

    The internal pass contributed the peak (``internal_frequency`` finite)
    and its SNR is ``nan`` (written since) or exactly ``0.0`` (earlier
    writers' value for a non-positive internal noise; the internal noise is
    not stored, and a detected peak's intensity is never exactly zero).
    """
    snr = np.asarray(internal_snr, dtype=float)
    contributed = np.isfinite(np.asarray(internal_frequency, dtype=float))
    return contributed & (np.isnan(snr) | (snr == 0.0))
