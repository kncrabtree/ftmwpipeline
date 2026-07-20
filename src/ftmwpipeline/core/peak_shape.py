"""
Time-domain envelope shape selector for the Stage 5 fit.

``PeakShape`` lives in ``core`` so settings dataclasses can reference it
without depending on the ``fitting`` package (which would cycle, since
``fitting.peak_model`` imports from ``core.data_structures``).
``fitting.peak_model`` re-exports it for backwards compatibility with the
rest of the package and any external callers.
"""

from __future__ import annotations

from enum import Enum


class PeakShape(str, Enum):
    """Time-domain envelope shape selector for the Stage 5 fit.

    ``LORENTZIAN`` (default) — envelope ``exp(-t/τ)``. The frequency-domain
    line is a finite-T-windowed Lorentzian. This is the historical Stage 5
    model; ``h_T`` / ``h_T_jacobian`` / ``effective_tau`` carry it.

    ``GAUSSIAN`` — envelope ``exp(-(t/τ_G)²)``. The frequency-domain line is
    a finite-T-windowed Gaussian. Motivated by the Voigt-deficit prototype
    on 2638 (per-window joint ``(τ_L, τ_G)`` LSQ degenerated to Gaussian-
    dominant with ``τ_L`` pinning at the upper bound), shipping as an
    alternative when the data is supersonic-beam-geometry-shaped rather
    than collisional-Lorentzian-shaped.

    A future ``VOIGT`` member is anticipated by the settings layer
    (``ShapeSpec`` discriminator); shipping Voigt gates on longer-T fixture
    data that can discriminate τ_L from τ_G (current 2638 cannot).
    """

    LORENTZIAN = "lorentzian"
    GAUSSIAN = "gaussian"

    @classmethod
    def coerce(cls, value: "PeakShape | str") -> "PeakShape":
        """Convert a shape-like value into a :class:`PeakShape`.

        Accepts the enum itself or one of the string members
        (case-insensitive). Useful at API boundaries where callers may
        pass either ``PeakShape.GAUSSIAN`` or ``"gaussian"``.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError(
            f"shape must be a PeakShape or one of "
            f"{[s.value for s in cls]}; got {value!r}"
        )
