"""Public vocabulary for a file's frequency-calibration state.

The calibration state is *derived, never stored*: it follows from the clock
declaration (``spur.clocks``) and whether a usable ``timebase_calibration`` is
present, so it can never disagree with what a ``frame="calibrated"`` call will
actually apply and never goes stale the way a persisted copy would. This module
publishes the vocabulary that derivation speaks -- the state strings and the
shape they are returned in -- so an external tool can label an axis, or decide
whether it may claim "calibrated" at all, from the same definition the pipeline
uses.

Reading the state itself is
:func:`ftmwpipeline.api.frequency_calibration` /
:meth:`ftmwpipeline.Pipeline.frequency_calibration` (CLI: ``timebase state``).

This module is intentionally dependency-free within the package (stdlib only),
the same discipline :mod:`ftmwpipeline.core.curation` documents for itself, so
it can be imported anywhere without a cycle.
"""

from dataclasses import dataclass
from typing import Literal, Optional

__all__ = ["CalibrationState", "CalibrationStamp"]

CalibrationState = Literal["rb_locked", "self_calibrated", "uncalibrated"]
"""How absolutely the file's frequency axis is calibrated.

``"rb_locked"``
    No unlocked clock is declared (or nothing is declared at all) -- the
    "assume Rb-locked unless told otherwise" default. The axis is absolutely
    calibrated as acquired and ``epsilon`` is identically zero, a null op.
``"self_calibrated"``
    An unlocked digitizer is declared **and** a ``timebase_calibration`` whose
    preconditions passed is present. The measured ``epsilon`` and its
    uncertainty are applied: the calibrated frame is
    ``f_corr = probe + (f_raw - probe) / (1 + epsilon)`` and
    ``sigma_epsilon * f_baseband`` enters the ``sigma_f`` budget.
``"uncalibrated"``
    An unlocked digitizer is declared but there is no usable self-calibration.
    Frequencies are reported as acquired and caveated; ``epsilon`` is zero
    because none is known, not because none is needed.

Only ``"self_calibrated"`` means the raw and calibrated frames actually differ.
These three strings are a stable public vocabulary: a consumer may pin a parser
on them, and they are the same strings
:class:`~ftmwpipeline.core.data_structures.FinalProducts.calibration_state`
carries.
"""


@dataclass(frozen=True)
class CalibrationStamp:
    """The frequency calibration a file is under *right now*.

    Everything needed to move between the raw and calibrated frames
    (:data:`~ftmwpipeline.core.curation.Frame`) and to reproduce the ``sigma_f``
    budget's calibration terms, read from the file without mutating it and
    without requiring any stage beyond the FID import to have run.

    This is the same six-tuple a freshly built
    :class:`~ftmwpipeline.core.data_structures.FinalProducts` is stamped with;
    comparing a persisted table's stamp against a current one is how a stale
    final-products table is detected. Read it rather than
    ``FinalProducts``' copy when the question is about the *file* -- the table
    describes the build that produced it, exists only once Stage 6 has run, and
    is what goes stale.

    Attributes
    ----------
    state : CalibrationState
        The derived calibration state. See :data:`CalibrationState`.
    epsilon : float
        Fractional timebase scale error that will be applied (``0.0`` unless
        ``state`` is ``"self_calibrated"``).
    sigma_epsilon : float
        1-sigma uncertainty on ``epsilon`` (``0.0`` when no calibration
        uncertainty applies).
    sigma_floor_khz : float
        The user-declared systematic accuracy floor (kHz) folded into every
        peak's budget -- ``0.0`` until ``set_sigma_floor`` declares one.
    probe_freq_mhz : float or None
        Probe/LO frequency (MHz) the calibrated frame is defined against.
        ``None`` only when the file carries no FID acquisition header to read
        it from, in which case no frame conversion is possible.
    sideband : str or None
        Sideband configuration (``"upper"`` / ``"lower"``), from the same
        header, ``None`` on the same terms. Recorded for completeness: the
        ``epsilon`` correction is applied in the baseband frame and is
        sideband-independent.
    """

    state: CalibrationState
    epsilon: float
    sigma_epsilon: float
    sigma_floor_khz: float
    probe_freq_mhz: Optional[float]
    sideband: Optional[str]
