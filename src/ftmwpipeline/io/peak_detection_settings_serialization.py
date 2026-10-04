"""
Persistence for :class:`~ftmwpipeline.core.peak_detection_settings.PeakDetectionSettings`.

The persisted record for a Stage 3 run's resolved knobs lives under
``processing_parameters/stage3_peaks``. The layout uses one HDF5 subgroup
per sub-dataclass so each block is independently inspectable with
``h5dump -p``:

.. code-block::

    processing_parameters/
      stage3_peaks/
        @creation_time
        @preset_name              (optional audit attr)
        promotion/
          @min_snr
          @internal_min_snr
          ...
        savgol/  primary_pass/  gap_pass/
        consumed/                 (what the run took from Stage 2b)
          @tau_basis_us
          @gap_shape
          @tau_basis_source

``consumed`` records the values the gap pass took from another stage's result
(:class:`Stage3Consumed`), not knobs: the settings resolver never reads it, and
re-running Stage 2b does not invalidate Stage 3, so this is the one place that
says which decay time and shape the matched filter actually used. A record
written by ``settings set`` (the sparse user layer) carries no ``consumed``
block; the stage is then not complete, and the next run writes one.

Unset (Optional-None) fields encode as the ``__None__`` sentinel string,
matching :mod:`ftmwpipeline.io.stage_fit_settings_serialization`,
:mod:`ftmwpipeline.io.tau_calibration_settings_serialization`, and
:mod:`ftmwpipeline.io.noise_settings_serialization`.

The path is intentionally distinct from the existing root-level
``/stage3_peaks`` group (the detected peak list -- frequencies,
intensities, SNRs, classifications). Settings (knobs) live here under
``processing_parameters/``; results live at the root. Same pattern
Stages 5 and 2 use (``processing_parameters/stage5_fit`` vs
``/stage5_fitting``, ``processing_parameters/stage2_noise`` vs
``/stage2_noise_result``).

The legacy ``processing_parameters/peak_detection`` group (carrying the
JSON-encoded ``parameters_used`` dict the older Stage 3 impl persisted)
is unrelated to this module; it stays in place as a back-compat shim
and is owned by ``_internal/stage3_impl.save_peak_parameters_impl``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import h5py

from ..core.peak_detection_settings import (
    PeakDetectionSettings,
)
from ..core.peak_detection_settings import from_attrs as peak_from_attrs
from ..core.peak_detection_settings import to_attrs as peak_to_attrs
from ._settings_serialization import (
    decode_attr,
    load_subblock_settings,
    save_settings,
    settings_block_present,
)
from .provenance import RecordProvenance, record_provenance

STAGE3_PEAKS_SETTINGS_PATH = "processing_parameters/stage3_peaks"

#: Field-set version of the ``stage3_peaks`` record this codec writes (see
#: :mod:`ftmwpipeline.io.provenance`). Version 2 added the ``consumed`` block.
STAGE3_PEAKS_FIELD_SET_VERSION = 2

_SUB_NAMES = ("promotion", "savgol", "primary_pass", "gap_pass")

#: The record's subgroup holding :class:`Stage3Consumed`.
_CONSUMED = "consumed"


@dataclass(frozen=True)
class Stage3Consumed:
    """The values a Stage 3 run took from Stage 2b's result.

    Attributes
    ----------
    tau_basis_us : float
        The decay time the gap pass's matched filter was built at (us).
    gap_shape : str
        The matched window's envelope: ``"lorentzian"`` (``exp(-t/tau)``) or
        ``"gaussian"`` (``exp(-(t/tau)^2)``).
    tau_basis_source : str
        Where ``tau_basis_us`` came from: ``"stage2b_tau_G_maj"`` (the Gaussian
        twin, under a Gaussian shape recommendation), ``"stage2b_tau_maj"``
        (the Lorentzian calibration) or ``"default_5us"`` (no calibration).
    """

    tau_basis_us: float
    gap_shape: str
    tau_basis_source: str

    def to_attrs(self) -> Dict[str, Any]:
        """The record's ``consumed`` attrs."""
        return {
            "tau_basis_us": float(self.tau_basis_us),
            "gap_shape": str(self.gap_shape),
            "tau_basis_source": str(self.tau_basis_source),
        }

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any]) -> "Stage3Consumed":
        """Inverse of :meth:`to_attrs`."""
        return cls(
            tau_basis_us=float(attrs["tau_basis_us"]),
            gap_shape=str(decode_attr(attrs["gap_shape"])),
            tau_basis_source=str(decode_attr(attrs["tau_basis_source"])),
        )


def save_peak_detection_settings_to_h5(
    file_path: str,
    settings: PeakDetectionSettings,
    *,
    preset_name: Optional[str] = None,
    consumed: Optional[Stage3Consumed] = None,
) -> None:
    """Persist a resolved :class:`PeakDetectionSettings` to ``processing_parameters/stage3_peaks``.

    Overwrites any prior group at that path. ``preset_name`` (if given) is
    recorded as a top-level attr for audit/reproducibility. ``consumed`` is
    what the run took from Stage 2b; a Stage 3 run always passes it, and only
    the sparse user layer (``settings set``) writes the record without it.
    """
    attrs = peak_to_attrs(settings)
    if consumed is not None:
        attrs[_CONSUMED] = consumed.to_attrs()
    save_settings(
        file_path,
        STAGE3_PEAKS_SETTINGS_PATH,
        attrs,
        field_set_version=STAGE3_PEAKS_FIELD_SET_VERSION,
        preset_name=preset_name,
    )


def load_peak_detection_consumed_from_h5(file_path: str) -> Optional[Stage3Consumed]:
    """The :class:`Stage3Consumed` the last Stage 3 run recorded, or ``None``.

    ``None`` when the file has no ``stage3_peaks`` record or the record has no
    ``consumed`` block (a record written before version 2, or the sparse user
    layer). :func:`peak_detection_settings_provenance` tells those apart.
    """
    with h5py.File(file_path, "r") as h5f:
        group = h5f.get(f"{STAGE3_PEAKS_SETTINGS_PATH}/{_CONSUMED}")
        if not isinstance(group, h5py.Group):
            return None
        return Stage3Consumed.from_attrs(group.attrs)


def load_peak_detection_settings_from_h5(
    file_path: str,
) -> Optional[PeakDetectionSettings]:
    """Return the persisted :class:`PeakDetectionSettings`, or ``None`` if absent.

    Tolerates missing sub-blocks (a partial group still loads); fields not
    present default to ``None``.
    """
    return load_subblock_settings(
        file_path, STAGE3_PEAKS_SETTINGS_PATH, _SUB_NAMES, peak_from_attrs
    )


def peak_detection_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage3_peaks`` settings block?"""
    return settings_block_present(file_path, STAGE3_PEAKS_SETTINGS_PATH)


def peak_detection_settings_provenance(file_path: str) -> Optional[RecordProvenance]:
    """The ``stage3_peaks`` record's field-set version against
    :data:`STAGE3_PEAKS_FIELD_SET_VERSION`, or ``None`` if the record is absent."""
    return record_provenance(
        file_path, STAGE3_PEAKS_SETTINGS_PATH, STAGE3_PEAKS_FIELD_SET_VERSION
    )


__all__ = [
    "STAGE3_PEAKS_SETTINGS_PATH",
    "STAGE3_PEAKS_FIELD_SET_VERSION",
    "Stage3Consumed",
    "load_peak_detection_consumed_from_h5",
    "peak_detection_settings_provenance",
    "save_peak_detection_settings_to_h5",
    "load_peak_detection_settings_from_h5",
    "peak_detection_settings_present",
]
