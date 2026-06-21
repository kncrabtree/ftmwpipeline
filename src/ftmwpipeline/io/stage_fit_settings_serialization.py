"""
Persistence for :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`.

The canonical record for a fit's resolved knobs lives under
``processing_parameters/stage5_fit`` (mirroring
``processing_parameters/ft_processing`` for Stage 1). The layout uses one
HDF5 subgroup per sub-dataclass so each block is independently inspectable
with ``h5dump -p`` and so future shape-specific parameter blocks
(``shape/voigt_params`` etc.) can attach without touching siblings:

.. code-block::

    processing_parameters/
      stage5_fit/
        @creation_time
        @preset_name              (optional audit attr)
        shape/
          @kind                   ("lorentzian" | "gaussian")
        tau/
          @max_decay_factor
          @tau_penalty_lambda
          ...
        seeder/
          @seeder_rchi2
          ...
        conservative/  penalties/  rescue/  thaw/  ...

Unset (Optional-None) fields encode as the ``__None__`` sentinel string,
matching :mod:`ftmwpipeline.io.fid_serialization` and
:class:`~ftmwpipeline.core.settings.FTSettings`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import h5py

from ..core.data_structures import ChirpWindow
from ..core.stage_fit_settings import (
    _SUB_NAMES,
    ClockSource,
    StageFitSettings,
    coerce_clock_sources,
)
from ..core.stage_fit_settings import from_attrs as stage_fit_from_attrs
from ..core.stage_fit_settings import to_attrs as stage_fit_to_attrs

logger = logging.getLogger(__name__)

STAGE_FIT_PATH = "processing_parameters/stage5_fit"

# Top-level audit attributes (always carried as-is, not part of to_attrs).
_AUDIT_ATTRS = ("creation_time", "preset_name")

# The Stage 2b recommended-shape attr lives on the Stage 2b calibration
# group(s). Both the Lorentzian-twin (``stage2b_tau_calibration``) and
# the Gaussian-twin (``stage2b_tau_G_calibration``) groups can carry the
# attr; the recommendation is shape-agnostic so the writer mirrors the
# same value to whichever groups exist and the reader takes the first
# concrete value it finds.
_STAGE2B_RECOMMENDED_SHAPE_ATTR = "recommended_shape"
_STAGE2B_VOTE_RATES_ATTR = "shape_vote_rates"
_STAGE2B_GROUP_PATHS = (
    "stage2b_tau_calibration",
    "stage2b_tau_G_calibration",
)
_NONE_SENTINEL = "__None__"


def _decode_attr(value: Any) -> Any:
    """Decode an HDF5 attribute value (handles bytes -> str)."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def save_stage_fit_settings_to_h5(
    file_path: str,
    settings: StageFitSettings,
    *,
    preset_name: Optional[str] = None,
) -> None:
    """Persist a resolved :class:`StageFitSettings` to ``processing_parameters/stage5_fit``.

    Overwrites any prior group at that path. ``preset_name`` (if given) is
    recorded as a top-level attr for audit/reproducibility — useful when a
    fit was driven by a named preset.
    """
    attrs = stage_fit_to_attrs(settings)
    with h5py.File(file_path, "a") as h5f:
        if STAGE_FIT_PATH in h5f:
            del h5f[STAGE_FIT_PATH]
        grp = h5f.create_group(STAGE_FIT_PATH)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        if preset_name is not None:
            grp.attrs["preset_name"] = preset_name
        # Shape: a subgroup carrying the discriminator + any future
        # shape-specific parameter blocks. When unset, write the sentinel
        # as a top-level attr (no subgroup).
        shape_val = attrs["shape"]
        if isinstance(shape_val, dict):
            shape_grp = grp.create_group("shape")
            for k, v in shape_val.items():
                shape_grp.attrs[k] = v
        else:
            grp.attrs["shape"] = shape_val
        # One subgroup per sub-dataclass; attrs hold field values or the
        # ``__None__`` sentinel.
        for sub_name, sub_attrs in attrs.items():
            if sub_name == "shape":
                continue
            sub_grp = grp.create_group(sub_name)
            for field_name, value in sub_attrs.items():
                sub_grp.attrs[field_name] = value


def load_stage_fit_settings_from_h5(file_path: str) -> Optional[StageFitSettings]:
    """Return the persisted :class:`StageFitSettings`, or ``None`` if absent.

    Tolerates missing sub-blocks (a partial group still loads); fields
    not present default to ``None``.
    """
    with h5py.File(file_path, "r") as h5f:
        if STAGE_FIT_PATH not in h5f:
            return None
        grp = h5f[STAGE_FIT_PATH]
        attrs_dict: Dict[str, Any] = {}
        # Shape — prefer the subgroup form; fall back to the top-level attr.
        if "shape" in grp and isinstance(grp["shape"], h5py.Group):
            shape_grp = grp["shape"]
            attrs_dict["shape"] = {
                k: _decode_attr(v) for k, v in shape_grp.attrs.items()
            }
        elif "shape" in grp.attrs:
            attrs_dict["shape"] = _decode_attr(grp.attrs["shape"])
        else:
            attrs_dict["shape"] = _NONE_SENTINEL
        # Sub-dataclass subgroups (canonical list, so a newly added block
        # such as ``spur`` / ``baseline`` round-trips without a second edit).
        for sub_name in _SUB_NAMES:
            if sub_name in grp and isinstance(grp[sub_name], h5py.Group):
                sub_grp = grp[sub_name]
                attrs_dict[sub_name] = {
                    k: _decode_attr(v) for k, v in sub_grp.attrs.items()
                }
            else:
                attrs_dict[sub_name] = {}
    return stage_fit_from_attrs(attrs_dict)


def stage_fit_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage5_fit`` block?"""
    try:
        with h5py.File(file_path, "r") as h5f:
            return STAGE_FIT_PATH in h5f
    except (OSError, KeyError):
        return False


def write_stage2b_recommended_shape(
    file_path: str,
    shape: Optional[str] = None,
    vote_rates: Optional[Dict[str, float]] = None,
) -> None:
    """Stamp the ``recommended_shape`` attr on every persisted Stage 2b group.

    The attr is the contract the Stage 5 resolver reads as its
    *recommended* layer; passing ``shape=None`` (the default) writes the
    ``__None__`` sentinel, which the resolver treats as "no
    recommendation." Concrete shape recommendations come from the
    Stage 2b 3-way L/G/V discriminator
    (:func:`~ftmwpipeline.fitting.tau_calibration.compute_shape_recommendation`)
    and are passed through this same attr.

    ``vote_rates`` carries the discriminator's per-model SNR-weighted vote
    fractions (keys ``"exp"``/``"gauss"``/``"voigt"``); they are stored as a
    JSON ``shape_vote_rates`` attr beside the verdict so the report can render
    the vote breakdown without recomputing it. Passing ``vote_rates=None``
    clears any stale breakdown, keeping it consistent with a reset verdict.

    The Lorentzian-twin (``stage2b_tau_calibration``) and Gaussian-twin
    (``stage2b_tau_G_calibration``) groups can each carry the attr; the
    recommendation is shape-agnostic so the same value is mirrored to
    whichever groups exist. No-op if neither group is present.
    """
    encoded = _NONE_SENTINEL if shape is None else str(shape)
    encoded_votes = None if vote_rates is None else json.dumps(dict(vote_rates))
    with h5py.File(file_path, "a") as h5f:
        for path in _STAGE2B_GROUP_PATHS:
            if path not in h5f:
                continue
            attrs = h5f[path].attrs
            attrs[_STAGE2B_RECOMMENDED_SHAPE_ATTR] = encoded
            if encoded_votes is None:
                attrs.pop(_STAGE2B_VOTE_RATES_ATTR, None)
            else:
                attrs[_STAGE2B_VOTE_RATES_ATTR] = encoded_votes


def read_stage2b_recommended_shape(file_path: str) -> Optional[str]:
    """Read the persisted Stage 2b recommended shape, or ``None`` if absent.

    Returns ``None`` for "no Stage 2b group present", "the attr is
    missing", or "the attr is the ``__None__`` sentinel". Callers
    don't need to distinguish these because the resolver treats them all
    as "no recommended layer." The Lorentzian-twin group is checked
    first; if it is absent or has no concrete recommendation the
    Gaussian-twin group is consulted.
    """
    try:
        with h5py.File(file_path, "r") as h5f:
            for path in _STAGE2B_GROUP_PATHS:
                if path not in h5f:
                    continue
                attr = h5f[path].attrs.get(_STAGE2B_RECOMMENDED_SHAPE_ATTR)
                if attr is None:
                    continue
                decoded = _decode_attr(attr)
                if decoded == _NONE_SENTINEL:
                    continue
                return str(decoded)
    except (OSError, KeyError):
        return None
    return None


def read_stage2b_vote_rates(file_path: str) -> Dict[str, float]:
    """Read the persisted Stage 2b shape-vote breakdown, or ``{}`` if absent.

    Returns the SNR-weighted per-model vote fractions
    (keys ``"exp"``/``"gauss"``/``"voigt"``) stamped beside the verdict, or an
    empty mapping when no Stage 2b group carries the attr. The Lorentzian-twin
    group is checked first, then the Gaussian twin.
    """
    try:
        with h5py.File(file_path, "r") as h5f:
            for path in _STAGE2B_GROUP_PATHS:
                if path not in h5f:
                    continue
                attr = h5f[path].attrs.get(_STAGE2B_VOTE_RATES_ATTR)
                if attr is None:
                    continue
                decoded = _decode_attr(attr)
                rates = json.loads(decoded)
                return {str(k): float(v) for k, v in rates.items()}
    except (OSError, KeyError, ValueError):
        return {}
    return {}


# The recommended clock declaration is stored as a JSON attr on the Stage 0
# FID group.  This is the natural home: the clock tree is instrument metadata
# that arrives at import time (Stage 0), before any Stage 5 processing, and
# it is analogous to the Stage 1 ``recommended_processing`` attr stored in the
# same group.  Storing it here (rather than alongside the Stage 2b shape
# recommendation) keeps source-derived recommendations co-located with the
# Stage 0 data they describe.
_STAGE0_GROUP = "stage0_fid_data"
_RECOMMENDED_CLOCKS_ATTR = "recommended_clock_sources"


def write_recommended_clock_sources(
    file_path: str,
    clock_sources: Optional[Tuple[ClockSource, ...]],
) -> None:
    """Persist the import-time recommended clock declaration on the Stage 0 group.

    Stores a JSON-encoded list of clock-source dicts as an attr on
    ``stage0_fid_data``.  Passing ``None`` writes the ``__None__`` sentinel
    (resolver reads as "no recommendation").  No-op when the Stage 0 group is
    absent (tolerant for unit tests against bare HDF5 files).

    Re-import over the same file overwrites the attr cleanly.
    """
    encoded: str
    if clock_sources is None:
        encoded = _NONE_SENTINEL
    else:
        encoded = json.dumps([c.to_dict() for c in clock_sources])
    try:
        with h5py.File(file_path, "a") as h5f:
            if _STAGE0_GROUP not in h5f:
                return
            h5f[_STAGE0_GROUP].attrs[_RECOMMENDED_CLOCKS_ATTR] = encoded
    except (OSError, KeyError):
        logger.warning("Could not write recommended clock sources to %s", file_path)


def read_recommended_clock_sources(
    file_path: str,
) -> Optional[Tuple[ClockSource, ...]]:
    """Read the import-time recommended clock declaration, or ``None`` if absent.

    Returns ``None`` for: Stage 0 group absent, attr missing, ``__None__``
    sentinel, or any parse error.  Callers treat all of these as "no
    recommendation."
    """
    try:
        with h5py.File(file_path, "r") as h5f:
            if _STAGE0_GROUP not in h5f:
                return None
            attr = h5f[_STAGE0_GROUP].attrs.get(_RECOMMENDED_CLOCKS_ATTR)
            if attr is None:
                return None
            decoded = _decode_attr(attr)
            if decoded == _NONE_SENTINEL:
                return None
            return coerce_clock_sources(decoded)
    except (OSError, KeyError, ValueError):
        return None


# The recommended chirp-window declaration is stored as a JSON attr on the
# Stage 0 FID group alongside the clock-sources attr.  It carries the
# instrument-declared chirp timing so the start detector can demote its
# sweep to a cross-check.  The same ``__None__`` sentinel and tolerant
# error-handling conventions as ``recommended_clock_sources`` apply.
_RECOMMENDED_CHIRP_WINDOW_ATTR = "recommended_chirp_window"


def write_recommended_chirp_window(
    file_path: str,
    chirp_window: Optional[ChirpWindow],
) -> None:
    """Persist the import-time recommended chirp-window declaration.

    Stores a JSON-encoded dict as an attr on ``stage0_fid_data``.  Passing
    ``None`` writes the ``__None__`` sentinel.  No-op when the Stage 0 group
    is absent.  Re-import over the same file overwrites the attr cleanly.
    """
    if chirp_window is None:
        encoded: str = _NONE_SENTINEL
    else:
        encoded = json.dumps(
            {
                "chirp_end_us": chirp_window.chirp_end_us,
                "chirp_start_us": (
                    chirp_window.chirp_start_us
                    if chirp_window.chirp_start_us is not None
                    else _NONE_SENTINEL
                ),
                "start_margin_us": (
                    chirp_window.start_margin_us
                    if chirp_window.start_margin_us is not None
                    else _NONE_SENTINEL
                ),
            }
        )
    try:
        with h5py.File(file_path, "a") as h5f:
            if _STAGE0_GROUP not in h5f:
                return
            h5f[_STAGE0_GROUP].attrs[_RECOMMENDED_CHIRP_WINDOW_ATTR] = encoded
    except (OSError, KeyError):
        logger.warning("Could not write recommended chirp window to %s", file_path)


def read_recommended_chirp_window(
    file_path: str,
) -> Optional[ChirpWindow]:
    """Read the import-time recommended chirp-window declaration, or ``None``.

    Returns ``None`` for: Stage 0 group absent, attr missing, ``__None__``
    sentinel, or any parse error.
    """
    try:
        with h5py.File(file_path, "r") as h5f:
            if _STAGE0_GROUP not in h5f:
                return None
            attr = h5f[_STAGE0_GROUP].attrs.get(_RECOMMENDED_CHIRP_WINDOW_ATTR)
            if attr is None:
                return None
            decoded = _decode_attr(attr)
            if decoded == _NONE_SENTINEL:
                return None
            d = json.loads(decoded)
            chirp_end_us = float(d["chirp_end_us"])
            raw_start = d.get("chirp_start_us")
            chirp_start_us: Optional[float] = (
                None
                if raw_start is None or raw_start == _NONE_SENTINEL
                else float(raw_start)
            )
            raw_margin = d.get("start_margin_us")
            start_margin_us: Optional[float] = (
                None
                if raw_margin is None or raw_margin == _NONE_SENTINEL
                else float(raw_margin)
            )
            return ChirpWindow(
                chirp_end_us=chirp_end_us,
                chirp_start_us=chirp_start_us,
                start_margin_us=start_margin_us,
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


__all__ = [
    "STAGE_FIT_PATH",
    "save_stage_fit_settings_to_h5",
    "load_stage_fit_settings_from_h5",
    "stage_fit_settings_present",
    "write_stage2b_recommended_shape",
    "read_stage2b_recommended_shape",
    "read_stage2b_vote_rates",
    "write_recommended_clock_sources",
    "read_recommended_clock_sources",
    "write_recommended_chirp_window",
    "read_recommended_chirp_window",
]
