"""
Persistence for :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`.

The persisted record for a fit's resolved knobs lives under
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
        consumed/                 (what the fit took from other stages)
          @tau_calibration_source
          @tau_maj_us  @sigma_tau_us
          @band_majorities        (JSON; the per-band table routed, or None)
          @timebase_epsilon  @timebase_sigma_epsilon
          @peak_survival_snr_floor

``consumed`` (:class:`Stage5Consumed`) holds the values the fit took from
another stage's result rather than from its knobs: the decay-time anchor from
Stage 2b, the timebase epsilon its spur window used, and the survival floor it
derived from the Stage 3 promotion cutoff. Neither a Stage 2b nor a timebase
re-run invalidates Stage 5, so this is what says what the fit used. The
settings resolver never reads it. A record written by ``settings set`` (the
sparse user layer) carries no ``consumed`` block; the stage is then not
complete, and the next fit writes one.

Unset (Optional-None) fields encode as the ``__None__`` sentinel string,
matching :mod:`ftmwpipeline.io.fid_serialization` and
:class:`~ftmwpipeline.core.settings.FTSettings`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import h5py

from .._internal.atomic import h5open
from .._internal.shared_utils import active_acquisition_us
from ..core.data_structures import ChirpWindow
from ..core.stage_fit_settings import (
    _SUB_NAMES,
    ClockSource,
    StageFitSettings,
    coerce_clock_sources,
)
from ..core.stage_fit_settings import from_attrs as stage_fit_from_attrs
from ..core.stage_fit_settings import to_attrs as stage_fit_to_attrs
from ..core.start_detection_settings import StartDetectionSettings
from ..fitting.active_ft import active_ft_bin_spacing_mhz
from ..fitting.tau_calibration import BandMajority
from ..preprocessing.start_detection import StartDetectionRecord, StartDetectionResult
from ._settings_serialization import (
    decode_attr,
    load_subblock_settings,
    save_settings,
    settings_block_present,
)
from .provenance import RecordProvenance, record_provenance

logger = logging.getLogger(__name__)

STAGE_FIT_PATH = "processing_parameters/stage5_fit"

#: Field-set version of the ``stage5_fit`` record this codec writes (see
#: :mod:`ftmwpipeline.io.provenance`). Version 2 added the ``consumed`` block
#: and made ``tau.fit_tau`` concrete (``True`` is its hard default).
STAGE5_FIT_FIELD_SET_VERSION = 2

#: The record's subgroup holding :class:`Stage5Consumed`.
_CONSUMED = "consumed"

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


@dataclass(frozen=True)
class Stage5Consumed:
    """The values a Stage 5 fit took from other stages' results.

    Attributes
    ----------
    tau_calibration_source : str
        Which band-wide decay-time anchor drove the fit: ``"override"`` (the
        ``tau.tau_maj_override_us`` / ``tau.sigma_tau_override_us`` pair),
        ``"persisted"`` (the Stage 2b calibration of the fit's shape) or
        ``"none"`` (no anchor).
    tau_maj_us, sigma_tau_us : float or None
        That band-wide anchor (us): the anchor of every window per-band
        routing did not reach. ``None`` when ``tau_calibration_source`` is
        ``"none"``.
    band_majorities : tuple of BandMajority or None
        The Stage 2b per-band table the fit routed each window's anchor
        through (:func:`~ftmwpipeline._internal.stage5_impl.resolve_window_tau_anchor`),
        or ``None`` when per-band routing was not used (switched off, an
        override pair, or no per-band table).
    timebase_epsilon, timebase_sigma_epsilon : float or None
        The timebase scale error and its 1-sigma uncertainty the spur gate's
        epsilon-aware match window used, or ``None`` when it used none (no
        clock declaration, spur gating off, or no timebase calibration).
    peak_survival_snr_floor : float
        The effective peak-survival SNR floor: ``peak_survival.snr_survival_floor``
        when set, otherwise the Stage 3 promotion cutoff times
        ``peak_survival.snr_survival_factor``.
    stft_spur_nominees : tuple of (float, bool) or None
        The Stage 2b spur clusters the spur gate took as nominees, as
        ``(center_freq_mhz, saturated)`` -- the only cluster fields the gate
        reads. Empty when the catalog was consulted but held none (or there
        was no calibration); ``None`` when the gate did not consult it (spur
        gating off or ``spur.use_stft_catalog`` false).
    """

    tau_calibration_source: str
    tau_maj_us: Optional[float]
    sigma_tau_us: Optional[float]
    band_majorities: Optional[Tuple[BandMajority, ...]]
    timebase_epsilon: Optional[float]
    timebase_sigma_epsilon: Optional[float]
    peak_survival_snr_floor: float
    stft_spur_nominees: Optional[Tuple[Tuple[float, bool], ...]] = None

    def to_attrs(self) -> Dict[str, Any]:
        """The record's ``consumed`` attrs (``None`` as ``__None__``)."""

        def _opt(value: Optional[float]) -> Any:
            return _NONE_SENTINEL if value is None else float(value)

        bands: Any = _NONE_SENTINEL
        if self.band_majorities is not None:
            bands = json.dumps(
                [
                    {
                        "label": str(b.label),
                        "freq_lo_mhz": float(b.freq_lo_mhz),
                        "freq_hi_mhz": float(b.freq_hi_mhz),
                        "n": int(b.n),
                        "tau_maj_us": float(b.tau_maj_us),
                        "sigma_tau_us": float(b.sigma_tau_us),
                    }
                    for b in self.band_majorities
                ]
            )
        return {
            "tau_calibration_source": str(self.tau_calibration_source),
            "tau_maj_us": _opt(self.tau_maj_us),
            "sigma_tau_us": _opt(self.sigma_tau_us),
            "band_majorities": bands,
            "timebase_epsilon": _opt(self.timebase_epsilon),
            "timebase_sigma_epsilon": _opt(self.timebase_sigma_epsilon),
            "peak_survival_snr_floor": float(self.peak_survival_snr_floor),
            "stft_spur_nominees": (
                _NONE_SENTINEL
                if self.stft_spur_nominees is None
                else json.dumps(
                    [[float(f), bool(sat)] for f, sat in self.stft_spur_nominees]
                )
            ),
        }

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any]) -> "Stage5Consumed":
        """Inverse of :meth:`to_attrs`."""

        def _opt(key: str) -> Optional[float]:
            value = decode_attr(attrs[key])
            return None if value == _NONE_SENTINEL else float(value)

        raw_bands = decode_attr(attrs["band_majorities"])
        raw_nominees = decode_attr(attrs["stft_spur_nominees"])
        bands: Optional[Tuple[BandMajority, ...]] = None
        if raw_bands != _NONE_SENTINEL:
            bands = tuple(
                BandMajority(
                    label=str(d["label"]),
                    freq_lo_mhz=float(d["freq_lo_mhz"]),
                    freq_hi_mhz=float(d["freq_hi_mhz"]),
                    n=int(d["n"]),
                    tau_maj_us=float(d["tau_maj_us"]),
                    sigma_tau_us=float(d["sigma_tau_us"]),
                )
                for d in json.loads(raw_bands)
            )
        return cls(
            tau_calibration_source=str(decode_attr(attrs["tau_calibration_source"])),
            tau_maj_us=_opt("tau_maj_us"),
            sigma_tau_us=_opt("sigma_tau_us"),
            band_majorities=bands,
            timebase_epsilon=_opt("timebase_epsilon"),
            timebase_sigma_epsilon=_opt("timebase_sigma_epsilon"),
            peak_survival_snr_floor=float(attrs["peak_survival_snr_floor"]),
            stft_spur_nominees=(
                None
                if raw_nominees == _NONE_SENTINEL
                else tuple((float(f), bool(sat)) for f, sat in json.loads(raw_nominees))
            ),
        )


def save_stage_fit_settings_to_h5(
    file_path: str,
    settings: StageFitSettings,
    *,
    preset_name: Optional[str] = None,
    consumed: Optional[Stage5Consumed] = None,
) -> None:
    """Persist a resolved :class:`StageFitSettings` to ``processing_parameters/stage5_fit``.

    Overwrites any prior group at that path. ``preset_name`` (if given) is
    recorded as a top-level attr for audit/reproducibility — useful when a
    fit was driven by a named preset. The set ``shape`` is a ``{"kind": ...}``
    subgroup (room for future shape-specific blocks); an unset shape is the
    ``__None__`` sentinel top-level attr -- both fall out of the generic
    dict-value-becomes-subgroup rule in ``save_settings``. ``consumed`` is what
    the fit took from other stages; a Stage 5 fit always passes it, and only
    the sparse user layer (``settings set``) writes the record without it.
    """
    attrs = stage_fit_to_attrs(settings)
    if consumed is not None:
        attrs[_CONSUMED] = consumed.to_attrs()
    save_settings(
        file_path,
        STAGE_FIT_PATH,
        attrs,
        field_set_version=STAGE5_FIT_FIELD_SET_VERSION,
        preset_name=preset_name,
    )


def _read_shape(grp: h5py.Group, attrs_dict: Dict[str, Any]) -> None:
    """Pull the ``shape`` discriminator off the group before the sub-blocks.

    Prefers the subgroup form (the set case), falls back to the top-level attr
    (the ``__None__`` sentinel), else stamps the sentinel so a missing shape
    decodes as unset.
    """
    if "shape" in grp and isinstance(grp["shape"], h5py.Group):
        attrs_dict["shape"] = {k: decode_attr(v) for k, v in grp["shape"].attrs.items()}
    elif "shape" in grp.attrs:
        attrs_dict["shape"] = decode_attr(grp.attrs["shape"])
    else:
        attrs_dict["shape"] = _NONE_SENTINEL


#: The legacy absolute spelling of ``spur.integer_tol_bins``, migrated at read
#: time (see :func:`_migrate_integer_tol`).
_LEGACY_INTEGER_TOL_ATTR = "integer_tol_mhz"


def declared_active_acquisition_us(h5f: h5py.File) -> Optional[float]:
    """The file's own *declared* active-region length ``end_us - start_us`` (us).

    Read from the persisted Stage 1 window plus the FID duration, through the
    one helper every stage uses, so the spacing a migrated knob is converted
    against is the spacing the fit will actually run at. ``None`` when the
    file does not carry enough to say.

    Public within the package (rather than a private helper of the knob
    migration it was written for) because it is the pre-Stage-5 half of every
    "what bin spacing does this file resolve against" question: the Stage 6
    snap-tolerance accessor falls back to it on a file that has not been
    fitted yet. Two readers of the declared active region would be two answers
    (``dev-docs/SCIENCE_STRATEGY.md`` Requirement 8).
    """
    from .._internal.stage1_impl import resolve_ft_settings_h5

    fid = h5f.get("stage0_fid_data/acquisition")
    if fid is None or "duration_us" not in fid.attrs:
        return None
    # The window comes from the one Stage 1 resolver, so this length is the
    # one the FT is built on. No ``ft_processing`` at all is not "cannot say":
    # it is a file that has persisted no window, which resolves to the
    # recommended window or, with none, both bounds unset -- the whole record.
    # Answering here is what lets a bare import resolve a bin-relative
    # tolerance.
    ft = resolve_ft_settings_h5(h5f)
    acquisition_us = active_acquisition_us(
        float(fid.attrs["duration_us"]), ft.start_us, ft.end_us
    )
    return acquisition_us if acquisition_us > 0.0 else None


def _optional_float(value: Any) -> Optional[float]:
    """A persisted bound as a float, treating the unset sentinel as unset."""
    decoded = decode_attr(value)
    if decoded is None or decoded == _NONE_SENTINEL:
        return None
    try:
        return float(decoded)
    except (TypeError, ValueError):
        return None


def _migrate_integer_tol(
    file_path: str, settings: Optional[StageFitSettings]
) -> Optional[StageFitSettings]:
    """Convert a legacy persisted ``spur.integer_tol_mhz`` to bins.

    The knob was an absolute frequency until it was redefined as a count of
    active-FT bins (``dev-docs/SCIENCE_STRATEGY.md`` Requirement 8). The
    conversion is **exact per file, not a default-and-hope**: a file carrying
    that setting also carries its own active region, so
    ``bins = mhz / (1 / T_active)`` is computable for that file alone. A
    reader must therefore never see a file lose the tolerance it was fitted
    with.

    Applies only when the file has no new-spelling value (a file written by
    this version wins over its own legacy attr) and when the active region is
    readable; otherwise the field stays ``None`` and the resolver's default
    applies, which is the same outcome an unset knob has always had.
    """
    if settings is None or settings.spur.integer_tol_bins is not None:
        return settings
    with h5open(file_path, "r") as h5f:
        group = h5f.get(f"{STAGE_FIT_PATH}/spur")
        if group is None or _LEGACY_INTEGER_TOL_ATTR not in group.attrs:
            return settings
        legacy = _optional_float(group.attrs[_LEGACY_INTEGER_TOL_ATTR])
        acquisition_us = declared_active_acquisition_us(h5f)
    if legacy is None or acquisition_us is None:
        return settings
    bins = legacy / active_ft_bin_spacing_mhz(acquisition_us)
    logger.info(
        "Migrating persisted spur.integer_tol_mhz=%g to spur.integer_tol_bins"
        "=%g (T_active=%g us) for %s",
        legacy,
        bins,
        acquisition_us,
        file_path,
    )
    return replace(settings, spur=replace(settings.spur, integer_tol_bins=bins))


def load_stage_fit_settings_from_h5(file_path: str) -> Optional[StageFitSettings]:
    """Return the persisted :class:`StageFitSettings`, or ``None`` if absent.

    Tolerates missing sub-blocks (a partial group still loads); fields
    not present default to ``None``. A file written before ``spur``'s
    integer tolerance became a bin count has it converted here, exactly
    (:func:`_migrate_integer_tol`).
    """
    settings = load_subblock_settings(
        file_path,
        STAGE_FIT_PATH,
        _SUB_NAMES,
        stage_fit_from_attrs,
        extra_top=_read_shape,
    )
    return _migrate_integer_tol(file_path, settings)


def stage_fit_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage5_fit`` block?"""
    return settings_block_present(file_path, STAGE_FIT_PATH)


def stage_fit_settings_provenance(file_path: str) -> Optional[RecordProvenance]:
    """The ``stage5_fit`` record's field-set version against
    :data:`STAGE5_FIT_FIELD_SET_VERSION`, or ``None`` if the record is absent."""
    return record_provenance(file_path, STAGE_FIT_PATH, STAGE5_FIT_FIELD_SET_VERSION)


def load_stage_fit_consumed_from_h5(file_path: str) -> Optional[Stage5Consumed]:
    """The :class:`Stage5Consumed` the last Stage 5 fit recorded, or ``None``.

    ``None`` when the file has no ``stage5_fit`` record or the record has no
    ``consumed`` block (a record written before version 2, or the sparse user
    layer). :func:`stage_fit_settings_provenance` tells those apart.
    """
    with h5open(file_path, "r") as h5f:
        group = h5f.get(f"{STAGE_FIT_PATH}/{_CONSUMED}")
        if not isinstance(group, h5py.Group):
            return None
        return Stage5Consumed.from_attrs(group.attrs)


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
    with h5open(file_path, "a") as h5f:
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
        with h5open(file_path, "r") as h5f:
            for path in _STAGE2B_GROUP_PATHS:
                if path not in h5f:
                    continue
                attr = h5f[path].attrs.get(_STAGE2B_RECOMMENDED_SHAPE_ATTR)
                if attr is None:
                    continue
                decoded = decode_attr(attr)
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
        with h5open(file_path, "r") as h5f:
            for path in _STAGE2B_GROUP_PATHS:
                if path not in h5f:
                    continue
                attr = h5f[path].attrs.get(_STAGE2B_VOTE_RATES_ATTR)
                if attr is None:
                    continue
                decoded = decode_attr(attr)
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


def _write_stage0_attr(file_path: str, attr: str, encoded: str, what: str) -> None:
    """Store *encoded* as the Stage 0 group's *attr* (a no-op when the group is
    absent), opening the file for write only when the stored value differs.

    An identical re-persist (a re-import of the same source) therefore writes
    nothing, so its transaction leaves the file untouched (§Crash safety).
    """
    try:
        with h5open(file_path, "r") as h5f:
            group = h5f.get(_STAGE0_GROUP)
            if group is None:
                return
            stored = group.attrs.get(attr)
            if isinstance(stored, bytes):
                stored = stored.decode("utf-8")
            if stored == encoded:
                return
    except (OSError, KeyError):
        pass  # Unreadable: the write below reports it.
    try:
        with h5open(file_path, "a") as h5f:
            if _STAGE0_GROUP not in h5f:
                return
            h5f[_STAGE0_GROUP].attrs[attr] = encoded
    except (OSError, KeyError):
        logger.warning("Could not write %s to %s", what, file_path)


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
    _write_stage0_attr(
        file_path, _RECOMMENDED_CLOCKS_ATTR, encoded, "recommended clock sources"
    )


def read_recommended_clock_sources(
    file_path: str,
) -> Optional[Tuple[ClockSource, ...]]:
    """Read the import-time recommended clock declaration, or ``None`` if absent.

    Returns ``None`` for: Stage 0 group absent, attr missing, ``__None__``
    sentinel, or any parse error.  Callers treat all of these as "no
    recommendation."
    """
    try:
        with h5open(file_path, "r") as h5f:
            if _STAGE0_GROUP not in h5f:
                return None
            attr = h5f[_STAGE0_GROUP].attrs.get(_RECOMMENDED_CLOCKS_ATTR)
            if attr is None:
                return None
            decoded = decode_attr(attr)
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
    _write_stage0_attr(
        file_path, _RECOMMENDED_CHIRP_WINDOW_ATTR, encoded, "recommended chirp window"
    )


def read_recommended_chirp_window(
    file_path: str,
) -> Optional[ChirpWindow]:
    """Read the import-time recommended chirp-window declaration, or ``None``.

    Returns ``None`` for: Stage 0 group absent, attr missing, ``__None__``
    sentinel, or any parse error.
    """
    try:
        with h5open(file_path, "r") as h5f:
            if _STAGE0_GROUP not in h5f:
                return None
            attr = h5f[_STAGE0_GROUP].attrs.get(_RECOMMENDED_CHIRP_WINDOW_ATTR)
            if attr is None:
                return None
            decoded = decode_attr(attr)
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


# The start-detection settings + sweep outcome are stored as a JSON attr on the
# Stage 0 FID group, alongside the chirp-window and clock-sources attrs. Unlike
# those import-time declarations, this one is written by ``start run`` /
# ``detect_start_time_impl`` (any time it is asked to stamp), whether or not a
# chirp collapse was found and whether or not a chirp-window declaration
# governs the final start_us -- it is the diagnostic record of what the sweep
# detector was actually given and actually found, so a report can replay the
# exact sweep instead of re-running it with guessed default knobs.
_RECOMMENDED_START_DETECTION_ATTR = "recommended_start_detection"


def write_recommended_start_detection(
    file_path: str,
    settings: StartDetectionSettings,
    result: StartDetectionResult,
) -> None:
    """Persist the start-detection settings + sweep outcome actually used.

    Stores a JSON-encoded record as an attr on ``stage0_fid_data``. No-op when
    the Stage 0 group is absent. Re-running detection over the same file
    overwrites the attr cleanly.
    """
    band = result.band_mhz
    encoded = json.dumps(
        {
            "sweep_max_us": settings.sweep_max_us,
            "step_us": settings.step_us,
            "floor_factor": settings.floor_factor,
            "floor_tail_us": settings.floor_tail_us,
            "guard_margin_us": settings.guard_margin_us,
            "min_chirp_drop_ratio": settings.min_chirp_drop_ratio,
            "band_min_mhz": (
                settings.band_min_mhz
                if settings.band_min_mhz is not None
                else _NONE_SENTINEL
            ),
            "band_max_mhz": (
                settings.band_max_mhz
                if settings.band_max_mhz is not None
                else _NONE_SENTINEL
            ),
            "chirp_end_us": result.chirp_end_us,
            "chirp_detected": result.chirp_detected,
            "floor": result.floor,
            "plateau": result.plateau,
            "resolved_band_min_mhz": band[0] if band is not None else _NONE_SENTINEL,
            "resolved_band_max_mhz": band[1] if band is not None else _NONE_SENTINEL,
        }
    )
    try:
        with h5open(file_path, "a") as h5f:
            if _STAGE0_GROUP not in h5f:
                return
            h5f[_STAGE0_GROUP].attrs[_RECOMMENDED_START_DETECTION_ATTR] = encoded
    except (OSError, KeyError):
        logger.warning(
            "Could not write recommended start-detection record to %s", file_path
        )


def read_recommended_start_detection(
    file_path: str,
) -> Optional[StartDetectionRecord]:
    """Read the persisted start-detection settings + sweep outcome, or ``None``.

    Returns ``None`` for: Stage 0 group absent, attr missing, or any parse
    error -- i.e. start-time detection has never been explicitly run
    (``start run`` / ``Pipeline.detect_start_time`` / ``api.detect_start_time``)
    on this file.
    """
    try:
        with h5open(file_path, "r") as h5f:
            if _STAGE0_GROUP not in h5f:
                return None
            attr = h5f[_STAGE0_GROUP].attrs.get(_RECOMMENDED_START_DETECTION_ATTR)
            if attr is None:
                return None
            decoded = decode_attr(attr)
            d = json.loads(decoded)

            def _opt(key: str) -> Optional[float]:
                v = d.get(key)
                return None if v is None or v == _NONE_SENTINEL else float(v)

            band_lo, band_hi = _opt("resolved_band_min_mhz"), _opt(
                "resolved_band_max_mhz"
            )
            band_mhz = (
                (band_lo, band_hi)
                if band_lo is not None and band_hi is not None
                else None
            )

            settings = StartDetectionSettings(
                sweep_max_us=float(d["sweep_max_us"]),
                step_us=float(d["step_us"]),
                floor_factor=float(d["floor_factor"]),
                floor_tail_us=float(d["floor_tail_us"]),
                guard_margin_us=float(d["guard_margin_us"]),
                min_chirp_drop_ratio=float(d["min_chirp_drop_ratio"]),
                band_min_mhz=_opt("band_min_mhz"),
                band_max_mhz=_opt("band_max_mhz"),
            )
            return StartDetectionRecord(
                settings=settings,
                chirp_end_us=float(d["chirp_end_us"]),
                chirp_detected=bool(d["chirp_detected"]),
                floor=float(d["floor"]),
                plateau=float(d["plateau"]),
                band_mhz=band_mhz,
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


__all__ = [
    "STAGE_FIT_PATH",
    "STAGE5_FIT_FIELD_SET_VERSION",
    "Stage5Consumed",
    "load_stage_fit_consumed_from_h5",
    "stage_fit_settings_provenance",
    "declared_active_acquisition_us",
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
    "write_recommended_start_detection",
    "read_recommended_start_detection",
]
