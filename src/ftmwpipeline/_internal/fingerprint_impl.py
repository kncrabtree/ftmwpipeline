"""The analysis fingerprint: one digest of every input that shaped the results.

Normative spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Analysis fingerprint. This
module is the read side; what every stage records so the fingerprint can be
computed from the file alone is :mod:`ftmwpipeline.io.provenance` and each
stage's codec.

Three pieces:

* :data:`_STAGE_RECORDS` -- the single, auditable map from each canonical stage
  to the records it is read from. Each entry names the canonical sub-key the
  record fills, the codec reader that returns its values, the provenance
  reader that says whether the record can be trusted, and the canonical keys
  it supplies (named in ``incomplete_provenance`` when it cannot be).
* :func:`canonical_fingerprint_inputs` -- the canonical input object the digest
  is taken over (private: for diagnosis and tests, not in the manifest).
* :func:`canonical_json` -- the canonical JSON encoding of that object.

A record is read only through its codec, never by walking the HDF5 layout.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

import h5py
import numpy as np

from ..contract import ANALYSIS_FINGERPRINT_SCHEMA, Stage, key_for_stage
from ..core import noise_settings as noise_core
from ..core import peak_detection_settings as peak_core
from ..core import stage_fit_settings as fit_core
from ..core import window_planning_settings as window_core
from ..core.settings import FT_PROCESSING_PATH
from ..core.stage_fit_settings import ClockSource, ShapeSpec
from ..core.tau_calibration_settings import (
    PRODUCER_FIELDS,
    PRODUCER_GAUSSIAN,
    PRODUCER_LORENTZIAN,
)
from ..file_manager import IncompleteProvenanceError, _load_stage_tracker
from ..io.environment_serialization import load_stage_environments
from ..io.fid_serialization import (
    load_acquisition_segments_from_hdf5,
    load_fid_from_hdf5,
)
from ..io.frequency_calibration_serialization import (
    frequency_calibration_provenance,
    load_frequency_calibration_record,
)
from ..io.noise_settings_serialization import (
    load_noise_settings_from_h5,
    noise_settings_provenance,
)
from ..io.peak_detection_settings_serialization import (
    load_peak_detection_consumed_from_h5,
    load_peak_detection_settings_from_h5,
    peak_detection_settings_provenance,
)
from ..io.provenance import RecordProvenance
from ..io.stage_fit_settings_serialization import (
    load_stage_fit_consumed_from_h5,
    load_stage_fit_settings_from_h5,
    stage_fit_settings_provenance,
)
from ..io.tau_calibration_settings_serialization import (
    load_tau_producer_settings_from_h5,
    tau_producer_settings_provenance,
)
from ..io.timebase_serialization import GROUP_PATH as TIMEBASE_GROUP_PATH
from ..io.timebase_serialization import (
    load_timebase_calibration_from_hdf5,
    timebase_calibration_provenance,
)
from ..io.window_planning_settings_serialization import (
    load_window_planning_settings_from_h5,
    window_planning_settings_provenance,
)
from .atomic import h5open
from .read_impl import _open
from .stage1_impl import _read_settings_layer, ft_settings_provenance

#: The canonical stage order of the input object (keys are sorted on encoding;
#: this order only drives the reading and the order of ``missing``).
CANONICAL_STAGES: Tuple[Stage, ...] = (
    Stage.DATA,
    Stage.FT,
    Stage.NOISE,
    Stage.TAU,
    Stage.TAU_G,
    Stage.TIMEBASE,
    Stage.PEAKS,
    Stage.WINDOWS,
    Stage.FIT,
    Stage.REVIEW,
)

#: Wire value of a stage that has not run (``"<stage>_absent"`` sibling).
NOT_RUN = "not_run"

#: The canonical key of a stage's analysis-epoch stamp.
EPOCH_KEY = "analysis_epoch"

#: The canonical sub-key of a consumer's consumed block.
CONSUMED_KEY = "consumed"


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------
def _encode_float(value: float) -> str:
    if math.isnan(value):
        return '"nan"'
    if math.isinf(value):
        return '"inf"' if value > 0 else '"-inf"'
    # ``repr`` is the shortest decimal string that round-trips to the same
    # binary64 value (``0.1``, ``1e-05``, ``25.0``, ``-0.0``). Every spelling it
    # produces for a finite float is a valid JSON number.
    return repr(value)


def _encode_str(value: str) -> str:
    out = ['"']
    for ch in value:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _encode(value: Any, path: str) -> str:
    # Order matters: ``bool`` is an ``int`` subclass and ``np.bool_`` is
    # neither, so booleans are recognised first and never spelled as numbers.
    if value is None:
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _encode_float(float(value))
    if isinstance(value, str):
        return _encode_str(value)
    if isinstance(value, ShapeSpec):
        return _encode({"kind": value.kind.value}, path)
    if isinstance(value, ClockSource):
        return _encode(value.to_dict(), path)
    if isinstance(value, Mapping):
        items = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"non-string key {key!r} at {path or '<root>'}")
            items.append(_encode_str(key) + ":" + _encode(value[key], f"{path}.{key}"))
        return "{" + ",".join(items) + "}"
    if isinstance(value, (list, tuple)):
        return (
            "["
            + ",".join(_encode(v, f"{path}[{i}]") for i, v in enumerate(value))
            + "]"
        )
    raise TypeError(
        f"cannot encode {type(value).__name__} at {path or '<root>'} in the "
        "canonical form"
    )


def canonical_json(obj: Any) -> bytes:
    """The canonical encoding of *obj* (§Analysis fingerprint, Canonical form).

    - Object keys sorted by code point at every level (Python ``str``
      ordering); keys must be strings.
    - A finite float as Python's ``repr`` spells it (the shortest decimal that
      round-trips to the same binary64 value), ``-0.0`` as ``-0.0``; ``nan`` /
      ``inf`` / ``-inf`` as those JSON strings. A numpy float is encoded as the
      Python float of the same value.
    - An integer (Python or numpy) without a decimal point; a boolean (Python
      or numpy) as ``true`` / ``false``, never as a number.
    - Tuples and lists as arrays; ``None`` (a setting resolved to unset) as
      ``null``.
    - :class:`~ftmwpipeline.core.stage_fit_settings.ShapeSpec` as
      ``{"kind": "<shape>"}``; :class:`~ftmwpipeline.core.stage_fit_settings.ClockSource`
      as ``{"freq_mhz", "locked", "label"}`` (its ``to_dict`` form).
    - Strings escape only ``"``, ``\\`` and control characters (``\\u00XX``);
      everything else is written as itself. The result is UTF-8 with no
      insignificant whitespace.

    Anything else raises :class:`TypeError`; there is no ``str()`` fallback.
    """
    return _encode(obj, "").encode("utf-8")


def digest_of(obj: Any) -> str:
    """SHA-256 of :func:`canonical_json` of *obj*, lowercase hex."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()


# ---------------------------------------------------------------------------
# Record readers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Read:
    """What a codec reader returned: canonical values, and the fields a record
    at its current version should hold but does not (relative dotted keys)."""

    values: Dict[str, Any]
    missing: Tuple[str, ...] = ()


def _plain(value: Any) -> Any:
    """A codec value in canonical-builder form: enums as their value, tuples
    and lists element-wise, numpy scalars as Python scalars."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _flat_fields(settings: Any, required: Any) -> _Read:
    """Every field of a flat settings dataclass; a field the resolver always
    fills (non-``None`` in *required*) but the record holds as ``None`` is
    missing."""
    values: Dict[str, Any] = {}
    missing: List[str] = []
    for f in dataclasses.fields(settings):
        value = getattr(settings, f.name)
        values[f.name] = _plain(value)
        if value is None and getattr(required, f.name) is not None:
            missing.append(f.name)
    return _Read(values, tuple(missing))


def _subblock_fields(settings: Any, required: Any, groups: Tuple[str, ...]) -> _Read:
    """Every field of every settings group, nested by group (see
    :func:`_flat_fields` for the missing rule)."""
    values: Dict[str, Any] = {}
    missing: List[str] = []
    for group in groups:
        block = _flat_fields(getattr(settings, group), getattr(required, group))
        values[group] = block.values
        missing.extend(f"{group}.{name}" for name in block.missing)
    return _Read(values, tuple(missing))


# -- data (Stage 0) ---------------------------------------------------------
_DATA_KEYS = ("probe_freq_mhz", "sideband", "spacing_s", "acquisition_segments")


def _array_digest(array: np.ndarray) -> str:
    """SHA-256 of an array as little-endian float64, C order (spec: data)."""
    data = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return hashlib.sha256(data.tobytes()).hexdigest()


def _segments_values(segments: Any) -> Optional[Dict[str, Any]]:
    """The stored acquisition segments, which the Stage 5 spur gate reads (its
    chirp-response probe and interleave comb): their layout scalars and a
    content digest per array."""
    if segments is None:
        return None
    patterns = segments.interleave_patterns
    return {
        "pre_record_us": float(segments.pre_record_us),
        "frame_period_us": float(segments.frame_period_us),
        "n_frames": int(segments.n_frames),
        "frame_selection": (
            None if segments.frame_selection is None else int(segments.frame_selection)
        ),
        "sample_dt": float(segments.sample_dt),
        "pre_record": _array_digest(segments.pre_record),
        "tail": _array_digest(segments.tail),
        "frames": None if segments.frames is None else _array_digest(segments.frames),
        "interleave_patterns": (
            None
            if patterns is None
            else {str(int(m)): _array_digest(v) for m, v in patterns.items()}
        ),
    }


def _read_data(file_path: str) -> Optional[_Read]:
    """The acquisition parameters the analysis read, as stored.

    Import-time overrides (probe frequency, sideband, sample spacing) are
    applied to the FID before it is stored, so the stored values are the ones
    every stage used. The spacing is hashed in the stored unit (seconds), so no
    conversion can merge two distinct values.
    """
    with h5open(file_path, "r") as h5f:
        group = h5f.get(key_for_stage(Stage.DATA))
        if not isinstance(group, h5py.Group):
            return None
        fid = load_fid_from_hdf5(group)
        segments = load_acquisition_segments_from_hdf5(group)
    return _Read(
        {
            "probe_freq_mhz": float(fid.probe_freq_mhz),
            "sideband": str(fid.sideband.value),
            "spacing_s": float(fid.spacing),
            "acquisition_segments": _segments_values(segments),
        }
    )


# -- ft (Stage 1) -----------------------------------------------------------
_FT_KEYS = ("start_us", "end_us", "units_power", "trim_min_mhz", "trim_max_mhz")


def _read_ft(file_path: str) -> Optional[_Read]:
    settings = _read_settings_layer(file_path, FT_PROCESSING_PATH)
    if settings is None:
        return None
    trim = settings.trim
    values = {
        "start_us": _plain(settings.start_us),
        "end_us": _plain(settings.end_us),
        "units_power": _plain(settings.units_power),
        "trim_min_mhz": None if trim is None else float(trim[0]),
        "trim_max_mhz": None if trim is None else float(trim[1]),
    }
    # At the current version the active region is always concrete and
    # ``units_power`` always resolved; only ``trim`` may be unset ("no trim").
    missing = tuple(
        name for name in ("start_us", "end_us", "units_power") if values[name] is None
    )
    return _Read(values, missing)


def _ft_provenance(file_path: str) -> Optional[RecordProvenance]:
    return ft_settings_provenance(file_path)


# -- noise (Stage 2) --------------------------------------------------------
_NOISE_KEYS = tuple(f.name for f in dataclasses.fields(noise_core.NoiseSettings))


def _read_noise(file_path: str) -> Optional[_Read]:
    settings = load_noise_settings_from_h5(file_path)
    if settings is None:
        return None
    return _flat_fields(settings, noise_core.resolve())


# -- tau / tau_g (Stage 2b twins) ---------------------------------------------
def _producer_keys(producer: str) -> Tuple[str, ...]:
    return tuple(PRODUCER_FIELDS[producer])


def _producer_reader(producer: str) -> Callable[[str], Optional[_Read]]:
    """A reader of *producer*'s own record (never the shared recipe)."""

    def read(file_path: str) -> Optional[_Read]:
        blocks = load_tau_producer_settings_from_h5(file_path, producer)
        if blocks is None:
            return None
        values: Dict[str, Any] = {}
        missing: List[str] = []
        # The record writes every field of the producer's map (``None`` is a
        # field resolved to unset), so a field absent from it is missing.
        for group, names in PRODUCER_FIELDS[producer].items():
            stored = blocks.get(group, {})
            values[group] = {}
            for name in names:
                if name in stored:
                    values[group][name] = _plain(stored[name])
                else:
                    values[group][name] = None
                    missing.append(f"{group}.{name}")
        return _Read(values, tuple(missing))

    return read


def _producer_provenance(producer: str) -> Callable[[str], Optional[RecordProvenance]]:
    def provenance(file_path: str) -> Optional[RecordProvenance]:
        return tau_producer_settings_provenance(file_path, producer)

    return provenance


# -- timebase ---------------------------------------------------------------
_TIMEBASE_KEYS = ("kappa_sys", "snr_min", "start_us", "end_us", "clock_sources")


def _read_timebase(file_path: str) -> Optional[_Read]:
    """The timebase calibration's inputs: its knobs, the active region it ran
    on and the clock declaration it resolved."""
    with h5open(file_path, "r") as h5f:
        group = h5f.get(TIMEBASE_GROUP_PATH)
        if not isinstance(group, h5py.Group):
            return None
        try:
            result = load_timebase_calibration_from_hdf5(group)
        except KeyError:
            # A record that lacks a field its version writes.
            return None
    clocks = result.clock_sources
    values = {
        "kappa_sys": float(result.kappa_sys),
        "snr_min": float(result.snr_min),
        "start_us": float(result.start_us),
        "end_us": float(result.end_us),
        "clock_sources": None if clocks is None else list(clocks),
    }
    return _Read(values, ("clock_sources",) if clocks is None else ())


def _timebase_provenance(file_path: str) -> Optional[RecordProvenance]:
    with h5open(file_path, "r") as h5f:
        group = h5f.get(TIMEBASE_GROUP_PATH)
        if not isinstance(group, h5py.Group):
            return None
        return timebase_calibration_provenance(group)


# -- peaks (Stage 3) ----------------------------------------------------------
_PEAK_GROUPS: Tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(peak_core.PeakDetectionSettings)
)


def _read_peaks(file_path: str) -> Optional[_Read]:
    settings = load_peak_detection_settings_from_h5(file_path)
    if settings is None:
        return None
    return _subblock_fields(settings, peak_core.resolve(), _PEAK_GROUPS)


def _read_peaks_consumed(file_path: str) -> Optional[_Read]:
    try:
        consumed = load_peak_detection_consumed_from_h5(file_path)
    except KeyError:
        # A block that lacks a field its version writes.
        return _Read({}, ("",))
    if consumed is None:
        return _Read({}, ("",))
    return _Read(
        {
            "tau_basis_us": float(consumed.tau_basis_us),
            "gap_shape": str(consumed.gap_shape),
            "tau_basis_source": str(consumed.tau_basis_source),
        }
    )


# -- windows (Stage 4) --------------------------------------------------------
_WINDOW_GROUPS: Tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(window_core.WindowPlanningSettings)
)


def _read_windows(file_path: str) -> Optional[_Read]:
    settings = load_window_planning_settings_from_h5(file_path)
    if settings is None:
        return None
    return _subblock_fields(settings, window_core.resolve(), _WINDOW_GROUPS)


# -- fit (Stage 5) ------------------------------------------------------------
_FIT_GROUPS: Tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(fit_core.StageFitSettings) if f.name != "shape"
)
_FIT_KEYS: Tuple[str, ...] = ("shape",) + _FIT_GROUPS


def _read_fit(file_path: str) -> Optional[_Read]:
    settings = load_stage_fit_settings_from_h5(file_path)
    if settings is None:
        return None
    block = _subblock_fields(settings, fit_core.resolve(), _FIT_GROUPS)
    values = dict(block.values)
    values["shape"] = settings.shape
    missing = block.missing + (("shape",) if settings.shape is None else ())
    return _Read(values, missing)


def _read_fit_consumed(file_path: str) -> Optional[_Read]:
    try:
        consumed = load_stage_fit_consumed_from_h5(file_path)
    except KeyError:
        # A block that lacks a field its version writes.
        return _Read({}, ("",))
    if consumed is None:
        return _Read({}, ("",))
    bands = consumed.band_majorities
    nominees = consumed.stft_spur_nominees
    return _Read(
        {
            "tau_calibration_source": str(consumed.tau_calibration_source),
            "tau_maj_us": _plain(consumed.tau_maj_us),
            "sigma_tau_us": _plain(consumed.sigma_tau_us),
            "band_majorities": (
                None
                if bands is None
                else [
                    {
                        "label": str(b.label),
                        "freq_lo_mhz": float(b.freq_lo_mhz),
                        "freq_hi_mhz": float(b.freq_hi_mhz),
                        "n": int(b.n),
                        "tau_maj_us": float(b.tau_maj_us),
                        "sigma_tau_us": float(b.sigma_tau_us),
                    }
                    for b in bands
                ]
            ),
            "timebase_epsilon": _plain(consumed.timebase_epsilon),
            "timebase_sigma_epsilon": _plain(consumed.timebase_sigma_epsilon),
            "peak_survival_snr_floor": float(consumed.peak_survival_snr_floor),
            "stft_spur_nominees": (
                None
                if nominees is None
                else [[float(f), bool(sat)] for f, sat in nominees]
            ),
        }
    )


# -- review (Stage 6) ---------------------------------------------------------
def _read_review(file_path: str) -> Optional[_Read]:
    with h5open(file_path, "r") as h5f:
        try:
            calibration = load_frequency_calibration_record(h5f)
        except KeyError:
            return _Read({}, ("sigma_floor_khz",))
    if calibration is None:
        return None
    return _Read({"sigma_floor_khz": float(calibration.sigma_floor_khz)})


def _read_calibration_clocks(file_path: str) -> Optional[_Read]:
    """The clock declaration the final products' calibration state is derived
    from, through Stage 6's own resolver. The products are derived on read, so
    this is the declaration in effect, not one recorded at ``review run``."""
    from .stage6_impl import _resolve_calibration_clocks

    return _Read({"calibration_clocks": list(_resolve_calibration_clocks(file_path))})


def _review_provenance(file_path: str) -> Optional[RecordProvenance]:
    with h5open(file_path, "r") as h5f:
        return frequency_calibration_provenance(h5f)


# ---------------------------------------------------------------------------
# The map: stage -> records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Record:
    """One record a stage's inputs are read from.

    Attributes
    ----------
    sub_key : str
        Where the record's values go under the stage's object: ``""`` merges
        them into it, anything else nests them under that key (``consumed``).
    reader : callable
        The codec reader: ``file_path -> _Read``, or ``None`` when the file
        holds no such record.
    provenance : callable or None
        The record's field-set version reader, ``None`` for a record that has
        no field-set version (Stage 0's acquisition parameters).
    keys : tuple of str
        The canonical keys the record supplies, relative to ``sub_key``
        (its settings groups, or its fields when it has none). Named in
        ``incomplete_provenance`` when the record is absent or untrusted.
    absent_values : dict or None
        Values to use when a completed stage has no such record at all, when
        the spec defines them (``review.sigma_floor_khz`` is ``0.0``).
        ``None``: an absent record is missing provenance.
    """

    sub_key: str
    reader: Callable[[str], Optional[_Read]]
    provenance: Optional[Callable[[str], Optional[RecordProvenance]]]
    keys: Tuple[str, ...]
    absent_values: Optional[Mapping[str, Any]] = None


#: Every input the fingerprint covers, by canonical stage. This is the one
#: place coverage is declared. Each settings record contributes every field it
#: writes at its current field-set version; each consumer's ``consumed`` block
#: is hashed as recorded. The Stage 2b twins read their own producer records,
#: never the shared ``stage2b_tau`` recipe, and the shape recommendation's
#: record is not read at all (its verdict is covered where it was used:
#: ``peaks.consumed`` and ``fit.shape``).
_STAGE_RECORDS: Mapping[Stage, Tuple[_Record, ...]] = {
    Stage.DATA: (_Record("", _read_data, None, _DATA_KEYS),),
    Stage.FT: (_Record("", _read_ft, _ft_provenance, _FT_KEYS),),
    Stage.NOISE: (_Record("", _read_noise, noise_settings_provenance, _NOISE_KEYS),),
    Stage.TAU: (
        _Record(
            "",
            _producer_reader(PRODUCER_LORENTZIAN),
            _producer_provenance(PRODUCER_LORENTZIAN),
            _producer_keys(PRODUCER_LORENTZIAN),
        ),
    ),
    Stage.TAU_G: (
        _Record(
            "",
            _producer_reader(PRODUCER_GAUSSIAN),
            _producer_provenance(PRODUCER_GAUSSIAN),
            _producer_keys(PRODUCER_GAUSSIAN),
        ),
    ),
    Stage.TIMEBASE: (
        _Record("", _read_timebase, _timebase_provenance, _TIMEBASE_KEYS),
    ),
    Stage.PEAKS: (
        _Record("", _read_peaks, peak_detection_settings_provenance, _PEAK_GROUPS),
        _Record(
            CONSUMED_KEY,
            _read_peaks_consumed,
            peak_detection_settings_provenance,
            ("",),
        ),
    ),
    Stage.WINDOWS: (
        _Record("", _read_windows, window_planning_settings_provenance, _WINDOW_GROUPS),
    ),
    Stage.FIT: (
        _Record("", _read_fit, stage_fit_settings_provenance, _FIT_KEYS),
        _Record(CONSUMED_KEY, _read_fit_consumed, stage_fit_settings_provenance, ("",)),
    ),
    Stage.REVIEW: (
        _Record(
            "",
            _read_review,
            _review_provenance,
            ("sigma_floor_khz",),
            absent_values={"sigma_floor_khz": 0.0},
        ),
        _Record("", _read_calibration_clocks, None, ("calibration_clocks",)),
    ),
}


def _dotted(stage: Stage, sub_key: str, rel: str) -> str:
    return ".".join(part for part in (stage.value, sub_key, rel) if part)


def _read_stage(
    file_path: str,
    stage: Stage,
    epoch: Optional[int],
    missing: List[str],
    newer: List[str],
) -> Dict[str, Any]:
    """The canonical object of one completed stage; appends every input it
    cannot vouch for to *missing*, or to *newer* when a newer engine wrote its
    record."""
    obj: Dict[str, Any] = {EPOCH_KEY: epoch}
    if epoch is None:
        missing.append(_dotted(stage, "", EPOCH_KEY))
    for record in _STAGE_RECORDS[stage]:
        unreadable = [_dotted(stage, record.sub_key, k) for k in record.keys]
        provenance = None if record.provenance is None else record.provenance(file_path)
        if record.provenance is not None and provenance is None:
            if record.absent_values is None:
                missing.extend(unreadable)
                continue
            read: Optional[_Read] = _Read(dict(record.absent_values))
        elif provenance is not None and provenance.is_newer:
            newer.extend(unreadable)
            continue
        elif provenance is not None and provenance.is_pre_provenance:
            missing.extend(unreadable)
            continue
        else:
            read = record.reader(file_path)
        if read is None:
            missing.extend(unreadable)
            continue
        missing.extend(_dotted(stage, record.sub_key, k) for k in read.missing)
        if record.sub_key:
            obj[record.sub_key] = read.values
        else:
            obj.update(read.values)
    return obj


def _completed_and_epochs(file_path: str) -> Tuple[Set[str], Dict[str, Optional[int]]]:
    """The completed stage keys and every stamped analysis epoch, read once
    through the format-gated open."""
    with _open(file_path) as h5f:
        completed = set(_load_stage_tracker(Path(file_path), h5f).completed_stages)
        environments = load_stage_environments(h5f)
    epochs = {key: env.analysis_epoch for key, env in environments.items()}
    return completed, epochs


def canonical_fingerprint_inputs(file_path: Union[str, Path]) -> Dict[str, Any]:
    """The canonical input object :func:`analysis_fingerprint_impl` hashes.

    Private (not in the contract manifest): for diagnosis and tests. Every
    canonical stage key is present; a stage not complete in the stage tracker
    is ``None`` with ``"<stage>_absent": "not_run"``.

    Raises
    ------
    IncompleteProvenanceError
        A completed stage cannot account for every input it used. ``missing``
        lists every such input as a dotted canonical key, across all stages;
        ``newer`` lists those whose records a newer engine wrote.
    """
    path = str(file_path)
    completed, epochs = _completed_and_epochs(path)
    obj: Dict[str, Any] = {}
    missing: List[str] = []
    newer: List[str] = []
    for stage in CANONICAL_STAGES:
        key = key_for_stage(stage)
        if key not in completed:
            obj[stage.value] = None
            obj[f"{stage.value}_absent"] = NOT_RUN
            continue
        obj[stage.value] = _read_stage(path, stage, epochs.get(key), missing, newer)
    if missing or newer:
        problems = []
        if missing:
            problems.append(
                "the file does not record every input its completed stages used "
                f"({', '.join(missing)}); re-run the stage(s) that own them to "
                "record them"
            )
        if newer:
            problems.append(
                "the file holds records written by a newer ftmwpipeline "
                f"({', '.join(newer)}); upgrade ftmwpipeline to read them"
            )
        raise IncompleteProvenanceError(
            missing,
            newer=newer,
            message=f"Cannot compute the analysis fingerprint: {'; '.join(problems)}.",
        )
    return obj


def analysis_fingerprint_impl(file_path: Union[str, Path]) -> Dict[str, Any]:
    """``{"schema": "ftmw/analysis_fingerprint@1", "digest": "<64 hex>"}``.

    The SHA-256 of the canonical JSON of :func:`canonical_fingerprint_inputs`.
    Never writes the file.
    """
    return {
        "schema": ANALYSIS_FINGERPRINT_SCHEMA,
        "digest": digest_of(canonical_fingerprint_inputs(file_path)),
    }


__all__ = [
    "CANONICAL_STAGES",
    "analysis_fingerprint_impl",
    "canonical_fingerprint_inputs",
    "canonical_json",
    "digest_of",
]
