"""
Persistence for the Stage 2b settings records.

Stage 2b keeps four settings records, all under ``processing_parameters/``:

* ``stage2b_tau`` -- the **recipe**: the full resolved
  :class:`~ftmwpipeline.core.tau_calibration_settings.TauCalibrationSettings`
  of the most recent Stage 2b run (either twin or the recommender). It is the
  *persisted* layer every Stage 2b run resolves against, so a no-arg follow-up
  call inherits the same recipe, and it is what ``settings show`` /
  ``settings set`` read and write. It is shared, so it says what the *next* run
  will use, not what any existing result used.
* ``stage2b_tau_calibration`` -- what the Lorentzian twin's current result
  used (canonical stage ``tau``).
* ``stage2b_tau_G_calibration`` -- what the Gaussian twin's current result used
  (canonical stage ``tau_g``).
* ``stage2b_shape_recommendation`` -- what the shape recommendation used, the
  Stage 1 values it consumed, the effective tau clip, and its verdict
  (``tau.recommendation``).

The three producer records are written only by their own producer, each at its
own field-set version, and hold exactly the fields that producer passes to its
kernel (:data:`~ftmwpipeline.core.tau_calibration_settings.PRODUCER_FIELDS`), so
running one producer never changes what another is read as having used. The
twins' effective tau clip is in their result groups (``scalars@tau_max_us``,
read through :mod:`ftmwpipeline.io.tau_calibration_serialization`).

The layout uses one HDF5 subgroup per sub-dataclass so each block is
independently inspectable with ``h5dump -p``:

.. code-block::

    processing_parameters/
      stage2b_tau/
        @creation_time
        @field_set_version
        @preset_name              (optional audit attr)
        stft/
          @n_seg
          @t_sigma
          ...
        polish/  aggregation/  band/  gaussian/  recommendation/
      stage2b_shape_recommendation/
        @creation_time  @field_set_version  @preset_name
        stft/  recommendation/
        consumed/   @start_us @end_us @trim_lo_mhz @trim_hi_mhz
        effective/  @tau_max_us
        verdict/    @recommended_shape @vote_rates (JSON)

Unset (Optional-None) fields encode as the ``__None__`` sentinel string,
matching :mod:`ftmwpipeline.io.stage_fit_settings_serialization`.
Tuple-valued fields (``tau_G_seeds``, ``band_edges_mhz``,
``band_labels``) round-trip via 1-D HDF5 datasets stored as attrs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import h5py
import numpy as np

from ..core.tau_calibration_settings import (
    PRODUCER_FIELDS,
    PRODUCER_GAUSSIAN,
    PRODUCER_LORENTZIAN,
    PRODUCER_RECOMMENDATION,
    TauCalibrationSettings,
)
from ..core.tau_calibration_settings import from_attrs as tau_settings_from_attrs
from ..core.tau_calibration_settings import (
    producer_values,
)
from ..core.tau_calibration_settings import to_attrs as tau_settings_to_attrs
from ._settings_serialization import (
    decode_attr,
    load_subblock_settings,
    save_settings,
    settings_block_present,
)
from .provenance import RecordProvenance, record_provenance

#: The shared Stage 2b recipe (the persisted layer for the next run).
STAGE2B_TAU_SETTINGS_PATH = "processing_parameters/stage2b_tau"

#: Field-set version of the ``stage2b_tau`` recipe this codec writes (see
#: :mod:`ftmwpipeline.io.provenance`).
STAGE2B_TAU_FIELD_SET_VERSION = 1

#: What the Lorentzian twin's result used (canonical stage ``tau``).
STAGE2B_LORENTZIAN_SETTINGS_PATH = "processing_parameters/stage2b_tau_calibration"
STAGE2B_LORENTZIAN_FIELD_SET_VERSION = 1

#: What the Gaussian twin's result used (canonical stage ``tau_g``).
STAGE2B_GAUSSIAN_SETTINGS_PATH = "processing_parameters/stage2b_tau_G_calibration"
STAGE2B_GAUSSIAN_FIELD_SET_VERSION = 1

#: What the shape recommendation used, and its verdict (``tau.recommendation``).
SHAPE_RECOMMENDATION_SETTINGS_PATH = (
    "processing_parameters/stage2b_shape_recommendation"
)
#: 2: the record gains stft.relative_gate_fraction and stft.sigma_x_full, which
#: the recommender now passes to its kernel (it ran at the kernel defaults).
SHAPE_RECOMMENDATION_FIELD_SET_VERSION = 2

# producer -> (record path, current field-set version)
_PRODUCER_RECORDS: Dict[str, Tuple[str, int]] = {
    PRODUCER_LORENTZIAN: (
        STAGE2B_LORENTZIAN_SETTINGS_PATH,
        STAGE2B_LORENTZIAN_FIELD_SET_VERSION,
    ),
    PRODUCER_GAUSSIAN: (
        STAGE2B_GAUSSIAN_SETTINGS_PATH,
        STAGE2B_GAUSSIAN_FIELD_SET_VERSION,
    ),
    PRODUCER_RECOMMENDATION: (
        SHAPE_RECOMMENDATION_SETTINGS_PATH,
        SHAPE_RECOMMENDATION_FIELD_SET_VERSION,
    ),
}

#: The Stage 1 values the recommendation consumed (it is not a tracked stage,
#: so a Stage 1 re-run does not invalidate it; its record says what it used).
SHAPE_RECOMMENDATION_CONSUMED_FIELDS = (
    "start_us",
    "end_us",
    "trim_lo_mhz",
    "trim_hi_mhz",
)

_SUB_NAMES = (
    "stft",
    "polish",
    "aggregation",
    "band",
    "gaussian",
    "recommendation",
)

# Tuple-valued attrs encoded as Python lists; HDF5 stores them as 1-D
# numpy arrays under the hood. The serialization module flattens these
# back to plain tuples on load via ``from_attrs``.
_TUPLE_FIELDS = {"tau_G_seeds", "band_edges_mhz", "band_labels"}

_NONE_SENTINEL = "__None__"


def _write_attr(grp: h5py.Group, name: str, value: Any) -> None:
    """Write one attr, lifting lists/tuples to numpy 1-D arrays."""
    if isinstance(value, list):
        grp.attrs[name] = list(value)
    else:
        grp.attrs[name] = value


def _read_sub_attrs(grp: h5py.Group) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, raw in grp.attrs.items():
        value = decode_attr(raw)
        if key in _TUPLE_FIELDS:
            # HDF5 may return numpy arrays for sequence attrs; pass them
            # through as iterables so the dataclass decoder can coerce to
            # tuples.
            if isinstance(value, str) and value == _NONE_SENTINEL:
                out[key] = _NONE_SENTINEL
                continue
            out[key] = [decode_attr(v) for v in value]
        else:
            out[key] = value
    return out


def save_tau_calibration_settings_to_h5(
    file_path: str,
    settings: TauCalibrationSettings,
    *,
    preset_name: Optional[str] = None,
) -> None:
    """Persist a resolved :class:`TauCalibrationSettings` to ``processing_parameters/stage2b_tau``.

    Overwrites any prior group at that path. ``preset_name`` (if given) is
    recorded as a top-level attr for audit/reproducibility -- useful when
    a calibration was driven by a named preset.
    """
    save_settings(
        file_path,
        STAGE2B_TAU_SETTINGS_PATH,
        tau_settings_to_attrs(settings),
        field_set_version=STAGE2B_TAU_FIELD_SET_VERSION,
        preset_name=preset_name,
        write_attr=_write_attr,
    )


def load_tau_calibration_settings_from_h5(
    file_path: str,
) -> Optional[TauCalibrationSettings]:
    """Return the persisted :class:`TauCalibrationSettings`, or ``None`` if absent.

    Tolerates missing sub-blocks (a partial group still loads); fields
    not present default to ``None``.
    """
    return load_subblock_settings(
        file_path,
        STAGE2B_TAU_SETTINGS_PATH,
        _SUB_NAMES,
        tau_settings_from_attrs,
        read_sub=_read_sub_attrs,
    )


def tau_calibration_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage2b_tau`` block?"""
    return settings_block_present(file_path, STAGE2B_TAU_SETTINGS_PATH)


def tau_calibration_settings_provenance(file_path: str) -> Optional[RecordProvenance]:
    """The ``stage2b_tau`` recipe's field-set version against
    :data:`STAGE2B_TAU_FIELD_SET_VERSION`, or ``None`` if the record is absent.

    The recipe is what the next run resolves against, not what any result
    used; :func:`tau_producer_settings_provenance` is the per-result reader.
    """
    return record_provenance(
        file_path, STAGE2B_TAU_SETTINGS_PATH, STAGE2B_TAU_FIELD_SET_VERSION
    )


# ---------------------------------------------------------------------------
# Per-producer records: what each Stage 2b result used
# ---------------------------------------------------------------------------
def _producer_record(producer: str) -> Tuple[str, int]:
    try:
        return _PRODUCER_RECORDS[producer]
    except KeyError:
        raise ValueError(
            f"unknown Stage 2b producer {producer!r}; expected one of "
            f"{sorted(_PRODUCER_RECORDS)}"
        ) from None


def _encode(value: Any) -> Any:
    if value is None:
        return _NONE_SENTINEL
    if isinstance(value, tuple):
        return list(value)
    return value


def _producer_attrs(settings: TauCalibrationSettings, producer: str) -> Dict[str, Any]:
    """The producer's sub-blocks, every field of its map, encoded for HDF5."""
    return {
        block: {name: _encode(value) for name, value in fields.items()}
        for block, fields in producer_values(settings, producer).items()
    }


def _save_producer_record(
    file_path: str,
    producer: str,
    attrs: Dict[str, Any],
    preset_name: Optional[str],
) -> None:
    path, version = _producer_record(producer)
    save_settings(
        file_path,
        path,
        attrs,
        field_set_version=version,
        preset_name=preset_name,
        write_attr=_write_attr,
    )


def save_tau_producer_settings_to_h5(
    file_path: str,
    producer: str,
    settings: TauCalibrationSettings,
    *,
    preset_name: Optional[str] = None,
) -> None:
    """Record what a Stage 2b twin's result used, in that twin's own record.

    *producer* is ``"lorentzian"`` or ``"gaussian"``; *settings* is the resolved
    bundle the twin ran with. Every field the twin consumes is written (a
    ``None`` is a field resolved to unset), at the twin's current field-set
    version, overwriting only that twin's prior record. The recommendation
    writes its record through :func:`save_shape_recommendation_record`, which
    carries its verdict too.
    """
    if producer not in (PRODUCER_LORENTZIAN, PRODUCER_GAUSSIAN):
        raise ValueError(
            f"save_tau_producer_settings_to_h5 records a tau twin "
            f"('lorentzian' or 'gaussian'); got {producer!r}"
        )
    _save_producer_record(
        file_path, producer, _producer_attrs(settings, producer), preset_name
    )


def _native(value: Any) -> Any:
    """A numpy scalar read back from HDF5 as the matching Python scalar."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_producer_blocks(grp: h5py.Group, producer: str) -> Dict[str, Dict[str, Any]]:
    """Decode the stored settings sub-blocks of a producer record.

    Only the fields actually stored are returned (a record at the current
    version stores every field of the producer's map).
    """
    blocks = PRODUCER_FIELDS[producer]
    raw: Dict[str, Dict[str, Any]] = {}
    for block in blocks:
        sub = grp.get(block)
        raw[block] = _read_sub_attrs(sub) if isinstance(sub, h5py.Group) else {}
    decoded = tau_settings_from_attrs(raw)
    return {
        block: {
            name: _native(getattr(getattr(decoded, block), name))
            for name in names
            if name in raw[block]
        }
        for block, names in blocks.items()
    }


def load_tau_producer_settings_from_h5(
    file_path: str, producer: str
) -> Optional[Dict[str, Dict[str, Any]]]:
    """What *producer*'s current result used, as ``{sub_block: {field: value}}``.

    *producer* is ``"lorentzian"``, ``"gaussian"`` or ``"recommendation"``.
    Returns ``None`` when the file holds no record for it (the producer has
    not run since records were kept). A ``None`` value in a record at the
    current version (:func:`tau_producer_settings_provenance`) means the field
    resolved to unset.
    """
    path, _version = _producer_record(producer)
    with h5py.File(file_path, "r") as h5f:
        grp = h5f.get(path)
        if not isinstance(grp, h5py.Group):
            return None
        return _read_producer_blocks(grp, producer)


def tau_producer_settings_provenance(
    file_path: str, producer: str
) -> Optional[RecordProvenance]:
    """The field-set version of *producer*'s record against its codec's
    current one, or ``None`` when the record is absent."""
    path, version = _producer_record(producer)
    return record_provenance(file_path, path, version)


@dataclass(frozen=True)
class ShapeRecommendationRecord:
    """What the shape recommendation used, and what it decided.

    Attributes
    ----------
    settings : dict
        ``{sub_block: {field: value}}`` for the knobs the recommender passes to
        its kernel (``PRODUCER_FIELDS["recommendation"]``).
    consumed : dict
        The Stage 1 values it ran on: ``start_us``, ``end_us``,
        ``trim_lo_mhz``, ``trim_hi_mhz``.
    tau_max_us : float or None
        The effective upper clip on the classifier's per-bin tau.
    recommended_shape : str or None
        The verdict: ``"lorentzian"``, ``"gaussian"``, or ``None`` for no clear
        winner.
    vote_rates : dict
        The SNR-weighted vote rate per model (``exp`` / ``gauss`` / ``voigt``).
    """

    settings: Dict[str, Dict[str, Any]]
    consumed: Dict[str, Optional[float]]
    tau_max_us: Optional[float]
    recommended_shape: Optional[str]
    vote_rates: Dict[str, float]


def save_shape_recommendation_record(
    file_path: str,
    settings: TauCalibrationSettings,
    *,
    consumed: Mapping[str, float],
    tau_max_us: Optional[float],
    recommended_shape: Optional[str],
    vote_rates: Mapping[str, float],
    preset_name: Optional[str] = None,
) -> None:
    """Persist the shape recommendation's own record, verdict included.

    Written whether or not any Stage 2b result group exists, at
    :data:`SHAPE_RECOMMENDATION_FIELD_SET_VERSION`, overwriting only the prior
    recommendation record. *consumed* must carry every
    :data:`SHAPE_RECOMMENDATION_CONSUMED_FIELDS` key.
    """
    missing = [k for k in SHAPE_RECOMMENDATION_CONSUMED_FIELDS if k not in consumed]
    if missing:
        raise ValueError(f"consumed is missing {missing}")
    attrs = _producer_attrs(settings, PRODUCER_RECOMMENDATION)
    attrs["consumed"] = {
        k: float(consumed[k]) for k in SHAPE_RECOMMENDATION_CONSUMED_FIELDS
    }
    attrs["effective"] = {
        "tau_max_us": _encode(None if tau_max_us is None else float(tau_max_us))
    }
    attrs["verdict"] = {
        "recommended_shape": _encode(recommended_shape),
        "vote_rates": json.dumps({str(k): float(v) for k, v in vote_rates.items()}),
    }
    _save_producer_record(file_path, PRODUCER_RECOMMENDATION, attrs, preset_name)


def _decoded_or_none(raw: Any) -> Any:
    value = decode_attr(raw)
    if isinstance(value, str) and value == _NONE_SENTINEL:
        return None
    return value


def load_shape_recommendation_record(
    file_path: str,
) -> Optional[ShapeRecommendationRecord]:
    """The shape recommendation's record, or ``None`` when the file has none.

    The record is removed when a primary Stage 2b calibration resets the
    verdict, so its presence means the verdict it holds is the one in effect.
    """
    with h5py.File(file_path, "r") as h5f:
        grp = h5f.get(SHAPE_RECOMMENDATION_SETTINGS_PATH)
        if not isinstance(grp, h5py.Group):
            return None
        settings = _read_producer_blocks(grp, PRODUCER_RECOMMENDATION)

        def _sub(name: str) -> Mapping[str, Any]:
            sub = grp.get(name)
            return sub.attrs if isinstance(sub, h5py.Group) else {}

        consumed_attrs = _sub("consumed")
        consumed: Dict[str, Optional[float]] = {
            k: (float(consumed_attrs[k]) if k in consumed_attrs else None)
            for k in SHAPE_RECOMMENDATION_CONSUMED_FIELDS
        }
        tau_raw = _decoded_or_none(_sub("effective").get("tau_max_us", _NONE_SENTINEL))
        verdict = _sub("verdict")
        shape = _decoded_or_none(verdict.get("recommended_shape", _NONE_SENTINEL))
        votes_raw = verdict.get("vote_rates")
        vote_rates: Dict[str, float] = (
            {}
            if votes_raw is None
            else {
                str(k): float(v) for k, v in json.loads(decode_attr(votes_raw)).items()
            }
        )
    return ShapeRecommendationRecord(
        settings=settings,
        consumed=consumed,
        tau_max_us=None if tau_raw is None else float(tau_raw),
        recommended_shape=None if shape is None else str(shape),
        vote_rates=vote_rates,
    )


def delete_shape_recommendation_record(file_path: str) -> bool:
    """Remove the shape recommendation's record; ``True`` if one was removed.

    Called when a primary Stage 2b calibration resets the verdict, so a
    withdrawn verdict is never read as the one in effect.
    """
    with h5py.File(file_path, "a") as h5f:
        if SHAPE_RECOMMENDATION_SETTINGS_PATH not in h5f:
            return False
        del h5f[SHAPE_RECOMMENDATION_SETTINGS_PATH]
        return True


def shape_recommendation_provenance(file_path: str) -> Optional[RecordProvenance]:
    """The recommendation record's field-set version against
    :data:`SHAPE_RECOMMENDATION_FIELD_SET_VERSION`, or ``None`` if absent."""
    return tau_producer_settings_provenance(file_path, PRODUCER_RECOMMENDATION)


__all__ = [
    "STAGE2B_TAU_SETTINGS_PATH",
    "STAGE2B_TAU_FIELD_SET_VERSION",
    "STAGE2B_LORENTZIAN_SETTINGS_PATH",
    "STAGE2B_LORENTZIAN_FIELD_SET_VERSION",
    "STAGE2B_GAUSSIAN_SETTINGS_PATH",
    "STAGE2B_GAUSSIAN_FIELD_SET_VERSION",
    "SHAPE_RECOMMENDATION_SETTINGS_PATH",
    "SHAPE_RECOMMENDATION_FIELD_SET_VERSION",
    "SHAPE_RECOMMENDATION_CONSUMED_FIELDS",
    "ShapeRecommendationRecord",
    "tau_calibration_settings_provenance",
    "save_tau_calibration_settings_to_h5",
    "load_tau_calibration_settings_from_h5",
    "tau_calibration_settings_present",
    "save_tau_producer_settings_to_h5",
    "load_tau_producer_settings_from_h5",
    "tau_producer_settings_provenance",
    "save_shape_recommendation_record",
    "load_shape_recommendation_record",
    "delete_shape_recommendation_record",
    "shape_recommendation_provenance",
]
