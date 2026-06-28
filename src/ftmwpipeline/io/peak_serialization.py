"""
Stage 3 peak-list serialization to HDF5.

Peaks are scientific output that may be **hand-edited** between Stage 3 and
Stage 4 (the detector and window assignment are deliberately decoupled, see the
Stage 3 plan), so the on-disk form is a flat, obvious set of equal-length
parallel arrays plus the detection parameters/provenance. The list is persisted
(not recomputed on demand).

HDF5 layout (under the caller-provided group, e.g. ``/stage3_peaks``)::

    frequency          [f8]  MHz, on the persisted user grid
    intensity          [f8]  magnitude, re-measured on the user spectrum
    index              [i8]  index into the user-grid spectrum
    snr                [f8]  excess-over-leakage SNR, (|X|-pedestal)/sigma,
                             vs the canonical Stage 2 noise (user grid)
    noise_std_local    [f8]
    classification     [str] "weak" | "medium" | "strong" | ""
    detection_pass     [str] "primary" | "gap" | ""
    internal_snr       [f8]  SNR on the internal zpf=1 detection grid
    internal_frequency [f8]  MHz on the internal detection grid
    leakage_pedestal   [f8]  local coherent-leakage pedestal subtracted in snr
    .attrs:
        n_peaks, creation_time, stage_name, parameters (JSON),
        promotion_min_snr, internal_min_snr

*All* detected peaks are stored. ``promotion_min_snr`` is the cutoff on the
user-grid ``snr`` deciding which peaks move to Stage 4; ``load`` derives a
per-peak ``properties['promoted']`` flag from it (it is not stored, so the
threshold stays the single source of truth and an edited list re-derives
cleanly). ``internal_snr``/``internal_frequency`` are detection provenance for
curation and snap-back diagnosis -- not consumed by Stage 4 -- and are
optional on read so pre-existing files still load.

Round-trip contract: ``load`` -> edit the list -> ``save`` -> ``load`` returns
the edited list. A malformed/edited group (missing required column, mismatched
lengths, unknown classification label) raises ``ValueError`` loudly rather
than silently dropping or guessing data.
"""

import json
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from ..core.data_structures import Peak, PeakClassification
from ._hdf5_helpers import nan_if_none, none_if_nan, stamp_stage_header

_COLUMNS = (
    "frequency",
    "intensity",
    "index",
    "snr",
    "noise_std_local",
    "classification",
    "detection_pass",
)

# Detection provenance: written by current code, optional on read.
_OPTIONAL_COLUMNS = (
    "internal_snr",
    "internal_frequency",
    "leakage_pedestal",
)

_VALID_CLASSES = {c.value for c in PeakClassification}


def _prop_float(peak: Peak, key: str) -> float:
    """Provenance value from ``peak.properties`` as float (NaN if absent)."""
    return nan_if_none(peak.properties.get(key))


def save_peaks_to_hdf5(
    peaks: List[Peak],
    h5_group: h5py.Group,
    parameters: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a peak list to an HDF5 group as parallel arrays.

    Parameters
    ----------
    peaks : list of Peak
        Peaks to persist (any order; not re-sorted here).
    h5_group : h5py.Group
        Destination group. Existing peak columns are overwritten.
    parameters : dict, optional
        Detection parameters/provenance, stored as a JSON attribute. If it
        carries ``promotion_min_snr`` / ``internal_min_snr`` those are also
        promoted to dedicated scalar attributes (the former drives the
        ``promoted`` flag on load).

    Raises
    ------
    ValueError
        If a peak carries an unknown classification value.
    """
    n = len(peaks)
    frequency: np.ndarray = np.empty(n, dtype="f8")
    intensity: np.ndarray = np.empty(n, dtype="f8")
    index: np.ndarray = np.empty(n, dtype="i8")
    snr: np.ndarray = np.empty(n, dtype="f8")
    noise_std_local: np.ndarray = np.empty(n, dtype="f8")
    internal_snr: np.ndarray = np.empty(n, dtype="f8")
    internal_frequency: np.ndarray = np.empty(n, dtype="f8")
    leakage_pedestal: np.ndarray = np.empty(n, dtype="f8")
    classification: List[str] = []
    detection_pass: List[str] = []

    for i, p in enumerate(peaks):
        frequency[i] = p.frequency
        intensity[i] = p.intensity
        index[i] = -1 if p.index is None else int(p.index)
        snr[i] = nan_if_none(p.snr)
        noise_std_local[i] = nan_if_none(p.noise_std_local)
        internal_snr[i] = _prop_float(p, "internal_snr")
        internal_frequency[i] = _prop_float(p, "internal_frequency")
        leakage_pedestal[i] = _prop_float(p, "leakage_pedestal")
        cls: Any = p.classification
        if cls is None:
            classification.append("")
        elif isinstance(cls, PeakClassification):
            classification.append(cls.value)
        else:
            raise ValueError(
                f"peak {i} has non-PeakClassification classification " f"{cls!r}"
            )
        detection_pass.append(str(p.properties.get("detection_pass", "")))

    str_dt = h5py.string_dtype(encoding="utf-8")
    payload = {
        "frequency": frequency,
        "intensity": intensity,
        "index": index,
        "snr": snr,
        "noise_std_local": noise_std_local,
        "internal_snr": internal_snr,
        "internal_frequency": internal_frequency,
        "leakage_pedestal": leakage_pedestal,
        "classification": np.array(classification, dtype=object),
        "detection_pass": np.array(detection_pass, dtype=object),
    }
    for name, data in payload.items():
        if name in h5_group:
            del h5_group[name]
        if name in ("classification", "detection_pass"):
            h5_group.create_dataset(name, data=data, dtype=str_dt)
        else:
            h5_group.create_dataset(name, data=data)

    params = parameters or {}
    stamp_stage_header(h5_group, "stage3_peaks", n_peaks=n)
    h5_group.attrs["parameters"] = json.dumps(params, default=str)
    for attr in ("promotion_min_snr", "internal_min_snr"):
        h5_group.attrs[attr] = nan_if_none(params.get(attr))


def load_peaks_from_hdf5(h5_group: h5py.Group) -> List[Peak]:
    """Load a peak list from an HDF5 group, validating structure loudly.

    A ``properties['promoted']`` flag is derived from the group's
    ``promotion_min_snr`` attribute (peak promoted iff its user-grid ``snr``
    meets the cutoff). Detection provenance columns, when present, are
    restored into ``properties``.

    Raises
    ------
    ValueError
        If a required column is missing, columns have mismatched lengths, or a
        classification label is not one of weak/medium/strong/"".
    """
    missing = [c for c in _COLUMNS if c not in h5_group]
    if missing:
        raise ValueError(f"stage3_peaks group missing required column(s): {missing}")

    present_optional = [c for c in _OPTIONAL_COLUMNS if c in h5_group]
    cols = {c: h5_group[c][:] for c in (*_COLUMNS, *present_optional)}
    lengths = {c: len(v) for c, v in cols.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"stage3_peaks columns have mismatched lengths: {lengths}")

    promotion_attr = h5_group.attrs.get("promotion_min_snr")
    promotion_min_snr: Optional[float] = None
    if promotion_attr is not None:
        promotion_attr = float(promotion_attr)
        if not np.isnan(promotion_attr):
            promotion_min_snr = promotion_attr

    peaks: List[Peak] = []
    for i in range(next(iter(lengths.values()))):
        raw_class = cols["classification"][i]
        if isinstance(raw_class, bytes):
            raw_class = raw_class.decode("utf-8")
        if raw_class not in _VALID_CLASSES and raw_class != "":
            raise ValueError(
                f"peak {i} has invalid classification {raw_class!r}; "
                f"expected one of {sorted(_VALID_CLASSES)} or empty"
            )
        raw_pass = cols["detection_pass"][i]
        if isinstance(raw_pass, bytes):
            raw_pass = raw_pass.decode("utf-8")

        idx = int(cols["index"][i])
        snr_val = float(cols["snr"][i])
        nsl_val = float(cols["noise_std_local"][i])

        extra: Dict[str, Any] = {"detection_pass": raw_pass}
        for opt in present_optional:
            extra[opt] = none_if_nan(cols[opt][i])
        if promotion_min_snr is not None:
            extra["promoted"] = not np.isnan(snr_val) and snr_val >= promotion_min_snr

        peaks.append(
            Peak(
                frequency=float(cols["frequency"][i]),
                intensity=float(cols["intensity"][i]),
                index=None if idx < 0 else idx,
                snr=none_if_nan(snr_val),
                noise_std_local=none_if_nan(nsl_val),
                classification=raw_class if raw_class != "" else None,
                **extra,
            )
        )
    return peaks
