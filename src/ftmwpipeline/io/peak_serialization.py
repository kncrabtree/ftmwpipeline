"""
Stage 3 peak-list serialization to HDF5.

Peaks are scientific output that may be **hand-edited** between Stage 3 and
Stage 4 (the detector and window assignment are deliberately decoupled, see the
Stage 3 plan), so the on-disk form is a flat, obvious set of equal-length
parallel arrays plus the detection parameters/provenance. The list is persisted
(not recomputed on demand).

HDF5 layout (under the caller-provided group, e.g. ``/stage3_peaks``)::

    frequency          [f8]  MHz
    intensity          [f8]  magnitude
    index              [i8]  index into the gap/reference spectrum grid
    snr                [f8]
    noise_std_local    [f8]
    classification     [str] "weak" | "medium" | "strong" | ""
    detection_pass     [str] "primary" | "gap" | ""
    .attrs:
        n_peaks, creation_time, stage_name, parameters (JSON)

Round-trip contract: ``load`` -> edit the list -> ``save`` -> ``load`` returns
the edited list. A malformed/edited group (missing column, mismatched lengths,
unknown classification label) raises ``ValueError`` loudly rather than
silently dropping or guessing data.
"""

from datetime import datetime
import json
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from ..core.data_structures import Peak, PeakClassification

_COLUMNS = (
    "frequency",
    "intensity",
    "index",
    "snr",
    "noise_std_local",
    "classification",
    "detection_pass",
)

_VALID_CLASSES = {c.value for c in PeakClassification}


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
        Detection parameters/provenance, stored as a JSON attribute.

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
    classification: List[str] = []
    detection_pass: List[str] = []

    for i, p in enumerate(peaks):
        frequency[i] = p.frequency
        intensity[i] = p.intensity
        index[i] = -1 if p.index is None else int(p.index)
        snr[i] = np.nan if p.snr is None else float(p.snr)
        noise_std_local[i] = (
            np.nan if p.noise_std_local is None else float(p.noise_std_local)
        )
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

    h5_group.attrs["n_peaks"] = n
    h5_group.attrs["creation_time"] = datetime.now().isoformat()
    h5_group.attrs["stage_name"] = "stage3_peaks"
    h5_group.attrs["parameters"] = json.dumps(parameters or {}, default=str)


def load_peaks_from_hdf5(h5_group: h5py.Group) -> List[Peak]:
    """Load a peak list from an HDF5 group, validating structure loudly.

    Parameters
    ----------
    h5_group : h5py.Group
        Group previously written by :func:`save_peaks_to_hdf5` (possibly
        hand-edited).

    Returns
    -------
    list of Peak

    Raises
    ------
    ValueError
        If a required column is missing, columns have mismatched lengths, or a
        classification label is not one of weak/medium/strong/"".
    """
    missing = [c for c in _COLUMNS if c not in h5_group]
    if missing:
        raise ValueError(f"stage3_peaks group missing required column(s): {missing}")

    cols = {c: h5_group[c][:] for c in _COLUMNS}
    lengths = {c: len(v) for c, v in cols.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"stage3_peaks columns have mismatched lengths: {lengths}")

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
        peaks.append(
            Peak(
                frequency=float(cols["frequency"][i]),
                intensity=float(cols["intensity"][i]),
                index=None if idx < 0 else idx,
                snr=None if np.isnan(snr_val) else snr_val,
                noise_std_local=None if np.isnan(nsl_val) else nsl_val,
                classification=raw_class if raw_class != "" else None,
                detection_pass=raw_pass,
            )
        )
    return peaks
