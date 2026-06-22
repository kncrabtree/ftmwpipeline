"""HDF5 serialization for :class:`TimebaseCalibrationResult`.

Persisted at ``/timebase_calibration``. The schema stores the scalar eps
estimate plus the per-tone diagnostic table; everything else is recomputable
from the FID and the persisted knobs.

HDF5 layout::

    /timebase_calibration/
        attrs:
            epsilon, sigma_epsilon, lattice_g_mhz, n_used, n_detected,
            kappa_sys, snr_min, sample_dt_us, start_us, end_us, span_us,
            preconditions_passed, creation_time
        preconditions_notes   dataset, str
        tones                 group:
            f_bb_mhz   (float64)
            k          (int64)
            df_mhz     (float64)
            sigma_mhz  (float64)
            snr        (float64)
            used       (bool)
            drift_control (bool)
        algorithm_info        group, attrs: method, version
"""

from __future__ import annotations

import h5py
import numpy as np

from ..fitting.timebase_calibration import (
    TimebaseCalibrationResult,
    TimebaseToneRead,
)
from ._hdf5_helpers import reset_group

SCHEMA_VERSION = "1.0"
GROUP_PATH = "timebase_calibration"


__all__ = [
    "save_timebase_calibration_to_hdf5",
    "load_timebase_calibration_from_hdf5",
    "SCHEMA_VERSION",
    "GROUP_PATH",
]


def save_timebase_calibration_to_hdf5(
    result: TimebaseCalibrationResult,
    h5_group: h5py.Group,
) -> None:
    """Save a :class:`TimebaseCalibrationResult` into the given HDF5 group.

    The group is wiped and rebuilt; pass a freshly-created group or accept
    that all of its contents will be replaced.
    """
    reset_group(h5_group, attrs=True)

    h5_group.attrs["epsilon"] = float(result.epsilon)
    h5_group.attrs["sigma_epsilon"] = float(result.sigma_epsilon)
    h5_group.attrs["lattice_g_mhz"] = float(result.lattice_g_mhz)
    h5_group.attrs["n_used"] = int(result.n_used)
    h5_group.attrs["n_detected"] = int(result.n_detected)
    h5_group.attrs["kappa_sys"] = float(result.kappa_sys)
    h5_group.attrs["snr_min"] = float(result.snr_min)
    h5_group.attrs["sample_dt_us"] = float(result.sample_dt_us)
    h5_group.attrs["start_us"] = float(result.start_us)
    h5_group.attrs["end_us"] = float(result.end_us)
    h5_group.attrs["span_us"] = float(result.span_us)
    h5_group.attrs["preconditions_passed"] = bool(result.preconditions_passed)

    notes = np.asarray(result.preconditions_notes, dtype=object)
    h5_group.create_dataset(
        "preconditions_notes",
        data=notes,
        dtype=h5py.string_dtype(encoding="utf-8"),
    )

    tg = h5_group.create_group("tones")
    tones = list(result.tone_reads)
    if tones:
        tg.create_dataset(
            "f_bb_mhz",
            data=np.asarray([t.f_bb_mhz for t in tones], dtype=np.float64),
        )
        tg.create_dataset("k", data=np.asarray([t.k for t in tones], dtype=np.int64))
        tg.create_dataset(
            "df_mhz", data=np.asarray([t.df_mhz for t in tones], dtype=np.float64)
        )
        tg.create_dataset(
            "sigma_mhz",
            data=np.asarray([t.sigma_mhz for t in tones], dtype=np.float64),
        )
        tg.create_dataset(
            "snr", data=np.asarray([t.snr for t in tones], dtype=np.float64)
        )
        tg.create_dataset("used", data=np.asarray([t.used for t in tones], dtype=bool))
        tg.create_dataset(
            "drift_control",
            data=np.asarray([t.drift_control for t in tones], dtype=bool),
        )
    else:
        tg.create_dataset("f_bb_mhz", data=np.zeros(0, dtype=np.float64))
        tg.create_dataset("k", data=np.zeros(0, dtype=np.int64))
        tg.create_dataset("df_mhz", data=np.zeros(0, dtype=np.float64))
        tg.create_dataset("sigma_mhz", data=np.zeros(0, dtype=np.float64))
        tg.create_dataset("snr", data=np.zeros(0, dtype=np.float64))
        tg.create_dataset("used", data=np.zeros(0, dtype=bool))
        tg.create_dataset("drift_control", data=np.zeros(0, dtype=bool))
    tg.attrs["n_tones"] = int(len(tones))

    ai = h5_group.create_group("algorithm_info")
    ai.attrs["method"] = "rb_locked_lattice_shared_eps"
    ai.attrs["version"] = SCHEMA_VERSION


def load_timebase_calibration_from_hdf5(
    h5_group: h5py.Group,
) -> TimebaseCalibrationResult:
    """Inverse of :func:`save_timebase_calibration_to_hdf5`."""
    a = dict(h5_group.attrs)

    tg = h5_group["tones"]
    f_bb = np.asarray(tg["f_bb_mhz"][:], dtype=np.float64)
    k_arr = np.asarray(tg["k"][:], dtype=np.int64)
    df_arr = np.asarray(tg["df_mhz"][:], dtype=np.float64)
    sigma_arr = np.asarray(tg["sigma_mhz"][:], dtype=np.float64)
    snr_arr = np.asarray(tg["snr"][:], dtype=np.float64)
    used_arr = np.asarray(tg["used"][:], dtype=bool)
    drift_arr = np.asarray(tg["drift_control"][:], dtype=bool)
    tones = tuple(
        TimebaseToneRead(
            f_bb_mhz=float(f_bb[i]),
            k=int(k_arr[i]),
            df_mhz=float(df_arr[i]),
            sigma_mhz=float(sigma_arr[i]),
            snr=float(snr_arr[i]),
            used=bool(used_arr[i]),
            drift_control=bool(drift_arr[i]),
        )
        for i in range(int(f_bb.size))
    )

    notes_raw = h5_group["preconditions_notes"][:]
    notes = tuple(
        (n.decode("utf-8") if isinstance(n, bytes) else str(n)) for n in notes_raw
    )

    return TimebaseCalibrationResult(
        epsilon=float(a["epsilon"]),
        sigma_epsilon=float(a["sigma_epsilon"]),
        n_used=int(a["n_used"]),
        n_detected=int(a["n_detected"]),
        lattice_g_mhz=float(a["lattice_g_mhz"]),
        tone_reads=tones,
        kappa_sys=float(a["kappa_sys"]),
        snr_min=float(a["snr_min"]),
        sample_dt_us=float(a["sample_dt_us"]),
        start_us=float(a["start_us"]),
        end_us=float(a["end_us"]),
        span_us=float(a["span_us"]),
        preconditions_passed=bool(a["preconditions_passed"]),
        preconditions_notes=notes,
    )
