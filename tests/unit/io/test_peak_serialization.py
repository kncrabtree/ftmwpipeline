"""
Tests for Stage 3 peak-list HDF5 serialization.

Covers the round-trip, the hand-edit-between-stages contract, and the
loud-failure requirement for malformed/edited groups.
"""

import h5py
import numpy as np
import pytest

from ftmwpipeline.core.data_structures import Peak, PeakClassification
from ftmwpipeline.io.peak_serialization import (
    save_peaks_to_hdf5,
    load_peaks_from_hdf5,
)


def _sample_peaks():
    return [
        Peak(
            frequency=30000.0,
            intensity=1.0,
            index=10,
            snr=300.0,
            noise_std_local=0.0033,
            classification=PeakClassification.STRONG,
            detection_pass="primary",
        ),
        Peak(
            frequency=31000.0,
            intensity=0.05,
            index=200,
            snr=5.0,
            noise_std_local=0.01,
            classification=PeakClassification.WEAK,
            detection_pass="gap",
        ),
        # Edge: unclassified, no index/snr.
        Peak(frequency=32000.0, intensity=0.5),
    ]


def test_round_trip_preserves_peaks(tmp_path):
    f = tmp_path / "peaks.h5"
    peaks = _sample_peaks()
    with h5py.File(f, "w") as h5:
        save_peaks_to_hdf5(
            peaks, h5.create_group("stage3_peaks"), parameters={"min_snr": 3.0}
        )
    with h5py.File(f, "r") as h5:
        loaded = load_peaks_from_hdf5(h5["stage3_peaks"])

    assert len(loaded) == 3
    a, b, c = loaded
    assert a.frequency == pytest.approx(30000.0)
    assert a.snr == pytest.approx(300.0)
    assert a.classification is PeakClassification.STRONG
    assert a.properties["detection_pass"] == "primary"
    assert b.classification is PeakClassification.WEAK
    assert b.properties["detection_pass"] == "gap"
    # Unclassified / missing fields survive as None.
    assert c.classification is None
    assert c.index is None
    assert c.snr is None
    assert c.properties["detection_pass"] == ""


def test_hand_edit_between_stages_round_trip(tmp_path):
    """Load -> edit/delete a peak -> save -> reload yields the edited list."""
    f = tmp_path / "peaks.h5"
    with h5py.File(f, "w") as h5:
        save_peaks_to_hdf5(_sample_peaks(), h5.create_group("stage3_peaks"))
    with h5py.File(f, "r") as h5:
        peaks = load_peaks_from_hdf5(h5["stage3_peaks"])

    # Curate: drop the unclassified one, shift a frequency, reclassify.
    peaks = peaks[:2]
    peaks[0].frequency = 30000.123
    peaks[1].classification = PeakClassification.MEDIUM

    with h5py.File(f, "a") as h5:
        del h5["stage3_peaks"]
        save_peaks_to_hdf5(peaks, h5.create_group("stage3_peaks"))
    with h5py.File(f, "r") as h5:
        reloaded = load_peaks_from_hdf5(h5["stage3_peaks"])

    assert len(reloaded) == 2
    assert reloaded[0].frequency == pytest.approx(30000.123)
    assert reloaded[1].classification is PeakClassification.MEDIUM


def test_missing_column_fails_loudly(tmp_path):
    f = tmp_path / "peaks.h5"
    with h5py.File(f, "w") as h5:
        save_peaks_to_hdf5(_sample_peaks(), h5.create_group("stage3_peaks"))
    with h5py.File(f, "a") as h5:
        del h5["stage3_peaks"]["snr"]  # simulate a botched hand-edit
    with h5py.File(f, "r") as h5:
        with pytest.raises(ValueError, match="missing required column"):
            load_peaks_from_hdf5(h5["stage3_peaks"])


def test_mismatched_lengths_fail_loudly(tmp_path):
    f = tmp_path / "peaks.h5"
    with h5py.File(f, "w") as h5:
        save_peaks_to_hdf5(_sample_peaks(), h5.create_group("stage3_peaks"))
    with h5py.File(f, "a") as h5:
        g = h5["stage3_peaks"]
        del g["frequency"]
        g.create_dataset("frequency", data=np.array([1.0, 2.0]))  # wrong len
    with h5py.File(f, "r") as h5:
        with pytest.raises(ValueError, match="mismatched lengths"):
            load_peaks_from_hdf5(h5["stage3_peaks"])


def test_invalid_classification_label_fails_loudly(tmp_path):
    f = tmp_path / "peaks.h5"
    with h5py.File(f, "w") as h5:
        save_peaks_to_hdf5(_sample_peaks(), h5.create_group("stage3_peaks"))
    with h5py.File(f, "a") as h5:
        g = h5["stage3_peaks"]
        del g["classification"]
        g.create_dataset(
            "classification",
            data=np.array(["strong", "bogus", ""], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    with h5py.File(f, "r") as h5:
        with pytest.raises(ValueError, match="invalid classification"):
            load_peaks_from_hdf5(h5["stage3_peaks"])
