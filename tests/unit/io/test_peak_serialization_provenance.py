"""
Tests for Stage 3 peak serialization: provenance columns and promotion flag.

Covers:
  - New optional columns internal_snr / internal_frequency written and round-tripped.
  - Group attrs promotion_min_snr / internal_min_snr stored correctly.
  - load_peaks_from_hdf5 derives properties['promoted'] from the promotion_min_snr
    attr (peak snr >= cutoff -> promoted=True; < cutoff -> promoted=False).
  - Backward compatibility: files written without the optional columns / attrs
    (simulating pre-promotion-flag saves) still load without error and without
    injecting internal_* or promoted keys.
"""

import h5py
import numpy as np
import pytest

from ftmwpipeline.core.data_structures import Peak, PeakClassification
from ftmwpipeline.io.peak_serialization import (
    _COLUMNS,
    load_peaks_from_hdf5,
    save_peaks_to_hdf5,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_peaks_with_provenance():
    """Two peaks: one promoted (snr=5.0 >= 3.0), one not (snr=2.1 < 3.0)."""
    promoted = Peak(
        frequency=28000.0,
        intensity=0.5,
        index=100,
        snr=5.0,
        noise_std_local=0.1,
        classification=PeakClassification.WEAK,
        detection_pass="primary",
        internal_snr=4.8,
        internal_frequency=28000.5,
    )
    not_promoted = Peak(
        frequency=29000.0,
        intensity=0.22,
        index=200,
        snr=2.1,
        noise_std_local=0.105,
        classification=PeakClassification.WEAK,
        detection_pass="gap",
        internal_snr=1.9,
        internal_frequency=28999.8,
    )
    return promoted, not_promoted


# ---------------------------------------------------------------------------
# Round-trip with provenance columns
# ---------------------------------------------------------------------------


class TestProvenanceRoundTrip:
    def test_optional_datasets_written(self, tmp_path):
        """save_peaks_to_hdf5 writes internal_snr and internal_frequency datasets."""
        f = tmp_path / "p.h5"
        peaks = list(_make_peaks_with_provenance())
        with h5py.File(f, "w") as h5:
            save_peaks_to_hdf5(
                peaks,
                h5.create_group("stage3_peaks"),
                parameters={"promotion_min_snr": 3.0, "internal_min_snr": 2.0},
            )
        with h5py.File(f, "r") as h5:
            grp = h5["stage3_peaks"]
            assert "internal_snr" in grp, "internal_snr dataset missing"
            assert "internal_frequency" in grp, "internal_frequency dataset missing"
            assert grp["internal_snr"].dtype == np.float64
            assert grp["internal_frequency"].dtype == np.float64

    def test_promotion_attrs_written(self, tmp_path):
        """promotion_min_snr and internal_min_snr written as scalar group attrs."""
        f = tmp_path / "p.h5"
        peaks = list(_make_peaks_with_provenance())
        with h5py.File(f, "w") as h5:
            save_peaks_to_hdf5(
                peaks,
                h5.create_group("stage3_peaks"),
                parameters={"promotion_min_snr": 3.0, "internal_min_snr": 2.0},
            )
        with h5py.File(f, "r") as h5:
            grp = h5["stage3_peaks"]
            assert float(grp.attrs["promotion_min_snr"]) == pytest.approx(3.0)
            assert float(grp.attrs["internal_min_snr"]) == pytest.approx(2.0)

    def test_round_trip_core_fields(self, tmp_path):
        """frequency, intensity, snr, classification survive the round-trip."""
        f = tmp_path / "p.h5"
        p_promoted, p_not = _make_peaks_with_provenance()
        with h5py.File(f, "w") as h5:
            save_peaks_to_hdf5(
                [p_promoted, p_not],
                h5.create_group("stage3_peaks"),
                parameters={"promotion_min_snr": 3.0, "internal_min_snr": 2.0},
            )
        with h5py.File(f, "r") as h5:
            loaded = load_peaks_from_hdf5(h5["stage3_peaks"])

        assert len(loaded) == 2
        a, b = loaded
        assert a.frequency == pytest.approx(28000.0)
        assert a.snr == pytest.approx(5.0)
        assert a.classification is PeakClassification.WEAK
        assert b.frequency == pytest.approx(29000.0)
        assert b.snr == pytest.approx(2.1)

    def test_internal_snr_and_frequency_restored_in_properties(self, tmp_path):
        """internal_snr and internal_frequency are present in properties after load."""
        f = tmp_path / "p.h5"
        p_promoted, p_not = _make_peaks_with_provenance()
        with h5py.File(f, "w") as h5:
            save_peaks_to_hdf5(
                [p_promoted, p_not],
                h5.create_group("stage3_peaks"),
                parameters={"promotion_min_snr": 3.0, "internal_min_snr": 2.0},
            )
        with h5py.File(f, "r") as h5:
            loaded = load_peaks_from_hdf5(h5["stage3_peaks"])

        a, b = loaded
        assert "internal_snr" in a.properties
        assert "internal_frequency" in a.properties
        assert a.properties["internal_snr"] == pytest.approx(4.8)
        assert a.properties["internal_frequency"] == pytest.approx(28000.5)
        assert b.properties["internal_snr"] == pytest.approx(1.9)
        assert b.properties["internal_frequency"] == pytest.approx(28999.8)

    def test_promoted_flag_derived_from_promotion_attr(self, tmp_path):
        """properties['promoted'] correctly reflects snr >= promotion_min_snr."""
        f = tmp_path / "p.h5"
        p_promoted, p_not = _make_peaks_with_provenance()
        with h5py.File(f, "w") as h5:
            save_peaks_to_hdf5(
                [p_promoted, p_not],
                h5.create_group("stage3_peaks"),
                parameters={"promotion_min_snr": 3.0, "internal_min_snr": 2.0},
            )
        with h5py.File(f, "r") as h5:
            loaded = load_peaks_from_hdf5(h5["stage3_peaks"])

        a, b = loaded
        # snr=5.0 >= 3.0 -> promoted
        assert a.properties["promoted"] is True
        # snr=2.1 < 3.0 -> not promoted
        assert b.properties["promoted"] is False

    def test_promoted_is_not_a_stored_column(self, tmp_path):
        """The 'promoted' flag must NOT be stored as an HDF5 dataset (derived on load)."""
        f = tmp_path / "p.h5"
        peaks = list(_make_peaks_with_provenance())
        with h5py.File(f, "w") as h5:
            save_peaks_to_hdf5(
                peaks,
                h5.create_group("stage3_peaks"),
                parameters={"promotion_min_snr": 3.0, "internal_min_snr": 2.0},
            )
        with h5py.File(f, "r") as h5:
            assert "promoted" not in h5["stage3_peaks"]

    def test_detection_pass_round_trips(self, tmp_path):
        """properties['detection_pass'] is preserved through save/load."""
        f = tmp_path / "p.h5"
        peaks = list(_make_peaks_with_provenance())
        with h5py.File(f, "w") as h5:
            save_peaks_to_hdf5(
                peaks,
                h5.create_group("stage3_peaks"),
                parameters={"promotion_min_snr": 3.0, "internal_min_snr": 2.0},
            )
        with h5py.File(f, "r") as h5:
            loaded = load_peaks_from_hdf5(h5["stage3_peaks"])
        assert loaded[0].properties["detection_pass"] == "primary"
        assert loaded[1].properties["detection_pass"] == "gap"


# ---------------------------------------------------------------------------
# Promotion boundary: exactly at the cutoff
# ---------------------------------------------------------------------------


class TestPromotionBoundary:
    def test_snr_exactly_at_cutoff_is_promoted(self, tmp_path):
        """A peak whose snr == promotion_min_snr exactly must be promoted."""
        cutoff = 4.0
        p = Peak(
            frequency=30000.0,
            intensity=0.4,
            snr=cutoff,
            noise_std_local=0.1,
            classification=PeakClassification.WEAK,
            detection_pass="primary",
        )
        f = tmp_path / "boundary.h5"
        with h5py.File(f, "w") as h5:
            save_peaks_to_hdf5(
                [p],
                h5.create_group("stage3_peaks"),
                parameters={"promotion_min_snr": cutoff},
            )
        with h5py.File(f, "r") as h5:
            loaded = load_peaks_from_hdf5(h5["stage3_peaks"])
        assert loaded[0].properties["promoted"] is True

    def test_snr_just_below_cutoff_not_promoted(self, tmp_path):
        """A peak just below the cutoff must not be promoted."""
        cutoff = 4.0
        p = Peak(
            frequency=30001.0,
            intensity=0.39,
            snr=cutoff - 1e-9,
            noise_std_local=0.1,
            classification=PeakClassification.WEAK,
            detection_pass="primary",
        )
        f = tmp_path / "just_below.h5"
        with h5py.File(f, "w") as h5:
            save_peaks_to_hdf5(
                [p],
                h5.create_group("stage3_peaks"),
                parameters={"promotion_min_snr": cutoff},
            )
        with h5py.File(f, "r") as h5:
            loaded = load_peaks_from_hdf5(h5["stage3_peaks"])
        assert loaded[0].properties["promoted"] is False


# ---------------------------------------------------------------------------
# Backward compatibility: old file (no optional columns / no promotion attr)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def _build_old_format_group(self, h5_group: h5py.Group, peaks) -> None:
        """Write peaks in the pre-promotion format: 7 required columns only, no
        promotion_min_snr / internal_min_snr attrs and no optional datasets."""
        # Use save then strip the new additions (simulates an old file).
        save_peaks_to_hdf5(peaks, h5_group, parameters={})
        # Remove the optional columns that the old format didn't have.
        for col in ("internal_snr", "internal_frequency"):
            if col in h5_group:
                del h5_group[col]
        # Set promotion attrs to NaN (as if they were never written).
        h5_group.attrs["promotion_min_snr"] = float("nan")
        h5_group.attrs["internal_min_snr"] = float("nan")

    def test_old_file_loads_without_error(self, tmp_path):
        """load_peaks_from_hdf5 must succeed on a pre-promotion-flag file."""
        f = tmp_path / "old.h5"
        peaks = [
            Peak(
                frequency=28000.0,
                intensity=0.5,
                index=100,
                snr=5.0,
                noise_std_local=0.1,
                classification=PeakClassification.WEAK,
                detection_pass="primary",
            )
        ]
        with h5py.File(f, "w") as h5:
            self._build_old_format_group(h5.create_group("stage3_peaks"), peaks)
        with h5py.File(f, "r") as h5:
            loaded = load_peaks_from_hdf5(h5["stage3_peaks"])
        assert len(loaded) == 1
        assert loaded[0].frequency == pytest.approx(28000.0)

    def test_old_file_has_no_internal_keys_in_properties(self, tmp_path):
        """Old files lack internal_snr / internal_frequency in properties."""
        f = tmp_path / "old.h5"
        peaks = [
            Peak(
                frequency=28000.0,
                intensity=0.5,
                index=100,
                snr=5.0,
                noise_std_local=0.1,
                classification=PeakClassification.WEAK,
                detection_pass="primary",
            )
        ]
        with h5py.File(f, "w") as h5:
            self._build_old_format_group(h5.create_group("stage3_peaks"), peaks)
        with h5py.File(f, "r") as h5:
            loaded = load_peaks_from_hdf5(h5["stage3_peaks"])
        p = loaded[0]
        assert "internal_snr" not in p.properties
        assert "internal_frequency" not in p.properties

    def test_old_file_has_no_promoted_key_in_properties(self, tmp_path):
        """Without a valid promotion_min_snr attr, no 'promoted' key is set."""
        f = tmp_path / "old.h5"
        peaks = [
            Peak(
                frequency=28000.0,
                intensity=0.5,
                snr=5.0,
                noise_std_local=0.1,
                detection_pass="primary",
            )
        ]
        with h5py.File(f, "w") as h5:
            self._build_old_format_group(h5.create_group("stage3_peaks"), peaks)
        with h5py.File(f, "r") as h5:
            loaded = load_peaks_from_hdf5(h5["stage3_peaks"])
        assert "promoted" not in loaded[0].properties

    def test_old_file_core_fields_still_valid(self, tmp_path):
        """frequency / snr / classification are still correct from an old file."""
        f = tmp_path / "old_core.h5"
        peaks = [
            Peak(
                frequency=31000.0,
                intensity=0.8,
                snr=12.0,
                noise_std_local=0.066,
                classification=PeakClassification.MEDIUM,
                detection_pass="primary",
            )
        ]
        with h5py.File(f, "w") as h5:
            self._build_old_format_group(h5.create_group("stage3_peaks"), peaks)
        with h5py.File(f, "r") as h5:
            loaded = load_peaks_from_hdf5(h5["stage3_peaks"])
        p = loaded[0]
        assert p.frequency == pytest.approx(31000.0)
        assert p.snr == pytest.approx(12.0)
        assert p.classification is PeakClassification.MEDIUM
