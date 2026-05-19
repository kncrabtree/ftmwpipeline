"""
Stage 3 integration tests: store-all / promotion / provenance on real 2638 data.

Behaviour under test:
  - detect_peaks_impl stores ALL detected peaks (no pre-filter at promotion floor).
  - internal_min_snr == 2.0 (DEFAULT_INTERNAL_MIN_SNR) for default call.
  - promotion_min_snr == 3.0 (DEFAULT_MIN_SNR) for default call.
  - n_peaks > n_promoted > 0: some peaks exist below the promotion cutoff.
  - Every peak with user-grid snr >= 3.0 has properties['promoted'] is True;
    those with snr < 3.0 have properties['promoted'] is False.
  - properties carry internal_snr and internal_frequency; NOT internal_index.
  - On-disk: stage3_peaks group has internal_snr/internal_frequency datasets
    and promotion_min_snr/internal_min_snr attrs.
  - load_peaks_impl returns n_promoted matching the in-memory count.
  - Varying the promotion cutoff:
      min_snr=1.5  -> internal_min_snr=1.5  (below floor -> floor follows)
      min_snr=10.0 -> internal_min_snr=2.0  (above floor -> floor stays at 2.0)
  - Visualization: show_snr_histogram=True -> 2 axes; default -> 1 axis.

All slow tests are marked @pytest.mark.integration and @pytest.mark.slow.
A module-scoped fixture runs detection once; individual tests reuse the result.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import h5py

from ftmwpipeline._internal.stage0_impl import import_data_impl
from ftmwpipeline._internal.stage3_impl import (
    detect_peaks_impl,
    load_peaks_impl,
    visualize_peaks_impl,
)
from ftmwpipeline.preprocessing.peak_detection import (
    DEFAULT_INTERNAL_MIN_SNR,
    DEFAULT_MIN_SNR,
)
import ftmwpipeline.api as ftmw

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TRIM = (26500.0, 40000.0)


# ---------------------------------------------------------------------------
# Module-scoped fixture: one detection run shared across tests in this file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stage3_result(exp_2638_data_path, tmp_path_factory):
    """
    Prepare 2638 through Stage 2 with zpf=1/expf_us=5.0/trim, then run
    detect_peaks_impl with default settings.  The file path is also returned
    so individual tests can open it for on-disk checks.

    NOTE: per the CLAUDE.md instructions the task specifies zpf=1 for Stage 3
    tests; this matches the reference Stage 3 test suite.
    """
    tmp = tmp_path_factory.mktemp("stage3_promotion")
    fp = str(tmp / "exp.ftmw")
    import_data_impl(fp, source=exp_2638_data_path)
    ftmw.compute_ft(fp, zpf=1, expf_us=5.0, trim=TRIM)
    ftmw.estimate_noise(fp)
    result = detect_peaks_impl(fp)
    return result, fp


# ---------------------------------------------------------------------------
# Core promotion / floor assertions
# ---------------------------------------------------------------------------

class TestPromotionDefaults:
    def test_internal_min_snr_is_floor(self, stage3_result):
        """internal_min_snr must equal DEFAULT_INTERNAL_MIN_SNR (2.0) on a default run."""
        r, _ = stage3_result
        assert r["internal_min_snr"] == pytest.approx(DEFAULT_INTERNAL_MIN_SNR)

    def test_promotion_min_snr_is_default(self, stage3_result):
        """promotion_min_snr must equal DEFAULT_MIN_SNR (3.0) on a default run."""
        r, _ = stage3_result
        assert r["promotion_min_snr"] == pytest.approx(DEFAULT_MIN_SNR)

    def test_total_peaks_exceeds_promoted(self, stage3_result):
        """Store-all: n_peaks > n_promoted (sub-threshold peaks are kept)."""
        r, _ = stage3_result
        assert r["n_peaks"] > r["n_promoted"], (
            f"n_peaks={r['n_peaks']}, n_promoted={r['n_promoted']}; "
            "expected some sub-promotion-threshold peaks in the stored list"
        )

    def test_some_peaks_are_promoted(self, stage3_result):
        """At least some peaks must clear the promotion threshold."""
        r, _ = stage3_result
        assert r["n_promoted"] > 0, "No peaks promoted at SNR>=3.0"

    def test_promoted_flag_true_when_snr_above_cutoff(self, stage3_result):
        """Every peak with snr >= 3.0 must have properties['promoted'] is True."""
        r, _ = stage3_result
        cutoff = r["promotion_min_snr"]
        bad = [
            p for p in r["peaks"]
            if p.snr is not None and p.snr >= cutoff
            and not p.properties.get("promoted")
        ]
        assert not bad, (
            f"{len(bad)} peaks with snr>=cutoff lack promoted=True: "
            + ", ".join(f"{p.frequency:.2f} MHz (snr={p.snr:.2f})" for p in bad[:5])
        )

    def test_promoted_flag_false_when_snr_below_cutoff(self, stage3_result):
        """Every peak with snr < 3.0 must have properties['promoted'] is False."""
        r, _ = stage3_result
        cutoff = r["promotion_min_snr"]
        bad = [
            p for p in r["peaks"]
            if p.snr is not None and p.snr < cutoff
            and p.properties.get("promoted") is not False
        ]
        assert not bad, (
            f"{len(bad)} peaks with snr<cutoff lack promoted=False: "
            + ", ".join(f"{p.frequency:.2f} MHz (snr={p.snr:.2f})" for p in bad[:5])
        )

    def test_no_mismatched_promoted_flags(self, stage3_result):
        """Zero mismatches between snr threshold and promoted flag."""
        r, _ = stage3_result
        cutoff = r["promotion_min_snr"]
        peaks = r["peaks"]
        mismatches = 0
        for p in peaks:
            if p.snr is None:
                continue
            expected = p.snr >= cutoff
            actual = p.properties.get("promoted")
            if actual is not expected:
                mismatches += 1
        assert mismatches == 0, f"{mismatches} promoted-flag mismatches found"


# ---------------------------------------------------------------------------
# Provenance properties
# ---------------------------------------------------------------------------

class TestProvenanceProperties:
    def test_internal_snr_in_properties(self, stage3_result):
        """Every peak must have internal_snr in properties."""
        r, _ = stage3_result
        peaks = r["peaks"]
        missing = [p for p in peaks if "internal_snr" not in p.properties]
        assert not missing, (
            f"{len(missing)} peaks lack properties['internal_snr']"
        )

    def test_internal_frequency_in_properties(self, stage3_result):
        """Every peak must have internal_frequency in properties."""
        r, _ = stage3_result
        peaks = r["peaks"]
        missing = [p for p in peaks if "internal_frequency" not in p.properties]
        assert not missing, (
            f"{len(missing)} peaks lack properties['internal_frequency']"
        )

    def test_internal_index_absent(self, stage3_result):
        """internal_index must NOT be in peak properties (intentionally dropped)."""
        r, _ = stage3_result
        peaks = r["peaks"]
        present = [p for p in peaks if "internal_index" in p.properties]
        assert not present, (
            f"{len(present)} peaks unexpectedly carry 'internal_index' in properties"
        )

    def test_internal_intensity_absent(self, stage3_result):
        """internal_intensity must NOT be in peak properties (intentionally dropped)."""
        r, _ = stage3_result
        peaks = r["peaks"]
        present = [p for p in peaks if "internal_intensity" in p.properties]
        assert not present, (
            f"{len(present)} peaks unexpectedly carry 'internal_intensity' in properties"
        )


# ---------------------------------------------------------------------------
# On-disk structure: HDF5 datasets and attrs
# ---------------------------------------------------------------------------

class TestOnDiskStructure:
    def test_internal_snr_dataset_present(self, stage3_result):
        """stage3_peaks group must contain an internal_snr dataset."""
        _, fp = stage3_result
        with h5py.File(fp, "r") as h5:
            assert "internal_snr" in h5["stage3_peaks"], (
                "internal_snr dataset missing from stage3_peaks group"
            )

    def test_internal_frequency_dataset_present(self, stage3_result):
        """stage3_peaks group must contain an internal_frequency dataset."""
        _, fp = stage3_result
        with h5py.File(fp, "r") as h5:
            assert "internal_frequency" in h5["stage3_peaks"], (
                "internal_frequency dataset missing from stage3_peaks group"
            )

    def test_promotion_min_snr_attr(self, stage3_result):
        """promotion_min_snr group attr must equal 3.0."""
        _, fp = stage3_result
        with h5py.File(fp, "r") as h5:
            attr = float(h5["stage3_peaks"].attrs["promotion_min_snr"])
        assert attr == pytest.approx(DEFAULT_MIN_SNR)

    def test_internal_min_snr_attr(self, stage3_result):
        """internal_min_snr group attr must equal 2.0."""
        _, fp = stage3_result
        with h5py.File(fp, "r") as h5:
            attr = float(h5["stage3_peaks"].attrs["internal_min_snr"])
        assert attr == pytest.approx(DEFAULT_INTERNAL_MIN_SNR)

    def test_load_peaks_impl_n_promoted_matches(self, stage3_result):
        """load_peaks_impl must return n_promoted matching in-memory count."""
        r, fp = stage3_result
        loaded = load_peaks_impl(fp)
        # n_promoted from load should match n_promoted from detect
        assert loaded["n_promoted"] == r["n_promoted"], (
            f"detect returned n_promoted={r['n_promoted']} but load "
            f"returned n_promoted={loaded['n_promoted']}"
        )

    def test_load_peaks_impl_n_peaks_matches(self, stage3_result):
        """load_peaks_impl must return the same total peak count."""
        r, fp = stage3_result
        loaded = load_peaks_impl(fp)
        assert loaded["n_peaks"] == r["n_peaks"]

    def test_load_peaks_impl_promotion_min_snr_matches(self, stage3_result):
        """load_peaks_impl must return the stored promotion_min_snr."""
        r, fp = stage3_result
        loaded = load_peaks_impl(fp)
        assert loaded["promotion_min_snr"] == pytest.approx(r["promotion_min_snr"])


# ---------------------------------------------------------------------------
# Variable promotion cutoff (function-scoped; re-run detect on module file)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stage3_file(stage3_result):
    """Return just the file path from the module fixture (pre-Stage-2 already done)."""
    _, fp = stage3_result
    return fp


def test_min_snr_below_floor_lowers_internal_floor(stage3_file):
    """min_snr=1.5 (below DEFAULT_INTERNAL_MIN_SNR=2.0) -> internal_min_snr=1.5."""
    r = detect_peaks_impl(stage3_file, min_snr=1.5)
    assert r["internal_min_snr"] == pytest.approx(1.5), (
        f"Expected internal_min_snr=1.5, got {r['internal_min_snr']}"
    )
    assert r["promotion_min_snr"] == pytest.approx(1.5)


def test_min_snr_above_floor_keeps_internal_floor(stage3_file):
    """min_snr=10.0 -> internal_min_snr stays at 2.0 (floor is not raised)."""
    r = detect_peaks_impl(stage3_file, min_snr=10.0)
    assert r["internal_min_snr"] == pytest.approx(DEFAULT_INTERNAL_MIN_SNR), (
        f"Expected internal_min_snr=2.0, got {r['internal_min_snr']}"
    )
    assert r["promotion_min_snr"] == pytest.approx(10.0)


def test_lower_internal_floor_yields_at_least_as_many_peaks(stage3_file, stage3_result):
    """Detection at floor=1.5 must yield >= as many stored peaks as the default (2.0)."""
    default_n = stage3_result[0]["n_peaks"]
    r_low = detect_peaks_impl(stage3_file, min_snr=1.5)
    assert r_low["n_peaks"] >= default_n, (
        f"Lower floor (1.5) yielded fewer peaks ({r_low['n_peaks']}) than "
        f"default floor (2.0, {default_n} peaks)"
    )


# ---------------------------------------------------------------------------
# Visualization: axis count
# ---------------------------------------------------------------------------

def test_visualize_peaks_default_one_axis(stage3_file):
    """Default visualize_peaks_impl must return a figure with exactly 1 axis."""
    import matplotlib.pyplot as plt
    # Restore default run so there are peaks on disk to visualize.
    detect_peaks_impl(stage3_file)  # re-run with default params
    fig = visualize_peaks_impl(stage3_file, interactive=False)
    try:
        assert len(fig.get_axes()) == 1, (
            f"Expected 1 axis for default plot, got {len(fig.get_axes())}"
        )
    finally:
        plt.close(fig)


def test_visualize_peaks_snr_histogram_two_axes(stage3_file):
    """show_snr_histogram=True must return a figure with exactly 2 axes."""
    import matplotlib.pyplot as plt
    fig = visualize_peaks_impl(stage3_file, show_snr_histogram=True, interactive=False)
    try:
        assert len(fig.get_axes()) == 2, (
            f"Expected 2 axes with snr_histogram, got {len(fig.get_axes())}"
        )
    finally:
        plt.close(fig)
