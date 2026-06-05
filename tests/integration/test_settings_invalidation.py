"""
Integration tests for downstream invalidation when canonical FT settings change.

Behaviour under test (Behaviour A):
  - compute_ft (persist=True / user-driven) writes resolved settings to
    processing_parameters/ft_processing.
  - If the resolved settings DIFFER from the previously persisted record,
    every stage built on the FT (stage2_noise_result, stage3_peaks) is
    invalidated: its HDF5 group is deleted and it is removed from
    completed_stages.  A logging.WARNING is emitted.
  - An identical re-persist (same resolved settings) is idempotent: Stage 2
    / Stage 3 data are NOT touched.

All tests use the real 2638 experiment.
"""

import json
import logging

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage0_impl import import_data_impl
from ftmwpipeline._internal.stage1_impl import compute_ft_impl
from ftmwpipeline._internal.stage2_impl import compute_noise_estimation_impl
from ftmwpipeline._internal.stage3_impl import detect_peaks_impl

pytestmark = pytest.mark.integration

TRIM = (26500.0, 40000.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _completed(file_path: str):
    with h5py.File(file_path, "r") as h5f:
        return json.loads(h5f["pipeline_stages"].attrs.get("completed_stages", "[]"))


def _has_group(file_path: str, group: str) -> bool:
    with h5py.File(file_path, "r") as h5f:
        return group in h5f


def _prep_through_stage2(tmp_path, data_path) -> str:
    """Import 2638, compute FT, estimate noise; return path as str."""
    fp = str(tmp_path / "exp.ftmw")
    import_data_impl(fp, source=data_path)
    ftmw.compute_ft(fp, zpf=2, expf_us=5.0, trim=TRIM)
    ftmw.estimate_noise(fp)  # exercises the scatter settings chain
    return fp


# ---------------------------------------------------------------------------
# Stage 2 present after initial pipeline build
# ---------------------------------------------------------------------------

class TestInitialBuild:
    def test_stage2_group_present_and_completed(self, exp_2638_data_path, tmp_path):
        """After import -> compute_ft -> estimate_noise, stage2 is persisted."""
        fp = _prep_through_stage2(tmp_path, exp_2638_data_path)

        assert _has_group(fp, "stage2_noise_result"), (
            "stage2_noise_result HDF5 group missing after estimate_noise"
        )
        assert "stage2_noise_result" in _completed(fp), (
            "stage2_noise_result not in completed_stages"
        )
        assert "stage1_complex_ft" in _completed(fp)


# ---------------------------------------------------------------------------
# Idempotent re-persist: same settings -> Stage 2 survives
# ---------------------------------------------------------------------------

class TestIdempotentRepersist:
    def test_same_settings_leave_stage2_intact(self, exp_2638_data_path, tmp_path, caplog):
        """Re-running compute_ft with IDENTICAL settings must not invalidate Stage 2."""
        fp = _prep_through_stage2(tmp_path, exp_2638_data_path)

        with caplog.at_level(logging.WARNING, logger="ftmwpipeline"):
            ftmw.compute_ft(fp, zpf=2, expf_us=5.0, trim=TRIM)

        # Stage 2 must still be present.
        assert _has_group(fp, "stage2_noise_result"), (
            "Idempotent re-persist deleted stage2_noise_result — bug"
        )
        assert "stage2_noise_result" in _completed(fp), (
            "Idempotent re-persist removed stage2 from completed_stages — bug"
        )

        # No invalidation warning should have fired.
        invalidation_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "invalidated" in r.message.lower()
        ]
        assert not invalidation_warnings, (
            f"Unexpected invalidation warning on identical re-persist: "
            f"{invalidation_warnings}"
        )


# ---------------------------------------------------------------------------
# Changed settings -> Stage 2 invalidated
# ---------------------------------------------------------------------------

class TestChangedSettingsInvalidatesStage2:
    def test_changed_zpf_invalidates_stage2(self, exp_2638_data_path, tmp_path, caplog):
        """Changing zpf from 2 to 3 must delete stage2 and remove from completed."""
        fp = _prep_through_stage2(tmp_path, exp_2638_data_path)

        with caplog.at_level(logging.WARNING, logger="ftmwpipeline"):
            # Change zpf; this changes the resolved canonical settings.
            ftmw.compute_ft(fp, zpf=3, expf_us=5.0, trim=TRIM)

        # Stage 2 data must be gone.
        assert not _has_group(fp, "stage2_noise_result"), (
            "stage2_noise_result group persists after canonical settings change"
        )
        assert "stage2_noise_result" not in _completed(fp), (
            "stage2_noise_result remains in completed_stages after settings change"
        )

        # Stage 1 must still be completed (only downstream stages invalidated).
        assert "stage1_complex_ft" in _completed(fp)

        # A WARNING must have been logged mentioning invalidation.
        invalidation_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "invalidated" in r.message.lower()
        ]
        assert invalidation_warnings, (
            "No invalidation WARNING logged when canonical settings changed"
        )

    def test_changed_trim_invalidates_stage2(self, exp_2638_data_path, tmp_path, caplog):
        """Changing trim must invalidate Stage 2."""
        fp = _prep_through_stage2(tmp_path, exp_2638_data_path)

        with caplog.at_level(logging.WARNING, logger="ftmwpipeline"):
            ftmw.compute_ft(fp, zpf=2, expf_us=5.0, trim=(27000.0, 39000.0))

        assert not _has_group(fp, "stage2_noise_result"), (
            "stage2_noise_result survives a trim change"
        )
        assert "stage2_noise_result" not in _completed(fp)
        assert "stage1_complex_ft" in _completed(fp)


# ---------------------------------------------------------------------------
# Full chain: Stage 3 also invalidated when settings change
# ---------------------------------------------------------------------------

class TestChangedSettingsInvalidatesChain:
    def test_both_stage2_and_stage3_invalidated(self, exp_2638_data_path, tmp_path, caplog):
        """After compute_ft -> estimate_noise -> detect_peaks, a settings change
        must invalidate BOTH stage2_noise_result and stage3_peaks."""
        fp = _prep_through_stage2(tmp_path, exp_2638_data_path)

        # Run Stage 3 (detection is slow; run once at impl level).
        detect_peaks_impl(fp)

        assert _has_group(fp, "stage3_peaks"), "Stage 3 not persisted after detect_peaks"
        assert "stage3_peaks" in _completed(fp)

        # Now change canonical settings.
        with caplog.at_level(logging.WARNING, logger="ftmwpipeline"):
            ftmw.compute_ft(fp, zpf=3, expf_us=5.0, trim=TRIM)

        # Both downstream stages must be gone.
        assert not _has_group(fp, "stage2_noise_result"), (
            "stage2_noise_result persists after settings change (stage3 chain)"
        )
        assert not _has_group(fp, "stage3_peaks"), (
            "stage3_peaks persists after settings change"
        )
        assert "stage2_noise_result" not in _completed(fp)
        assert "stage3_peaks" not in _completed(fp)

        # Stage 1 unaffected.
        assert "stage1_complex_ft" in _completed(fp)

        # Warning logged.
        invalidation_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "invalidated" in r.message.lower()
        ]
        assert invalidation_warnings, "No invalidation warning for full-chain invalidation"
