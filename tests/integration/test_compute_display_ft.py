"""Integration tests for ``compute_display_ft`` (functional API + Pipeline).

The 2x-zero-padded active-region DISPLAY FT is the spectrum the Stage 5
report and 'fit show' detail panels render (see ``_padded_active_display_ft``
in ``_internal/stage5_impl.py``). These tests exercise the public accessor
against the session-scoped Stage 0+1 baseline only -- proving the dependency
is Stage 1 (the FID + canonical FT settings), not Stage 5.
"""

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.core.data_structures import ComplexFT

pytestmark = pytest.mark.integration


class TestComputeDisplayFt:
    def test_length_is_exactly_pad_factor_times_native(self, baseline_2638_stage1):
        """``pad_factor=2`` (the display default) must be exactly 2x the
        unpadded (``pad_factor=1``) active-region FFT length."""
        native = ftmw.compute_display_ft(baseline_2638_stage1, pad_factor=1)
        padded = ftmw.compute_display_ft(baseline_2638_stage1, pad_factor=2)
        assert padded.n_points == 2 * native.n_points

    def test_default_pad_factor_is_two(self, baseline_2638_stage1):
        default = ftmw.compute_display_ft(baseline_2638_stage1)
        explicit = ftmw.compute_display_ft(baseline_2638_stage1, pad_factor=2)
        assert default.n_points == explicit.n_points
        assert default.metadata["pad_factor"] == 2

    def test_frequencies_ascending(self, baseline_2638_stage1):
        display_ft = ftmw.compute_display_ft(baseline_2638_stage1)
        assert np.all(np.diff(display_ft.freq_array) > 0)

    def test_runs_on_stage1_only_file(self, baseline_2638_stage1):
        """``baseline_2638_stage1`` has no Stage 2-6 -- the call must not
        require a persisted fit."""
        display_ft = ftmw.compute_display_ft(baseline_2638_stage1)
        assert isinstance(display_ft, ComplexFT)
        assert display_ft.n_points > 0
        assert display_ft.complex_spectrum.shape == display_ft.freq_array.shape

    def test_metadata_carries_display_style(self, baseline_2638_stage1):
        display_ft = ftmw.compute_display_ft(baseline_2638_stage1)
        assert isinstance(display_ft.metadata["amplitude_scale"], float)
        assert isinstance(display_ft.metadata["units_label"], str)
        assert display_ft.metadata["pad_factor"] == 2

    def test_pipeline_and_functional_api_identical(self, baseline_2638_stage1):
        pipe = Pipeline.open(baseline_2638_stage1)
        pipeline_ft = pipe.compute_display_ft()
        api_ft = ftmw.compute_display_ft(baseline_2638_stage1)

        np.testing.assert_array_equal(pipeline_ft.freq_array, api_ft.freq_array)
        np.testing.assert_array_equal(
            pipeline_ft.complex_spectrum, api_ft.complex_spectrum
        )
        assert pipeline_ft.metadata == api_ft.metadata

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            ftmw.compute_display_ft(tmp_path / "does_not_exist.ftmw")
