"""Integration tests for ``compute_display_ft`` (functional API + Pipeline).

The 2x-zero-padded DISPLAY FT is the spectrum the Stage 5 report and
'fit show' detail panels render (see ``_padded_active_display_ft`` in
``_internal/stage5_impl.py``). ``baseline_2638_stage1`` persists a real
narrowing trim (``(26500, 40000)`` MHz, out of a wider native active
region), so these tests exercise both invariants ``compute_display_ft``
must hold against ``compute_ft(from_saved_params=True)``: same frequency
EXTENT (the padding interpolates within the canonical band, it never grows
it), at ``pad_factor``x the DENSITY. They also exercise the public accessor
against the session-scoped Stage 0+1 baseline only -- proving the dependency
is Stage 1 (the FID + canonical FT settings, including the trim), not
Stage 5.
"""

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.core.data_structures import ComplexFT

pytestmark = pytest.mark.integration


class TestComputeDisplayFt:
    def test_frequency_range_matches_canonical(self, baseline_2638_stage1):
        """The trimmed display band's extent must match
        ``compute_ft(from_saved_params=True)``'s own band, within one padded
        bin at each edge -- padding must never widen the band beyond what
        Stage 1's trim kept."""
        canonical = ftmw.compute_ft(baseline_2638_stage1, from_saved_params=True)
        display_ft = ftmw.compute_display_ft(baseline_2638_stage1)

        canon_min, canon_max = (
            float(np.min(canonical.freq_array)),
            float(np.max(canonical.freq_array)),
        )
        disp_min, disp_max = (
            float(np.min(display_ft.freq_array)),
            float(np.max(display_ft.freq_array)),
        )
        bin_spacing = float(display_ft.freq_array[1] - display_ft.freq_array[0])

        assert abs(disp_min - canon_min) < bin_spacing
        assert abs(disp_max - canon_max) < bin_spacing

    def test_density_is_pad_factor_times_native(self, baseline_2638_stage1):
        """Density, not extent, is what ``pad_factor`` controls once the
        band is trimmed to the canonical range: ``pad_factor=2``'s bin
        spacing must be half of ``pad_factor=1``'s, and its point count
        within the same trimmed band must be ~2x (exact equality is not
        guaranteed -- independently-trimmed grids can differ by an edge bin)."""
        native = ftmw.compute_display_ft(baseline_2638_stage1, pad_factor=1)
        padded = ftmw.compute_display_ft(baseline_2638_stage1, pad_factor=2)

        native_spacing = float(native.freq_array[1] - native.freq_array[0])
        padded_spacing = float(padded.freq_array[1] - padded.freq_array[0])
        assert padded_spacing == pytest.approx(native_spacing / 2.0, rel=1e-9)

        assert abs(padded.n_points - 2 * native.n_points) <= 2

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
