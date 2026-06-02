"""2638 no-regression acceptance test for the scatter noise estimator.

The §9(b) acceptance check from
``dev-docs/research/noise-snr-scaling/report.md``: on the 2638 calibration
fixture the region-aware scatter estimator reads ~1.0× the frame-difference
truth, versus the old estimator's ~1.4× (the gap on 2638 is the √2 complex-RMS
convention plus a negligible pedestal — 2638 sits at SNR ~700, far below where
the pedestal failure bites). So the scatter σ must come out essentially equal to
the adaptive σ here — the precondition for Stage 3/4/5 outputs holding when the
estimator is eventually made the default.

The full Stage 3/4/5 output confirmation is a deliberate manual gate before the
default is flipped (see the report §9 and the project lead's sign-off note); this
test guards the noise-level precondition that makes that flip safe.
"""

import shutil

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage2_impl import load_noise_result_impl
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_scatter


@pytest.mark.integration
class TestScatterNoRegression2638:
    def test_scatter_sane_level_on_2638(
        self, baseline_2638_stage1_raw, tmp_path
    ):
        """Scatter σ on the raw (production) 2638 FT is finite, positive, and
        reads a sane noise floor.

        2638 sits near the pedestal-free regime (SNR ~700), so the scatter
        estimator should report a smooth, mostly-noise floor here. The
        level-agreement against the frame-difference truth (and the contrast
        with the retired level-based estimator) is documented in
        ``dev-docs/research/noise-snr-scaling/report.md`` §9.
        """
        fp = tmp_path / "scatter_2638.ftmw"
        shutil.copy(baseline_2638_stage1_raw, fp)

        ft = ftmw.compute_ft(fp)
        nr_scatter = estimate_noise_scatter(ft.freq_array, ft.magnitude_spectrum)

        assert nr_scatter.bin_info["algorithm"] == "scatter_highpass_region_aware"
        assert np.all(np.isfinite(nr_scatter.rms_noise))
        assert np.all(nr_scatter.rms_noise > 0)
        assert nr_scatter.rms_noise.shape == ft.freq_array.shape

        # The 2638 spectrum is mostly quiet, so the self-mask keeps the vast
        # majority of bins as noise.
        assert nr_scatter.bin_info["noise_fraction"] > 0.9

    def test_scatter_persists_and_reloads_exactly_on_2638(
        self, baseline_2638_stage1_raw, tmp_path
    ):
        """The persisted scatter σ reloads bit-for-bit (verbatim storage)."""
        fp = tmp_path / "scatter_2638_rt.ftmw"
        shutil.copy(baseline_2638_stage1_raw, fp)

        nr = ftmw.estimate_noise(fp)
        reloaded = load_noise_result_impl(fp)["noise_result"]

        np.testing.assert_array_equal(reloaded.rms_noise, nr.rms_noise)
        np.testing.assert_array_equal(reloaded.noise_mask, nr.noise_mask)
        assert reloaded.bin_info["algorithm"] == "scatter_highpass_region_aware"
