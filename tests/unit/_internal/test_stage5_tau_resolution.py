"""Unit tests for the Stage 5 tau-calibration resolver.

``_resolve_tau_calibration_for_fit`` decides which ``(tau_maj_us,
sigma_tau_us)`` pair drives the fit: an explicit override beats a
persisted Stage 2b calibration, which beats no calibration at all.
The two overrides are an atomic pair; supplying only one raises.

Pure-function tests; no file I/O. End-to-end precedence (override
actually beats persisted on disk) is covered by the integration
suite.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pytest

from ftmwpipeline._internal.stage5_impl import _resolve_tau_calibration_for_fit
from ftmwpipeline.fitting.tau_calibration import (
    GMMBimodality,
    TauCalibrationResult,
)


def _make_persisted(
    tau_maj_us: float = 6.5, sigma_tau_us: float = 1.7,
) -> TauCalibrationResult:
    """Build a minimal TauCalibrationResult for tests (only the two scalars matter)."""
    empty_int = np.array([], dtype=np.int64)
    empty_float = np.array([], dtype=np.float64)
    bm = GMMBimodality(
        n=0,
        mu1=float("nan"), sigma1=float("nan"),
        mu_a=float("nan"), sigma_a=float("nan"),
        mu_b=float("nan"), sigma_b=float("nan"),
        pi_a=float("nan"),
        aic1=float("nan"), aic2=float("nan"),
        delta_aic=float("nan"),
        two_component_preferred=False,
        dominant_weight=float("nan"),
    )
    return TauCalibrationResult(
        tau_maj_us=tau_maj_us,
        sigma_tau_us=sigma_tau_us,
        n_contributors=0,
        n_spur_bins=0,
        spur_clusters=(),
        bimodality=bm,
        pearson_r_log_snr_vs_tau=float("nan"),
        pearson_r_freq_vs_tau=float("nan"),
        frequency_thirds=(),
        contributor_bin_indices=empty_int,
        contributor_taus_us=empty_float,
        contributor_snrs=empty_float,
        contributor_freqs_mhz=empty_float,
        n_seg=10,
        t_sigma=5.0,
        tau_max_us=63.25,
        rss_gate_factor=5.0,
        sample_dt_us=0.020,
        start_us=0.0,
        end_us=12.65,
        probe_freq_mhz=40000.0,
        sideband="lower",
        trim_lo_mhz=26500.0,
        trim_hi_mhz=40000.0,
        sigma_x_full=1.0,
        sigma_frame=1.0,
        snr_weighted=True,
        preconditions_passed=True,
        preconditions_notes=("ok", "ok", "ok"),
    )


class TestResolveTauCalibrationForFit:
    """Precedence: override > persisted > none."""

    def test_no_calibration_no_override_returns_none(self) -> None:
        tau, sigma, source = _resolve_tau_calibration_for_fit(
            None, None, None,
        )
        assert tau is None
        assert sigma is None
        assert source == "none"

    def test_persisted_only(self) -> None:
        persisted = _make_persisted(tau_maj_us=6.5, sigma_tau_us=1.7)
        tau, sigma, source = _resolve_tau_calibration_for_fit(
            persisted, None, None,
        )
        assert tau == pytest.approx(6.5)
        assert sigma == pytest.approx(1.7)
        assert source == "persisted"

    def test_override_no_persisted(self) -> None:
        tau, sigma, source = _resolve_tau_calibration_for_fit(
            None, 8.0, 2.0,
        )
        assert tau == pytest.approx(8.0)
        assert sigma == pytest.approx(2.0)
        assert source == "override"

    def test_override_beats_persisted(self) -> None:
        persisted = _make_persisted(tau_maj_us=6.5, sigma_tau_us=1.7)
        tau, sigma, source = _resolve_tau_calibration_for_fit(
            persisted, 9.0, 0.5,
        )
        assert tau == pytest.approx(9.0)
        assert sigma == pytest.approx(0.5)
        assert source == "override"

    def test_partial_override_raises(self) -> None:
        # tau set, sigma missing
        with pytest.raises(ValueError, match="together"):
            _resolve_tau_calibration_for_fit(None, 8.0, None)
        # sigma set, tau missing
        with pytest.raises(ValueError, match="together"):
            _resolve_tau_calibration_for_fit(None, None, 2.0)

    def test_non_positive_override_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _resolve_tau_calibration_for_fit(None, 0.0, 1.0)
        with pytest.raises(ValueError, match="positive"):
            _resolve_tau_calibration_for_fit(None, 8.0, -0.1)
        with pytest.raises(ValueError, match="positive"):
            _resolve_tau_calibration_for_fit(None, -5.0, 1.0)

    def test_partial_override_raises_even_with_persisted(self) -> None:
        """The half-set override is still ambiguous, regardless of persisted."""
        persisted = _make_persisted()
        with pytest.raises(ValueError, match="together"):
            _resolve_tau_calibration_for_fit(persisted, 8.0, None)
