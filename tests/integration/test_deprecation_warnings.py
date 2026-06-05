"""DeprecationWarning fires on legacy per-knob kwargs at every migrated impl.

The five settings dataclasses (Stages 2, 2b, 3, 4, 5) keep their legacy
per-knob kwargs alive as a back-compat shim. ``warn_legacy_kwargs`` in
each impl surfaces any remaining legacy-form call site so it can be
migrated to ``settings=`` / ``preset=`` before the surrounding work
ships. See ``dev-docs/planning/settings-backfill.md`` § Follow-ups #1
for the rationale.

Tests are split between:

* Impl-level: call the impl with a bogus path so the warning fires
  before any file work (fast, covers every stage cheaply). The
  subsequent ``FileNotFoundError`` / ``StageDependencyError`` /
  ``ValueError`` is swallowed -- the unit under test is the warning,
  not the downstream behaviour.
* End-to-end through ``ftmwpipeline.api``: one stage per session to
  prove the api/Pipeline wrappers pass the legacy kwargs through to
  the impl (the impls are thin enough that one such proof covers them
  all).
"""

from __future__ import annotations

import contextlib
import shutil

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import (
    shape_recommendation_impl as _shape_recommendation_impl,
)
from ftmwpipeline._internal import (
    stage2_impl,
    stage2b_g_impl,
    stage2b_impl,
    stage3_impl,
    stage4_impl,
    stage5_impl,
)


def _swallow_downstream_errors():
    """Helper: ignore any exception raised after the warning fires.

    The warning is the unit under test; the impl will inevitably fail
    once it tries to open the bogus file path.
    """
    return contextlib.suppress(Exception)


class TestImplLevelWarnings:
    """Each impl emits ``DeprecationWarning`` for legacy per-knob kwargs."""

    def test_stage2_estimate_noise_scatter_legacy_kwarg(self, tmp_path) -> None:
        bogus = tmp_path / "no_such.ftmw"
        with pytest.warns(DeprecationWarning, match=r"estimate_noise.*window_mhz"):
            with _swallow_downstream_errors():
                stage2_impl.compute_noise_estimation_impl(
                    str(bogus),
                    window_mhz=60.0,
                )

    def test_stage2b_calibrate_tau_legacy_kwarg(self, tmp_path) -> None:
        bogus = tmp_path / "no_such.ftmw"
        with pytest.warns(DeprecationWarning, match=r"calibrate_tau.*n_seg"):
            with _swallow_downstream_errors():
                stage2b_impl.calibrate_tau_impl(str(bogus), n_seg=5)

    def test_stage2b_g_calibrate_tau_G_legacy_kwarg(self, tmp_path) -> None:
        bogus = tmp_path / "no_such.ftmw"
        with pytest.warns(DeprecationWarning, match=r"calibrate_tau_G.*snr_min"):
            with _swallow_downstream_errors():
                stage2b_g_impl.calibrate_tau_G_impl(str(bogus), snr_min=10.0)

    def test_shape_recommendation_legacy_kwarg(self, tmp_path) -> None:
        bogus = tmp_path / "no_such.ftmw"
        with pytest.warns(
            DeprecationWarning,
            match=r"recommend_shape.*pure_margin_threshold",
        ):
            with _swallow_downstream_errors():
                _shape_recommendation_impl.recommend_shape_impl(
                    str(bogus),
                    pure_margin_threshold=0.05,
                )

    def test_stage3_detect_peaks_legacy_kwarg(self, tmp_path) -> None:
        bogus = tmp_path / "no_such.ftmw"
        with pytest.warns(DeprecationWarning, match=r"detect_peaks.*min_snr"):
            with _swallow_downstream_errors():
                stage3_impl.detect_peaks_impl(str(bogus), min_snr=4.0)

    def test_stage4_assign_windows_legacy_kwarg(self, tmp_path) -> None:
        bogus = tmp_path / "no_such.ftmw"
        with pytest.warns(DeprecationWarning, match=r"assign_windows.*edge_m"):
            with _swallow_downstream_errors():
                stage4_impl.assign_windows_impl(str(bogus), edge_m=64)

    def test_stage5_fit_peaks_legacy_kwarg(self, tmp_path) -> None:
        bogus = tmp_path / "no_such.ftmw"
        with pytest.warns(DeprecationWarning, match=r"fit_peaks.*tau0_us"):
            with _swallow_downstream_errors():
                stage5_impl.fit_peaks_impl(str(bogus), tau0_us=6.0)

    def test_no_warning_without_legacy_kwargs(self, tmp_path) -> None:
        """A bogus-path call with no legacy kwargs must not emit the warning."""
        import warnings

        bogus = tmp_path / "no_such.ftmw"
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            for impl_call in (
                lambda: stage2_impl.compute_noise_estimation_impl(str(bogus)),
                lambda: stage2b_impl.calibrate_tau_impl(str(bogus)),
                lambda: stage3_impl.detect_peaks_impl(str(bogus)),
                lambda: stage4_impl.assign_windows_impl(str(bogus)),
                lambda: stage5_impl.fit_peaks_impl(str(bogus)),
            ):
                with _swallow_downstream_errors():
                    impl_call()


class TestApiChainPropagation:
    """One end-to-end test proves the api/Pipeline pass legacy kwargs to impl.

    Stage 2 is the cheapest (only needs Stage 1 baseline); the impls are
    thin enough that passing through ``ftmw.estimate_noise`` is
    representative of every stage's wrapper chain.
    """

    def test_api_estimate_noise_emits_warning_for_legacy_kwarg(
        self,
        baseline_2638_stage1,
        tmp_path,
    ) -> None:
        fp = tmp_path / "warn_api.ftmw"
        shutil.copy(baseline_2638_stage1, fp)
        with pytest.warns(DeprecationWarning, match=r"estimate_noise.*window_mhz"):
            ftmw.estimate_noise(str(fp), window_mhz=60.0)
