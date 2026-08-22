"""What the Stage 6 fit context estimates noise over, and what it does not.

``build_stage5_fit_context`` used to estimate the active-FT noise twice: once
over the whole active FT, then again over the analysis band, overwriting the
first result there. On 2638 the active FT spans 15960-40960 MHz against a
26500-40000 MHz band, so 46% of the first estimate covered bins the second
never touched -- and on the Stage 6 replay path nothing reads those bins:

* ``Stage5FitContext.active_rms`` has no reader anywhere;
* ``rms_for_fit`` is only ever sliced by a window's ``freq_range``
  (``materialize_window``), and Stage 4 plans windows on the *trimmed* grid,
  so no window can reach outside the band;
* the spur detector, which does read out there, runs only on a first fit --
  a replay reinstates the persisted catalog without probing.

So the replay path now estimates only the band it will read and leaves the
rest NaN. These tests pin all three halves of that: the in-band values are
unchanged, the out-of-band values are gone, and the first-fit path -- which
does need them -- still has them.

Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.active_ft_support import _persisted_scatter_knobs
from ftmwpipeline.preprocessing.noise_estimation import estimate_active_ft_noise

pytestmark = [pytest.mark.integration]


def _band_mask(fit_ctx) -> np.ndarray:
    freq = np.asarray(fit_ctx.active_ft.freq_mhz, dtype=float)
    trim = fit_ctx.trim_range
    assert trim is not None, "fixture must carry a trim range to be meaningful"
    return (freq >= float(min(trim))) & (freq <= float(max(trim)))


@pytest.fixture(scope="module")
def replay_ctx(stage5_multi_source):
    """The fit context a Stage 6 verb builds (spur catalog replayed)."""
    return s6._build_shared_fit_ctx(str(stage5_multi_source)).fit_ctx


def test_the_replay_path_estimates_only_the_analysis_band(replay_ctx):
    in_band = _band_mask(replay_ctx)
    rms = np.asarray(replay_ctx.rms_for_fit, dtype=float)

    assert in_band.any(), "no in-band bins; the fixture cannot exercise this"
    assert not in_band.all(), (
        "the active FT does not extend past the analysis band on this fixture, "
        "so there is no out-of-band region to skip and this test proves nothing"
    )
    assert not np.isnan(rms[in_band]).any(), "in-band sigma must be finite"
    assert np.isnan(rms[~in_band]).all(), (
        "out-of-band sigma must be NaN on the replay path -- a plausible-looking "
        "number there would hide a reader that should not exist"
    )


def test_the_in_band_values_are_the_ones_the_old_code_produced(
    replay_ctx, stage5_multi_source
):
    """The saving must be free: skipping the full estimate cannot move a
    single in-band sigma, because the in-band estimate was always computed
    from the in-band bins alone."""
    in_band = _band_mask(replay_ctx)
    rms = np.asarray(replay_ctx.rms_for_fit, dtype=float)
    freq = np.asarray(replay_ctx.active_ft.freq_mhz, dtype=float)

    direct = np.asarray(
        estimate_active_ft_noise(
            freq[in_band],
            np.asarray(replay_ctx.active_ft.complex_spectrum)[in_band],
            **_persisted_scatter_knobs(str(stage5_multi_source)),
        ).rms_noise,
        dtype=float,
    )
    assert np.array_equal(rms[in_band], direct), (
        "the in-band sigma is no longer bit-identical to a direct in-band "
        "estimate; this change was supposed to remove work, not alter it"
    )


def test_no_planned_window_ever_reads_the_skipped_region(
    replay_ctx, stage5_multi_source
):
    """The safety argument, asserted rather than reasoned about.

    Every window's slice of ``rms_for_fit`` must be finite. A NaN here would
    mean a window reaching outside the analysis band -- the case that makes
    skipping the out-of-band estimate unsafe.
    """
    rms = np.asarray(replay_ctx.rms_for_fit, dtype=float)
    freq = np.asarray(replay_ctx.active_ft.freq_mhz, dtype=float)
    plan = s6.effective_window_plan(str(stage5_multi_source))
    assert plan.windows, "fixture must carry a plan"

    touching_nan = []
    for w in plan.windows:
        lo, hi = sorted(float(v) for v in w.freq_range)
        mask = (freq >= lo) & (freq <= hi)
        if bool(mask.any()) and bool(np.isnan(rms[mask]).any()):
            touching_nan.append(int(w.window_id))

    assert not touching_nan, (
        f"window(s) {touching_nan[:5]} slice into the un-estimated region; "
        f"Stage 4 is supposed to plan windows on the trimmed grid"
    )


def test_the_first_fit_path_still_estimates_the_whole_active_ft(
    stage5_multi_source, monkeypatch
):
    """The skip is scoped to a replay, and that scope is the safety argument.

    On a first fit the spur detector runs, and it is handed the full-length
    noise array -- so out there the estimate is still needed and must still
    be finite. Driven by capturing the arguments a Stage 6 context build
    resolves and re-running the builder with the catalog withheld, which is
    exactly what distinguishes the two paths.
    """
    from ftmwpipeline._internal import stage5_impl as s5

    captured: dict = {}
    real = s5.build_stage5_fit_context

    def spy(path, resolved, persisted_cal, shape_enum, replay_spur_catalog=None):
        captured.update(
            path=path,
            resolved=resolved,
            persisted_cal=persisted_cal,
            shape_enum=shape_enum,
        )
        return real(
            path,
            resolved,
            persisted_cal,
            shape_enum,
            replay_spur_catalog=replay_spur_catalog,
        )

    monkeypatch.setattr(s5, "build_stage5_fit_context", spy)
    s6._build_shared_fit_ctx(str(stage5_multi_source))
    assert captured, "the Stage 6 build did not reach build_stage5_fit_context"

    first_fit = real(
        captured["path"],
        captured["resolved"],
        captured["persisted_cal"],
        captured["shape_enum"],
        replay_spur_catalog=None,
    )
    rms = np.asarray(first_fit.rms_for_fit, dtype=float)
    in_band = _band_mask(first_fit)

    assert not np.isnan(rms).any(), (
        "the first-fit path left un-estimated bins; the spur detector reads "
        "the whole active FT and would see NaN"
    )
    assert in_band.any() and not in_band.all()
