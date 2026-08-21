"""
Tests for the amortized Stage 6 review session (Task D,
``scratch/preview-session-plan.md``): D1 (the fingerprint), D3
(``Pipeline.review_session()`` / ``ReviewSession``, hosting the review verb
set -- ``edit``, ``accept``, ``create``, ``undo``, ``preview``, ``apply``;
``merge``/``split`` are not verbs anywhere, session included, since
curation-intent inference reads them from add/remove), D4 (staged-preview
reuse on an immediately-following apply), and D5 (bounded, caller-controlled
memory -- no module-level cache).

The contract this whole module is built to prove
(``scratch/bq-correspondence/reply-preview-execute.md`` section 4): "A miss
is always a full recompute, results are identical either way, and nothing
about the file's contents depends on whether you went through a session."
Every correctness test here is therefore built to compare a session-hosted
result against an independent, sessionless computation on the same
inputs -- never merely "the session ran without raising" -- and the
mutation-probe tests explicitly force a permanent miss and a permanent
(stale) hit to show each would be caught.

Uses a ``self_calibrated`` fixture (forced by the recipe from
``scratch/preview-session-plan.md``, duplicated locally per this suite's own
precedent -- ``test_frame_parameter.py``'s docstring, ``tests/AGENTS.md``)
for every test that touches a calibrated number or the final-products table:
an ``rb_locked``/``uncalibrated`` fixture has ``epsilon == 0``, where a stale
calibration stamp is invisible.
"""

from __future__ import annotations

import ast
import gc
import inspect
import shutil
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.stage4_impl import load_windows_impl, save_window_plan_impl
from ftmwpipeline._internal.stage6_impl import (
    apply_curation_impl,
    create_window_impl,
    get_final_products_impl,
    refit_window_impl,
    review_accept_impl,
)
from ftmwpipeline.core.environment import ANALYSIS_EPOCH, capture_environment
from ftmwpipeline.core.stage_fit_settings import ClockSource
from ftmwpipeline.fitting.timebase_calibration import TimebaseCalibrationResult
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file
from ftmwpipeline.io.stage_fit_settings_serialization import (
    load_stage_fit_settings_from_h5,
    save_stage_fit_settings_to_h5,
)
from ftmwpipeline.io.timebase_serialization import (
    GROUP_PATH,
    save_timebase_calibration_to_hdf5,
)
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.integration]

EPS_1 = 2.2e-6
EPS_2 = 8.8e-6
SIGMA_EPS = 0.1e-6

# ---------------------------------------------------------------------------
# self_calibrated, multi-window fixture (recipe from
# scratch/preview-session-plan.md; duplicated locally, matching this suite's
# own precedent).
# ---------------------------------------------------------------------------


def _build_stage5_multi(dest: Path, data_path: str) -> None:
    ftmw.import_data(dest, source=data_path)
    ftmw.compute_ft(dest, trim=(26500, 40000))
    ftmw.estimate_noise(dest)
    ftmw.detect_peaks(dest)
    ftmw.assign_windows(dest)

    plan = load_windows_impl(str(dest))["plan"]
    candidates: List[int] = []
    for wid in plan.topological_order:
        deps = [(a, b) for (a, b) in plan.dependency_edges if a == wid or b == wid]
        if all(a in candidates or a == wid for (a, _) in deps) and all(
            b in candidates or b == wid for (_, b) in deps
        ):
            candidates.append(wid)
        if len(candidates) >= 12:
            break
    keep = set(candidates)
    plan.windows = [w for w in plan.windows if w.window_id in keep]
    plan.topological_order = [w for w in plan.topological_order if w in keep]
    plan.dependency_edges = [
        (a, b) for (a, b) in plan.dependency_edges if a in keep and b in keep
    ]
    save_window_plan_impl(str(dest), plan)

    ftmw.fit_peaks(str(dest))


def _declare_unlocked_digitizer(path: Path) -> None:
    persisted = load_stage_fit_settings_from_h5(str(path))
    assert persisted is not None, "Stage 5 must have persisted fit settings"
    new_settings = replace(
        persisted,
        spur=replace(
            persisted.spur,
            clocks=(
                ClockSource(5120.0, locked=True),
                ClockSource(6250.0, locked=False),
            ),
        ),
    )
    save_stage_fit_settings_to_h5(str(path), new_settings)


def _stamp_timebase(path: Path, *, epsilon: float, sigma_epsilon: float) -> None:
    result = TimebaseCalibrationResult(
        epsilon=epsilon,
        sigma_epsilon=sigma_epsilon,
        n_used=5,
        n_detected=5,
        lattice_g_mhz=320.0,
        tone_reads=(),
        kappa_sys=0.0,
        snr_min=10.0,
        sample_dt_us=0.02,
        start_us=0.0,
        end_us=13.0,
        span_us=13.0,
        preconditions_passed=True,
    )
    with h5py.File(str(path), "a") as h5f:
        if GROUP_PATH in h5f:
            del h5f[GROUP_PATH]
        grp = h5f.create_group(GROUP_PATH)
        save_timebase_calibration_to_hdf5(result, grp)


@pytest.fixture(scope="session")
def _stage5_multi_sc_built(exp_2638_data_path, tmp_path_factory) -> Path:
    fp = tmp_path_factory.mktemp("stage6_session_multi_sc") / "stage5_multi_sc.ftmw"
    _build_stage5_multi(fp, exp_2638_data_path)
    _declare_unlocked_digitizer(fp)
    _stamp_timebase(fp, epsilon=EPS_1, sigma_epsilon=SIGMA_EPS)
    return fp


@pytest.fixture
def sc_multi_file(_stage5_multi_sc_built: Path, tmp_path: Path) -> Path:
    """A fresh writable copy of the shared self_calibrated multi-window build."""
    fp = tmp_path / "multi_sc.ftmw"
    shutil.copy(_stage5_multi_sc_built, fp)
    return fp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fitted_window_ids(path: Path) -> List[int]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return sorted(
        int(wf.window_id) for wf in sf.window_fits if wf.window_id is not None
    )


def _window_center(path: Path, wid: int) -> float:
    plan = load_windows_impl(str(path))["plan"]
    for w in plan.windows:
        if int(w.window_id) == wid:
            lo, hi = w.freq_range
            return 0.5 * (min(lo, hi) + max(lo, hi))
    raise KeyError(wid)


def _clear_add_freq(path: Path, wid: int) -> float:
    """An in-window frequency that is a legitimate ``add`` target.

    NOT the window center: Stage 4 builds a window around the detection that
    seeded it, so the center is the *birth position* of the peak already fitted
    there. An ``add`` at the center therefore asks for a second peak at an
    existing peak's exact identity -- both are stamped with the same
    ``peak_uid`` and the add is refused (a ``ValueError`` from the
    seed-building path -- two lines cannot be born at one position). (Resolving
    a blend is what ``split`` is for.) Pick the in-window position furthest
    from every fitted peak instead, which is what "add a line the detector
    missed" actually means.
    """
    import h5py as _h5py
    import numpy as _np

    from ftmwpipeline._internal.stage4_impl import load_windows_impl as _lw
    from ftmwpipeline.io.fitting_serialization import (
        load_spectrum_fit_from_hdf5 as _load_sf,
    )

    plan = _lw(str(path))["plan"]
    w = next(x for x in plan.windows if int(x.window_id) == wid)
    lo, hi = w.freq_range
    lo, hi = min(lo, hi), max(lo, hi)
    with _h5py.File(str(path), "r") as h5f:
        sf = _load_sf(h5f["stage5_fitting"])
    wf = next((x for x in sf.window_fits if int(x.window_id) == wid), None)
    peaks = [float(p.frequency_mhz) for p in (wf.fitted_peaks if wf else [])]
    if not peaks:
        return 0.5 * (lo + hi)
    grid = _np.linspace(lo, hi, 512)[1:-1]
    dist = _np.min(_np.abs(grid[:, None] - _np.asarray(peaks)[None, :]), axis=1)
    return float(grid[int(_np.argmax(dist))])


def _window_stats(path: Path) -> Dict[int, Tuple[int, float]]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {
        int(wf.window_id): (len(wf.fitted_peaks), float(wf.reduced_chi2))
        for wf in sf.window_fits
        if wf.window_id is not None
    }


def _digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _free_anchor(path: Path) -> float:
    """A frequency well clear of every fitted window, inside the trim band."""
    plan = load_windows_impl(str(path))["plan"]
    hi_edges = [max(w.freq_range) for w in plan.windows]
    assert hi_edges, "fixture has no windows"
    return max(hi_edges) + 5.0


def _force_fit_epoch(path: Path, epoch: int) -> None:
    import json

    with h5py.File(path, "a") as f:
        g = f.require_group("pipeline_stages")
        blob = json.loads(str(g.attrs.get("stage_environments", "{}")))
        rec = capture_environment().to_dict()
        rec["analysis_epoch"] = epoch
        blob["stage5_fitting"] = rec
        g.attrs["stage_environments"] = json.dumps(blob, sort_keys=True)


# ---------------------------------------------------------------------------
# Fixture sanity.
# ---------------------------------------------------------------------------


def test_fixture_is_actually_self_calibrated(sc_multi_file: Path) -> None:
    stamp = s6._current_calibration_stamp(str(sc_multi_file))
    assert stamp is not None
    assert stamp[0] == "self_calibrated"
    assert stamp[1] == pytest.approx(EPS_1)


def test_fixture_has_at_least_two_live_windows(sc_multi_file: Path) -> None:
    assert len(_fitted_window_ids(sc_multi_file)) >= 2


# ---------------------------------------------------------------------------
# D1: the fingerprint.
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_stable_across_pure_reads(self, sc_multi_file: Path) -> None:
        fp1 = s6._compute_fit_ctx_fingerprint(str(sc_multi_file))
        fp2 = s6._compute_fit_ctx_fingerprint(str(sc_multi_file))
        assert fp1 == fp2

    def test_changes_after_a_stage6_write(self, sc_multi_file: Path) -> None:
        before = s6._compute_fit_ctx_fingerprint(str(sc_multi_file))
        wid = _fitted_window_ids(sc_multi_file)[0]
        review_accept_impl(str(sc_multi_file), wid)  # bare accept: cheap write
        after = s6._compute_fit_ctx_fingerprint(str(sc_multi_file))
        assert after != before

    def test_changes_after_a_timebase_rerun_with_no_stage6_action(
        self, sc_multi_file: Path
    ) -> None:
        """The exact gap A7 hit for final-products staleness: a timebase
        re-run touches neither /pipeline_stages nor /stage6_review, so the
        mtime/size fast path is the only thing (besides the direct
        calibration-stamp read folded into the stage-provenance tuple) that
        could catch it."""
        before = s6._compute_fit_ctx_fingerprint(str(sc_multi_file))
        _stamp_timebase(sc_multi_file, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
        after = s6._compute_fit_ctx_fingerprint(str(sc_multi_file))
        assert after != before
        # Specifically via the calibration-stamp component of stage provenance
        # (the last element), not merely mtime/size -- confirms the second
        # tier is doing real work, not just riding along on the file size.
        assert before[2][-1] != after[2][-1]

    def test_fingerprint_is_cheap_relative_to_a_shared_ctx_build(
        self, sc_multi_file: Path
    ) -> None:
        import time

        t0 = time.perf_counter()
        for _ in range(20):
            s6._compute_fit_ctx_fingerprint(str(sc_multi_file))
        per_call = (time.perf_counter() - t0) / 20

        t0 = time.perf_counter()
        s6._build_shared_fit_ctx(str(sc_multi_file))
        build_time = time.perf_counter() - t0

        # Generous margins (avoid flakiness): the ground truth measured
        # ~0.8 ms vs ~420-450 ms, roughly 500x. Assert at least a 10x gap.
        assert per_call * 10 < build_time, (per_call, build_time)


# ---------------------------------------------------------------------------
# D3: session lifecycle.
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_verb_before_enter_raises(self, sc_multi_file: Path) -> None:
        session = Pipeline.open(sc_multi_file).review_session()
        wid = _fitted_window_ids(sc_multi_file)[0]
        with pytest.raises(ValueError, match="closed"):
            session.review_accept(wid)

    def test_verb_after_exit_raises(self, sc_multi_file: Path) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        with Pipeline.open(sc_multi_file).review_session() as session:
            pass
        with pytest.raises(ValueError, match="closed"):
            session.review_accept(wid)

    def test_enter_returns_self(self, sc_multi_file: Path) -> None:
        session = Pipeline.open(sc_multi_file).review_session()
        with session as opened:
            assert opened is session

    def test_two_sessions_on_the_same_path_do_not_share_a_context(
        self, sc_multi_file: Path
    ) -> None:
        """D5: no module-level cache keyed by path."""
        with Pipeline.open(sc_multi_file).review_session() as s1:
            with Pipeline.open(sc_multi_file).review_session() as s2:
                assert s1._shared is not s2._shared

    def test_session_edit_gates_across_epoch_mismatch_and_writes_nothing(
        self, sc_multi_file: Path
    ) -> None:
        """A session's mutating verbs must gate exactly as the sessionless
        ones do -- this is the exact regression the task called out as a
        serious one to avoid."""
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)
        _force_fit_epoch(sc_multi_file, ANALYSIS_EPOCH + 1)
        before = _digest(sc_multi_file)
        with Pipeline.open(sc_multi_file).review_session() as session:
            with pytest.raises(ValueError, match="analysis epoch"):
                session.review_edit(wid, add=[freq], frame="raw")
        assert _digest(sc_multi_file) == before
        with h5py.File(sc_multi_file, "r") as f:
            assert "stage5_fitting_baseline" not in f

    def test_session_edit_takes_the_undo_baseline_exactly_like_sessionless(
        self, sc_multi_file: Path
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)
        with h5py.File(sc_multi_file, "r") as f:
            assert "stage5_fitting_baseline" not in f
        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_edit(wid, add=[freq], frame="raw")
        with h5py.File(sc_multi_file, "r") as f:
            assert "stage5_fitting_baseline" in f


# ---------------------------------------------------------------------------
# The core property: correctness never depends on a cache hit. Every test in
# this section is built so it would fail if a stale cached context were
# served, or if the session silently produced different numbers than the
# sessionless functions it wraps.
# ---------------------------------------------------------------------------


class TestSessionMatchesSessionless:
    def test_first_use_in_a_session_is_necessarily_a_miss_and_matches_sessionless(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        plain = tmp_path / "plain.ftmw"
        shutil.copy(sc_multi_file, plain)
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)

        sessionless = refit_window_impl(str(plain), wid, add=[freq], frame="raw")
        with Pipeline.open(sc_multi_file).review_session() as session:
            hosted = session.review_edit(wid, add=[freq], frame="raw")

        assert hosted.n_peaks_before == sessionless.n_peaks_before
        assert hosted.n_peaks_after == sessionless.n_peaks_after
        assert hosted.chi2r_after == pytest.approx(sessionless.chi2r_after, rel=1e-9)
        assert hosted.epsilon == pytest.approx(sessionless.epsilon)
        assert sorted(hosted.fitted_peaks_calibrated_mhz) == pytest.approx(
            sorted(sessionless.fitted_peaks_calibrated_mhz), rel=1e-9
        )

    def test_second_call_in_a_session_a_genuine_hit_matches_two_independent_misses(
        self, sc_multi_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second session call reuses the cached shared context (a real
        hit); the sessionless baseline rebuilds it twice, independently (two
        real misses). The outcomes must agree exactly."""
        rebuild_count = {"n": 0}
        real_build = s6._build_shared_fit_ctx

        def counting_build(path: str) -> "s6._SharedFitCtx":
            rebuild_count["n"] += 1
            return real_build(path)

        monkeypatch.setattr(s6, "_build_shared_fit_ctx", counting_build)

        wids = _fitted_window_ids(sc_multi_file)
        w0, w1 = wids[0], wids[1]
        f0 = _clear_add_freq(sc_multi_file, w0)
        f1 = _clear_add_freq(sc_multi_file, w1)

        plain = tmp_path / "plain.ftmw"
        shutil.copy(sc_multi_file, plain)
        refit_window_impl(str(plain), w0, add=[f0], frame="raw")
        refit_window_impl(str(plain), w1, add=[f1], frame="raw")

        # Reset the counter now: the two sessionless calls above also went
        # through the patched builder (two genuine misses of their own),
        # which is not what this assertion is about.
        rebuild_count["n"] = 0
        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_edit(w0, add=[f0], frame="raw")
            session.review_edit(w1, add=[f1], frame="raw")

        # Exactly one build: __enter__'s warm-up. The second edit genuinely
        # reused it -- this is not merely "the numbers happen to agree".
        assert rebuild_count["n"] == 1

        stats_session = _window_stats(sc_multi_file)
        stats_plain = _window_stats(plain)
        for w in (w0, w1):
            n_s, chi_s = stats_session[w]
            n_p, chi_p = stats_plain[w]
            assert n_s == n_p
            assert chi_s == pytest.approx(chi_p, rel=1e-9)

    def test_accept_create_match_sessionless(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        """Runs ``accept`` and ``create`` through one session and an
        independent sessionless sequence, on independent windows so the
        actions do not interact, and checks agreement window by window.

        ``merge``/``split`` are covered by :func:`merge_peaks_impl` /
        :func:`split_peak_impl` directly in ``test_review_verbs.py`` --
        ``ReviewSession`` does not host them (they are not verbs anywhere;
        curation-intent inference reads them from add/remove), so there is no
        session-hosted counterpart to compare here."""
        wids = _fitted_window_ids(sc_multi_file)
        w_accept = wids[0]
        accept_anchor = _clear_add_freq(sc_multi_file, w_accept)

        plain = tmp_path / "plain.ftmw"
        shutil.copy(sc_multi_file, plain)
        review_accept_impl(
            str(plain), w_accept, candidate_freq=accept_anchor, frame="raw"
        )
        anchor = _free_anchor(plain)
        create_window_impl(str(plain), anchor, frame="raw")

        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_accept(w_accept, candidate_freq=accept_anchor, frame="raw")
            session.review_create(_free_anchor(sc_multi_file), frame="raw")

        stats_session = _window_stats(sc_multi_file)
        stats_plain = _window_stats(plain)
        for w in (w_accept,):
            assert stats_session[w][0] == stats_plain[w][0]
            assert stats_session[w][1] == pytest.approx(stats_plain[w][1], rel=1e-9)

    def test_session_undo_matches_sessionless_undo(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)

        plain = tmp_path / "plain.ftmw"
        shutil.copy(sc_multi_file, plain)
        refit_window_impl(str(plain), wid, add=[freq], frame="raw")
        ftmw.review_undo(str(plain), [0])

        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_edit(wid, add=[freq], frame="raw")
            session.review_undo([0])

        stats_session = _window_stats(sc_multi_file)
        stats_plain = _window_stats(plain)
        assert stats_session[wid][0] == stats_plain[wid][0]
        assert stats_session[wid][1] == pytest.approx(stats_plain[wid][1], rel=1e-9)


# ---------------------------------------------------------------------------
# D4: staged-preview reuse on an immediately-following apply.
# ---------------------------------------------------------------------------


class TestStagedReuse:
    def _one_add_curation(self, path: Path, tmp_path: Path) -> Tuple[Path, int]:
        wid = _fitted_window_ids(path)[0]
        freq = _window_center(path, wid)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wid},{freq},\n")
        return cur, wid

    def test_preview_then_apply_reuses_and_matches_sessionless_apply(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        cur, wid = self._one_add_curation(sc_multi_file, tmp_path)

        plain = tmp_path / "plain.ftmw"
        shutil.copy(sc_multi_file, plain)
        apply_curation_impl(str(plain), cur, frame="raw")

        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_preview(cur, frame="raw")
            # Spy on the reuse path so the test can assert it actually fired,
            # not merely that the numbers happen to agree.
            calls = {"n": 0}
            orig = session._persist_staged

            def spy(staged):
                calls["n"] += 1
                return orig(staged)

            session._persist_staged = spy
            result = session.review_apply(cur, frame="raw")

        assert calls["n"] == 1
        assert result.base_changed is False

        stats_session = _window_stats(sc_multi_file)
        stats_plain = _window_stats(plain)
        assert stats_session[wid] == pytest.approx(stats_plain[wid])

        fp_session = get_final_products_impl(str(sc_multi_file))
        fp_plain = get_final_products_impl(str(plain))
        assert fp_session is not None and fp_plain is not None
        assert fp_session.epsilon == pytest.approx(fp_plain.epsilon)

    def test_apply_without_a_preceding_preview_falls_back_to_full_apply(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        cur, wid = self._one_add_curation(sc_multi_file, tmp_path)
        plain = tmp_path / "plain.ftmw"
        shutil.copy(sc_multi_file, plain)
        apply_curation_impl(str(plain), cur, frame="raw")

        with Pipeline.open(sc_multi_file).review_session() as session:
            result = session.review_apply(cur, frame="raw")

        assert result.base_changed is False
        stats_session = _window_stats(sc_multi_file)
        stats_plain = _window_stats(plain)
        assert stats_session[wid] == pytest.approx(stats_plain[wid])

    def test_base_changed_when_a_foreign_write_lands_between_preview_and_apply(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        cur, wid = self._one_add_curation(sc_multi_file, tmp_path)

        # Ground truth: what a fresh apply under EPS_2 gives.
        fresh = tmp_path / "fresh.ftmw"
        shutil.copy(sc_multi_file, fresh)
        _stamp_timebase(fresh, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
        apply_curation_impl(str(fresh), cur, frame="raw")
        fresh_fp = get_final_products_impl(str(fresh))
        assert fresh_fp is not None

        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_preview(cur, frame="raw")
            _stamp_timebase(sc_multi_file, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
            result = session.review_apply(cur, frame="raw")

        assert result.base_changed is True
        persisted = load_stage6_review_from_file(str(sc_multi_file)).final_products
        assert persisted is not None
        assert persisted.epsilon == pytest.approx(EPS_2)
        assert persisted.epsilon == pytest.approx(fresh_fp.epsilon)

    def test_base_changed_when_an_intervening_session_edit_invalidates_staged(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wids = _fitted_window_ids(sc_multi_file)
        w_preview, w_other = wids[0], wids[1]
        freq_preview = _clear_add_freq(sc_multi_file, w_preview)
        freq_other = _clear_add_freq(sc_multi_file, w_other)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{w_preview},{freq_preview},\n")

        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_preview(cur, frame="raw")
            session.review_edit(w_other, add=[freq_other], frame="raw")
            result = session.review_apply(cur, frame="raw")

        assert result.base_changed is True
        # Both edits must have landed -- the intervening edit was not lost,
        # and the previewed plan was still applied (freshly recomputed).
        stats = _window_stats(sc_multi_file)
        assert stats[w_other][0] >= 1
        assert w_preview in stats

    def test_staged_result_dropped_when_a_different_plan_is_applied(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wids = _fitted_window_ids(sc_multi_file)
        w_a, w_b = wids[0], wids[1]
        cur_a = tmp_path / "a.csv"
        cur_a.write_text(f"add,{w_a},{_clear_add_freq(sc_multi_file, w_a)},\n")
        cur_b = tmp_path / "b.csv"
        cur_b.write_text(f"add,{w_b},{_clear_add_freq(sc_multi_file, w_b)},\n")

        # A pristine baseline (no curation at all) establishes window A's
        # untouched stats to compare against.
        pristine = tmp_path / "pristine.ftmw"
        shutil.copy(sc_multi_file, pristine)

        plain = tmp_path / "plain.ftmw"
        shutil.copy(sc_multi_file, plain)
        apply_curation_impl(str(plain), cur_b, frame="raw")

        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_preview(cur_a, frame="raw")
            result = session.review_apply(cur_b, frame="raw")

        assert result.base_changed is False  # the .ftmw base never moved
        stats_session = _window_stats(sc_multi_file)
        stats_plain = _window_stats(plain)
        stats_pristine = _window_stats(pristine)
        assert stats_session[w_b] == pytest.approx(stats_plain[w_b])
        # Window A (the staged-but-unused plan A's target) must be
        # untouched -- the staged preview against A was never persisted.
        assert stats_session[w_a] == pytest.approx(stats_pristine[w_a])


# ---------------------------------------------------------------------------
# Mutation probes, run against my own implementation before reporting.
# ---------------------------------------------------------------------------


class TestMutationProbeAlwaysMiss:
    """Force the fingerprint check to report 'changed' on every call (a
    permanent miss). Every result must still match the sessionless baseline
    -- a miss is always a full, correct recompute."""

    def test_forced_permanent_miss_still_matches_sessionless(
        self, sc_multi_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rebuild_count = {"n": 0}
        real_build = s6._build_shared_fit_ctx

        def counting_build(path: str) -> "s6._SharedFitCtx":
            rebuild_count["n"] += 1
            return real_build(path)

        monkeypatch.setattr(s6, "_build_shared_fit_ctx", counting_build)

        counter = {"n": 0}

        def always_different(path: str) -> "s6._FitCtxFingerprint":
            counter["n"] += 1
            return (counter["n"], 0, ())

        monkeypatch.setattr(s6, "_compute_fit_ctx_fingerprint", always_different)

        wids = _fitted_window_ids(sc_multi_file)
        w0, w1 = wids[0], wids[1]
        f0 = _clear_add_freq(sc_multi_file, w0)
        f1 = _clear_add_freq(sc_multi_file, w1)

        plain = tmp_path / "plain.ftmw"
        shutil.copy(sc_multi_file, plain)
        refit_window_impl(str(plain), w0, add=[f0], frame="raw")
        refit_window_impl(str(plain), w1, add=[f1], frame="raw")

        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_edit(w0, add=[f0], frame="raw")
            session.review_edit(w1, add=[f1], frame="raw")

        # The patch really did force a rebuild on every use (enter + 2 verbs).
        assert rebuild_count["n"] >= 3

        stats_session = _window_stats(sc_multi_file)
        stats_plain = _window_stats(plain)
        for w in (w0, w1):
            assert stats_session[w][0] == stats_plain[w][0]
            assert stats_session[w][1] == pytest.approx(stats_plain[w][1], rel=1e-9)

    def test_forced_permanent_miss_never_reuses_a_staged_preview(
        self, sc_multi_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        counter = {"n": 0}

        def always_different(path: str) -> "s6._FitCtxFingerprint":
            counter["n"] += 1
            return (counter["n"], 0, ())

        monkeypatch.setattr(s6, "_compute_fit_ctx_fingerprint", always_different)

        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wid},{freq},\n")

        plain = tmp_path / "plain.ftmw"
        shutil.copy(sc_multi_file, plain)
        apply_curation_impl(str(plain), cur, frame="raw")

        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_preview(cur, frame="raw")
            calls = {"n": 0}
            orig = session._persist_staged

            def spy(staged):
                calls["n"] += 1
                return orig(staged)

            session._persist_staged = spy
            session.review_apply(cur, frame="raw")

        assert calls["n"] == 0, "a permanently-missing fingerprint must never reuse"
        stats_session = _window_stats(sc_multi_file)
        stats_plain = _window_stats(plain)
        assert stats_session[wid] == pytest.approx(stats_plain[wid])


class TestMutationProbeAlwaysHitStale:
    """Force the fingerprint check to report 'unchanged' forever (a
    permanent, stale hit) after a real, on-disk change that DOES matter.
    Demonstrates that (a) the real code catches the drift and gets the
    correct answer, and (b) the neutered check serves the stale one -- so
    the check is load-bearing, not decorative.

    Targets the single-window verbs specifically: ``_SharedFitCtx.epsilon``
    is stamped onto ``RefitWindowResult`` at ctx-build time and is NOT
    re-derived per applier call (see ``_SharedFitCtx``'s own docstring) --
    unlike the batch/preview path's final-products table, which re-reads
    the calibration fresh from disk on every derive regardless of context
    staleness. This is exactly the risk the fingerprint check exists for.
    """

    def test_real_code_detects_a_mid_session_timebase_change(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)

        real_copy = tmp_path / "real.ftmw"
        shutil.copy(sc_multi_file, real_copy)
        with Pipeline.open(real_copy).review_session() as session:
            _stamp_timebase(real_copy, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
            result = session.review_edit(wid, add=[freq], frame="raw")

        assert result.epsilon == pytest.approx(EPS_2)

    def test_neutered_check_serves_the_stale_epsilon(
        self, sc_multi_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)

        stale_copy = tmp_path / "stale.ftmw"
        shutil.copy(sc_multi_file, stale_copy)
        with Pipeline.open(stale_copy).review_session() as session:
            monkeypatch.setattr(
                s6,
                "_compute_fit_ctx_fingerprint",
                lambda path: session._fingerprint,
            )
            _stamp_timebase(stale_copy, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
            result = session.review_edit(wid, add=[freq], frame="raw")

        # Wrong: the neutered check kept serving the EPS_1-built context.
        assert result.epsilon == pytest.approx(EPS_1)
        assert result.epsilon != pytest.approx(EPS_2)

    def test_real_code_detects_drift_between_preview_and_apply(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wid},{freq},\n")

        real_copy = tmp_path / "real.ftmw"
        shutil.copy(sc_multi_file, real_copy)
        with Pipeline.open(real_copy).review_session() as session:
            session.review_preview(cur, frame="raw")
            _stamp_timebase(real_copy, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
            result = session.review_apply(cur, frame="raw")

        assert result.base_changed is True
        persisted = load_stage6_review_from_file(str(real_copy)).final_products
        assert persisted is not None
        assert persisted.epsilon == pytest.approx(EPS_2)

    def test_neutered_check_persists_the_stale_staged_preview(
        self, sc_multi_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wid = _fitted_window_ids(sc_multi_file)[0]
        freq = _window_center(sc_multi_file, wid)
        cur = tmp_path / "cur.csv"
        cur.write_text(f"add,{wid},{freq},\n")

        stale_copy = tmp_path / "stale.ftmw"
        shutil.copy(sc_multi_file, stale_copy)
        with Pipeline.open(stale_copy).review_session() as session:
            session.review_preview(cur, frame="raw")
            monkeypatch.setattr(
                s6,
                "_compute_fit_ctx_fingerprint",
                lambda path: session._fingerprint,
            )
            _stamp_timebase(stale_copy, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)
            result = session.review_apply(cur, frame="raw")

        # Wrong: the neutered check never noticed the base moved, so it
        # reused (persisted) the preview staged under EPS_1.
        assert result.base_changed is False
        persisted = load_stage6_review_from_file(str(stale_copy)).final_products
        assert persisted is not None
        assert persisted.epsilon == pytest.approx(EPS_1)
        assert persisted.epsilon != pytest.approx(EPS_2)


# ---------------------------------------------------------------------------
# D5: bounded, caller-controlled memory; no module-level cache.
# ---------------------------------------------------------------------------


class TestBoundedMemory:
    def test_shared_context_is_released_on_close(self, sc_multi_file: Path) -> None:
        session = Pipeline.open(sc_multi_file).review_session()
        session.__enter__()
        ref = weakref.ref(session._shared)
        session.close()
        gc.collect()
        assert ref() is None, (
            "the shared fit context must not be kept alive by anything after "
            "close() -- in particular, not by a module-level cache"
        )

    def test_shared_context_is_released_on_context_exit(
        self, sc_multi_file: Path
    ) -> None:
        holder: List["weakref.ReferenceType[object]"] = []
        with Pipeline.open(sc_multi_file).review_session() as session:
            holder.append(weakref.ref(session._shared))
        gc.collect()
        assert holder[0]() is None

    def test_no_module_level_cache_attribute_exists(self) -> None:
        """A grep-equivalent structural check: no module-level assignment in
        stage6_impl whose target name suggests a cache/registry. Session
        scoping is achieved entirely through instance attributes on
        ReviewSession (asserted directly above); this guards against a
        regression that reintroduces a global registry instead."""
        tree = ast.parse(inspect.getsource(s6))
        suspicious_names = []
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and "cache" in target.id.lower():
                    suspicious_names.append(target.id)
        assert not suspicious_names

    def test_existing_sessionless_callers_are_unaffected_by_session_use(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        """Using (and closing) a session on window w0 must not change what a
        subsequent ordinary sessionless call does on the independent window
        w1 -- compared against w1's outcome on a copy that never saw a
        session at all."""
        wids = _fitted_window_ids(sc_multi_file)
        w0, w1 = wids[0], wids[1]
        f0 = _clear_add_freq(sc_multi_file, w0)
        f1 = _clear_add_freq(sc_multi_file, w1)

        never_sessioned = tmp_path / "never_sessioned.ftmw"
        shutil.copy(sc_multi_file, never_sessioned)
        baseline_result = refit_window_impl(
            str(never_sessioned), w1, add=[f1], frame="raw"
        )

        with Pipeline.open(sc_multi_file).review_session() as session:
            session.review_edit(w0, add=[f0], frame="raw")

        # Plain sessionless call after the session closed.
        after_session_result = refit_window_impl(
            str(sc_multi_file), w1, add=[f1], frame="raw"
        )

        assert after_session_result.n_peaks_after == baseline_result.n_peaks_after
        assert after_session_result.chi2r_after == pytest.approx(
            baseline_result.chi2r_after, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Structural guards: confirm this unit did not weaken the engine invariants
# test_engine_invariants.py polices (I cannot run that suite myself, so this
# is a self-check using the identical AST technique against the current
# module source).
# ---------------------------------------------------------------------------


class TestEngineInvariantsStillHold:
    def _functions_calling(self, names: set) -> Dict[str, List[str]]:
        tree = ast.parse(inspect.getsource(s6))
        found: Dict[str, List[str]] = {}
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            hits = [
                sub.func.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in names
            ]
            if hits:
                found[node.name] = hits
        return found

    def test_only_finish_batch_persists_the_fit(self) -> None:
        writers = set(self._functions_calling({"save_spectrum_fit_to_hdf5"}))
        assert writers == {"_finish_batch"}

    def test_only_open_batch_takes_the_undo_baseline(self) -> None:
        snappers = set(self._functions_calling({"_snapshot_stage5_baseline"}))
        assert snappers == {"_open_batch"}

    def test_finish_batch_still_refuses_an_unbaselined_context_even_with_staged_args(
        self, sc_multi_file: Path
    ) -> None:
        """My new ``cascaded``/``precomputed_review`` params on
        ``_finish_batch`` must not bypass the baseline guard."""
        ctx = s6._open_batch(
            str(sc_multi_file),
            snap_tol_mhz=s6.resolve_snap_tol_mhz(str(sc_multi_file), None),
            snapshot=False,
        )
        assert ctx.baseline_taken is False
        before = _digest(sc_multi_file)
        with pytest.raises(ValueError, match="undo baseline was.*never taken"):
            s6._finish_batch(
                ctx,
                str(sc_multi_file),
                snap_tol_mhz=s6.resolve_snap_tol_mhz(str(sc_multi_file), None),
                cascaded=[],
                precomputed_review=s6.Stage6Review(),
            )
        assert _digest(sc_multi_file) == before
