"""W4 -- the preview surfaces the implied structure.

BlackQuill's acceptance condition for W3's implicit window creation
(``scratch/intent-driven-windowing-plan.md``, W4, and
``scratch/bq-correspondence/ask-intent-driven-windowing.md``): letting an
``add`` mint the window it needs is acceptable to them *iff* ``review_preview``
/ ``apply --dry-run`` reports the implied create (mode, extent, ``depends_on``)
so a UI can render "this add creates a window at A-B MHz" before apply -- and
iff the preview-to-apply byte-for-byte guarantee covers the pair as one plan.

The data already existed and had no home: :class:`CreateWindowResult` (W3)
already carries ``mode``, ``freq_range``, ``n_points``, ``n_contributors`` and
``depends_on``, and the ``created_window`` decision-log evidence records the
same set. This adds five additive ``created_window_*`` fields to
``PreviewWindowResult`` and to ``RefitWindowResult`` -- ``None`` on any window
a batch did not create or widen -- and wires the CLI (``review preview``,
``review apply --dry-run``, and ``review edit``) to print them.

New module rather than an extension of ``test_review_preview.py``: this is a
narrow, self-contained surface (five fields on two dataclasses, plus their CLI
rendering) with the same fixture needs as W3's own module
(``test_implied_create.py``) rather than W3's -- keeping it separate mirrors
how W3 got its own module instead of growing ``test_derived_window.py``.

Uses ``stage5_multi_file`` (several live fitted windows), like
``test_implied_create.py``. Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import h5py
import pytest

from ftmwpipeline._internal.stage4_impl import load_windows_impl
from ftmwpipeline._internal.stage6_impl import (
    apply_curation_impl,
    refit_window_impl,
    review_log_impl,
    review_preview_impl,
)
from ftmwpipeline.cli.review_commands import cmd_review_apply, cmd_review_preview
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers -- duplicated from test_implied_create.py per this suite's own
# precedent (tests/AGENTS.md; test_frame_parameter.py's docstring) rather
# than imported across test files.
# ---------------------------------------------------------------------------


def _load_spectrum_fit(path: Path):
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _first_peak(path: Path) -> Tuple[int, int]:
    """``(window_id, peak_uid)`` of the first fitted peak carrying a stamped
    uid -- a target for an ORDINARY (non-implied-create) edit."""
    sf = _load_spectrum_fit(path)
    for wf in sf.window_fits:
        if wf.window_id is None:
            continue
        for p in wf.fitted_peaks:
            if p.peak_uid is not None:
                return int(wf.window_id), int(p.peak_uid)
    pytest.skip(
        "expected stage5_multi_file to carry at least one fitted peak with "
        "a stamped peak_uid"
    )


def _gap_anchor(path: Path) -> float:
    """A molecular frequency inside the analysis band but outside every
    PLANNED window (live or dead) -- reliably triggers ``mode="created"``
    rather than ``mode="widened"``. Mirrors ``test_implied_create.py``."""
    plan = load_windows_impl(str(path))["plan"]
    spans = sorted((min(w.freq_range), max(w.freq_range)) for w in plan.windows)
    for (_lo1, hi1), (lo2, _hi2) in zip(spans, spans[1:]):
        if lo2 - hi1 > 4.0:
            return 0.5 * (hi1 + lo2)
    pytest.skip("no gap between planned windows wide enough to create into")


def _bin_width_mhz(path: Path) -> float:
    plan = load_windows_impl(str(path))["plan"]
    w = next(
        w
        for w in plan.windows
        if "grid_span" in w.diagnostics
        and w.diagnostics["grid_span"][1] > w.diagnostics["grid_span"][0]
    )
    lo, hi = w.freq_range
    span = w.diagnostics["grid_span"]
    return abs(hi - lo) / (int(span[1]) - int(span[0]))


def _live_ranges(path: Path) -> List[Tuple[int, float, float]]:
    """``(window_id, lo, hi)`` for every window that carries a Stage 5 fit,
    ascending by ``lo``. Mirrors ``test_implied_create.py``."""
    sf = _load_spectrum_fit(path)
    plan = load_windows_impl(str(path))["plan"]
    by_id = {int(w.window_id): w for w in plan.windows}
    out: List[Tuple[int, float, float]] = []
    for wf in sf.window_fits:
        if wf.window_id is None:
            continue
        w = by_id.get(int(wf.window_id))
        if w is None:
            continue
        lo, hi = w.freq_range
        out.append((int(wf.window_id), min(lo, hi), max(lo, hi)))
    return sorted(out, key=lambda t: t[1])


def _write_curation(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ---------------------------------------------------------------------------
# PreviewWindowResult: an implied create matches what the apply installs
# ---------------------------------------------------------------------------


def test_preview_of_implied_create_matches_the_apply(stage5_multi_file, tmp_path):
    base = stage5_multi_file
    freq = _gap_anchor(base)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    preview = review_preview_impl(base, cur)
    assert len(preview.windows) == 1
    wid = next(iter(preview.windows))
    w = preview.windows[wid]

    assert w.created_window_mode == "created"
    assert w.created_window_freq_range is not None
    assert w.created_window_n_points is not None and w.created_window_n_points > 0
    assert w.created_window_n_contributors is not None
    assert w.created_window_depends_on is not None
    # A preview never persists.
    assert review_log_impl(base) == []

    # Apply the SAME curation to an independent copy of the SAME base file
    # and compare against what actually landed -- never hard-coded.
    applied_path = tmp_path / "applied.ftmw"
    shutil.copy(base, applied_path)
    apply_curation_impl(applied_path, cur)

    log = review_log_impl(applied_path)
    assert len(log) == 1
    assert log[0].window_id == wid
    created = log[0].evidence["created_window"]
    assert created["mode"] == w.created_window_mode
    assert (
        created["freq_min_mhz"],
        created["freq_max_mhz"],
    ) == w.created_window_freq_range
    assert created["n_points"] == w.created_window_n_points
    assert created["n_contributors"] == w.created_window_n_contributors
    assert created["depends_on"] == w.created_window_depends_on


def test_preview_of_ordinary_edit_leaves_structure_none(stage5_multi_file, tmp_path):
    path = stage5_multi_file
    wid, uid = _first_peak(path)
    cur = _write_curation(tmp_path, "ordinary.csv", f"remove,{wid},uid:{uid},\n")

    preview = review_preview_impl(path, cur)
    assert wid in preview.windows
    w = preview.windows[wid]

    assert w.created_window_mode is None
    assert w.created_window_freq_range is None
    assert w.created_window_n_points is None
    assert w.created_window_n_contributors is None
    assert w.created_window_depends_on is None


def test_preview_widened_mode_is_distinguishable_from_created(
    stage5_multi_file, tmp_path
):
    path = stage5_multi_file
    step = _bin_width_mhz(path)
    live = _live_ranges(path)
    wid, _lo, hi = live[-1]
    anchor = hi + 2 * step
    cur = _write_curation(tmp_path, "widen.csv", f"add,,{anchor},\n")

    preview = review_preview_impl(path, cur)
    assert wid in preview.windows
    w = preview.windows[wid]

    assert w.created_window_mode == "widened"
    assert w.created_window_mode != "created"
    assert w.created_window_freq_range is not None
    assert w.created_window_n_points is not None
    assert w.created_window_n_contributors is not None
    assert w.created_window_depends_on is not None


# ---------------------------------------------------------------------------
# RefitWindowResult: review_edit's own result carries (or omits) the same
# ---------------------------------------------------------------------------


def test_refit_result_carries_structure_for_implied_create(stage5_multi_file):
    path = stage5_multi_file
    freq = _gap_anchor(path)

    result = refit_window_impl(str(path), None, add=[freq])

    assert result.created_window_mode == "created"
    assert result.created_window_freq_range is not None
    assert result.created_window_n_points is not None
    assert result.created_window_n_contributors is not None
    assert result.created_window_depends_on is not None

    entry = review_log_impl(path)[0]
    created = entry.evidence["created_window"]
    assert created["mode"] == result.created_window_mode
    assert (
        created["freq_min_mhz"],
        created["freq_max_mhz"],
    ) == result.created_window_freq_range
    assert created["n_points"] == result.created_window_n_points
    assert created["n_contributors"] == result.created_window_n_contributors
    assert created["depends_on"] == result.created_window_depends_on


def test_refit_result_leaves_structure_none_for_ordinary_edit(stage5_multi_file):
    path = stage5_multi_file
    wid, uid = _first_peak(path)

    result = refit_window_impl(str(path), wid, remove=[f"uid:{uid}"])

    assert result.created_window_mode is None
    assert result.created_window_freq_range is None
    assert result.created_window_n_points is None
    assert result.created_window_n_contributors is None
    assert result.created_window_depends_on is None


# ---------------------------------------------------------------------------
# Session staged reuse: the previewed outcome, byte for byte, for a plan
# containing an implied create.
# ---------------------------------------------------------------------------


def test_session_staged_reuse_persists_implied_create_from_preview(
    stage5_multi_file, tmp_path
):
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    with Pipeline.open(path).review_session() as session:
        preview = session.review_preview(cur)
        assert len(preview.windows) == 1
        wid = next(iter(preview.windows))
        w = preview.windows[wid]
        assert w.created_window_mode == "created"

        # Spy on the reuse path so the test proves it actually fired, not
        # merely that the numbers happen to agree (test_review_session.py's
        # own D4 pattern).
        calls = {"n": 0}
        orig_persist = session._persist_staged

        def spy(staged):
            calls["n"] += 1
            return orig_persist(staged)

        session._persist_staged = spy
        session.review_apply(cur)

    assert calls["n"] == 1

    log = review_log_impl(path)
    assert len(log) == 1
    assert log[0].window_id == wid
    created = log[0].evidence["created_window"]
    assert created["mode"] == w.created_window_mode
    assert (
        created["freq_min_mhz"],
        created["freq_max_mhz"],
    ) == w.created_window_freq_range
    assert created["n_points"] == w.created_window_n_points
    assert created["n_contributors"] == w.created_window_n_contributors
    assert created["depends_on"] == w.created_window_depends_on


# ---------------------------------------------------------------------------
# CLI: review preview / review apply --dry-run print the implied structure
# ---------------------------------------------------------------------------


def test_cli_review_preview_prints_the_implied_structure(
    stage5_multi_file, tmp_path, capsys
):
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    preview = review_preview_impl(path, cur)
    wid = next(iter(preview.windows))
    w = preview.windows[wid]
    assert w.created_window_freq_range is not None
    lo, hi = w.created_window_freq_range

    rc = cmd_review_preview(
        argparse.Namespace(file_path=str(path), curation_file=cur, verbose=False)
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "created" in out
    assert f"{lo:.4f}" in out
    assert f"{hi:.4f}" in out


def test_cli_review_apply_dry_run_prints_the_implied_structure(
    stage5_multi_file, tmp_path, capsys
):
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    preview = review_preview_impl(path, cur)
    wid = next(iter(preview.windows))
    w = preview.windows[wid]
    assert w.created_window_freq_range is not None
    lo, hi = w.created_window_freq_range

    rc = cmd_review_apply(
        argparse.Namespace(file_path=str(path), curation_file=cur, dry_run=True)
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "created" in out
    assert f"{lo:.4f}" in out
    assert f"{hi:.4f}" in out
    # A dry run still writes nothing.
    assert review_log_impl(path) == []


def test_cli_review_apply_dry_run_without_a_create_prints_nothing_extra(
    stage5_multi_file, tmp_path, capsys
):
    """The narrow-gating check: an ordinary dry run (no create in the plan)
    must not pay for (or print) the extra in-memory preview pass."""
    path = stage5_multi_file
    wid, uid = _first_peak(path)
    cur = _write_curation(tmp_path, "ordinary.csv", f"remove,{wid},uid:{uid},\n")

    rc = cmd_review_apply(
        argparse.Namespace(file_path=str(path), curation_file=cur, dry_run=True)
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "created" not in out
    assert "widened" not in out
