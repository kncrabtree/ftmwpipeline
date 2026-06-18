"""Tests for the end-to-end ``run`` orchestration (``run_pipeline_impl``).

Fast unit tests drive the orchestrator against a fake Pipeline (no fixture fit
needed) to pin the stage ordering, stop-at-first-failure, the non-fatal timebase
skip, report gating, and the dual-interface delegation; plus direct tests of the
``StageProgress`` display. One integration test runs a real narrow-band build.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.progress import ProgressHandler, StageProgress
from ftmwpipeline._internal.run_impl import run_pipeline_impl
from ftmwpipeline.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Fake Pipeline for fast orchestration tests
# ---------------------------------------------------------------------------


class _FakePipe:
    """Records the order of stage calls; can fail or raise on demand."""

    def __init__(self, *, fail_on=None, timebase_exc=None):
        self.calls = []
        self._fail_on = fail_on
        self._timebase_exc = timebase_exc

    def _record(self, name):
        self.calls.append(name)
        if name == self._fail_on:
            raise RuntimeError(f"boom in {name}")

    def detect_start_time(self, **k):
        self._record("start")

    def compute_ft(self, **k):
        self._record("ft")

    def estimate_noise(self, **k):
        self._record("noise")

    def calibrate_tau(self, **k):
        self._record("tau")

    def detect_peaks(self, **k):
        self._record("peaks")

    def assign_windows(self, **k):
        self._record("windows")

    def fit_peaks(self, **k):
        self._record("fit")

    def calibrate_timebase(self, **k):
        self._record("timebase")
        if self._timebase_exc is not None:
            raise self._timebase_exc

    def review_run(self, **k):
        self._record("review")

    def report_run(self, **k):
        self._record("report")
        return {"table": "lines.csv", "html": "report.html"}


@pytest.fixture
def patch_pipeline(monkeypatch):
    """Patch ``Pipeline.create`` to return a shared fake (no file written)."""

    def _install(pipe):
        monkeypatch.setattr(Pipeline, "create", classmethod(lambda cls, *a, **k: pipe))
        return pipe

    return _install


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_run_full_sequence_in_order(patch_pipeline):
    pipe = patch_pipeline(_FakePipe())
    res = run_pipeline_impl("src", output="x.ftmw", trim=(26500, 40000), progress=False)
    assert res["status"] == "success"
    # import (the fake's create), then every stage in canonical order.
    assert pipe.calls == [
        "start",
        "ft",
        "noise",
        "tau",
        "peaks",
        "windows",
        "fit",
        "timebase",
        "review",
    ]
    assert res["completed_stages"][0] == "import"
    assert res["completed_stages"][-1] == "review"
    assert res["timebase"] == "calibrated"


def test_run_requires_trim(patch_pipeline):
    patch_pipeline(_FakePipe())
    with pytest.raises(ValueError, match="trim"):
        run_pipeline_impl("src", output="x.ftmw", trim=None, progress=False)


def test_run_stops_at_first_failure(patch_pipeline):
    pipe = patch_pipeline(_FakePipe(fail_on="peaks"))
    res = run_pipeline_impl("src", output="x.ftmw", trim=(1, 2), progress=False)
    assert res["status"] == "error"
    assert res["failed_stage"] == "peaks"
    assert "boom in peaks" in res["error"]
    # windows / fit never ran.
    assert "windows" not in pipe.calls and "fit" not in pipe.calls
    assert "peaks" not in res["completed_stages"]


def test_timebase_failure_warns_and_skips(patch_pipeline):
    pipe = patch_pipeline(
        _FakePipe(timebase_exc=ValueError("No instrument clock declaration found."))
    )
    stream = io.StringIO()
    res = run_pipeline_impl(
        "src", output="x.ftmw", trim=(1, 2), progress=True, progress_stream=stream
    )
    # Non-fatal: the run still succeeds and reaches review.
    assert res["status"] == "success"
    assert res["timebase"] == "skipped"
    assert "review" in pipe.calls
    assert "timebase" not in res["completed_stages"]
    out = stream.getvalue()
    assert "timebase calibration skipped" in out and "clock declaration" in out


def test_no_cal_skips_timebase(patch_pipeline):
    pipe = patch_pipeline(_FakePipe())
    res = run_pipeline_impl(
        "src", output="x.ftmw", trim=(1, 2), calibrate=False, progress=False
    )
    assert "timebase" not in pipe.calls
    assert res["timebase"] == "not_requested"


def test_no_start_detect_skips_start(patch_pipeline):
    pipe = patch_pipeline(_FakePipe())
    run_pipeline_impl(
        "src", output="x.ftmw", trim=(1, 2), detect_start=False, progress=False
    )
    assert "start" not in pipe.calls
    assert pipe.calls[0] == "ft"


def test_report_gated_on_flag(patch_pipeline):
    pipe = patch_pipeline(_FakePipe())
    res = run_pipeline_impl(
        "src", output="x.ftmw", trim=(1, 2), report=True, progress=False
    )
    assert "report" in pipe.calls
    assert res["report"] == {"table": "lines.csv", "html": "report.html"}

    pipe2 = patch_pipeline(_FakePipe())
    res2 = run_pipeline_impl("src", output="x.ftmw", trim=(1, 2), progress=False)
    assert "report" not in pipe2.calls and res2["report"] is None


def test_cross_interface_delegation(patch_pipeline):
    """api.run_pipeline and Pipeline.build route through run_pipeline_impl."""
    patch_pipeline(_FakePipe())
    via_impl = run_pipeline_impl("src", output="x.ftmw", trim=(1, 2), progress=False)
    patch_pipeline(_FakePipe())  # re-arm the patched create
    via_api = ftmw.run_pipeline("src", output="x.ftmw", trim=(1, 2), progress=False)
    patch_pipeline(_FakePipe())
    via_pipe = Pipeline.build("src", output="x.ftmw", trim=(1, 2), progress=False)
    assert (
        via_impl["completed_stages"]
        == via_api["completed_stages"]
        == via_pipe["completed_stages"]
    )


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------


class _Stream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def test_stage_banner_and_done():
    s = _Stream(tty=False)
    p = StageProgress(2, stream=s)
    with p.stage("noise"):
        pass
    out = s.getvalue()
    assert "[1/2] noise" in out and "✓" in out


def test_stage_marks_failure_and_reraises():
    s = _Stream(tty=False)
    p = StageProgress(1, stream=s)
    with pytest.raises(RuntimeError):
        with p.stage("fit"):
            raise RuntimeError("nope")
    assert "✗ failed" in s.getvalue()


def test_substep_non_tty_emits_decile_lines():
    s = _Stream(tty=False)
    p = StageProgress(1, stream=s)
    with p.stage("fit"):
        for n in range(1, 11):
            p.substep(n, 10)
    out = s.getvalue()
    assert "fit: 50% (5/10)" in out
    assert "fit: 100% (10/10)" in out


def test_substep_tty_uses_carriage_return():
    s = _Stream(tty=True)
    p = StageProgress(1, stream=s)
    with p.stage("fit"):
        p.substep(3, 10)
    assert "\r" in s.getvalue() and "30% (3/10)" in s.getvalue()


def test_disabled_progress_is_silent():
    s = _Stream(tty=False)
    p = StageProgress(1, stream=s, enabled=False)
    with p.stage("fit"):
        p.substep(1, 2)
    assert s.getvalue() == ""


def test_handler_bridges_window_log_to_substep():
    """A plan_execution per-window INFO record renders as a percentage."""
    s = _Stream(tty=False)
    p = StageProgress(1, stream=s)
    handler = ProgressHandler(p)
    with p.stage("fit"):
        rec = logging.LogRecord(
            name="ftmwpipeline.fitting.plan_execution",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="window %d/%d w%d [%.1f-%.1f MHz]: %d peaks",
            args=(5, 10, 42, 1.0, 2.0, 3),
            exc_info=None,
        )
        handler.emit(rec)
    assert "fit: 50% (5/10)" in s.getvalue()


# ---------------------------------------------------------------------------
# Integration: a real narrow-band build
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_run_end_to_end_real(tmp_path):
    data = Path("examples/blackchirp_data/2638")
    if not data.exists():
        pytest.skip("Experiment 2638 data not available")
    out = tmp_path / "run_2638.ftmw"
    res = ftmw.run_pipeline(
        str(data),
        output=str(out),
        trim=(38000, 40000),
        report=True,
        report_output_dir=str(tmp_path / "report"),
        progress=False,
    )
    assert res["status"] == "success", res["error"]
    assert res["completed_stages"][0] == "import"
    assert "review" in res["completed_stages"]
    assert out.exists()

    # The file is finalized: Stage 6 review consolidated a final-products table.
    import h5py

    with h5py.File(out, "r") as f:
        assert "stage6_review" in f
    # --report emitted both artifacts.
    assert res["report"] and res["report"]["table"] and res["report"]["html"]
    assert Path(res["report"]["html"]).exists()
