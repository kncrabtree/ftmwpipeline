"""Tests for the end-to-end ``run`` orchestration (``run_pipeline_impl``).

Fast unit tests drive the orchestrator against a fake Pipeline (no fixture fit
needed) to pin the stage ordering, stop-at-first-failure, the non-fatal timebase
skip, report gating, and the dual-interface delegation; plus direct tests of the
``StageProgress`` display. Also covers the namespaced per-knob passthrough
(``--start.*`` / ``--ft.*`` / ``--noise.*`` / ... / ``--fit.*``) added on top of
``run``: CLI parsing/reconstruction, CLI-vs-api parity, and one cheap real-data
check that a namespaced knob has a genuine effect. One integration test runs a
real narrow-band build end to end.
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.progress import ProgressHandler, StageProgress
from ftmwpipeline._internal.run_impl import run_pipeline_impl
from ftmwpipeline.cli._argspec import settings_from_namespace
from ftmwpipeline.cli.main import main as run_cli
from ftmwpipeline.cli.run_commands import (
    _stage_settings_params,
    _start_detection_params_from_namespace,
    register_run_command,
)
from ftmwpipeline.core.settings import FTSettings
from ftmwpipeline.core.stage_fit_settings import StageFitSettings, TauSubSettings
from ftmwpipeline.core.start_detection_settings import StartDetectionSettings
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
        "timebase",
        "noise",
        "tau",
        "peaks",
        "windows",
        "fit",
        "review",
    ]
    assert res["completed_stages"][0] == "import"
    assert res["completed_stages"][-1] == "review"
    assert res["timebase"] == "calibrated"


def test_timebase_runs_before_noise(patch_pipeline):
    """Timebase self-calibration runs right after FT, before noise (C3 Part 1)."""
    patch_pipeline(_FakePipe())
    res = run_pipeline_impl("src", output="x.ftmw", trim=(26500, 40000), progress=False)
    stages = res["completed_stages"]
    assert "timebase" in stages and "noise" in stages
    assert stages.index("timebase") < stages.index("noise")


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
# Namespaced per-knob passthrough (--start.* / --ft.* / ... / --fit.*)
# ---------------------------------------------------------------------------


def _run_parser() -> argparse.ArgumentParser:
    """Build the real ``run`` parser (mirrors the registered CLI surface)."""
    parser = argparse.ArgumentParser(prog="ftmwpipeline")
    sub = parser.add_subparsers()
    register_run_command(sub)
    return parser


def _parse_run(extra: list) -> argparse.Namespace:
    return _run_parser().parse_args(["run", "raw.dat", "--trim", "8000:18000"] + extra)


class TestNamespacedKnobParsing:
    """The ``run`` parser accepts namespaced per-stage flags and reconstructs
    the correct sparse settings instance, leaving everything else unset."""

    def test_accepts_start_ft_and_nested_fit_flags(self):
        ns = _parse_run(
            [
                "--start.guard-margin-us",
                "1.0",
                "--ft.start-us",
                "2.5",
                "--fit.tau.max-decay-factor",
                "0.9",
            ]
        )
        assert getattr(ns, "start.guard_margin_us") == 1.0
        assert getattr(ns, "ft.start_us") == 2.5
        assert getattr(ns, "fit.tau.max_decay_factor") == 0.9
        # Untouched knobs in the same namespaces stay unset.
        assert getattr(ns, "start.sweep_max_us") is None
        assert getattr(ns, "ft.end_us") is None
        assert getattr(ns, "fit.tau.tau0_us") is None

    def test_ft_trim_is_an_alias_for_trim(self):
        """``--ft.trim`` is a second option string on the canonical ``--trim``
        flag (same dest), not a separate namespaced flag -- so it is excluded
        from the generated ``--ft.*`` group (that generation would otherwise
        collide with this alias) and instead routes to the same ``args.trim``."""
        ns_canonical = _run_parser().parse_args(
            ["run", "raw.dat", "--trim", "8000:18000"]
        )
        ns_alias = _run_parser().parse_args(
            ["run", "raw.dat", "--ft.trim", "8000:18000"]
        )
        assert ns_canonical.trim == (8000.0, 18000.0)
        assert ns_alias.trim == ns_canonical.trim

    def test_reconstructs_ft_settings(self):
        ns = _parse_run(["--ft.start-us", "2.5"])
        ft = settings_from_namespace(ns, FTSettings, prefix="ft")
        assert ft.start_us == 2.5
        assert ft.end_us is None
        assert ft.units_power is None
        assert ft.trim is None

    def test_reconstructs_nested_fit_settings(self):
        ns = _parse_run(["--fit.tau.max-decay-factor", "0.9"])
        fit = settings_from_namespace(ns, StageFitSettings, prefix="fit")
        assert fit.tau.max_decay_factor == 0.9
        assert fit.tau.tau0_us is None  # sibling field in the same sub-block
        assert fit.shape is None
        assert fit.rescue.max_rounds is None  # untouched sub-block

    def test_start_detection_params_from_namespace(self):
        ns = _parse_run(["--start.guard-margin-us", "1.0"])
        params = _start_detection_params_from_namespace(ns)
        assert params == {"settings": StartDetectionSettings(guard_margin_us=1.0)}

    def test_start_detection_params_none_when_unset(self):
        ns = _parse_run([])
        assert _start_detection_params_from_namespace(ns) is None

    def test_stage_settings_params_none_when_unset(self):
        ns = _parse_run([])
        assert _stage_settings_params(ns, StageFitSettings, "fit") is None

    def test_stage_settings_params_set_when_given(self):
        ns = _parse_run(["--fit.tau.max-decay-factor", "0.9"])
        params = _stage_settings_params(ns, StageFitSettings, "fit")
        assert params == {
            "settings": StageFitSettings(tau=TauSubSettings(max_decay_factor=0.9))
        }


class _CapturingPipe(_FakePipe):
    """A :class:`_FakePipe` that also records the kwargs each stage received."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.received: dict = {}

    def detect_start_time(self, **k):
        self.received["start"] = k
        super().detect_start_time(**k)

    def compute_ft(self, **k):
        self.received["ft"] = k
        super().compute_ft(**k)

    def fit_peaks(self, **k):
        self.received["fit"] = k
        super().fit_peaks(**k)


def test_cli_namespaced_knobs_match_api_explicit_params(patch_pipeline):
    """Driving ``run`` via argv with namespaced knobs reaches the same
    stage-call kwargs as ``api.run_pipeline`` given the equivalent explicit
    ``start_detection_params`` / ``ft_params`` / ``fit_params``."""
    pipe_cli = patch_pipeline(_CapturingPipe())
    argv = [
        "run",
        "src",
        "--output",
        "x.ftmw",
        "--trim",
        "1:2",
        "--start.guard-margin-us",
        "1.0",
        "--ft.start-us",
        "2.5",
        "--fit.tau.max-decay-factor",
        "0.9",
        "--quiet",
    ]
    assert run_cli(argv) == 0

    pipe_api = patch_pipeline(_CapturingPipe())
    ftmw.run_pipeline(
        "src",
        output="x.ftmw",
        trim=(1.0, 2.0),
        progress=False,
        start_detection_params={
            "settings": StartDetectionSettings(guard_margin_us=1.0)
        },
        ft_params={"start_us": 2.5},
        fit_params={
            "settings": StageFitSettings(tau=TauSubSettings(max_decay_factor=0.9))
        },
    )

    assert pipe_cli.received["start"] == pipe_api.received["start"]
    assert pipe_cli.received["ft"] == pipe_api.received["ft"]
    assert pipe_cli.received["fit"] == pipe_api.received["fit"]
    # The other (un-namespaced) stages saw no override either way.
    assert pipe_cli.received["ft"] == {"start_us": 2.5, "trim": (1.0, 2.0)}


@pytest.mark.integration
def test_start_guard_margin_knob_changes_recommended_start(tmp_path):
    """A cheap, real-data check that ``--start.guard-margin-us`` (routed
    through ``start_detection_params``) has a genuine effect: the stamped
    Stage-0 recommended ``start_us`` moves by exactly the change in margin,
    since both share the same detected chirp end. Only Stage 0 runs -- no
    need to drive the full pipeline to exercise this knob.
    """
    data = Path("examples/blackchirp_data/1512")
    if not data.exists():
        pytest.skip("Experiment 1512 data not available")

    default_file = tmp_path / "default.ftmw"
    overridden_file = tmp_path / "overridden.ftmw"

    pipe_default = Pipeline.create(default_file, source=data, force=True)
    default_start = pipe_default.detect_start_time().start_us

    pipe_override = Pipeline.create(overridden_file, source=data, force=True)
    override_start = pipe_override.detect_start_time(
        **_start_detection_params_from_namespace(
            _parse_run(["--start.guard-margin-us", "5.0"])
        )
    ).start_us

    assert override_start != default_start
    assert override_start - default_start == pytest.approx(5.0 - 0.67, abs=1e-6)


@pytest.mark.integration
def test_ft_start_us_knob_overrides_ft(tmp_path):
    """``--ft.start-us`` (routed through ``ft_params``) actually changes the
    resulting FT relative to the default (no explicit start_us)."""
    data = Path("examples/blackchirp_data/1512")
    if not data.exists():
        pytest.skip("Experiment 1512 data not available")

    pipe = Pipeline.create(tmp_path / "ft.ftmw", source=data, force=True)
    ft_default = pipe.compute_ft(trim=(26500, 40000))

    ns = _parse_run(["--ft.start-us", "2.5"])
    ft_overrides = settings_from_namespace(ns, FTSettings, prefix="ft").overrides()
    assert ft_overrides == {"start_us": 2.5}
    ft_overridden = pipe.compute_ft(trim=(26500, 40000), **ft_overrides)

    import numpy as np

    assert not np.allclose(
        ft_default.magnitude_spectrum, ft_overridden.magnitude_spectrum
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


def _window_log_record(msg: str, args: tuple) -> logging.LogRecord:
    return logging.LogRecord(
        name="ftmwpipeline.fitting.plan_execution",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def test_handler_bridges_window_log_to_substep():
    """A plan_execution progress INFO record renders as a percentage."""
    s = _Stream(tty=False)
    p = StageProgress(1, stream=s)
    handler = ProgressHandler(p)
    with p.stage("fit"):
        handler.emit(_window_log_record("window %d/%d", (5, 10)))
    assert "fit: 50% (5/10)" in s.getvalue()


def test_handler_ignores_per_window_detail_line():
    """The worker's per-window detail line must not drive the bar.

    It carries a window id, not a completion count; only the dedicated
    ``window n/total`` progress record does.
    """
    s = _Stream(tty=False)
    p = StageProgress(1, stream=s)
    handler = ProgressHandler(p)
    with p.stage("fit"):
        handler.emit(
            _window_log_record(
                "w%d [%.1f-%.1f MHz]: %d peaks, chi2r=%.3g, %.1fs",
                (42, 1.0, 2.0, 3, 1.1, 0.5),
            )
        )
    assert "%" not in s.getvalue()


def test_substep_never_goes_backwards():
    """A count at or below the high-water mark is dropped, not rendered.

    Guards the stage-level display against an emitter that legitimately
    replays sub-steps (the fit walk's sequential redo after an accepted thaw).
    """
    s = _Stream(tty=False)
    p = StageProgress(1, stream=s)
    with p.stage("fit"):
        p.substep(9, 10)
        p.substep(3, 10)
        p.substep(1, 10)
    out = s.getvalue()
    assert "90% (9/10)" in out
    assert "30%" not in out and "10%" not in out


def test_substep_high_water_mark_resets_per_stage():
    """Each stage starts its own bar; a later stage is not gated by an earlier."""
    s = _Stream(tty=False)
    p = StageProgress(2, stream=s)
    with p.stage("fit"):
        p.substep(10, 10)
    with p.stage("report"):
        p.substep(1, 10)
    assert "report: 10% (1/10)" in s.getvalue()


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
