"""
Tests for A3 (curation-file frame header + epsilon stamp) and A5 (the
frame-mismatch diagnostic), from ``scratch/preview-session-plan.md``, "Task
A3+A5 -- curation-file frame, and the mismatch diagnostic".

Fixture recipe duplicated from ``test_frame_parameter.py`` (that file's own
stated precedent: each test file states its own fixture contract rather than
importing it) -- an ``rb_locked``/``uncalibrated`` fixture has ``epsilon ==
0``, where every frame bug (and this diagnostic) is invisible, so every test
that matters here forces a ``self_calibrated`` fixture.

A3's drift refusal is the one defect-shaped claim in this unit (a stale
calibrated curation file is silently absorbed today, absent the check) --
``TestHeaderEpsilonDrift`` is written and exercised test-first; see that
class's docstring for the red-state record.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage6_impl import (
    FittingResult,
    _current_calibration_stamp,
    apply_curation_impl,
    parse_curation_file,
    review_preview_impl,
)
from ftmwpipeline.cli.review_commands import cmd_review_apply, cmd_review_preview
from ftmwpipeline.core.data_structures import SpectrumFit
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
EPS_1_SMALL_DRIFT = 4.4e-6
"""A smaller post-staging drift than ``EPS_2`` (delta 2.2e-6, not 6.6e-6) --
on this fixture's ~14.3 GHz baseband that keeps the resulting frame error
(~31.6 kHz) under the 50 kHz snap tolerance, so disabling the drift check
(``TestHeaderEpsilonDrift.test_probe_defect_is_real``) demonstrates genuine
silent absorption rather than a peak-missing failure for an unrelated
reason."""
SIGMA_EPS = 0.1e-6


# ---------------------------------------------------------------------------
# self_calibrated fixture recipe (scratch/preview-session-plan.md, "Verified
# facts for later units").
# ---------------------------------------------------------------------------


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


def _make_self_calibrated(
    path: Path, *, epsilon: float, sigma_epsilon: float = SIGMA_EPS
) -> None:
    _declare_unlocked_digitizer(path)
    _stamp_timebase(path, epsilon=epsilon, sigma_epsilon=sigma_epsilon)


@pytest.fixture(scope="session")
def _stage5_sc_small_built(_stage5_small_built: Path, tmp_path_factory) -> Path:
    fp = tmp_path_factory.mktemp("stage6_curhdr_sc") / "stage5_sc_small.ftmw"
    shutil.copy(_stage5_small_built, fp)
    _make_self_calibrated(fp, epsilon=EPS_1)
    return fp


@pytest.fixture
def sc_file(_stage5_sc_small_built: Path, tmp_path: Path) -> Path:
    fp = tmp_path / "sc.ftmw"
    shutil.copy(_stage5_sc_small_built, fp)
    return fp


# A5's diagnostic needs several distinct matched candidates in one batch --
# the shared 3-window small build above typically keeps only one or two
# fitted peaks (most windows fail their Stage 5 gate on this narrow a slice).
# Reuse test_curation.py's wider recipe (12 dependency-free windows) locally,
# per tests/AGENTS.md's "check for an override" note: this is a distinct,
# file-local fixture, not the conftest one.


def _build_stage5_multi(dest: Path, data_path: str) -> None:
    from ftmwpipeline._internal.stage4_impl import (
        load_windows_impl,
        save_window_plan_impl,
    )

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


@pytest.fixture(scope="session")
def _stage5_multi_sc_built(exp_2638_data_path: str, tmp_path_factory) -> Path:
    fp = tmp_path_factory.mktemp("stage6_curhdr_multi_sc") / "stage5_multi_sc.ftmw"
    _build_stage5_multi(fp, exp_2638_data_path)
    _make_self_calibrated(fp, epsilon=EPS_1)
    return fp


@pytest.fixture
def sc_multi_file(_stage5_multi_sc_built: Path, tmp_path: Path) -> Path:
    fp = tmp_path / "sc_multi.ftmw"
    shutil.copy(_stage5_multi_sc_built, fp)
    return fp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_spectrum_fit(path: Path) -> SpectrumFit:
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _fitted_windows_with_n_peaks(path: Path, n: int) -> List[FittingResult]:
    sf = _load_spectrum_fit(path)
    return [wf for wf in sf.window_fits if len(wf.fitted_peaks) >= n]


def _ref_calibrated(f_raw: float, *, probe: float, eps: float) -> float:
    """Independent re-derivation of ``f_corr = probe + (f_raw-probe)/(1+eps)``."""
    return probe + (f_raw - probe) / (1.0 + eps)


def _write_curation(tmp_path: Path, text: str, name: str = "cur.csv") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# A3: the file-level header and its precedence with the ``frame`` argument.
# ---------------------------------------------------------------------------


class TestHeaderParsing:
    def test_no_header_is_none_none(self, tmp_path: Path) -> None:
        ops = parse_curation_file(_write_curation(tmp_path, "add,5,100.0,\n"))
        assert ops.header.frame is None
        assert ops.header.epsilon is None
        # Backward compatibility: still a plain list of CurationOp.
        assert len(ops) == 1
        assert ops[0].action == "add"

    def test_frame_raw_header_parsed(self, tmp_path: Path) -> None:
        ops = parse_curation_file(
            _write_curation(tmp_path, "# frame: raw\nadd,5,100.0,\n")
        )
        assert ops.header.frame == "raw"
        assert ops.header.epsilon is None

    def test_frame_calibrated_header_requires_epsilon(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires an 'epsilon' header"):
            parse_curation_file(
                _write_curation(tmp_path, "# frame: calibrated\nadd,5,100.0,\n")
            )

    def test_epsilon_header_requires_frame_calibrated(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires a 'frame: calibrated'"):
            parse_curation_file(
                _write_curation(tmp_path, "# epsilon: 2.2e-6\nadd,5,100.0,\n")
            )

    def test_frame_calibrated_with_epsilon_parses(self, tmp_path: Path) -> None:
        ops = parse_curation_file(
            _write_curation(
                tmp_path, "# frame: calibrated\n# epsilon: 2.2e-6\nadd,5,100.0,\n"
            )
        )
        assert ops.header.frame == "calibrated"
        assert ops.header.epsilon == pytest.approx(2.2e-6)

    def test_bad_frame_value_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="'frame' header must be"):
            parse_curation_file(
                _write_curation(tmp_path, "# frame: sideways\nadd,5,100.0,\n")
            )

    def test_conflicting_frame_headers_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="conflicting 'frame' header"):
            parse_curation_file(
                _write_curation(
                    tmp_path, "# frame: raw\n# frame: calibrated\nadd,5,100.0,\n"
                )
            )

    def test_ordinary_comments_still_ignored(self, tmp_path: Path) -> None:
        ops = parse_curation_file(
            _write_curation(
                tmp_path,
                "# just a note about window 5\nadd,5,100.0,\n# trailing note\n",
            )
        )
        assert ops.header.frame is None
        assert len(ops) == 1


class TestHeaderEpsilonDrift:
    """A3's defect-shaped claim: a curation file staged calibrated at one
    epsilon, applied after the target file's calibration has moved, must be
    REFUSED rather than silently resolved against the (now wrong) raw
    frequencies.

    Written and run test-first. Red state, observed directly by temporarily
    disabling the epsilon-comparison block in ``_resolve_curation_frame``
    (``stage6_impl.py``, guarded ``if header.epsilon is not None and stamp
    is not None:``) and re-running just this test:

    ``AssertionError: Regex pattern did not match. Expected regex: 'frame
    drift' Actual message: 'curation action 1 (edit window 2: remove
    26613.4872) failed: remove=26613.4872 MHz: no fitted peak within 0.050
    MHz (closest is at 26613.5819 MHz, distance=0.0947 MHz)'``

    With the check disabled, the file-staged-``EPS_1``,
    drifted-to-``EPS_2`` curation file was NOT refused -- it was converted to
    raw using the file's CURRENT epsilon (``_frame_to_raw`` always uses
    :func:`_current_calibration_stamp`, never the header's stamped value),
    landing ~94.7 kHz from the true raw frequency (``delta_eps *
    f_baseband``, the same arithmetic the A7 staleness bug hit). At this
    particular drift that offset happens to exceed the 50 kHz snap
    tolerance, so the batch still raised -- but a generic, unrelated "no
    fitted peak" error instead of the specific, actionable frame-drift one,
    which is itself a worse failure mode for a caller to debug. At a
    SMALLER drift the same missing check produces true silent
    absorption -- no error at all, against the wrong peak -- reproduced
    directly below via ``test_probe_defect_is_real``, which monkeypatches
    the check away (the sanctioned red-state reconstruction method for this
    environment, per ``scratch/preview-session-plan.md``'s process rules)
    rather than re-editing ``src/``. Restoring the epsilon-comparison block
    turns this test green again.
    """

    def _stage_calibrated_remove(
        self, sc_file: Path, tmp_path: Path
    ) -> Tuple[Path, int, float]:
        """Build a curation file that removes a real fitted peak, declared
        via the header as calibrated and stamped at the file's current
        (``EPS_1``) epsilon."""
        wins = _fitted_windows_with_n_peaks(sc_file, 1)
        assert wins, "fixture has no fitted window"
        wf = wins[0]
        f_raw = float(wf.fitted_peaks[0].frequency_mhz)

        stamp = _current_calibration_stamp(str(sc_file))
        assert stamp is not None and stamp[0] == "self_calibrated"
        eps, probe = stamp[1], stamp[4]
        f_cal = _ref_calibrated(f_raw, probe=probe, eps=eps)

        wid = int(wf.window_id) if wf.window_id is not None else -1
        cur = _write_curation(
            tmp_path,
            f"# frame: calibrated\n# epsilon: {eps!r}\nremove,{wid},{f_cal!r},\n",
        )
        return cur, wid, f_raw

    def test_apply_refuses_when_header_epsilon_disagrees_with_current(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        cur, _wid, _f_raw = self._stage_calibrated_remove(sc_file, tmp_path)

        # Drift the file's calibration after the curation file was staged.
        _stamp_timebase(sc_file, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)

        with pytest.raises(ValueError, match="frame drift") as excinfo:
            apply_curation_impl(str(sc_file), cur)
        msg = str(excinfo.value)
        assert f"{EPS_1:.6e}" in msg
        assert f"{EPS_2:.6e}" in msg

        # Refused calls leave the file untouched.
        review = load_stage6_review_from_file(str(sc_file))
        assert not review.decision_log

    def test_preview_refuses_identically(self, sc_file: Path, tmp_path: Path) -> None:
        cur, _wid, _f_raw = self._stage_calibrated_remove(sc_file, tmp_path)
        _stamp_timebase(sc_file, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)

        with pytest.raises(ValueError, match="frame drift"):
            review_preview_impl(str(sc_file), cur)

    def test_apply_succeeds_when_epsilon_unchanged(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        """Sanity: the SAME header, with NO intervening drift, must not be
        refused -- only a genuine mismatch triggers this."""
        cur, wid, f_raw = self._stage_calibrated_remove(sc_file, tmp_path)
        n_before = len(_load_spectrum_fit(sc_file).window_fits[0].fitted_peaks)
        result = apply_curation_impl(str(sc_file), cur)
        assert result.applied == 1
        wf_after = next(
            wf for wf in _load_spectrum_fit(sc_file).window_fits if wf.window_id == wid
        )
        assert all(
            abs(float(p.frequency_mhz) - f_raw) > 1e-6 for p in wf_after.fitted_peaks
        )

    def test_probe_defect_is_real(
        self, sc_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reconstruct the pre-fix defect via monkeypatch (sanctioned
        red-state method for this environment): with the epsilon comparison
        disabled, the SAME staged-then-drifted file is silently absorbed --
        the batch applies without error, against the wrong raw frequency.

        Uses a SMALL drift (``EPS_1`` -> ``EPS_1_SMALL_DRIFT``, not the
        larger ``EPS_2`` the refusal tests use) so the resulting frame error
        stays under the 50 kHz snap tolerance and is genuinely absorbed
        rather than missing every peak outright -- with the larger ``EPS_2``
        drift, disabling the check produces a DIFFERENT, also-bad outcome
        (a generic "no fitted peak within tolerance" failure, since that
        drift's ~95 kHz error exceeds the snap tolerance) which is a less
        pointed demonstration of the silent-absorption failure mode this
        check specifically exists to prevent.
        """
        import ftmwpipeline._internal.stage6_impl as s6

        cur, wid, f_raw = self._stage_calibrated_remove(sc_file, tmp_path)
        _stamp_timebase(sc_file, epsilon=EPS_1_SMALL_DRIFT, sigma_epsilon=SIGMA_EPS)

        real_isclose = s6.math.isclose
        monkeypatch.setattr(
            s6.math,
            "isclose",
            lambda *a, **k: True,  # pretend every epsilon always matches
        )
        try:
            result = apply_curation_impl(str(sc_file), cur)
        finally:
            monkeypatch.setattr(s6.math, "isclose", real_isclose)

        # The defect: no refusal, and the removed frequency is silently
        # resolved (it lands within the 50 kHz snap tolerance) against the
        # peak nearest the STALE (EPS_1) conversion, computed against the
        # CURRENT (EPS_1_SMALL_DRIFT) epsilon -- exactly the drift the check
        # exists to catch, and it goes through with no error raised at all.
        assert result.applied == 1
        wf_after = next(
            wf for wf in _load_spectrum_fit(sc_file).window_fits if wf.window_id == wid
        )
        assert all(
            abs(float(p.frequency_mhz) - f_raw) > 1e-6 for p in wf_after.fitted_peaks
        )


class TestHeaderFramePrecedence:
    def test_header_alone_wins(self, sc_file: Path, tmp_path: Path) -> None:
        """Header declares raw, no ``frame`` argument passed -- omitting the
        argument would normally be refused on a self_calibrated file, but
        the header alone resolves it."""
        wins = _fitted_windows_with_n_peaks(sc_file, 1)
        wf = wins[0]
        f_raw = float(wf.fitted_peaks[0].frequency_mhz)
        cur = _write_curation(
            tmp_path, f"# frame: raw\nremove,{wf.window_id},{f_raw!r},\n"
        )
        result = apply_curation_impl(str(sc_file), cur)
        assert result.applied == 1

    def test_argument_alone_wins(self, sc_file: Path, tmp_path: Path) -> None:
        wins = _fitted_windows_with_n_peaks(sc_file, 1)
        wf = wins[0]
        f_raw = float(wf.fitted_peaks[0].frequency_mhz)
        cur = _write_curation(tmp_path, f"remove,{wf.window_id},{f_raw!r},\n")
        result = apply_curation_impl(str(sc_file), cur, frame="raw")
        assert result.applied == 1

    def test_agreeing_header_and_argument_succeed(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        wins = _fitted_windows_with_n_peaks(sc_file, 1)
        wf = wins[0]
        f_raw = float(wf.fitted_peaks[0].frequency_mhz)
        cur = _write_curation(
            tmp_path, f"# frame: raw\nremove,{wf.window_id},{f_raw!r},\n"
        )
        result = apply_curation_impl(str(sc_file), cur, frame="raw")
        assert result.applied == 1

    def test_disagreeing_header_and_argument_refused(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        wins = _fitted_windows_with_n_peaks(sc_file, 1)
        wf = wins[0]
        f_raw = float(wf.fitted_peaks[0].frequency_mhz)
        cur = _write_curation(
            tmp_path, f"# frame: raw\nremove,{wf.window_id},{f_raw!r},\n"
        )
        with pytest.raises(ValueError, match="frame disagreement"):
            apply_curation_impl(str(sc_file), cur, frame="calibrated")
        # Refused calls leave the file untouched.
        assert not load_stage6_review_from_file(str(sc_file)).decision_log

    def test_neither_present_falls_back_to_normal_rule(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        """Absent both header and argument, the ordinary rule applies:
        refuse on a self_calibrated file."""
        wins = _fitted_windows_with_n_peaks(sc_file, 1)
        wf = wins[0]
        f_raw = float(wf.fitted_peaks[0].frequency_mhz)
        cur = _write_curation(tmp_path, f"remove,{wf.window_id},{f_raw!r},\n")
        with pytest.raises(ValueError, match="frame is required"):
            apply_curation_impl(str(sc_file), cur)
        assert not load_stage6_review_from_file(str(sc_file)).decision_log

    def test_neither_present_is_inert_on_rb_locked_file(
        self, stage5_small_source: Path, tmp_path: Path
    ) -> None:
        fp = tmp_path / "rb.ftmw"
        shutil.copy(stage5_small_source, fp)
        wf = _fitted_windows_with_n_peaks(fp, 1)[0]
        f_raw = float(wf.fitted_peaks[0].frequency_mhz)
        cur = _write_curation(tmp_path, f"remove,{wf.window_id},{f_raw!r},\n")
        result = apply_curation_impl(str(fp), cur)
        assert result.applied == 1


class TestCrossInterfaceHeader:
    """CLI, Pipeline and the functional API agree on the header-drift refusal."""

    def _stage(self, source: Path, dest: Path) -> Path:
        shutil.copy(source, dest)
        wins = _fitted_windows_with_n_peaks(dest, 1)
        wf = wins[0]
        f_raw = float(wf.fitted_peaks[0].frequency_mhz)
        stamp = _current_calibration_stamp(str(dest))
        assert stamp is not None
        eps, probe = stamp[1], stamp[4]
        f_cal = _ref_calibrated(f_raw, probe=probe, eps=eps)
        cur = dest.parent / f"{dest.stem}_cur.csv"
        cur.write_text(
            f"# frame: calibrated\n# epsilon: {eps!r}\n"
            f"remove,{wf.window_id},{f_cal!r},\n"
        )
        return cur

    def test_drift_refusal_agrees_across_interfaces(
        self, sc_file: Path, tmp_path: Path
    ) -> None:
        paths: Dict[str, Path] = {}
        curs: Dict[str, Path] = {}
        for name in ("impl", "pipe", "api", "cli"):
            p = tmp_path / f"{name}.ftmw"
            paths[name] = p
            curs[name] = self._stage(sc_file, p)

        for p in paths.values():
            _stamp_timebase(p, epsilon=EPS_2, sigma_epsilon=SIGMA_EPS)

        with pytest.raises(ValueError, match="frame drift"):
            apply_curation_impl(str(paths["impl"]), curs["impl"])
        with pytest.raises(ValueError, match="frame drift"):
            Pipeline.open(paths["pipe"]).review_apply(curs["pipe"])
        with pytest.raises(ValueError, match="frame drift"):
            ftmw.review_apply(str(paths["api"]), curs["api"])

        rc = cmd_review_apply(
            argparse.Namespace(
                file_path=str(paths["cli"]),
                curation_file=str(curs["cli"]),
                dry_run=False,
                frame=None,
                verbose=False,
            )
        )
        assert rc == 1

        for p in paths.values():
            assert not load_stage6_review_from_file(str(p)).decision_log


# ---------------------------------------------------------------------------
# A5: the frame-mismatch diagnostic.
# ---------------------------------------------------------------------------


def _build_batch_curation(
    sc_file: Path, tmp_path: Path, *, n: int, correct_frame: bool, name: str
) -> Path:
    """Build a curation file that removes ``n`` fitted peaks (drawn from
    whichever windows the fixture happens to have fitted -- possibly all
    from one window, since the shared small build typically keeps only one
    window with a surviving fit), either with correctly-converted raw
    frequencies (``correct_frame=True``) or with calibrated-equivalent
    frequencies declared raw (the omitted-conversion mistake,
    ``correct_frame=False``)."""
    sf = _load_spectrum_fit(sc_file)
    stamp = _current_calibration_stamp(str(sc_file))
    assert stamp is not None
    eps, probe = stamp[1], stamp[4]

    targets: List[str] = []
    count = 0
    for wf in sf.window_fits:
        if wf.window_id is None:
            continue
        for peak in wf.fitted_peaks:
            f_raw = float(peak.frequency_mhz)
            if correct_frame:
                submitted = f_raw
            else:
                submitted = _ref_calibrated(f_raw, probe=probe, eps=eps)
            targets.append(f"remove,{wf.window_id},{submitted!r},\n")
            count += 1
            if count >= n:
                break
        if count >= n:
            break
    assert count >= n, f"fixture does not have {n} fitted peaks"
    text = "".join(targets)
    return _write_curation(tmp_path, text, name=name)


class TestFrameMismatchDiagnostic:
    def test_advisory_fires_on_systematically_mismatched_batch(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        cur = _build_batch_curation(
            sc_multi_file, tmp_path, n=3, correct_frame=False, name="mismatch.csv"
        )
        result = apply_curation_impl(str(sc_multi_file), cur, frame="raw")
        assert result.applied >= 1
        joined = " ".join(result.warnings)
        assert "frame" in joined.lower()
        assert "calibrated" in joined.lower()

    def test_advisory_fires_on_preview_too(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        cur = _build_batch_curation(
            sc_multi_file, tmp_path, n=3, correct_frame=False, name="mismatch.csv"
        )
        result = review_preview_impl(str(sc_multi_file), cur, frame="raw")
        joined = " ".join(result.warnings)
        assert "frame" in joined.lower()

    def test_advisory_silent_on_correctly_framed_batch(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        """The false-positive check: an ordinary, correctly-declared-raw
        batch (the frequencies really are raw, nothing was mis-declared)
        must NOT trigger the advisory."""
        cur = _build_batch_curation(
            sc_multi_file, tmp_path, n=3, correct_frame=True, name="correct.csv"
        )
        result = apply_curation_impl(str(sc_multi_file), cur, frame="raw")
        assert result.applied >= 1
        assert result.warnings == []

    def test_advisory_silent_when_frame_declared_calibrated(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        """When the caller correctly declares calibrated, the conversion
        already happens and no systematic residual remains -- inert."""
        cur = _build_batch_curation(
            sc_multi_file, tmp_path, n=3, correct_frame=False, name="mismatch.csv"
        )
        result = apply_curation_impl(str(sc_multi_file), cur, frame="calibrated")
        assert result.applied >= 1
        assert result.warnings == []

    def test_advisory_silent_below_minimum_candidate_count(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        """Only 1-2 mismatched candidates: too few to call it a batch-wide
        signature -- must not fire (keeps the false-positive rate down)."""
        cur = _build_batch_curation(
            sc_multi_file, tmp_path, n=1, correct_frame=False, name="one.csv"
        )
        result = apply_curation_impl(str(sc_multi_file), cur, frame="raw")
        assert result.warnings == []

    def test_advisory_inert_on_rb_locked_file(
        self, stage5_small_source: Path, tmp_path: Path
    ) -> None:
        """epsilon == 0 on an rb_locked file -- the two frames coincide, so
        there is no mismatch signature to detect at all."""
        fp = tmp_path / "rb.ftmw"
        shutil.copy(stage5_small_source, fp)
        sf = _load_spectrum_fit(fp)
        targets: List[str] = []
        for wf in sf.window_fits:
            if wf.fitted_peaks and wf.window_id is not None:
                f_raw = float(wf.fitted_peaks[0].frequency_mhz)
                targets.append(f"remove,{wf.window_id},{f_raw!r},\n")
            if len(targets) >= 3:
                break
        cur = _write_curation(tmp_path, "".join(targets))
        result = apply_curation_impl(str(fp), cur)
        assert result.warnings == []

    def test_advisory_is_never_a_refusal(
        self, sc_multi_file: Path, tmp_path: Path
    ) -> None:
        """A5 must never block the batch -- it is advisory only."""
        cur = _build_batch_curation(
            sc_multi_file, tmp_path, n=3, correct_frame=False, name="mismatch.csv"
        )
        result = apply_curation_impl(str(sc_multi_file), cur, frame="raw")
        assert result.applied >= 1  # not raised, not zero
