"""``analysis_fingerprint`` on real files (CONTRACT_STRATEGY §Analysis
fingerprint).

Checks the spec's promises, not the implementation's choices: the digest is the
same across API / ``Pipeline`` / CLI; two independent builds agree; completing a
stage or changing a recorded input moves it; a write that is not an input does
not; a file that cannot account for an input is refused with every gap listed;
an incomplete stage reads as not run whatever records remain.

Builds come from the session-scoped baselines in ``conftest.py`` (read only);
every test that mutates a file copies it into ``tmp_path`` first. The canonical
input object is read through the private ``canonical_fingerprint_inputs``
diagnostic so a test can say *which* key moved.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline._internal.fingerprint_impl import (
    canonical_fingerprint_inputs,
    digest_of,
)
from ftmwpipeline.cli.main import main
from ftmwpipeline.contract import ANALYSIS_FINGERPRINT_SCHEMA, Stage, key_for_stage
from ftmwpipeline.core.noise_settings import NoiseSettings
from ftmwpipeline.core.stage_fit_settings import ClockSource, ShapeSpec
from ftmwpipeline.file_manager import (
    IncompleteProvenanceError,
    PipelineFileNotFoundError,
)
from ftmwpipeline.io.environment_serialization import (
    _STAGE_ENVS_ATTR,
    _STAGES_GROUP,
)
from ftmwpipeline.io.noise_settings_serialization import (
    STAGE2_NOISE_FIELD_SET_VERSION,
    STAGE2_NOISE_SETTINGS_PATH,
    load_noise_settings_from_h5,
)
from ftmwpipeline.io.provenance import (
    FIELD_SET_VERSION_ATTR,
    write_field_set_version,
)
from ftmwpipeline.io.stage_fit_settings_serialization import STAGE_FIT_PATH
from ftmwpipeline.io.timebase_serialization import (
    _NONE_SENTINEL,
)
from ftmwpipeline.io.timebase_serialization import GROUP_PATH as TIMEBASE_PATH

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _digest(path) -> str:
    return ftmw.analysis_fingerprint(str(path))["digest"]


def _inputs(path) -> Dict[str, Any]:
    return canonical_fingerprint_inputs(str(path))


def _flat(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Every leaf of *obj* by dotted key (list items by ``[i]``)."""
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        if not obj:
            out[prefix] = []
        for i, v in enumerate(obj):
            out.update(_flat(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _differing(a: Dict[str, Any], b: Dict[str, Any]) -> Set[str]:
    fa, fb = _flat(a), _flat(b)
    return {
        k
        for k in set(fa) | set(fb)
        if repr(fa.get(k, _MISSING)) != repr(fb.get(k, _MISSING))
    }


_MISSING = object()


def _copy(src, tmp_path: Path, name: str = "work.ftmw") -> str:
    dst = tmp_path / name
    shutil.copy(src, dst)
    return str(dst)


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _strip_version(path: str, group: str) -> None:
    with h5py.File(path, "a") as f:
        del f[group].attrs[FIELD_SET_VERSION_ATTR]


def _bump_version(path: str, group: str, current: int) -> None:
    with h5py.File(path, "a") as f:
        write_field_set_version(f[group].attrs, current + 1)


def _edit_environments(path: str, edit: Callable[[Dict[str, Any]], None]) -> None:
    with h5py.File(path, "a") as f:
        attrs = f[_STAGES_GROUP].attrs
        blob = json.loads(attrs[_STAGE_ENVS_ATTR])
        edit(blob)
        attrs[_STAGE_ENVS_ATTR] = json.dumps(blob, sort_keys=True)


def _noise_missing() -> Set[str]:
    return {f"noise.{f.name}" for f in dataclasses.fields(NoiseSettings)}


def _cli(argv: List[str], capsys):
    rc = main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


@pytest.fixture
def fit_file(baseline_2638_stage5_small, tmp_path) -> str:
    """A private writable copy of the small Stage 5 build (no review)."""
    return _copy(baseline_2638_stage5_small, tmp_path)


@pytest.fixture
def reviewed_file(stage5_reviewed_2638, tmp_path) -> str:
    """A private writable copy of the reviewed build."""
    return _copy(stage5_reviewed_2638, tmp_path)


# ---------------------------------------------------------------------------
# Cross-interface
# ---------------------------------------------------------------------------
def test_api_pipeline_and_cli_return_the_same_payload(stage5_reviewed_2638, capsys):
    # Mutation: any one interface computing its own digest, a differently shaped
    # payload, an envelope that drops or renames the schema, or a read that
    # writes the file.
    path = str(stage5_reviewed_2638)
    before = _md5(path)
    via_api = ftmw.analysis_fingerprint(path)
    via_pipeline = Pipeline.open(path).analysis_fingerprint()
    rc, out, err = _cli(
        ["read", "analysis_fingerprint", path, "--format", "json"], capsys
    )
    assert rc == 0 and err == ""
    via_cli = json.loads(out)
    assert via_api == via_pipeline == via_cli
    assert set(via_api) == {"schema", "digest"}
    assert via_api["schema"] == ANALYSIS_FINGERPRINT_SCHEMA
    assert via_api["schema"] == "ftmw/analysis_fingerprint@1"
    assert _HEX64.match(via_api["digest"])
    assert via_api["digest"] == digest_of(_inputs(path))
    assert _md5(path) == before


def test_incomplete_provenance_is_the_same_error_on_every_interface(fit_file, capsys):
    # Mutation: an interface that swallows the refusal, hashes anyway, or wraps
    # it in another error type / exit code.
    _strip_version(fit_file, STAGE2_NOISE_SETTINGS_PATH)
    expected = _noise_missing()
    with pytest.raises(IncompleteProvenanceError) as api_err:
        ftmw.analysis_fingerprint(fit_file)
    with pytest.raises(IncompleteProvenanceError) as pipe_err:
        Pipeline.open(fit_file).analysis_fingerprint()
    assert set(api_err.value.missing) == set(pipe_err.value.missing) == expected
    rc, out, err = _cli(
        ["read", "analysis_fingerprint", fit_file, "--format", "json"], capsys
    )
    assert rc == 1 and out == ""
    payload = json.loads(err)
    assert payload["schema"] == "ftmw/error@1"
    assert payload["code"] == "incomplete_provenance"
    assert set(payload["missing"]) == expected


def test_missing_file_is_not_found_on_every_interface(tmp_path, capsys):
    gone = str(tmp_path / "gone.ftmw")
    with pytest.raises(PipelineFileNotFoundError):
        ftmw.analysis_fingerprint(gone)
    with pytest.raises(PipelineFileNotFoundError):
        Pipeline.open(gone)
    rc, out, err = _cli(
        ["read", "analysis_fingerprint", gone, "--format", "json"], capsys
    )
    assert rc == 1 and out == ""
    payload = json.loads(err)
    assert payload["schema"] == "ftmw/error@1" and payload["code"] == "not_found"


# ---------------------------------------------------------------------------
# Shape of the canonical input object
# ---------------------------------------------------------------------------
def test_every_stage_key_is_present_and_unrun_stages_say_so(
    baseline_2638_stage4_small,
):
    # Spec: every stage key is always present; a stage that has not run is null
    # with a sibling "<stage>_absent": "not_run". Mutation: omitting the key,
    # or hashing the leftover record of an unrun stage.
    inputs = _inputs(baseline_2638_stage4_small)
    for stage in Stage:
        assert stage.value in inputs
    for stage in (Stage.TAU, Stage.TAU_G, Stage.TIMEBASE, Stage.FIT, Stage.REVIEW):
        assert inputs[stage.value] is None, stage
        assert inputs[f"{stage.value}_absent"] == "not_run"
    for stage in (Stage.DATA, Stage.FT, Stage.NOISE, Stage.PEAKS, Stage.WINDOWS):
        assert isinstance(inputs[stage.value], dict), stage
        assert f"{stage.value}_absent" not in inputs
        assert isinstance(inputs[stage.value]["analysis_epoch"], int)


def test_a_complete_build_records_the_documented_keys(stage5_reviewed_2638):
    inputs = _inputs(stage5_reviewed_2638)
    assert set(inputs["data"]) == {
        "analysis_epoch",
        "probe_freq_mhz",
        "sideband",
        "spacing_s",
        "acquisition_segments",
    }
    assert inputs["data"]["sideband"] in ("upper", "lower")
    assert inputs["data"]["spacing_s"] > 0.0
    assert set(inputs["ft"]) == {
        "analysis_epoch",
        "start_us",
        "end_us",
        "units_power",
        "trim_min_mhz",
        "trim_max_mhz",
    }
    assert (inputs["ft"]["trim_min_mhz"], inputs["ft"]["trim_max_mhz"]) == (
        26500.0,
        40000.0,
    )
    assert set(inputs["noise"]) == {"analysis_epoch"} | {
        f.name for f in dataclasses.fields(NoiseSettings)
    }
    assert set(inputs["peaks"]) >= {"promotion", "savgol", "consumed"}
    assert set(inputs["peaks"]["consumed"]) == {
        "tau_basis_us",
        "gap_shape",
        "tau_basis_source",
    }
    assert set(inputs["windows"]) >= {"coherence", "clustering", "contributor"}
    assert set(inputs["fit"]) >= {"shape", "tau", "peak_survival", "consumed"}
    assert set(inputs["fit"]["consumed"]) == {
        "tau_calibration_source",
        "tau_maj_us",
        "sigma_tau_us",
        "band_majorities",
        "timebase_epsilon",
        "timebase_sigma_epsilon",
        "peak_survival_snr_floor",
        "stft_spur_nominees",
    }
    assert isinstance(inputs["fit"]["shape"], ShapeSpec)
    assert set(inputs["review"]) == {
        "analysis_epoch",
        "sigma_floor_khz",
        "calibration_clocks",
    }
    # Results are not inputs: no fitted frequency, line list or review log.
    flat = _flat(inputs)
    assert not any("fitted" in k or "review_log" in k for k in flat)


def test_the_two_decay_calibrations_and_the_timebase_are_read_from_their_records(
    tau_timebase_calibrated_2638,
):
    # Spec: tau and tau_g come from each twin's own record; the timebase hashes
    # its knobs and clock declaration, never its result (epsilon).
    # Mutation: reading the shared recipe for both twins (tau would then carry
    # "gaussian", or tau_g would lack it); hashing epsilon.
    inputs = _inputs(tau_timebase_calibrated_2638)
    assert set(inputs["tau"]) >= {"stft", "polish", "aggregation", "band"}
    assert "gaussian" not in inputs["tau"]
    assert set(inputs["tau_g"]) >= {"stft", "aggregation", "band", "gaussian"}
    assert set(inputs["timebase"]) == {
        "analysis_epoch",
        "kappa_sys",
        "snr_min",
        "start_us",
        "end_us",
        "clock_sources",
    }
    clocks = inputs["timebase"]["clock_sources"]
    assert [c.label for c in clocks] == ["upconv", "downconv", "awg"]
    assert [c.freq_mhz for c in clocks] == [5760.0, 5120.0, 16000.0]
    assert all(c.locked is True for c in clocks)
    for stage in ("fit", "peaks", "windows", "review"):
        assert inputs[stage] is None and inputs[f"{stage}_absent"] == "not_run"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_two_fresh_imports_give_the_same_digest(exp_2638_data_path, tmp_path):
    # Spec: the same FID and the same recorded inputs give the same digest.
    # Mutation: hashing a timestamp, a path, a random id or the package/host
    # environment (the two imports are written at different times, to
    # different paths).
    a, b = str(tmp_path / "a.ftmw"), str(tmp_path / "b.ftmw")
    ftmw.import_data(a, source=exp_2638_data_path)
    ftmw.import_data(b, source=exp_2638_data_path)
    assert _digest(a) == _digest(b)
    inputs = _inputs(a)
    assert isinstance(inputs["data"], dict)
    for stage in Stage:
        if stage is not Stage.DATA:
            assert inputs[stage.value] is None
            assert inputs[f"{stage.value}_absent"] == "not_run"


def test_an_independently_rerun_fit_gives_the_same_digest(
    baseline_2638_stage4_small, baseline_2638_stage5_small, tmp_path
):
    # Mutation: any fit output (not input) leaking into the digest, or a
    # nondeterministic value (timing, ordering of a set) in an input block.
    path = _copy(baseline_2638_stage4_small, tmp_path, "refit.ftmw")
    ftmw.fit_peaks(path)
    assert _digest(path) == _digest(baseline_2638_stage5_small)


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------
def test_completing_a_stage_changes_the_digest(
    baseline_2638_stage4_small, baseline_2638_stage5_small, stage5_reviewed_2638
):
    # Spec: a stage that has not run contributes not_run, "so the digest
    # changes as stages complete". Mutation: omitting an unrun stage's key.
    d4 = _digest(baseline_2638_stage4_small)
    d5 = _digest(baseline_2638_stage5_small)
    d6 = _digest(stage5_reviewed_2638)
    assert len({d4, d5, d6}) == 3


def test_changing_a_stage5_knob_changes_only_that_key(
    baseline_2638_stage4_small, baseline_2638_stage5_small, tmp_path
):
    # Mutation: leaving a Stage 5 settings group (here ``conservative``) out of
    # the canonical object, or hashing the pre-run value instead of the
    # persisted one.
    path = _copy(baseline_2638_stage4_small, tmp_path, "knob.ftmw")
    ftmw.settings_set(path, "stage5.conservative.significance", 0.04)
    ftmw.fit_peaks(path)
    base = _inputs(baseline_2638_stage5_small)
    changed = _inputs(path)
    assert changed["fit"]["conservative"]["significance"] == 0.04
    assert _differing(base, changed) == {"fit.conservative.significance"}
    assert _digest(path) != _digest(baseline_2638_stage5_small)


def test_changing_a_noise_knob_changes_only_that_key(baseline_2638_stage2, tmp_path):
    # Mutation: leaving Stage 2's settings out of the digest.
    path = _copy(baseline_2638_stage2, tmp_path, "noise.ftmw")
    ftmw.settings_set(path, "stage2.window_mhz", "111")
    ftmw.estimate_noise(path)
    assert _differing(_inputs(baseline_2638_stage2), _inputs(path)) == {
        "noise.window_mhz"
    }
    assert _digest(path) != _digest(baseline_2638_stage2)


def test_changing_the_declared_clocks_changes_the_timebase_digest(
    tau_timebase_calibrated_2638, tmp_path
):
    # Mutation: leaving the clock declaration out of the timebase inputs.
    path = _copy(tau_timebase_calibrated_2638, tmp_path, "clocks.ftmw")
    base = _inputs(path)
    ftmw.calibrate_timebase(
        path,
        clocks=[
            ClockSource(freq_mhz=5760.0, locked=True, label="upconv"),
            ClockSource(freq_mhz=5120.0, locked=True, label="downconv"),
        ],
    )
    diff = _differing(base, _inputs(path))
    assert diff and all(k.startswith("timebase.clock_sources") for k in diff)
    assert _digest(path) != _digest(tau_timebase_calibrated_2638)


def test_rerunning_the_timebase_unchanged_does_not_move_the_digest(
    tau_timebase_calibrated_2638, tmp_path
):
    # The calibration result (epsilon) and the write time are not inputs.
    # Mutation: hashing the result or a timestamp of the record.
    path = _copy(tau_timebase_calibrated_2638, tmp_path, "again.ftmw")
    before = _digest(path)
    ftmw.calibrate_timebase(
        path,
        clocks=[
            ClockSource(freq_mhz=5760.0, locked=True, label="upconv"),
            ClockSource(freq_mhz=5120.0, locked=True, label="downconv"),
            ClockSource(freq_mhz=16000.0, locked=True, label="awg"),
        ],
    )
    assert _digest(path) == before


def test_the_analysis_epoch_is_covered(fit_file):
    # Spec: the analysis epoch each persisted stage was produced under is
    # covered. Mutation: dropping ``analysis_epoch`` from a stage's object.
    before_inputs = _inputs(fit_file)
    before = _digest(fit_file)
    ft_key = key_for_stage(Stage.FT)

    def bump(blob: Dict[str, Any]) -> None:
        blob[ft_key]["analysis_epoch"] += 1

    _edit_environments(fit_file, bump)
    assert _differing(before_inputs, _inputs(fit_file)) == {"ft.analysis_epoch"}
    assert _digest(fit_file) != before


def test_the_declared_floor_is_an_input_once_review_is_complete(reviewed_file):
    # Spec: the declared sigma_floor_khz is covered. Mutation: leaving the
    # floor out, or reading it from a stale copy.
    base = _inputs(reviewed_file)
    before = _digest(reviewed_file)
    ftmw.set_sigma_floor(reviewed_file, 1.5)
    assert _inputs(reviewed_file)["review"]["sigma_floor_khz"] == 1.5
    assert _differing(base, _inputs(reviewed_file)) == {"review.sigma_floor_khz"}
    assert _digest(reviewed_file) != before


def test_the_floor_is_not_an_input_before_review_is_complete(fit_file):
    # Spec: an incomplete stage is not_run whatever records remain on disk.
    # set_sigma_floor writes /frequency_calibration but does not complete
    # review. Mutation: reading review's record without the tracker check.
    before = _digest(fit_file)
    ftmw.set_sigma_floor(fit_file, 2.5)
    assert _digest(fit_file) == before
    inputs = _inputs(fit_file)
    assert inputs["review"] is None and inputs["review_absent"] == "not_run"
    ftmw.review_run(fit_file)
    assert _inputs(fit_file)["review"]["sigma_floor_khz"] == 2.5
    assert _digest(fit_file) != before


def test_writes_that_are_not_inputs_leave_the_digest_alone(reviewed_file):
    # Excluded by the spec: curation decisions, write timestamps, audit
    # attributes. Mutation: hashing the review log, a stamp time or the
    # final-products table.
    before = _digest(reviewed_file)
    fit = ftmw.load_fit(reviewed_file)
    ftmw.review_accept(reviewed_file, int(fit.window_fits[0].window_id))
    assert _digest(reviewed_file) == before
    ftmw.review_run(reviewed_file)  # rewrites the layer and its timestamps
    assert _digest(reviewed_file) == before


# ---------------------------------------------------------------------------
# Review default
# ---------------------------------------------------------------------------
def test_a_completed_review_without_a_record_hashes_as_a_zero_floor(reviewed_file):
    # Spec: no /frequency_calibration on a completed review is 0.0, the same as
    # an explicit 0.0. Mutation: raising for the absent record, hashing null,
    # or spelling the default differently from the explicit value.
    with h5py.File(reviewed_file, "a") as f:
        if "frequency_calibration" in f:
            del f["frequency_calibration"]
    absent = _inputs(reviewed_file)["review"]
    assert absent["sigma_floor_khz"] == 0.0
    digest_absent = _digest(reviewed_file)
    ftmw.set_sigma_floor(reviewed_file, 0.0)
    with h5py.File(reviewed_file, "r") as f:
        assert "frequency_calibration" in f
    assert _inputs(reviewed_file)["review"] == absent
    assert _digest(reviewed_file) == digest_absent


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_a_pre_provenance_record_lists_every_key_it_supplies(fit_file):
    # Spec: a record with no field-set version is pre-provenance and raises,
    # listing the inputs. Mutation: trusting an unversioned record (hashing
    # None-filled fields), or listing only some of its keys.
    _strip_version(fit_file, STAGE2_NOISE_SETTINGS_PATH)
    with pytest.raises(IncompleteProvenanceError) as exc:
        ftmw.analysis_fingerprint(fit_file)
    assert exc.value.code == "incomplete_provenance"
    assert set(exc.value.missing) == _noise_missing()
    assert exc.value.newer == []
    for key in exc.value.missing:
        assert key in str(exc.value)
    assert "Re-run" in str(exc.value) or "re-run" in str(exc.value)
    assert "upgrade" not in str(exc.value)


def test_a_newer_field_set_version_is_refused(fit_file):
    # Spec: a record at a newer version than this build knows raises.
    # Mutation: treating only "older" as untrusted.
    _bump_version(fit_file, STAGE2_NOISE_SETTINGS_PATH, STAGE2_NOISE_FIELD_SET_VERSION)
    with pytest.raises(IncompleteProvenanceError) as exc:
        ftmw.analysis_fingerprint(fit_file)
    # a newer record is named in ``newer`` with an upgrade remedy, not a re-run
    assert set(exc.value.newer) == _noise_missing()
    assert exc.value.missing == []
    assert "upgrade ftmwpipeline" in str(exc.value)
    assert "re-run" not in str(exc.value)
    assert exc.value.to_dict()["newer"] == exc.value.newer


def test_a_fit_record_from_a_newer_build_names_its_groups_and_consumed_block(
    fit_file,
):
    # The Stage 5 record and its consumed block share one version, so both are
    # untrusted. Mutation: forgetting the consumed block, or listing another
    # stage.
    with h5py.File(fit_file, "r") as f:
        version = int(f[STAGE_FIT_PATH].attrs[FIELD_SET_VERSION_ATTR])
    _bump_version(fit_file, STAGE_FIT_PATH, version)
    with pytest.raises(IncompleteProvenanceError) as exc:
        ftmw.analysis_fingerprint(fit_file)
    newer = set(exc.value.newer)
    assert {"fit.shape", "fit.tau", "fit.consumed"} <= newer
    assert all(k.startswith("fit.") for k in newer)
    assert exc.value.missing == []


def test_a_missing_epoch_stamp_is_refused(fit_file):
    # Spec: a complete stage with no analysis-epoch stamp raises, listing
    # "<stage>.analysis_epoch". Mutation: hashing a null epoch.
    noise_key = key_for_stage(Stage.NOISE)
    _edit_environments(fit_file, lambda blob: blob.pop(noise_key))
    with pytest.raises(IncompleteProvenanceError) as exc:
        ftmw.analysis_fingerprint(fit_file)
    assert exc.value.missing == ["noise.analysis_epoch"]


def test_every_gap_across_stages_is_listed_at_once(fit_file):
    # Spec: the error lists every such input. Mutation: raising at the first
    # gap found.
    _strip_version(fit_file, STAGE2_NOISE_SETTINGS_PATH)
    ft_key = key_for_stage(Stage.FT)
    _edit_environments(fit_file, lambda blob: blob.pop(ft_key))
    with pytest.raises(IncompleteProvenanceError) as exc:
        ftmw.analysis_fingerprint(fit_file)
    assert set(exc.value.missing) == _noise_missing() | {"ft.analysis_epoch"}


def test_a_pre_provenance_timebase_record_is_refused(
    tau_timebase_calibrated_2638, tmp_path
):
    # Mutation: timebase left out of the version check.
    path = _copy(tau_timebase_calibrated_2638, tmp_path, "tb.ftmw")
    _strip_version(path, TIMEBASE_PATH)
    with pytest.raises(IncompleteProvenanceError) as exc:
        ftmw.analysis_fingerprint(path)
    assert set(exc.value.missing) == {
        "timebase.kappa_sys",
        "timebase.snr_min",
        "timebase.start_us",
        "timebase.end_us",
        "timebase.clock_sources",
    }


def test_a_current_timebase_record_without_clock_sources_is_refused(
    tau_timebase_calibrated_2638, tmp_path
):
    # Spec's own example: a version-2 timebase record without clock_sources.
    # Mutation: hashing the missing declaration as null.
    path = _copy(tau_timebase_calibrated_2638, tmp_path, "tb2.ftmw")
    with h5py.File(path, "a") as f:
        f[TIMEBASE_PATH].attrs["clock_sources"] = _NONE_SENTINEL
    with pytest.raises(IncompleteProvenanceError) as exc:
        ftmw.analysis_fingerprint(path)
    assert exc.value.missing == ["timebase.clock_sources"]


def test_a_refused_read_does_not_write(fit_file):
    _strip_version(fit_file, STAGE2_NOISE_SETTINGS_PATH)
    before = _md5(fit_file)
    with pytest.raises(IncompleteProvenanceError):
        ftmw.analysis_fingerprint(fit_file)
    assert _md5(fit_file) == before


# ---------------------------------------------------------------------------
# Incomplete stages read as not run
# ---------------------------------------------------------------------------
def test_an_incomplete_stage_is_not_run_even_with_a_sparse_record_left(fit_file):
    # Spec: a stage is read only when complete; a sparse record written by
    # ``settings set`` before a re-run is not read. The sparse record has no
    # field-set version, so reading it would raise.
    # Mutation: reading records without consulting the stage tracker.
    ftmw.settings_set(fit_file, "stage2.window_mhz", "111")
    left = load_noise_settings_from_h5(fit_file)
    assert left is not None and left.window_mhz == 111.0  # the record remains
    inputs = _inputs(fit_file)  # does not raise
    assert inputs["noise"] is None and inputs["noise_absent"] == "not_run"
    assert inputs["fit"] is None and inputs["fit_absent"] == "not_run"
    assert isinstance(inputs["data"], dict) and isinstance(inputs["ft"], dict)
    assert _HEX64.match(_digest(fit_file))


# ---------------------------------------------------------------------------
# Coverage found by review: inputs read outside the settings records
# ---------------------------------------------------------------------------
def test_the_effective_calibration_clocks_are_an_input(
    reviewed_file, monkeypatch: pytest.MonkeyPatch
):
    # Spec: review.calibration_clocks is the declaration Stage 6 derives the
    # calibration state from, read through its own resolver at read time.
    # Mutation: dropping the key, or hashing a declaration other than the one
    # the products use (a 'clocks clear' after review moved the products but
    # not the digest).
    from ftmwpipeline._internal import stage6_impl

    review = _inputs(reviewed_file)["review"]
    assert review["calibration_clocks"] == list(
        stage6_impl._resolve_calibration_clocks(reviewed_file)
    )
    before = _digest(reviewed_file)
    monkeypatch.setattr(
        stage6_impl,
        "_resolve_calibration_clocks",
        lambda path: (ClockSource(freq_mhz=6250.0, locked=False),),
    )
    assert _digest(reviewed_file) != before


def _write_segments(path: str, pre_record_scale: float) -> None:
    import numpy as np

    with h5py.File(path, "a") as f:
        data = f[key_for_stage(Stage.DATA)]
        if "acquisition_segments" in data:
            del data["acquisition_segments"]
        seg = data.create_group("acquisition_segments")
        seg.attrs["pre_record_us"] = 1.0
        seg.attrs["frame_period_us"] = 20.0
        seg.attrs["n_frames"] = 1
        seg.attrs["frame_selection"] = -1
        seg.attrs["sample_dt"] = 1e-11
        seg.create_dataset("pre_record", data=pre_record_scale * np.ones(16))
        seg.create_dataset("tail", data=np.zeros(8))


def test_acquisition_segments_are_an_input(reviewed_file):
    # Spec: data.acquisition_segments (the Stage 5 spur gate reads the
    # pre-record). Mutation: hashing only probe/sideband/spacing, so two files
    # whose spur gates saw different pre-records share a digest.
    assert _inputs(reviewed_file)["data"]["acquisition_segments"] is None
    plain = _digest(reviewed_file)
    _write_segments(reviewed_file, 1.0)
    segments = _inputs(reviewed_file)["data"]["acquisition_segments"]
    assert _HEX64.match(segments["pre_record"])
    with_segments = _digest(reviewed_file)
    _write_segments(reviewed_file, 2.0)
    assert len({plain, with_segments, _digest(reviewed_file)}) == 3


def test_a_consumed_block_missing_a_field_is_refused(fit_file):
    # Mutation: a codec KeyError escaping raw (losing every other gap) instead
    # of incomplete_provenance naming peaks.consumed.
    with h5py.File(fit_file, "a") as f:
        del f["processing_parameters/stage3_peaks/consumed"].attrs["gap_shape"]
    with pytest.raises(IncompleteProvenanceError) as exc:
        ftmw.analysis_fingerprint(fit_file)
    assert "peaks.consumed" in exc.value.missing


def test_a_current_floor_record_without_its_field_is_refused(reviewed_file):
    # Spec: only an absent record defaults to 0.0. Mutation: the codec's
    # .get(..., 0.0) silently filling a current record's missing field.
    ftmw.set_sigma_floor(reviewed_file, 1.5)
    with h5py.File(reviewed_file, "a") as f:
        del f["frequency_calibration"].attrs["sigma_floor_khz"]
    with pytest.raises(IncompleteProvenanceError) as exc:
        ftmw.analysis_fingerprint(reviewed_file)
    assert "review.sigma_floor_khz" in exc.value.missing
