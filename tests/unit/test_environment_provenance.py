"""
Analysis-environment provenance: what produced each stage, and what that gates.

The package claims a ``.ftmw`` is reproducible "given the file and a compatible
package version". These tests pin down the two things that claim needs: a record
of which environment produced each artifact, and a declared, machine-checkable
notion of compatibility.

The policy under test:

- **Record per stage.** Stages are written at different times, so a file whose
  Stage 5 fit and Stage 6 curation came from different code is representable and
  detectable. A single file-level stamp could not express that.
- **Reads are never gated.** Loading, inspecting, and reporting always work.
- **Extending forward warns.** Running a new stage into a file written by
  another environment is legitimate; it logs that the file now mixes them.
- **Splicing is refused.** A Stage 6 edit re-fits one window into a fit whose
  other windows came from earlier code; across an ``ANALYSIS_EPOCH`` change that
  would leave two fitting models in one product, so it is blocked until the user
  records an acknowledgement in the file.
- **Only the epoch gates.** Python/numpy/scipy/BLAS are advisory; an unknown
  epoch (a legacy file) is treated as compatible, never as incompatible.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import h5py
import pytest

import ftmwpipeline.api as ftmw
import ftmwpipeline.core.environment as envmod
from ftmwpipeline.core.environment import (
    ANALYSIS_EPOCH,
    EnvironmentRecord,
    capture_environment,
    describe_environment_drift,
    describe_runtime_drift,
    gating_fields_differ,
)
from ftmwpipeline.io.environment_serialization import (
    load_environment_ack,
    load_last_written_with,
    load_stage_environments,
    save_environment_ack,
    save_stage_environment,
)

# ---------------------------------------------------------------------------
# The record itself (pure, no file needed)
# ---------------------------------------------------------------------------


class TestCaptureEnvironment:
    def test_captures_the_running_environment(self):
        rec = capture_environment()
        assert rec.ftmwpipeline
        assert rec.analysis_epoch == ANALYSIS_EPOCH
        assert rec.python
        assert rec.numpy
        assert rec.scipy
        assert rec.h5py

    def test_blas_is_identified(self):
        """threadpoolctl is a hard dependency, so the runtime BLAS is knowable."""
        rec = capture_environment()
        assert rec.blas
        assert rec.blas != "unknown"

    def test_round_trips_through_a_dict(self):
        rec = capture_environment()
        assert EnvironmentRecord.from_dict(rec.to_dict()) == rec

    def test_summary_is_human_readable(self):
        rec = capture_environment()
        summary = rec.summary()
        assert rec.ftmwpipeline in summary
        assert f"epoch {ANALYSIS_EPOCH}" in summary


class TestGatingFields:
    def test_same_epoch_does_not_gate(self):
        a = capture_environment()
        b = EnvironmentRecord.from_dict({**a.to_dict(), "numpy": "0.0.1"})
        assert not gating_fields_differ(a, b), "numpy must be advisory, not gating"

    def test_different_epoch_gates(self):
        a = capture_environment()
        b = EnvironmentRecord.from_dict(
            {**a.to_dict(), "analysis_epoch": ANALYSIS_EPOCH + 1}
        )
        assert gating_fields_differ(a, b)

    def test_unknown_epoch_never_gates(self):
        """A legacy file must stay usable: unknown is not incompatible."""
        a = capture_environment()
        legacy = EnvironmentRecord.from_dict({**a.to_dict(), "analysis_epoch": None})
        assert not gating_fields_differ(a, legacy)
        assert not gating_fields_differ(a, None)


class TestDescribeDrift:
    def test_agreement_reports_nothing(self):
        rec = capture_environment()
        assert describe_environment_drift({"a": rec, "b": rec}) == []

    def test_single_record_reports_nothing(self):
        assert describe_environment_drift({"a": capture_environment()}) == []

    def test_names_the_field_and_the_stages(self):
        a = capture_environment()
        b = EnvironmentRecord.from_dict({**a.to_dict(), "numpy": "1.26.0"})
        lines = describe_environment_drift({"stage3_peaks": a, "stage5_fitting": b})
        assert any("numpy" in line for line in lines)
        assert any("stage5_fitting" in line for line in lines)

    def test_blas_is_not_treated_as_drift(self):
        """Thread count varies by machine; reporting it as drift would cry wolf."""
        a = capture_environment()
        b = EnvironmentRecord.from_dict({**a.to_dict(), "blas": "mkl (4 threads)"})
        assert describe_environment_drift({"x": a, "y": b}) == []


class TestDescribeRuntimeDrift:
    """The other question: was this file produced by the code running now?

    Cross-stage agreement cannot answer it -- a file stamped uniformly by one
    release has no internal drift at all and may still disagree with the
    interpreter about to edit it.
    """

    def test_agreement_reports_nothing(self):
        rec = capture_environment()
        assert describe_runtime_drift({"stage5_fitting": rec}, rec) == []

    def test_uniform_file_from_another_version_is_reported(self):
        cur = capture_environment()
        old = EnvironmentRecord.from_dict({**cur.to_dict(), "ftmwpipeline": "0.0.1"})
        lines = describe_runtime_drift(
            {"stage3_peaks": old, "stage5_fitting": old}, cur
        )
        assert describe_environment_drift({"a": old, "b": old}) == []
        assert any("ftmwpipeline" in line and "0.0.1" in line for line in lines)
        assert any(cur.ftmwpipeline in line for line in lines)

    def test_epoch_difference_is_reported(self):
        cur = capture_environment()
        other = EnvironmentRecord.from_dict(
            {**cur.to_dict(), "analysis_epoch": ANALYSIS_EPOCH + 1}
        )
        lines = describe_runtime_drift({"stage5_fitting": other}, cur)
        assert any(line.startswith("analysis_epoch:") for line in lines)

    def test_an_unstamped_file_states_nothing_to_disagree_with(self):
        assert describe_runtime_drift({}, capture_environment()) == []

    def test_blas_is_not_treated_as_drift(self):
        cur = capture_environment()
        other = EnvironmentRecord.from_dict({**cur.to_dict(), "blas": "mkl (4)"})
        assert describe_runtime_drift({"x": other}, cur) == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "t.h5"
        rec = capture_environment()
        with h5py.File(p, "w") as f:
            save_stage_environment(f, "stage5_fitting", rec)
        with h5py.File(p, "r") as f:
            envs = load_stage_environments(f)
            last = load_last_written_with(f)
        assert envs["stage5_fitting"] == rec
        assert last is not None and last.ftmwpipeline == rec.ftmwpipeline

    def test_stages_accumulate_independently(self, tmp_path):
        """Stamping one stage must not disturb another -- that is what makes a
        mixed-version file representable."""
        p = tmp_path / "t.h5"
        a = capture_environment()
        b = EnvironmentRecord.from_dict({**a.to_dict(), "ftmwpipeline": "9.9.9"})
        with h5py.File(p, "w") as f:
            save_stage_environment(f, "stage3_peaks", a)
            save_stage_environment(f, "stage5_fitting", b)
        with h5py.File(p, "r") as f:
            envs = load_stage_environments(f)
        assert envs["stage3_peaks"].ftmwpipeline == a.ftmwpipeline
        assert envs["stage5_fitting"].ftmwpipeline == "9.9.9"

    def test_rerunning_a_stage_replaces_its_entry(self, tmp_path):
        p = tmp_path / "t.h5"
        a = capture_environment()
        b = EnvironmentRecord.from_dict({**a.to_dict(), "ftmwpipeline": "9.9.9"})
        with h5py.File(p, "w") as f:
            save_stage_environment(f, "stage3_peaks", a)
            save_stage_environment(f, "stage3_peaks", b)
        with h5py.File(p, "r") as f:
            assert load_stage_environments(f)["stage3_peaks"].ftmwpipeline == "9.9.9"

    def test_absent_record_reads_as_empty(self, tmp_path):
        p = tmp_path / "t.h5"
        with h5py.File(p, "w") as f:
            f.create_group("pipeline_stages")
        with h5py.File(p, "r") as f:
            assert load_stage_environments(f) == {}
            assert load_last_written_with(f) is None
            assert load_environment_ack(f) is None

    def test_malformed_record_is_ignored_not_fatal(self, tmp_path):
        p = tmp_path / "t.h5"
        with h5py.File(p, "w") as f:
            f.require_group("pipeline_stages").attrs["stage_environments"] = "{oops"
        with h5py.File(p, "r") as f:
            assert load_stage_environments(f) == {}

    def test_ack_round_trip(self, tmp_path):
        p = tmp_path / "t.h5"
        rec = capture_environment()
        with h5py.File(p, "w") as f:
            save_environment_ack(f, rec, reason="known change")
        with h5py.File(p, "r") as f:
            ack = load_environment_ack(f)
        assert ack is not None
        assert ack["reason"] == "known change"
        assert (
            EnvironmentRecord.from_dict(ack["acknowledged_environment"]).analysis_epoch
            == ANALYSIS_EPOCH
        )


# ---------------------------------------------------------------------------
# Stamping through the real pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _stamped_file(tmp_path_factory) -> Path:
    data = Path("examples/blackchirp_data/2638")
    if not data.exists():
        pytest.skip("Experiment 2638 data not available")
    p = tmp_path_factory.mktemp("env") / "stamped.ftmw"
    ftmw.import_data(p, source=str(data))
    ftmw.compute_ft(p, trim=(26500, 27000))
    ftmw.estimate_noise(p)
    ftmw.detect_peaks(p)
    return p


@pytest.fixture
def stamped_file(_stamped_file, tmp_path) -> Path:
    dst = tmp_path / "stamped.ftmw"
    shutil.copy(_stamped_file, dst)
    return dst


class TestPipelineStamping:
    def test_import_stamps_stage0(self, stamped_file):
        with h5py.File(stamped_file, "r") as f:
            envs = load_stage_environments(f)
        assert "stage0_fid_data" in envs
        assert envs["stage0_fid_data"].analysis_epoch == ANALYSIS_EPOCH

    def test_every_persisted_stage_is_stamped(self, stamped_file):
        with h5py.File(stamped_file, "r") as f:
            envs = load_stage_environments(f)
        assert {"stage0_fid_data", "stage2_noise_result", "stage3_peaks"} <= set(envs)

    def test_stage1_is_not_stamped(self, stamped_file):
        """Stage 1 persists no artifact (the FT is recomputed on demand), so it
        is always the current environment and has nothing to stamp."""
        with h5py.File(stamped_file, "r") as f:
            envs = load_stage_environments(f)
        assert "stage1_complex_ft" not in envs

    def test_a_freshly_built_file_is_uniform(self, stamped_file):
        with h5py.File(stamped_file, "r") as f:
            envs = load_stage_environments(f)
        assert describe_environment_drift(envs) == []

    def test_info_reports_the_environment(self, stamped_file):
        info = ftmw.get_pipeline_info(str(stamped_file))
        assert info["stage_environments"]
        assert info["last_written_with"] is not None
        assert info["environment_drift"] == []
        assert info["environment_acknowledged"] is False


class TestMixedFileDetection:
    @staticmethod
    def _forge(path: Path, stage: str, **overrides) -> None:
        with h5py.File(path, "a") as f:
            g = f["pipeline_stages"]
            blob = json.loads(str(g.attrs["stage_environments"]))
            blob[stage] = {**blob[stage], **overrides}
            g.attrs["stage_environments"] = json.dumps(blob, sort_keys=True)

    def test_drift_is_detected_and_warned_about(self, stamped_file):
        self._forge(stamped_file, "stage3_peaks", ftmwpipeline="9.9.9")
        info = ftmw.get_pipeline_info(str(stamped_file))
        assert info["environment_drift"]
        assert any("different analysis environments" in w for w in info["warnings"])

    def test_a_mixed_file_is_still_valid(self, stamped_file):
        """Mixing environments is a caveat, not corruption."""
        self._forge(stamped_file, "stage3_peaks", ftmwpipeline="9.9.9")
        assert ftmw.validate_pipeline(str(stamped_file))["valid"] is True

    def test_reads_are_never_gated(self, stamped_file):
        self._forge(stamped_file, "stage3_peaks", analysis_epoch=ANALYSIS_EPOCH + 5)
        assert ftmw.load_peaks(str(stamped_file)) is not None
        assert ftmw.get_pipeline_info(str(stamped_file))["valid"] is True


class TestRuntimeMismatchDetection:
    """A file stamped uniformly by another version has no *internal* drift --
    the mismatch that matters is against the interpreter holding it."""

    @staticmethod
    def _forge_all(path: Path, **overrides) -> None:
        with h5py.File(path, "a") as f:
            g = f["pipeline_stages"]
            blob = json.loads(str(g.attrs["stage_environments"]))
            for stage in blob:
                blob[stage] = {**blob[stage], **overrides}
            g.attrs["stage_environments"] = json.dumps(blob, sort_keys=True)

    def test_info_reports_the_running_environment(self, stamped_file):
        info = ftmw.get_pipeline_info(str(stamped_file))
        assert info["current_environment"]["ftmwpipeline"]
        assert info["runtime_environment_drift"] == []

    def test_a_uniform_file_from_another_version_is_surfaced(self, stamped_file):
        self._forge_all(stamped_file, ftmwpipeline="0.0.1")
        info = ftmw.get_pipeline_info(str(stamped_file))
        assert info["environment_drift"] == [], "the file's stages still agree"
        assert any("ftmwpipeline" in line for line in info["runtime_environment_drift"])

    def test_an_epoch_mismatch_against_the_running_code_warns(self, stamped_file):
        self._forge_all(stamped_file, analysis_epoch=ANALYSIS_EPOCH + 5)
        info = ftmw.get_pipeline_info(str(stamped_file))
        assert any(
            line.startswith("analysis_epoch:")
            for line in info["runtime_environment_drift"]
        )
        assert any("different analysis epoch" in w for w in info["warnings"])
        assert info["valid"] is True, "a mismatch is a caveat, not corruption"


class TestLegacyRerunWarning:
    """The one case no gate can speak to: re-running a stage over a result that
    predates environment recording. Unknown epoch reads as compatible, so an
    arbitrary version gap -- and any numerical change in it -- applies in
    silence unless the re-run itself says so."""

    @staticmethod
    def _strip_stamps(path: Path) -> None:
        with h5py.File(path, "a") as f:
            g = f["pipeline_stages"]
            for key in ("stage_environments", "last_written_with"):
                if key in g.attrs:
                    del g.attrs[key]

    def test_rerun_of_an_unstamped_stage_warns(self, stamped_file, caplog):
        self._strip_stamps(stamped_file)
        with caplog.at_level(logging.WARNING):
            ftmw.estimate_noise(str(stamped_file))
        assert any(
            "reproducibility against the original run cannot be verified"
            in r.getMessage()
            for r in caplog.records
        )

    def test_a_stamped_rerun_is_silent(self, stamped_file, caplog):
        with caplog.at_level(logging.WARNING):
            ftmw.estimate_noise(str(stamped_file))
        assert not any("cannot be verified" in r.getMessage() for r in caplog.records)

    def test_completing_a_new_stage_on_a_legacy_file_is_silent(
        self, stamped_file, caplog
    ):
        """Only a *re-run* is unverifiable; a stage running for the first time
        has no earlier result to disagree with."""
        self._strip_stamps(stamped_file)
        with h5py.File(stamped_file, "a") as f:
            stages = json.loads(str(f["pipeline_stages"].attrs["completed_stages"]))
            f["pipeline_stages"].attrs["completed_stages"] = json.dumps(
                [s for s in stages if s != "stage3_peaks"]
            )
        with caplog.at_level(logging.WARNING):
            ftmw.detect_peaks(str(stamped_file))
        assert not any("cannot be verified" in r.getMessage() for r in caplog.records)
