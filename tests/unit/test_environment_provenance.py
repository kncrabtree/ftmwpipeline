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
