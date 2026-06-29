"""Cross-interface consistency test for keysight-mat import.

Verifies that importing a synthetic Keysight MAT file via the functional API,
Pipeline class, and CLI produces identical Stage 0 data (same FID array,
same probe/sideband metadata, same acquisition segments).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.io.fid_serialization import load_acquisition_segments_from_hdf5


def _make_synthetic_mat(tmp_path: Path) -> tuple:
    """Write a small synthetic Keysight MAT file with known content.

    Layout: 2 µs pre + 3 × 4 µs frames + 1 µs tail at 10 GSa/s.
    Frame k carries DC offset (k+1)*100 LSB, pre and tail are zero.
    """
    xinc = 1e-10  # 10 GSa/s
    pre_us, frame_us, tail_us = 2.0, 4.0, 1.0
    n_frames = 3
    pre_s = int(round(pre_us * 1e-6 / xinc))
    frame_s = int(round(frame_us * 1e-6 / xinc))
    tail_s = int(round(tail_us * 1e-6 / xinc))
    total = pre_s + n_frames * frame_s + tail_s

    record = np.zeros(total, dtype=np.int16)
    for k in range(n_frames):
        record[pre_s + k * frame_s : pre_s + (k + 1) * frame_s] = (k + 1) * 100

    mat_path = tmp_path / "synthetic.mat"
    with h5py.File(mat_path, "w") as f:
        ch = f.create_group("Channel_1")
        ch.create_dataset("Data", data=record.reshape(1, -1))
        ch.create_dataset("XInc", data=np.array([[xinc]]))
        ch.create_dataset("XOrg", data=np.array([[0.0]]))
        ch.create_dataset("YInc", data=np.array([[1.0]]))
        ch.create_dataset("YOrg", data=np.array([[0.0]]))
        frm = f.create_group("Frame")
        frm.create_dataset(
            "Model",
            data=np.array([ord(c) for c in "TestScope"], dtype=np.uint16).reshape(
                -1, 1
            ),
        )
        frm.create_dataset(
            "Serial",
            data=np.array([ord(c) for c in "SN000"], dtype=np.uint16).reshape(-1, 1),
        )

    layout = dict(
        pre_record_us=pre_us,
        frame_period_us=frame_us,
        n_frames=n_frames,
    )
    expected_avg = np.full(frame_s, (1 + 2 + 3) / 3 * 100.0)
    return mat_path, layout, expected_avg, frame_s, pre_s, tail_s


@pytest.mark.integration
class TestKeysightMatCrossInterface:
    """Identical Stage-0 data from all three interfaces on a synthetic MAT file."""

    @pytest.fixture(scope="class")
    def trio(self, tmp_path_factory):
        """Build three pipeline files — one per interface — and return their paths."""
        tmp = tmp_path_factory.mktemp("keysight_mat_trio")
        mat_path, layout, expected_avg, frame_s, pre_s, tail_s = _make_synthetic_mat(
            tmp
        )

        # Functional API
        fp_api = tmp / "via_api.ftmw"
        ftmw.import_data(fp_api, source=str(mat_path), **layout)

        # Pipeline class
        fp_pipeline = tmp / "via_pipeline.ftmw"
        Pipeline.create(fp_pipeline, source=str(mat_path), **layout)

        # CLI
        fp_cli = tmp / "via_cli.ftmw"
        cmd = [
            sys.executable,
            "-m",
            "ftmwpipeline",
            "data",
            "import",
            str(fp_cli),
            str(mat_path),
            "--pre-record-us",
            str(layout["pre_record_us"]),
            "--frame-period-us",
            str(layout["frame_period_us"]),
            "--n-frames",
            str(layout["n_frames"]),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert (
            result.returncode == 0
        ), f"CLI import failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        return {
            "api": fp_api,
            "pipeline": fp_pipeline,
            "cli": fp_cli,
            "mat_path": mat_path,
            "layout": layout,
            "expected_avg": expected_avg,
            "frame_s": frame_s,
            "pre_s": pre_s,
            "tail_s": tail_s,
        }

    def test_identical_fid_data(self, trio):
        fid_api = ftmw.load_fid(trio["api"])
        fid_pipeline = ftmw.load_fid(trio["pipeline"])
        fid_cli = ftmw.load_fid(trio["cli"])

        np.testing.assert_array_equal(fid_api.data, fid_pipeline.data)
        np.testing.assert_array_equal(fid_api.data, fid_cli.data)

    def test_fid_matches_expected_average(self, trio):
        fid = ftmw.load_fid(trio["api"])
        np.testing.assert_allclose(fid.data, trio["expected_avg"], rtol=1e-12)

    def test_fid_metadata_consistent(self, trio):
        fid_api = ftmw.load_fid(trio["api"])
        fid_pipeline = ftmw.load_fid(trio["pipeline"])
        fid_cli = ftmw.load_fid(trio["cli"])

        for fid in (fid_api, fid_pipeline, fid_cli):
            assert fid.probe_freq_mhz == 0.0
            assert fid.sideband.value == "upper"
            assert fid.n_points == trio["frame_s"]

    def test_segments_present_and_correct(self, trio):
        for key in ("api", "pipeline", "cli"):
            with h5py.File(trio[key], "r") as h5f:
                segs = load_acquisition_segments_from_hdf5(h5f["stage0_fid_data"])
            assert segs is not None, f"{key} file has no segments"
            assert segs.n_frames == trio["layout"]["n_frames"]
            assert len(segs.pre_record) == trio["pre_s"]
            assert len(segs.tail) == trio["tail_s"]
