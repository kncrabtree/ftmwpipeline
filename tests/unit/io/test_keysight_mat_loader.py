"""Unit tests for the Keysight MATLAB v7.3 scope-record loader.

Tests cover:
- Format detection (keysight-mat wins; generic hdf5 does NOT claim the file)
- Loader slicing: coherent average and single-frame selection
- keep_frames round-trip
- Pre-record and tail contents (exact)
- Re-import no-op (same params)
- Different frame param refuses without force, succeeds with force
- Old-file backward compat (CSV pipeline file loads cleanly, no segments)
- Acquisition segment serialization round-trip through a .ftmw file
- Interleave-offset cleanup: estimation, subtraction, sequential factors,
  comb removal, off-by-default, pattern round-trip, clock_sources injection
"""

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

from ftmwpipeline.core.data_structures import FID, Sideband
from ftmwpipeline.file_manager import (
    PipelineExistsError,
    SourceMetadata,
    create_pipeline_file,
)
from ftmwpipeline.io.acquisition_layout import (
    AcquisitionLayout,
    SlicedRecord,
    apply_interleave_cleanup,
    estimate_interleave_pattern,
    slice_record,
    subtract_interleave_pattern,
)
from ftmwpipeline.io.data_loaders import detect_format
from ftmwpipeline.io.data_loaders.keysight_mat import KeysightMatLoader
from ftmwpipeline.io.fid_serialization import (
    load_acquisition_segments_from_hdf5,
    load_fid_from_hdf5,
    save_fid_to_hdf5,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mat_file(
    tmp_path: Path,
    *,
    n_samples: int = 100_000,
    xinc: float = 7.8125e-12,
    yinc: float = 1.0,
    yorg: float = 0.0,
    channel: str = "Channel_1",
    model: str = "TestScope",
    serial: str = "SN12345",
    data: Optional[np.ndarray] = None,
    filename: str = "test.mat",
) -> Path:
    """Write a minimal Keysight MATLAB v7.3 .mat file using h5py."""
    if data is None:
        rng = np.random.default_rng(42)
        data = rng.integers(-1000, 1000, size=n_samples, dtype=np.int16)

    mat_path = tmp_path / filename
    with h5py.File(mat_path, "w") as f:
        ch = f.create_group(channel)
        ch.create_dataset("Data", data=data.reshape(1, -1))
        ch.create_dataset("XInc", data=np.array([[xinc]]))
        ch.create_dataset("XOrg", data=np.array([[0.0]]))
        ch.create_dataset("YInc", data=np.array([[yinc]]))
        ch.create_dataset("YOrg", data=np.array([[yorg]]))

        frame_grp = f.create_group("Frame")
        frame_grp.create_dataset(
            "Model",
            data=np.array([ord(c) for c in model], dtype=np.uint16).reshape(-1, 1),
        )
        frame_grp.create_dataset(
            "Serial",
            data=np.array([ord(c) for c in serial], dtype=np.uint16).reshape(-1, 1),
        )

    return mat_path


def _make_segmented_mat(
    tmp_path: Path, *, n_frames: int = 3, yinc: float = 1.0
) -> tuple:
    """Create a synthetic segmented record with a known per-frame tone.

    Layout: 2 µs pre-record + n_frames × 4 µs + 1 µs tail at 10 GSa/s.
    Frame k has a DC offset of (k+1)*100 LSB to make averaging predictable.
    """
    xinc = 1e-10  # 10 GSa/s
    pre_us, frame_us, tail_us = 2.0, 4.0, 1.0
    pre_s = int(round(pre_us * 1e-6 / xinc))
    frame_s = int(round(frame_us * 1e-6 / xinc))
    tail_s = int(round(tail_us * 1e-6 / xinc))
    total = pre_s + n_frames * frame_s + tail_s

    # Build record: pre=0, each frame has offset (k+1)*100, tail=0
    record = np.zeros(total, dtype=np.int16)
    for k in range(n_frames):
        start = pre_s + k * frame_s
        record[start : start + frame_s] = (k + 1) * 100

    mat_path = _make_mat_file(
        tmp_path,
        n_samples=total,
        xinc=xinc,
        yinc=yinc,
        data=record,
        filename="segmented.mat",
    )

    layout = dict(
        pre_record_us=pre_us,
        frame_period_us=frame_us,
        n_frames=n_frames,
    )
    expected_avg = np.full(
        frame_s, float(sum(range(1, n_frames + 1))) / n_frames * 100 * yinc
    )
    return mat_path, layout, expected_avg, pre_s, frame_s, tail_s, xinc, record


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestFormatDetection:
    def test_keysight_mat_detected(self, tmp_path):
        mat_path = _make_mat_file(tmp_path)
        fmt = detect_format(mat_path)
        assert fmt == "keysight-mat", f"Expected keysight-mat, got {fmt!r}"

    def test_ftmw_hdf5_loader_does_not_claim_keysight_mat(self, tmp_path):
        """The native ftmw-hdf5 loader must NOT claim a Keysight .mat file."""
        from ftmwpipeline.io.data_loaders.ftmw_hdf5 import FtmwHdf5Loader

        mat_path = _make_mat_file(tmp_path)
        assert not FtmwHdf5Loader().can_load(mat_path)

    def test_non_mat_not_detected(self, tmp_path):
        other = tmp_path / "data.h5"
        with h5py.File(other, "w") as f:
            f.create_group("fid_data")
        fmt = detect_format(other)
        assert fmt != "keysight-mat"

    def test_multi_channel_without_xinc_not_detected(self, tmp_path):
        bad = tmp_path / "bad.mat"
        with h5py.File(bad, "w") as f:
            ch = f.create_group("Channel_1")
            ch.create_dataset("Data", data=np.zeros((1, 10), dtype=np.int16))
            # No XInc — should not be detected as keysight-mat
        loader = KeysightMatLoader()
        assert not loader.can_load(bad)


# ---------------------------------------------------------------------------
# AcquisitionLayout + slice_record
# ---------------------------------------------------------------------------


class TestAcquisitionLayout:
    def test_frozen_dataclass_valid(self):
        layout = AcquisitionLayout(
            pre_record_us=12.5, frame_period_us=20.0, n_frames=19
        )
        assert layout.n_frames == 19

    def test_invalid_n_frames(self):
        with pytest.raises(ValueError, match="n_frames"):
            AcquisitionLayout(pre_record_us=0, frame_period_us=1.0, n_frames=0)

    def test_invalid_frame_index(self):
        with pytest.raises(ValueError, match="frame index"):
            AcquisitionLayout(pre_record_us=0, frame_period_us=1.0, n_frames=3, frame=5)

    def test_slice_average(self):
        xinc = 1e-10
        n_frames = 3
        pre_s, frame_s, tail_s = 200, 400, 100
        record = np.zeros(pre_s + n_frames * frame_s + tail_s)
        for k in range(n_frames):
            record[pre_s + k * frame_s : pre_s + (k + 1) * frame_s] = (k + 1) * 10.0

        layout = AcquisitionLayout(
            pre_record_us=pre_s * xinc * 1e6,
            frame_period_us=frame_s * xinc * 1e6,
            n_frames=n_frames,
        )
        result = slice_record(record, layout, xinc)
        expected_avg = np.full(frame_s, (1 + 2 + 3) / 3 * 10.0)
        np.testing.assert_allclose(result.science_fid, expected_avg)

    def test_slice_single_frame(self):
        xinc = 1e-10
        n_frames = 3
        pre_s, frame_s = 200, 400
        record = np.zeros(pre_s + n_frames * frame_s)
        for k in range(n_frames):
            record[pre_s + k * frame_s : pre_s + (k + 1) * frame_s] = float(k + 1)

        layout = AcquisitionLayout(
            pre_record_us=pre_s * xinc * 1e6,
            frame_period_us=frame_s * xinc * 1e6,
            n_frames=n_frames,
            frame=1,
        )
        result = slice_record(record, layout, xinc)
        np.testing.assert_allclose(result.science_fid, np.full(frame_s, 2.0))

    def test_slice_keep_frames(self):
        xinc = 1e-10
        n_frames = 3
        pre_s, frame_s = 200, 400
        record = np.arange(pre_s + n_frames * frame_s, dtype=np.float64)
        layout = AcquisitionLayout(
            pre_record_us=pre_s * xinc * 1e6,
            frame_period_us=frame_s * xinc * 1e6,
            n_frames=n_frames,
            keep_frames=True,
        )
        result = slice_record(record, layout, xinc)
        assert result.frames is not None
        assert result.frames.shape == (n_frames, frame_s)

    def test_slice_keep_frames_false(self):
        xinc = 1e-10
        n_frames = 2
        pre_s, frame_s = 100, 200
        record = np.zeros(pre_s + n_frames * frame_s)
        layout = AcquisitionLayout(
            pre_record_us=pre_s * xinc * 1e6,
            frame_period_us=frame_s * xinc * 1e6,
            n_frames=n_frames,
            keep_frames=False,
        )
        result = slice_record(record, layout, xinc)
        assert result.frames is None

    def test_pre_record_exact(self):
        xinc = 1e-10
        pre_val = 77.0
        pre_s, frame_s = 200, 400
        record = np.zeros(pre_s + frame_s)
        record[:pre_s] = pre_val
        layout = AcquisitionLayout(
            pre_record_us=pre_s * xinc * 1e6,
            frame_period_us=frame_s * xinc * 1e6,
            n_frames=1,
        )
        result = slice_record(record, layout, xinc)
        np.testing.assert_allclose(result.pre_record, np.full(pre_s, pre_val))

    def test_tail_exact(self):
        xinc = 1e-10
        tail_val = 55.0
        pre_s, frame_s, tail_s = 100, 200, 50
        record = np.zeros(pre_s + frame_s + tail_s)
        record[pre_s + frame_s :] = tail_val
        layout = AcquisitionLayout(
            pre_record_us=pre_s * xinc * 1e6,
            frame_period_us=frame_s * xinc * 1e6,
            n_frames=1,
        )
        result = slice_record(record, layout, xinc)
        np.testing.assert_allclose(result.tail, np.full(tail_s, tail_val))

    def test_record_too_short(self):
        xinc = 1e-10
        record = np.zeros(100)  # too short
        layout = AcquisitionLayout(
            pre_record_us=100 * xinc * 1e6,
            frame_period_us=200 * xinc * 1e6,
            n_frames=2,
        )
        with pytest.raises(ValueError, match="requires"):
            slice_record(record, layout, xinc)


# ---------------------------------------------------------------------------
# Keysight MAT Loader (load_fid)
# ---------------------------------------------------------------------------


class TestKeysightMatLoader:
    def test_load_average(self, tmp_path):
        n_frames = 3
        mat_path, layout, expected_avg, pre_s, frame_s, tail_s, xinc, _ = (
            _make_segmented_mat(tmp_path, n_frames=n_frames)
        )
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)
        assert fid.n_points == frame_s
        np.testing.assert_allclose(fid.data, expected_avg, rtol=1e-12)

    def test_load_single_frame(self, tmp_path):
        n_frames = 3
        mat_path, layout, _, pre_s, frame_s, tail_s, xinc, record = _make_segmented_mat(
            tmp_path, n_frames=n_frames
        )
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, frame=1, **layout)
        # Frame 1 has offset 200 LSB × yinc (1.0) = 200.0
        np.testing.assert_allclose(fid.data, np.full(frame_s, 200.0), rtol=1e-12)

    def test_keep_frames_round_trip(self, tmp_path):
        n_frames = 3
        mat_path, layout, _, pre_s, frame_s, tail_s, xinc, _ = _make_segmented_mat(
            tmp_path, n_frames=n_frames
        )
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, keep_frames=True, **layout)
        frames = fid.metadata.get("_sliced_frames")
        assert frames is not None
        assert frames.shape == (n_frames, frame_s)

    def test_pre_record_contents(self, tmp_path):
        mat_path, layout, _, pre_s, frame_s, tail_s, xinc, record = _make_segmented_mat(
            tmp_path, n_frames=3
        )
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)
        pre = fid.metadata["_sliced_pre_record"]
        np.testing.assert_allclose(pre, record[:pre_s].astype(float), rtol=1e-12)

    def test_tail_contents(self, tmp_path):
        mat_path, layout, _, pre_s, frame_s, tail_s, xinc, record = _make_segmented_mat(
            tmp_path, n_frames=3
        )
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)
        tail = fid.metadata["_sliced_tail"]
        total_frame_s = 3 * frame_s
        np.testing.assert_allclose(
            tail, record[pre_s + total_frame_s :].astype(float), rtol=1e-12
        )

    def test_probe_and_sideband(self, tmp_path):
        """Direct sampling: probe=0, upper sideband."""
        mat_path, layout, _, *_ = _make_segmented_mat(tmp_path)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)
        assert fid.probe_freq_mhz == 0.0
        assert fid.sideband == Sideband.UPPER

    def test_shots_from_avg(self, tmp_path):
        n_frames = 5
        mat_path, layout, _, *_ = _make_segmented_mat(tmp_path, n_frames=n_frames)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)
        assert fid.shots == n_frames

    def test_shots_single_frame(self, tmp_path):
        mat_path, layout, _, *_ = _make_segmented_mat(tmp_path)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, frame=0, **layout)
        assert fid.shots == 1

    def test_multiple_channels_error(self, tmp_path):
        """Loading a multi-channel file without specifying channel raises."""
        mat_path = tmp_path / "multi.mat"
        with h5py.File(mat_path, "w") as f:
            for ch_name in ["Channel_1", "Channel_2"]:
                ch = f.create_group(ch_name)
                ch.create_dataset("Data", data=np.zeros((1, 100), dtype=np.int16))
                ch.create_dataset("XInc", data=np.array([[1e-10]]))
                ch.create_dataset("XOrg", data=np.array([[0.0]]))
                ch.create_dataset("YInc", data=np.array([[1.0]]))
                ch.create_dataset("YOrg", data=np.array([[0.0]]))
        loader = KeysightMatLoader()
        with pytest.raises(Exception, match="channel"):
            loader.load_fid(
                mat_path,
                pre_record_us=0.001,
                frame_period_us=0.005,
                n_frames=2,
            )

    def test_explicit_channel_selection(self, tmp_path):
        """Specifying channel= selects the correct channel."""
        mat_path = tmp_path / "multi2.mat"
        ch1_data = np.zeros((1, 1000), dtype=np.int16)
        ch2_data = np.full((1, 1000), 50, dtype=np.int16)
        with h5py.File(mat_path, "w") as f:
            for ch_name, d in [("Channel_1", ch1_data), ("Channel_2", ch2_data)]:
                ch = f.create_group(ch_name)
                ch.create_dataset("Data", data=d)
                ch.create_dataset("XInc", data=np.array([[1e-10]]))
                ch.create_dataset("XOrg", data=np.array([[0.0]]))
                ch.create_dataset("YInc", data=np.array([[1.0]]))
                ch.create_dataset("YOrg", data=np.array([[0.0]]))
        # Layout: 1000 samples; 100 pre + 2 * 450 frame = 1000 exactly
        xinc = 1e-10
        pre_us = 100 * xinc * 1e6
        frame_us = 450 * xinc * 1e6
        loader = KeysightMatLoader()
        fid = loader.load_fid(
            mat_path,
            channel="Channel_2",
            pre_record_us=pre_us,
            frame_period_us=frame_us,
            n_frames=2,
        )
        # Both frames of Channel_2 are all 50 LSB, average = 50.0
        assert np.all(fid.data == 50.0)

    def test_validate_source(self, tmp_path):
        mat_path = _make_mat_file(tmp_path)
        loader = KeysightMatLoader()
        result = loader.validate_source(mat_path)
        assert result["valid"]
        assert "n_samples" in result["metadata"]
        assert "model" in result["metadata"]

    def test_required_parameters(self):
        loader = KeysightMatLoader()
        req = loader.get_required_parameters()
        assert "pre_record_us" in req
        assert "frame_period_us" in req
        assert "n_frames" in req

    def test_missing_required_raises(self, tmp_path):
        mat_path, layout, *_ = _make_segmented_mat(tmp_path)
        loader = KeysightMatLoader()
        incomplete = {k: v for k, v in layout.items() if k != "n_frames"}
        with pytest.raises(Exception, match="n_frames"):
            loader.load_fid(mat_path, **incomplete)

    def test_yinc_scaling_applied(self, tmp_path):
        """YInc × raw should produce the correct voltage."""
        yinc = 0.5e-3
        mat_path, layout, _, pre_s, frame_s, tail_s, xinc, record = _make_segmented_mat(
            tmp_path, yinc=yinc
        )
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, frame=0, **layout)
        # Frame 0 has raw value 100 LSB, so volts = 100 * yinc
        np.testing.assert_allclose(fid.data, np.full(frame_s, 100.0 * yinc), rtol=1e-9)


# ---------------------------------------------------------------------------
# Segment serialization through a .ftmw pipeline file
# ---------------------------------------------------------------------------


class TestAcquisitionSegmentSerialization:
    def test_round_trip_no_frames(self, tmp_path):
        """Segments persist and reload correctly without per-frame data."""
        mat_path, layout, _, pre_s, frame_s, tail_s, xinc, _ = _make_segmented_mat(
            tmp_path, n_frames=3
        )
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)

        pipeline_path = tmp_path / "test_seg.ftmw"
        src_meta = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters=layout,
        )
        create_pipeline_file(pipeline_path, fid, src_meta)

        with h5py.File(pipeline_path, "r") as h5f:
            segs = load_acquisition_segments_from_hdf5(h5f["stage0_fid_data"])

        assert segs is not None
        assert segs.n_frames == 3
        assert segs.pre_record_us == layout["pre_record_us"]
        assert segs.frame_period_us == layout["frame_period_us"]
        assert segs.frame_selection is None
        assert len(segs.pre_record) == pre_s
        assert len(segs.tail) == tail_s
        assert segs.frames is None  # keep_frames=False

    def test_round_trip_with_frames(self, tmp_path):
        """Per-frame data persists when keep_frames=True."""
        mat_path, layout, _, pre_s, frame_s, tail_s, xinc, _ = _make_segmented_mat(
            tmp_path, n_frames=3
        )
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, keep_frames=True, **layout)

        pipeline_path = tmp_path / "test_frames.ftmw"
        src_meta = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters={**layout, "keep_frames": True},
        )
        create_pipeline_file(pipeline_path, fid, src_meta)

        with h5py.File(pipeline_path, "r") as h5f:
            segs = load_acquisition_segments_from_hdf5(h5f["stage0_fid_data"])

        assert segs is not None
        assert segs.frames is not None
        assert segs.frames.shape == (3, frame_s)

    def test_frame_selection_round_trip(self, tmp_path):
        mat_path, layout, _, pre_s, frame_s, tail_s, xinc, _ = _make_segmented_mat(
            tmp_path, n_frames=3
        )
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, frame=2, **layout)

        pipeline_path = tmp_path / "test_framesel.ftmw"
        src_meta = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters={**layout, "frame": 2},
        )
        create_pipeline_file(pipeline_path, fid, src_meta)

        with h5py.File(pipeline_path, "r") as h5f:
            segs = load_acquisition_segments_from_hdf5(h5f["stage0_fid_data"])

        assert segs is not None
        assert segs.frame_selection == 2

    def test_old_file_no_segments_loads_cleanly(self, tmp_path):
        """Files without acquisition_segments load without error."""
        from ftmwpipeline.core.data_structures import FIDProcessingParameters

        fid = FID(
            data=np.zeros(1000),
            spacing=2e-11,
            probe_freq_mhz=18000.0,
            sideband=Sideband.UPPER,
        )
        pipeline_path = tmp_path / "old_format.ftmw"
        src_meta = SourceMetadata(
            source_path=tmp_path / "dummy.csv",
            format_name="csv",
            loader_parameters={},
        )
        # Write using save_fid_to_hdf5 directly (no segments in metadata)
        with h5py.File(pipeline_path, "w") as h5f:
            grp = h5f.create_group("stage0_fid_data")
            save_fid_to_hdf5(fid, grp)
            segs = load_acquisition_segments_from_hdf5(grp)

        assert segs is None

    def test_segment_data_correct(self, tmp_path):
        """Segment arrays round-trip byte-for-byte."""
        mat_path, layout, _, pre_s, frame_s, tail_s, xinc, record = _make_segmented_mat(
            tmp_path, n_frames=3
        )
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, keep_frames=True, **layout)

        pipeline_path = tmp_path / "test_data.ftmw"
        src_meta = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters={**layout, "keep_frames": True},
        )
        create_pipeline_file(pipeline_path, fid, src_meta)

        with h5py.File(pipeline_path, "r") as h5f:
            segs = load_acquisition_segments_from_hdf5(h5f["stage0_fid_data"])

        assert segs is not None
        # Pre-record
        np.testing.assert_allclose(segs.pre_record, record[:pre_s].astype(float))
        # Tail
        np.testing.assert_allclose(
            segs.tail, record[pre_s + 3 * frame_s :].astype(float)
        )
        # Frames: frame k = (k+1)*100
        assert segs.frames is not None
        for k in range(3):
            np.testing.assert_allclose(
                segs.frames[k], np.full(frame_s, float((k + 1) * 100))
            )


# ---------------------------------------------------------------------------
# Re-import semantics
# ---------------------------------------------------------------------------


class TestReImportSemantics:
    def test_same_params_no_op(self, tmp_path):
        """Re-import with identical source + layout is a safe no-op."""
        mat_path, layout, *_ = _make_segmented_mat(tmp_path, n_frames=2)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)

        pipeline_path = tmp_path / "reimport.ftmw"
        src_meta = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters=layout,
        )
        create_pipeline_file(pipeline_path, fid, src_meta)

        # Second create with same params must not raise
        result = create_pipeline_file(pipeline_path, fid, src_meta)
        assert result == pipeline_path

    def test_different_frame_refuses_without_force(self, tmp_path):
        """Different frame selection on same source raises PipelineExistsError."""
        mat_path, layout, *_ = _make_segmented_mat(tmp_path, n_frames=3)
        loader = KeysightMatLoader()

        params_a = {**layout}
        fid_a = loader.load_fid(mat_path, **params_a)
        src_a = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters=params_a,
        )
        pipeline_path = tmp_path / "frame_refusal.ftmw"
        create_pipeline_file(pipeline_path, fid_a, src_a)

        params_b = {**layout, "frame": 1}
        fid_b = loader.load_fid(mat_path, **params_b)
        src_b = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters=params_b,
        )
        with pytest.raises(PipelineExistsError):
            create_pipeline_file(pipeline_path, fid_b, src_b)

    def test_different_frame_succeeds_with_force(self, tmp_path):
        """force=True overwrites even when frame selection changed."""
        mat_path, layout, *_ = _make_segmented_mat(tmp_path, n_frames=3)
        loader = KeysightMatLoader()

        params_a = {**layout}
        fid_a = loader.load_fid(mat_path, **params_a)
        src_a = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters=params_a,
        )
        pipeline_path = tmp_path / "frame_force.ftmw"
        create_pipeline_file(pipeline_path, fid_a, src_a)

        params_b = {**layout, "frame": 0}
        fid_b = loader.load_fid(mat_path, **params_b)
        src_b = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters=params_b,
        )
        result = create_pipeline_file(pipeline_path, fid_b, src_b, force=True)
        assert result == pipeline_path


class TestPrivateMetadataNotPersisted:
    """Underscore-prefixed loader transport keys stay out of the JSON metadata."""

    def test_sliced_arrays_absent_from_experimental_metadata(self, tmp_path):
        mat_path, layout, *_ = _make_segmented_mat(tmp_path, n_frames=3)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)
        src = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters=dict(layout),
        )
        pipeline_path = tmp_path / "private_meta.ftmw"
        create_pipeline_file(pipeline_path, fid, src)

        with h5py.File(pipeline_path, "r") as f:
            raw = f["stage0_fid_data/metadata/experimental_data"][()]
            experimental = json.loads(raw)
        assert not any(k.startswith("_") for k in experimental), experimental.keys()
        # The segments themselves live in the dedicated group, not the JSON.
        with h5py.File(pipeline_path, "r") as f:
            assert "acquisition_segments" in f["stage0_fid_data"]


# ---------------------------------------------------------------------------
# Interleave-offset cleanup — pure functions
# ---------------------------------------------------------------------------


class TestInterleaveCleanupPureFunctions:
    """Tests for estimate_interleave_pattern / subtract_interleave_pattern."""

    def test_estimate_recovers_planted_pattern(self):
        """Per-phase means are recovered exactly from a pure-offset record."""
        m = 4
        rng = np.random.default_rng(0)
        pattern_true = np.array([10.0, -20.0, 5.0, -3.0])
        n = 1000 * m
        # Record whose only content is the periodic offset (no noise)
        record = pattern_true[np.arange(n) % m]
        estimated = estimate_interleave_pattern(record, m)
        np.testing.assert_allclose(estimated, pattern_true, atol=1e-10)

    def test_subtract_zeroes_comb(self):
        """Subtracting the estimated pattern from the record zeroes the comb."""
        m = 8
        n = 800
        pattern_true = np.arange(m, dtype=float) * 3.7 - 10.0
        record = pattern_true[np.arange(n) % m]
        pattern_est = estimate_interleave_pattern(record, m)
        corrected = subtract_interleave_pattern(record, pattern_est)
        np.testing.assert_allclose(corrected, 0.0, atol=1e-10)

    def test_comb_vanishes_in_fft(self):
        """After subtraction the fs/M comb lines drop to the noise floor."""
        m = 16
        n = 16 * 1000
        # Noise + offset comb
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(n)
        pattern_true = np.linspace(-100.0, 100.0, m)
        record = noise + pattern_true[np.arange(n) % m]

        pattern_est = estimate_interleave_pattern(record, m)
        corrected = subtract_interleave_pattern(record, pattern_est)

        # Power at k·(n/m) bins (the comb harmonics) should drop substantially
        spec_before = np.abs(np.fft.rfft(record))
        spec_after = np.abs(np.fft.rfft(corrected))
        comb_bins = [k * (n // m) for k in range(1, m // 2 + 1)]
        power_before = sum(spec_before[b] ** 2 for b in comb_bins)
        power_after = sum(spec_after[b] ** 2 for b in comb_bins)
        assert power_after < power_before * 1e-6, (
            f"Comb power not suppressed: before={power_before:.3g}, "
            f"after={power_after:.3g}"
        )

    def test_sequential_two_factors(self):
        """Sequential [4, 16] factors each remove their respective comb layer."""
        m1, m2 = 4, 16
        n = 16 * 500
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(n) * 0.1
        pat4 = np.array([5.0, -3.0, 2.0, -4.0])
        pat16 = np.linspace(-8.0, 8.0, 16)

        record = noise + pat4[np.arange(n) % m1] + pat16[np.arange(n) % m2]
        quiet = record[:n]  # whole record is quiet for this test

        cleaned, patterns = apply_interleave_cleanup(record, quiet, [m1, m2])

        assert m1 in patterns
        assert m2 in patterns
        # After cleanup, residual rms should be close to the noise rms
        noise_rms = np.std(noise)
        assert (
            np.std(cleaned) < noise_rms * 5
        ), f"Residual rms {np.std(cleaned):.3g} far above noise {noise_rms:.3g}"

    def test_estimate_truncates_to_multiple_of_m(self):
        """Segments not a multiple of M are truncated, not errored."""
        m = 7
        pattern_true = np.arange(m, dtype=float)
        n = 3 * m + 3  # 3 extra samples
        record = pattern_true[np.arange(n) % m]
        estimated = estimate_interleave_pattern(record, m)
        np.testing.assert_allclose(estimated, pattern_true, atol=1e-10)

    def test_estimate_too_short_raises(self):
        """Fewer samples than M raises ValueError."""
        with pytest.raises(ValueError, match="fewer than"):
            estimate_interleave_pattern(np.array([1.0, 2.0]), m=4)

    def test_subtract_empty_pattern_noop(self):
        record = np.array([1.0, 2.0, 3.0])
        result = subtract_interleave_pattern(record, np.array([]))
        np.testing.assert_allclose(result, record)


# ---------------------------------------------------------------------------
# Interleave cleanup wired into the loader
# ---------------------------------------------------------------------------


def _make_mat_with_offset_comb(
    tmp_path: Path,
    *,
    m: int = 4,
    n_frames: int = 2,
    xinc: float = 1e-10,
    pre_us: float = 2.0,
    frame_us: float = 4.0,
    tail_us: float = 0.5,
) -> tuple:
    """Synthetic segmented .mat with a planted mod-M offset comb.

    The record contains pure per-phase offsets with no other signal so that
    after cleanup the science FID is identically zero.

    Returns (mat_path, layout_kwargs, pattern_true, xinc).
    """
    pre_s = int(round(pre_us * 1e-6 / xinc))
    frame_s = int(round(frame_us * 1e-6 / xinc))
    tail_s = int(round(tail_us * 1e-6 / xinc))
    total = pre_s + n_frames * frame_s + tail_s

    pattern_true = np.arange(1, m + 1, dtype=np.float64) * 50.0
    record = pattern_true[np.arange(total) % m].astype(np.int16)

    mat_path = _make_mat_file(
        tmp_path,
        n_samples=total,
        xinc=xinc,
        yinc=1.0,
        yorg=0.0,
        data=record,
        filename="offset_comb.mat",
    )
    layout = dict(pre_record_us=pre_us, frame_period_us=frame_us, n_frames=n_frames)
    return mat_path, layout, pattern_true, xinc


class TestInterleaveCleanupInLoader:
    def test_off_by_default(self, tmp_path):
        """Without interleave_factors the science FID is unmodified."""
        mat_path, layout, pattern_true, xinc = _make_mat_with_offset_comb(tmp_path)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)
        # With no cleanup the FID contains the raw offset comb (non-zero)
        assert np.any(fid.data != 0.0)

    def test_cleanup_zeroes_science_fid(self, tmp_path):
        """After applying the correct factor, the science FID is zeroed."""
        m = 4
        mat_path, layout, pattern_true, xinc = _make_mat_with_offset_comb(tmp_path, m=m)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, interleave_factors=[m], **layout)
        np.testing.assert_allclose(fid.data, 0.0, atol=1e-8)

    def test_pattern_matches_planted(self, tmp_path):
        """The recovered pattern equals the planted offset (in LSB/yinc units)."""
        m = 4
        mat_path, layout, pattern_true, xinc = _make_mat_with_offset_comb(tmp_path, m=m)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, interleave_factors=[m], **layout)
        patterns = fid.metadata.get("_interleave_patterns")
        assert patterns is not None
        assert m in patterns
        np.testing.assert_allclose(patterns[m], pattern_true, atol=1e-8)

    def test_no_pattern_metadata_when_off(self, tmp_path):
        """No _interleave_patterns key when interleave_factors is not given."""
        mat_path, layout, *_ = _make_mat_with_offset_comb(tmp_path)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)
        assert "_interleave_patterns" not in fid.metadata

    def test_clock_sources_injected(self, tmp_path):
        """clock_sources is populated with fs/M entries when factors are given."""
        m = 4
        xinc = 1e-10  # 10 GSa/s
        mat_path, layout, *_ = _make_mat_with_offset_comb(tmp_path, m=m, xinc=xinc)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, interleave_factors=[m], **layout)
        clocks = fid.metadata.get("clock_sources")
        assert clocks is not None
        assert len(clocks) == 1
        expected_freq = 1.0 / xinc / 1e6 / m  # fs / m in MHz
        assert abs(clocks[0]["freq_mhz"] - expected_freq) < 1e-3
        assert clocks[0]["locked"] is True
        assert "interleave_m" in clocks[0]["label"]

    def test_no_clock_sources_when_off(self, tmp_path):
        """clock_sources is absent when interleave_factors is not given."""
        mat_path, layout, *_ = _make_mat_with_offset_comb(tmp_path)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)
        assert "clock_sources" not in fid.metadata

    def test_multiple_factors_two_clocks(self, tmp_path):
        """Two interleave factors produce two clock_sources entries."""
        m1, m2 = 4, 16
        xinc = 1e-10
        # Build a record with two comb layers
        pre_us, frame_us = 2.0, 4.0
        n_frames = 2
        pre_s = int(round(pre_us * 1e-6 / xinc))
        frame_s = int(round(frame_us * 1e-6 / xinc))
        total = pre_s + n_frames * frame_s
        pat4 = np.arange(1, m1 + 1, dtype=np.float64)
        pat16 = np.arange(1, m2 + 1, dtype=np.float64) * 0.5
        record = (pat4[np.arange(total) % m1] + pat16[np.arange(total) % m2]).astype(
            np.int16
        )
        mat_path = _make_mat_file(
            tmp_path,
            n_samples=total,
            xinc=xinc,
            yinc=1.0,
            data=record,
            filename="two_comb.mat",
        )
        layout = dict(pre_record_us=pre_us, frame_period_us=frame_us, n_frames=n_frames)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, interleave_factors=[m1, m2], **layout)
        clocks = fid.metadata.get("clock_sources")
        assert clocks is not None
        assert len(clocks) == 2
        fs_mhz = 1.0 / xinc / 1e6
        freqs = {round(c["freq_mhz"], 3) for c in clocks}
        assert round(fs_mhz / m1, 3) in freqs
        assert round(fs_mhz / m2, 3) in freqs


# ---------------------------------------------------------------------------
# Interleave pattern round-trip through a .ftmw file
# ---------------------------------------------------------------------------


class TestInterleavePatternSerialization:
    def test_patterns_round_trip(self, tmp_path):
        """Estimated patterns persist to and reload from the .ftmw file."""
        m = 4
        mat_path, layout, pattern_true, xinc = _make_mat_with_offset_comb(tmp_path, m=m)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, interleave_factors=[m], **layout)

        pipeline_path = tmp_path / "interleave_rt.ftmw"
        src = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters={**layout, "interleave_factors": [m]},
        )
        create_pipeline_file(pipeline_path, fid, src)

        import h5py

        with h5py.File(pipeline_path, "r") as h5f:
            segs = load_acquisition_segments_from_hdf5(h5f["stage0_fid_data"])

        assert segs is not None
        assert segs.interleave_patterns is not None
        assert m in segs.interleave_patterns
        np.testing.assert_allclose(segs.interleave_patterns[m], pattern_true, atol=1e-8)

    def test_no_patterns_on_old_files(self, tmp_path):
        """Files without interleave datasets load with interleave_patterns=None."""
        mat_path, layout, *_ = _make_mat_with_offset_comb(tmp_path)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, **layout)  # no cleanup

        pipeline_path = tmp_path / "no_patterns.ftmw"
        src = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters=layout,
        )
        create_pipeline_file(pipeline_path, fid, src)

        import h5py

        with h5py.File(pipeline_path, "r") as h5f:
            segs = load_acquisition_segments_from_hdf5(h5f["stage0_fid_data"])

        assert segs is not None
        assert segs.interleave_patterns is None

    def test_two_factor_patterns_both_stored(self, tmp_path):
        """Both patterns from a two-factor cleanup are persisted."""
        m1, m2 = 4, 8
        pre_us, frame_us = 2.0, 4.0
        xinc = 1e-10
        n_frames = 2
        pre_s = int(round(pre_us * 1e-6 / xinc))
        frame_s = int(round(frame_us * 1e-6 / xinc))
        total = pre_s + n_frames * frame_s
        pat4 = np.arange(1, m1 + 1, dtype=np.float64) * 10.0
        pat8 = np.arange(1, m2 + 1, dtype=np.float64) * 2.0
        record = (pat4[np.arange(total) % m1] + pat8[np.arange(total) % m2]).astype(
            np.int16
        )
        mat_path = _make_mat_file(
            tmp_path,
            n_samples=total,
            xinc=xinc,
            yinc=1.0,
            data=record,
            filename="two_factor.mat",
        )
        layout = dict(pre_record_us=pre_us, frame_period_us=frame_us, n_frames=n_frames)
        loader = KeysightMatLoader()
        fid = loader.load_fid(mat_path, interleave_factors=[m1, m2], **layout)

        pipeline_path = tmp_path / "two_factor.ftmw"
        src = SourceMetadata(
            source_path=mat_path,
            format_name="keysight-mat",
            loader_parameters={**layout, "interleave_factors": [m1, m2]},
        )
        create_pipeline_file(pipeline_path, fid, src)

        import h5py

        with h5py.File(pipeline_path, "r") as h5f:
            segs = load_acquisition_segments_from_hdf5(h5f["stage0_fid_data"])

        assert segs is not None
        assert segs.interleave_patterns is not None
        assert m1 in segs.interleave_patterns
        assert m2 in segs.interleave_patterns
