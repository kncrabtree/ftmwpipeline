"""Cross-interface consistency for the resolved curation snap tolerance.

``api.refit_snap_tol_mhz``, ``Pipeline.refit_snap_tol_mhz`` and the CLI
``review snap-tolerance`` must report the same MHz value for the same file.
The tolerance is defined as a count of active-FT bins
(``dev-docs/SCIENCE_STRATEGY.md`` Requirement 8), so each interface has to
resolve it from the file's own active region -- three resolutions that could
drift apart is the exact surface the accessor exists to replace.

The fixtures are cheap synthetic imports at deliberately different acquisition
lengths: the accessor is claimed to answer on a file that has been through
nothing but the import, and to move with the acquisition length, and both
claims should be tested at their own level rather than behind a full build.
"""

import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.core.curation import REFIT_SNAP_TOL_BINS

# (n_samples, spacing_us) -> T_active of 5.0 / 12.65 / 40.0 us.
_LENGTHS = {"short": (2500, 0.002), "reference": (6325, 0.002), "long": (20000, 0.002)}


def _make_native_ftmw(tmp_path, label):
    """A minimal imported .ftmw of a chosen duration (native-HDF5 source)."""
    n, spacing_us = _LENGTHS[label]
    src = tmp_path / f"{label}_src.h5"
    sig = np.cos(2 * np.pi * np.arange(n) * 0.01)
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = spacing_us
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=sig)
    out = tmp_path / f"{label}.ftmw"
    ftmw.import_data(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


def _cli_snap_tol(path, *args):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftmwpipeline",
            "review",
            "snap-tolerance",
            str(path),
            *args,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


@pytest.mark.integration
class TestSnapToleranceCrossInterface:
    @pytest.mark.parametrize("label", sorted(_LENGTHS))
    def test_all_three_interfaces_agree(self, tmp_path, label):
        path = _make_native_ftmw(tmp_path, label)

        from_api = ftmw.refit_snap_tol_mhz(str(path))
        from_pipeline = Pipeline(str(path)).refit_snap_tol_mhz()
        from_cli = json.loads(_cli_snap_tol(path, "--format", "json").stdout)

        assert from_pipeline == from_api
        assert from_cli["snap_tol_mhz"] == pytest.approx(from_api, rel=1e-12)
        assert from_cli["snap_tol_bins"] == REFIT_SNAP_TOL_BINS
        assert from_cli["snap_tol_mhz"] == pytest.approx(
            from_cli["snap_tol_bins"] * from_cli["bin_spacing_mhz"], rel=1e-12
        )
        assert from_cli["bin_spacing_mhz"] == pytest.approx(
            1.0 / from_cli["acquisition_us"], rel=1e-12
        )

    def test_the_answer_moves_with_the_acquisition_length(self, tmp_path):
        """Three files, three answers -- one bin count.

        A single-length suite cannot see this, which is why the constant
        survived as an absolute frequency for as long as it did.
        """
        resolved = {
            label: ftmw.refit_snap_tol_mhz(str(_make_native_ftmw(tmp_path, label)))
            for label in _LENGTHS
        }
        assert resolved["short"] > resolved["reference"] > resolved["long"]
        assert resolved["short"] / resolved["long"] == pytest.approx(8.0, rel=1e-9)

    def test_text_output_names_the_bin_count_and_the_grid(self, tmp_path):
        path = _make_native_ftmw(tmp_path, "reference")
        out = _cli_snap_tol(path).stdout
        assert "0.625 active-FT bins" in out
        assert "T_active = 12.65000 us" in out

    def test_cli_missing_file_is_a_user_error(self, tmp_path):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ftmwpipeline",
                "review",
                "snap-tolerance",
                str(tmp_path / "nope.ftmw"),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1
        assert "data import" in (proc.stdout + proc.stderr)

    def test_no_interface_mutates_the_file(self, tmp_path):
        """All three are reads: none of them may touch the file."""
        path = _make_native_ftmw(tmp_path, "reference")
        before = path.read_bytes()

        ftmw.refit_snap_tol_mhz(str(path))
        Pipeline(str(path)).refit_snap_tol_mhz()
        _cli_snap_tol(path)
        _cli_snap_tol(path, "--format", "json")

        assert path.read_bytes() == before
