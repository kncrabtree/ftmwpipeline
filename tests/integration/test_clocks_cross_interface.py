"""Cross-interface consistency for the ``clocks`` declaration surface.

The CLI ``clocks``, ``api.set_clock_sources`` / ``get_clock_sources``, and the
``Pipeline`` methods must all read and write the same recommended-clock-sources
declaration on a ``.ftmw`` file.
"""

import subprocess
import sys

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline


def _make_native_ftmw(tmp_path):
    """Create a minimal imported .ftmw (native-HDF5 source, no embedded clocks)."""
    src = tmp_path / "src.h5"
    sig = np.cos(2 * np.pi * np.arange(1024) * 0.01)
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.02
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=sig)
    out = tmp_path / "exp.ftmw"
    ftmw.import_data(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


def _as_tuples(clocks):
    return [(c.freq_mhz, c.locked, c.label) for c in (clocks or ())]


@pytest.mark.integration
class TestClocksCrossInterface:
    def test_api_set_and_get(self, tmp_path):
        path = _make_native_ftmw(tmp_path)
        assert ftmw.get_clock_sources(str(path)) is None
        result = ftmw.set_clock_sources(
            str(path),
            [
                {"freq_mhz": 5760.0, "locked": True, "label": "synth"},
                {"freq_mhz": 6250.0, "locked": False, "label": "digitizer"},
            ],
        )
        assert _as_tuples(result) == [
            (5760.0, True, "synth"),
            (6250.0, False, "digitizer"),
        ]
        assert _as_tuples(ftmw.get_clock_sources(str(path))) == _as_tuples(result)

    def test_pipeline_matches_api(self, tmp_path):
        path = _make_native_ftmw(tmp_path)
        ftmw.set_clock_sources(
            str(path), [{"freq_mhz": 5760.0, "locked": True, "label": "synth"}]
        )
        pipe = Pipeline(str(path))
        assert _as_tuples(pipe.get_clock_sources()) == [(5760.0, True, "synth")]

        pipe.set_clock_sources([{"freq_mhz": 16000.0, "label": "awg"}], replace=False)
        assert _as_tuples(ftmw.get_clock_sources(str(path))) == [
            (5760.0, True, "synth"),
            (16000.0, True, "awg"),
        ]

    def test_cli_matches_api(self, tmp_path):
        path = _make_native_ftmw(tmp_path)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ftmwpipeline",
                "clocks",
                "set",
                str(path),
                "5760:locked:synth",
                "6250:free:digitizer",
            ],
            check=True,
            capture_output=True,
        )
        assert _as_tuples(ftmw.get_clock_sources(str(path))) == [
            (5760.0, True, "synth"),
            (6250.0, False, "digitizer"),
        ]

    def test_remove_and_clear(self, tmp_path):
        path = _make_native_ftmw(tmp_path)
        ftmw.set_clock_sources(
            str(path),
            [
                {"freq_mhz": 5760.0, "label": "synth"},
                {"freq_mhz": 6250.0, "locked": False, "label": "digitizer"},
            ],
        )
        remaining = ftmw.remove_clock_sources(str(path), [6250.0])
        assert _as_tuples(remaining) == [(5760.0, True, "synth")]
        ftmw.clear_clock_sources(str(path))
        assert ftmw.get_clock_sources(str(path)) is None
