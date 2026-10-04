"""Absence in ``read_metadata`` and ``get_pipeline_info`` (Wave 3).

Pure-function tests of the conversion rules; the fresh-file behaviour is
covered by the cross-interface read tests.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from ftmwpipeline._internal.info_impl import environment_info_fields
from ftmwpipeline._internal.read_impl import (
    _stage5_acquisition_absence,
    _tau_section_absence,
    _timebase_absence,
    format_metadata_impl,
)
from ftmwpipeline.cli.info_commands import _print_environment
from ftmwpipeline.cli.timebase_commands import _print_epsilon_and_lattice
from ftmwpipeline.contract import Absent

_ENV = {
    "ftmwpipeline": "1.0.0",
    "analysis_epoch": 4,
    "python": "3.11.0",
    "numpy": "2.0",
    "scipy": "1.13",
    "h5py": "3.10",
    "blas": "openblas",
    "platform": "linux",
}


def test_timebase_without_tones_is_undefined() -> None:
    out: Dict[str, Any] = {
        "timebase.epsilon": 0.0,
        "timebase.sigma_epsilon": math.inf,
        "timebase.lattice_g_mhz": 0.0,
        "timebase.n_used": 0,
    }
    _timebase_absence(out)
    assert out["timebase.epsilon"] is Absent.UNDEFINED
    assert out["timebase.sigma_epsilon"] is Absent.UNDEFINED
    assert out["timebase.lattice_g_mhz"] is Absent.UNDEFINED
    assert out["timebase.n_used"] == 0


def test_timebase_with_tones_keeps_values() -> None:
    out: Dict[str, Any] = {
        "timebase.epsilon": 2e-6,
        "timebase.sigma_epsilon": 7e-8,
        "timebase.lattice_g_mhz": 640.0,
        "timebase.n_used": 11,
    }
    before = dict(out)
    _timebase_absence(out)
    assert out == before


def test_tau_nonfinite_and_unset_vote_are_undefined() -> None:
    out: Dict[str, Any] = {
        "tau.tau_maj_us": math.nan,
        "tau.sigma_tau_us": math.nan,
        "tau.n_contributors": 0,
        "tau.recommended_shape": None,
        "tau_g.tau_maj_us": 3.0,
    }
    _tau_section_absence(out, "tau")
    assert out["tau.tau_maj_us"] is Absent.UNDEFINED
    assert out["tau.sigma_tau_us"] is Absent.UNDEFINED
    assert out["tau.recommended_shape"] is Absent.UNDEFINED
    assert out["tau.n_contributors"] == 0
    assert out["tau_g.tau_maj_us"] == 3.0


@pytest.mark.parametrize(
    "stored, expected",
    [
        (None, Absent.NOT_RUN),
        (0.0, Absent.UNDEFINED),
        (-1.0, Absent.UNDEFINED),
        (math.nan, Absent.UNDEFINED),
        (12.5, 12.5),
    ],
)
def test_stage5_acquisition(stored: Any, expected: Any) -> None:
    assert _stage5_acquisition_absence(stored) == expected


def test_format_metadata_renders_absent_as_empty() -> None:
    meta = {"file.format_version": Absent.NOT_RUN, "fid.shots": 3}
    assert "file.format_version," in format_metadata_impl(meta, "csv").splitlines()
    assert json.loads(format_metadata_impl(meta, "json"))["file.format_version"] is None


def test_info_environment_stamped() -> None:
    fields = environment_info_fields(
        {
            "format_version": "1.0",
            "created_with": "1.0.0",
            "stage_environments": {"stage3_peaks": dict(_ENV, analysis_epoch=None)},
            "last_written_with": dict(_ENV),
            "environment_drift": [],
            "runtime_environment_drift": [],
            "current_environment": dict(_ENV),
            "environment_acknowledged": False,
        }
    )
    assert fields["format_version"] == "1.0"
    assert fields["environment_drift"] == []
    assert fields["stage_environments"]["stage3_peaks"]["analysis_epoch"] is (
        Absent.NOT_RUN
    )
    assert fields["last_written_with"] == _ENV
    assert fields["environment_acknowledged"] is False


def test_info_environment_unstamped_file() -> None:
    fields = environment_info_fields(
        {
            "format_version": None,
            "created_with": None,
            "stage_environments": {},
            "last_written_with": None,
            "environment_drift": [],
            "runtime_environment_drift": [],
            "current_environment": dict(_ENV),
            "environment_acknowledged": False,
        }
    )
    for key in (
        "format_version",
        "created_with",
        "stage_environments",
        "last_written_with",
        "environment_drift",
        "runtime_environment_drift",
    ):
        assert fields[key] is Absent.NOT_RUN, key
    assert fields["current_environment"] == _ENV


def test_info_environment_validation_failure_fabricates_nothing() -> None:
    fields = environment_info_fields({"valid": False, "errors": ["boom"]})
    for key in (
        "format_version",
        "created_with",
        "stage_environments",
        "last_written_with",
        "environment_drift",
        "runtime_environment_drift",
        "environment_acknowledged",
    ):
        assert fields[key] is Absent.UNDEFINED, key
    assert fields["current_environment"]["python"]


def test_info_cli_prints_absent_environment(capsys: pytest.CaptureFixture) -> None:
    info = {
        "last_written_with": Absent.NOT_RUN,
        "stage_environments": Absent.NOT_RUN,
        "environment_drift": Absent.NOT_RUN,
        "runtime_environment_drift": Absent.NOT_RUN,
        "current_environment": dict(_ENV),
        "environment_acknowledged": False,
    }
    _print_environment(info)
    out = capsys.readouterr().out
    assert "not recorded" in out
    assert "Absent" not in out


def test_info_cli_prints_partial_stamp(capsys: pytest.CaptureFixture) -> None:
    env = dict(_ENV, analysis_epoch=Absent.NOT_RUN)
    _print_environment(
        {
            "last_written_with": env,
            "stage_environments": {"stage3_peaks": env},
            "environment_drift": [],
            "runtime_environment_drift": [],
        }
    )
    assert "Absent" not in capsys.readouterr().out


def test_timebase_cli_names_the_undefined_cases(
    capsys: pytest.CaptureFixture,
) -> None:
    # Mutation: print eps/lattice unconditionally ("+0.000 +- inf ppm").
    _print_epsilon_and_lattice(
        SimpleNamespace(
            n_used=0, epsilon=0.0, sigma_epsilon=math.inf, lattice_g_mhz=0.0
        )
    )
    out = capsys.readouterr().out
    assert "undefined; no usable lattice tones" in out
    assert "undefined; no locked lattice" in out
    assert "inf" not in out


def test_timebase_cli_prints_measured_values(capsys: pytest.CaptureFixture) -> None:
    _print_epsilon_and_lattice(
        SimpleNamespace(
            n_used=11, epsilon=2e-6, sigma_epsilon=7e-8, lattice_g_mhz=640.0
        )
    )
    out = capsys.readouterr().out
    assert "+2.000 +- 0.070 ppm" in out
    assert "640.0 MHz" in out
