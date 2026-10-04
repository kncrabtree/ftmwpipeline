"""``settings set`` enforces a field's declared ``choices``, on every interface.

``dev-docs/CONTRACT_STRATEGY.md`` §Errors: a setting value outside what the
field declares is ``bad_setting`` (``path`` the registry path, ``expected``
what would have been accepted, ``value`` what the caller passed), raised before
anything is written. The functional API, the ``Pipeline`` class and the CLI
refuse identically and leave the file byte-for-byte unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline.cli.main import main
from ftmwpipeline.file_manager import BadSettingError
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.integration]

_KNOB = "stage5.conservative.n_eff_kind"
_CHOICES = ("perplexity_log1p_snr", "kish_mag_sq", "kish_mag", "hard_radius")
_EXPECTED = "one of " + ", ".join(repr(c) for c in _CHOICES)


@pytest.fixture
def imported(tmp_path) -> Path:
    src = tmp_path / "src.h5"
    t = np.arange(2048) * 0.002
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.002
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=np.cos(2 * np.pi * 37.0 * t) * np.exp(-t / 3.0))
    out = tmp_path / "exp.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _value(path: Path):
    rows = {r.path: r for r in ftmw.settings_show(path, include_advanced=True)}
    return rows[_KNOB].value


def _refuse(via: str, path: Path, value, capsys) -> dict:
    """The refusal's ``to_dict()`` (CLI: the error JSON on stderr)."""
    if via == "api":
        with pytest.raises(BadSettingError) as excinfo:
            ftmw.settings_set(path, _KNOB, value)
        return excinfo.value.to_dict()
    if via == "pipeline":
        with pytest.raises(BadSettingError) as excinfo:
            Pipeline.open(path).settings_set(_KNOB, value)
        return excinfo.value.to_dict()
    capsys.readouterr()
    rc = main(["settings", "set", str(path), _KNOB, str(value), "--json"])
    cap = capsys.readouterr()
    assert rc == 1 and cap.out == ""
    return json.loads(cap.err)


def test_all_three_interfaces_refuse_a_bad_choice_identically(imported, capsys):
    before = _md5(imported)
    errors = [
        _refuse(via, imported, "bogus", capsys) for via in ("api", "pipeline", "cli")
    ]
    assert errors[0] == errors[1] == errors[2]
    error = errors[0]
    assert error["schema"] == "ftmw/error@1" and error["code"] == "bad_setting"
    assert error["path"] == _KNOB
    assert error["expected"] == _EXPECTED
    assert error["value"] == "bogus"
    assert _md5(imported) == before  # not one byte written
    assert _value(imported) == "perplexity_log1p_snr"  # still the default


@pytest.mark.parametrize("via", ["api", "pipeline", "cli"])
def test_a_declared_choice_is_accepted_and_persists(imported, via, capsys):
    if via == "api":
        ftmw.settings_set(imported, _KNOB, "kish_mag")
    elif via == "pipeline":
        Pipeline.open(imported).settings_set(_KNOB, "kish_mag")
    else:
        assert main(["settings", "set", str(imported), _KNOB, "kish_mag"]) == 0
    capsys.readouterr()
    assert _value(imported) == "kish_mag"


@pytest.mark.parametrize("via", ["api", "pipeline", "cli"])
def test_unset_is_never_refused_as_a_bad_choice(imported, via, capsys):
    ftmw.settings_set(imported, _KNOB, "hard_radius")
    if via == "api":
        ftmw.settings_unset(imported, _KNOB)
    elif via == "pipeline":
        Pipeline.open(imported).settings_unset(_KNOB)
    else:
        assert main(["settings", "unset", str(imported), _KNOB]) == 0
    capsys.readouterr()
    assert _value(imported) == "perplexity_log1p_snr"


def test_the_refusal_is_still_a_value_error(imported):
    """Existing ``except ValueError`` clauses around ``settings_set`` keep
    working (the typed error subclasses what it replaced)."""
    with pytest.raises(ValueError):
        ftmw.settings_set(imported, _KNOB, "bogus")
