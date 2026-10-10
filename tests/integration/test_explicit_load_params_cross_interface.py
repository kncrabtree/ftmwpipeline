"""Explicit load parameters agree across the CLI, ``Pipeline`` and the API.

Contract 20: an explicit chirp end + margin on a Blackchirp source records the
derived recommended start over the file's ``FidStartUs`` on every interface,
and ``ft run`` inherits it; a parameter the format does not accept is the same
``bad_setting`` (``path`` the parameter name) everywhere, with nothing written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.cli.main import main
from ftmwpipeline.file_manager import BadSettingError

pytestmark = [pytest.mark.integration]

_DATA_2638 = Path(__file__).resolve().parents[2] / "examples/blackchirp_data/2638"


def _recommended_start(path: Path) -> Optional[float]:
    with h5py.File(path, "r") as h5f:
        raw = h5f["stage0_fid_data/recommended_processing"].attrs["start_us"]
    return None if isinstance(raw, (bytes, str)) else float(raw)


def _ft_start(path: Path) -> float:
    with h5py.File(path, "r") as h5f:
        return float(h5f["processing_parameters/ft_processing"].attrs["start_us"])


@pytest.mark.skipif(not _DATA_2638.exists(), reason="2638 example data absent")
def test_explicit_chirp_end_and_margin_agree_and_ft_inherits(tmp_path: Path) -> None:
    params: Dict[str, Any] = {"chirp_end_us": 3.6, "start_margin_us": 1.0}
    api_path = tmp_path / "api.ftmw"
    pipe_path = tmp_path / "pipe.ftmw"
    cli_path = tmp_path / "cli.ftmw"

    ftmw.import_data(api_path, _DATA_2638, **params)
    Pipeline.create(pipe_path, _DATA_2638, **params)
    rc = main(
        [
            "data",
            "import",
            str(cli_path),
            str(_DATA_2638),
            "--chirp-end-us",
            "3.6",
            "--start-margin-us",
            "1.0",
        ]
    )
    assert rc == 0

    for path in (api_path, pipe_path, cli_path):
        assert _recommended_start(path) == pytest.approx(4.6), path

    # The FT inherits the recommendation (FidStartUs 2.35 does not win).
    ftmw.compute_ft(api_path, trim=(26500.0, 40000.0))
    assert _ft_start(api_path) == pytest.approx(4.6)
    rc = main(["ft", "run", str(cli_path), "--trim", "26500:40000"])
    assert rc == 0
    assert _ft_start(cli_path) == pytest.approx(4.6)


@pytest.mark.skipif(not _DATA_2638.exists(), reason="2638 example data absent")
def test_an_inapplicable_parameter_is_the_same_refusal_everywhere(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(BadSettingError) as api_exc:
        ftmw.import_data(tmp_path / "api.ftmw", _DATA_2638, n_frames=4)
    with pytest.raises(BadSettingError) as pipe_exc:
        Pipeline.create(tmp_path / "pipe.ftmw", _DATA_2638, n_frames=4)
    capsys.readouterr()
    rc = main(
        [
            "data",
            "import",
            str(tmp_path / "cli.ftmw"),
            str(_DATA_2638),
            "--n-frames",
            "4",
            "--json",
        ]
    )
    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
    payload = json.loads(lines[-1])

    assert rc == 1
    assert api_exc.value.path == "n_frames"
    assert api_exc.value.to_dict() == pipe_exc.value.to_dict() == payload
    assert payload["code"] == "bad_setting" and payload["value"] == 4
    assert not list(tmp_path.glob("*.ftmw"))
