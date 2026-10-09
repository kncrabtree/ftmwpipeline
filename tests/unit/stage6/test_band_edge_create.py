"""A line at the analysis-band edge gets an ordinary window.

The active-FT grid is not trimmed (on 2638 it spans 15960-40960 MHz against a
26500-40000 MHz band), and the Stage 6 replay path estimates noise only inside
the band. A window created for an add within a half-width of the band edge used
to be centered on the anchor regardless, reaching past the edge onto NaN noise:
the fit then died on a bare ``amp_floor is required`` ValueError, sinking every
other action in the batch with it. The planner now clamps the window to the
band -- full width, shifted to end at the edge -- and these tests pin that on
every interface.

Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline, to_jsonable
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline.cli.main import main
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("every_write_is_reference"),
]

#: How far inside the band edge the anchor sits: well within a window
#: half-width (32 bins, ~2.7 MHz on 2638) and within the 8-bin floor.
_INSET_MHZ = 0.2


def _band_edges(path: Path) -> Tuple[float, float]:
    """The first and last active-FT grid points inside the analysis band."""
    ctx = s6._build_shared_fit_ctx(str(path)).fit_ctx
    assert ctx.trim_range is not None, "fixture must carry a trim range"
    freq = np.asarray(ctx.active_ft.freq_mhz, dtype=float)
    lo, hi = min(ctx.trim_range), max(ctx.trim_range)
    in_band = freq[(freq >= lo) & (freq <= hi)]
    assert in_band.size < freq.size, (
        "the active FT does not extend past the band on this fixture, so there "
        "is no edge to fall off and these tests prove nothing"
    )
    return float(in_band.min()), float(in_band.max())


def _trim(path: Path) -> Tuple[float, float]:
    ctx = s6._build_shared_fit_ctx(str(path)).fit_ctx
    assert ctx.trim_range is not None
    return float(min(ctx.trim_range)), float(max(ctx.trim_range))


def _edge_anchor(path: Path, side: str) -> float:
    lo, hi = _band_edges(path)
    anchor = lo + _INSET_MHZ if side == "low" else hi - _INSET_MHZ
    for w in s6.effective_window_plan(str(path)).windows:
        wlo, whi = sorted(float(v) for v in w.freq_range)
        assert not (wlo <= anchor <= whi), (
            f"window {w.window_id} already covers the {side} band edge on this "
            "fixture; the edge create cannot be exercised"
        )
    return anchor


def _interior_add(path: Path) -> Tuple[int, float]:
    """A fitted window and an in-window frequency well clear of its peaks."""
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    wf = next(w for w in sf.window_fits if w.fitted_peaks and w.window is not None)
    assert wf.window is not None and wf.window_id is not None
    lo, hi = sorted(float(v) for v in wf.window.freq_range)
    peaks = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    clear = max(
        (lo + (hi - lo) * t / 40 for t in range(4, 37)),
        key=lambda x: min(abs(x - q) for q in peaks),
    )
    return int(wf.window_id), clear


def _csv(tmp_path: Path, name: str, rows: List[str]) -> Path:
    cur = tmp_path / name
    cur.write_text("".join(f"{r}\n" for r in rows))
    return cur


def _copy(src: Path, tmp_path: Path, name: str) -> str:
    dst = tmp_path / name
    shutil.copy(src, dst)
    return str(dst)


def _cli_json(argv: List[str], capsys: Any) -> Dict[str, Any]:
    capsys.readouterr()
    rc = main(argv + ["--json"])
    out = capsys.readouterr().out
    assert rc == 0, out
    payload: Dict[str, Any] = json.loads(out)
    return payload


@pytest.mark.parametrize("side", ["low", "high"])
def test_an_edge_add_creates_a_full_window_ending_at_the_band_edge(
    stage5_reviewed_source: Path, tmp_path: Path, side: str
) -> None:
    src = stage5_reviewed_source
    anchor = _edge_anchor(src, side)
    preview = ftmw.review_preview(
        str(src), str(_csv(tmp_path, "edge.csv", [f"add,,{anchor},"])), frame="raw"
    )
    (pw,) = preview.created_windows
    t_lo, t_hi = _trim(src)
    lo, hi = pw.freq_range
    assert pw.mode == "created"
    assert t_lo <= lo <= anchor <= hi <= t_hi

    # Same width as a window created in open space: shifted, not shrunk.
    open_anchor = 0.5 * (t_lo + t_hi)
    open_ = ftmw.review_preview(
        str(src), str(_csv(tmp_path, "open.csv", [f"add,,{open_anchor},"])), frame="raw"
    ).created_windows[0]
    assert pw.n_points == open_.n_points


def test_an_edge_add_applies_and_persists(
    stage5_reviewed_source: Path, tmp_path: Path
) -> None:
    path = _copy(stage5_reviewed_source, tmp_path, "live.ftmw")
    anchor = _edge_anchor(Path(path), "low")
    result = ftmw.review_apply(
        path, str(_csv(tmp_path, "edge.csv", [f"add,,{anchor},"])), frame="raw"
    )
    (pw,) = result.created_windows
    kinds = [e.kind for e in ftmw.review_log(path)]
    assert kinds == ["add"]  # an implied create logs as its add
    lo, hi = sorted(
        float(v)
        for v in next(
            w
            for w in s6.effective_window_plan(path).windows
            if int(w.window_id) == pw.window_id
        ).freq_range
    )
    assert _trim(Path(path))[0] <= lo <= anchor <= hi


def test_an_edge_add_does_not_sink_the_rest_of_its_batch(
    stage5_reviewed_source: Path, tmp_path: Path
) -> None:
    src = stage5_reviewed_source
    anchor = _edge_anchor(src, "low")
    wid, interior = _interior_add(src)
    cur = _csv(tmp_path, "batch.csv", [f"add,,{anchor},", f"add,{wid},{interior},"])
    preview = ftmw.review_preview(str(src), str(cur), frame="raw")
    (pw,) = preview.created_windows
    assert {pw.window_id, wid} <= set(preview.windows)


def test_the_edge_preview_agrees_on_every_interface(
    stage5_reviewed_source: Path, tmp_path: Path, capsys: Any
) -> None:
    src = stage5_reviewed_source
    anchor = _edge_anchor(src, "high")
    cur = _csv(tmp_path, "edge.csv", [f"add,,{anchor},"])

    api = ftmw.review_preview(str(src), str(cur), frame="raw")
    pipe = Pipeline.open(str(src)).review_preview(str(cur), frame="raw")
    cli = _cli_json(["review", "preview", str(src), str(cur), "--frame", "raw"], capsys)
    assert api.created_windows == pipe.created_windows
    assert cli["created_windows"] == to_jsonable(api.created_windows)
    assert api.created_windows[0].mode == "created"


def test_review_create_at_the_edge_agrees_on_every_interface(
    stage5_reviewed_source: Path, tmp_path: Path, capsys: Any
) -> None:
    src = stage5_reviewed_source
    anchor = _edge_anchor(src, "low")
    api = ftmw.review_create(_copy(src, tmp_path, "a.ftmw"), anchor, frame="raw")
    pipe = Pipeline.open(_copy(src, tmp_path, "p.ftmw")).review_create(
        anchor, frame="raw"
    )
    cli = _cli_json(
        [
            "review",
            "create",
            _copy(src, tmp_path, "c.ftmw"),
            "--at",
            str(anchor),
            "--frame",
            "raw",
        ],
        capsys,
    )["summary"]
    assert api.mode == pipe.mode == "created"
    assert api.freq_range == pipe.freq_range
    assert api.n_points == pipe.n_points
    assert _trim(src)[0] <= api.freq_range[0] <= anchor
    assert cli["window_id"] == api.window_id
    assert cli["mode"] == api.mode
    assert [cli["freq_lo_mhz"], cli["freq_hi_mhz"]] == list(api.freq_range)
    assert cli["n_points"] == api.n_points
