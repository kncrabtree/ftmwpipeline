"""Unit tests for the knob-agnostic sweep engine, using synthetic knobs so no
pipeline build is needed."""

from pathlib import Path

import pytest

from ftmwpipeline._internal.tuning.engine import (
    SweepRow,
    run_scan,
    run_scan_batch,
)
from ftmwpipeline._internal.tuning.registry import KnobSpec


def _dummy_ftmw(tmp_path: Path) -> Path:
    """A stand-in input file the engine copies; synthetic runs ignore it."""
    fp = tmp_path / "in.ftmw"
    fp.write_bytes(b"not-a-real-hdf5")
    return fp


def _make_spec(*, plot=None, direction="none", primary_metric=None) -> KnobSpec:
    # run returns the grid value verbatim; metric exposes it as column "m".
    return KnobSpec(
        path="stageT.block.knob",
        stage="stageT",
        requires="stage0_fid_data",
        help="synthetic test knob",
        inst_sensitivity="N",
        default_grid=(1.0, 2.0, 3.0),
        run=lambda path, value: value,
        metric=lambda result: {"m": result, "k": result * 10},
        metric_columns=("m", "k"),
        primary_metric=primary_metric,
        direction=direction,
        plot=plot,
    )


def test_sweep_returns_row_per_grid_value(tmp_path):
    spec = _make_spec()
    res = run_scan(spec, _dummy_ftmw(tmp_path), grid=[1.0, 2.0],
                   output_dir=tmp_path)
    assert len(res.rows) == 2
    assert [r.value for r in res.rows] == [1.0, 2.0]
    assert res.rows[0].metrics == {"m": 1.0, "k": 10.0}


def test_default_grid_used_when_none(tmp_path):
    spec = _make_spec()
    res = run_scan(spec, _dummy_ftmw(tmp_path), output_dir=tmp_path)
    assert [r.value for r in res.rows] == [1.0, 2.0, 3.0]


def test_csv_written_to_output_dir(tmp_path):
    spec = _make_spec()
    res = run_scan(spec, _dummy_ftmw(tmp_path), grid=[1.0], output_dir=tmp_path)
    assert res.csv_path is not None
    assert res.csv_path.parent == tmp_path
    assert res.csv_path.exists()
    content = res.csv_path.read_text()
    assert "knob,m,k" in content  # leaf column + metric columns


def test_table_only_fallback_when_no_plot(tmp_path):
    spec = _make_spec(plot=None)
    res = run_scan(spec, _dummy_ftmw(tmp_path), grid=[1.0], output_dir=tmp_path)
    assert res.plot_path is None
    assert res.as_table()  # table always renders


def test_plot_adapter_writes_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot(spec, rows, ctx):
        fig, ax = plt.subplots()
        ax.plot([r.value for r in rows], [r.metrics["m"] for r in rows])
        return fig

    spec = _make_spec(plot=plot)
    res = run_scan(spec, _dummy_ftmw(tmp_path), grid=[1.0, 2.0],
                   output_dir=tmp_path)
    assert res.plot_path is not None
    assert res.plot_path.exists()


def test_recommender_picks_min(tmp_path):
    spec = _make_spec(direction="min", primary_metric="m")
    res = run_scan(spec, _dummy_ftmw(tmp_path), grid=[3.0, 1.0, 2.0],
                   output_dir=tmp_path)
    assert res.recommendation is not None
    assert res.recommendation.value == 1.0
    assert res.recommendation.metric == "m"


def test_recommender_picks_max(tmp_path):
    spec = _make_spec(direction="max", primary_metric="m")
    res = run_scan(spec, _dummy_ftmw(tmp_path), grid=[3.0, 1.0, 2.0],
                   output_dir=tmp_path)
    assert res.recommendation.value == 3.0


def test_no_recommendation_when_direction_none(tmp_path):
    spec = _make_spec(direction="none")
    res = run_scan(spec, _dummy_ftmw(tmp_path), grid=[1.0, 2.0],
                   output_dir=tmp_path)
    assert res.recommendation is None
    # apply instructions are always emitted
    assert "To apply a chosen value" in res.apply_instructions


def test_input_file_not_mutated(tmp_path):
    spec = _make_spec()
    fp = _dummy_ftmw(tmp_path)
    before = fp.read_bytes()
    run_scan(spec, fp, grid=[1.0, 2.0], output_dir=tmp_path / "out")
    assert fp.read_bytes() == before


def test_progress_callback_invoked_per_value(tmp_path):
    spec = _make_spec()
    calls = []
    run_scan(spec, _dummy_ftmw(tmp_path), grid=[1.0, 2.0, 3.0],
             output_dir=tmp_path, progress=lambda d, t, v: calls.append((d, t, v)))
    assert calls == [(1, 3, 1.0), (2, 3, 2.0), (3, 3, 3.0)]


def test_quiet_suppresses_default_reporter(tmp_path, capsys):
    spec = _make_spec()
    run_scan(spec, _dummy_ftmw(tmp_path), grid=[1.0, 2.0],
             output_dir=tmp_path, quiet=True)
    err = capsys.readouterr().err
    assert "Scanning" not in err


def test_default_reporter_writes_progress_to_stderr(tmp_path, capsys):
    spec = _make_spec()
    run_scan(spec, _dummy_ftmw(tmp_path), grid=[1.0, 2.0], output_dir=tmp_path)
    err = capsys.readouterr().err
    assert "Scanning stageT.block.knob" in err
    assert "[2/2]" in err


def test_output_dir_created_and_contains_artifacts(tmp_path):
    spec = _make_spec()
    out = tmp_path / "nested" / "out"
    res = run_scan(spec, _dummy_ftmw(tmp_path), grid=[1.0], output_dir=out)
    assert out.is_dir()
    assert res.csv_path.parent == out


def _named_spec(path: str, *, fail: bool = False) -> KnobSpec:
    def _run(p, value):
        if fail:
            raise RuntimeError("required stage missing")
        return value
    return KnobSpec(
        path=path, stage=path.split(".")[0], requires="stage0_fid_data",
        help="synthetic", inst_sensitivity="N", default_grid=(1.0, 2.0),
        run=_run, metric=lambda r: {"m": r}, metric_columns=("m",),
    )


def test_batch_runs_each_knob_and_returns_item_per_spec(tmp_path):
    specs = [_named_spec("s.b.k1"), _named_spec("s.b.k2")]
    items = run_scan_batch(specs, _dummy_ftmw(tmp_path),
                           output_dir=tmp_path, quiet=True)
    assert [it.knob for it in items] == ["s.b.k1", "s.b.k2"]
    assert all(it.ok and it.error is None for it in items)
    assert all(it.result is not None and it.result.csv_path.exists()
               for it in items)


def test_batch_continues_past_a_failing_knob(tmp_path):
    specs = [_named_spec("s.b.ok1"), _named_spec("s.b.bad", fail=True),
             _named_spec("s.b.ok2")]
    items = run_scan_batch(specs, _dummy_ftmw(tmp_path),
                           output_dir=tmp_path, quiet=True)
    assert [it.ok for it in items] == [True, False, True]
    bad = items[1]
    assert bad.result is None
    assert "required stage missing" in bad.error
