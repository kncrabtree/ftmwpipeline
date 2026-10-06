"""Bad value arguments to the Stage 6 / read / report / model calls are typed.

Each refusal reaches the caller through the PUBLIC functional API as
``BadSettingError`` (code ``bad_setting``) carrying the offending argument's
``path``, and is still the ``ValueError`` it used to be, so an ``except
ValueError`` written before the contract keeps working.

Mutation: reverting any converted site to a bare ``raise ValueError(...)``
fails the ``BadSettingError`` check; changing the ``path`` string at a site
fails the ``path`` check.
"""

from __future__ import annotations

import h5py
import pytest

import ftmwpipeline.api as api
from ftmwpipeline.cli import main
from ftmwpipeline.file_manager import BadSettingError, PipelineFileError
from ftmwpipeline.io.fitting_serialization import read_fit_window_coverage

pytestmark = [
    pytest.mark.unit,
    # Design G1: every write here persists the reference replay of its log.
    pytest.mark.usefixtures("every_write_is_reference"),
]


def _first_window(path) -> int:
    with h5py.File(path, "r") as h5f:
        return next(
            c.window_id for c in read_fit_window_coverage(h5f["stage5_fitting"])
        )


def _check(exc: pytest.ExceptionInfo, path: str) -> None:
    err = exc.value
    assert isinstance(err, BadSettingError)
    assert isinstance(err, (ValueError, PipelineFileError))
    assert err.path == path
    d = err.to_dict()
    assert d["code"] == "bad_setting"
    assert d["path"] == path


def test_read_surface_bad_choices(stage5_small_file):
    f = stage5_small_file
    with pytest.raises(ValueError) as exc:
        api.read_table(f, "no_such_table")
    _check(exc, "table")
    with pytest.raises(ValueError) as exc:
        api.window_model(f, _first_window(f), grid="bogus")
    _check(exc, "grid")


def test_review_edit_and_apply_bad_arguments(stage5_small_file, tmp_path):
    f = stage5_small_file
    # bare edit (no add, no remove), with or without a window id
    with pytest.raises(ValueError) as exc:
        api.review_edit(f)
    _check(exc, "add")
    with pytest.raises(ValueError) as exc:
        api.review_edit(f, _first_window(f))
    _check(exc, "add")
    # add takes a frequency, not a uid
    with pytest.raises(ValueError) as exc:
        api.review_edit(f, _first_window(f), add=["uid:7"])
    _check(exc, "add")
    # malformed peak token: named by the argument the caller wrote
    with pytest.raises(ValueError) as exc:
        api.review_edit(f, _first_window(f), remove=["not-a-peak"])
    _check(exc, "remove")
    with pytest.raises(ValueError) as exc:
        api.review_edit(f, _first_window(f), add=["not-a-peak"])
    _check(exc, "add")
    # log_prefix beyond the (empty) decision log
    cur = tmp_path / "c.csv"
    cur.write_text("")
    with pytest.raises(ValueError) as exc:
        api.review_apply(f, cur, log_prefix=99)
    _check(exc, "log_prefix")


def test_report_bad_choices(stage5_reviewed_file, tmp_path):
    f = stage5_reviewed_file
    with pytest.raises(ValueError) as exc:
        api.report_table(f, fmt="docx")
    _check(exc, "format")
    with pytest.raises(ValueError) as exc:
        api.report_run(f, output_dir=tmp_path / "o1", scope="bogus")
    _check(exc, "scope")
    with pytest.raises(ValueError) as exc:
        api.report_run(f, output_dir=tmp_path / "o2", windows="bogus")
    _check(exc, "windows")
    with pytest.raises(ValueError) as exc:
        api.report_run(f, output_dir=tmp_path / "o3", emit_table=False, emit_html=False)
    _check(exc, "emit_table")
    with pytest.raises(ValueError) as exc:
        api.report_table(f, catalog=tmp_path / "missing_catalog.cat")
    _check(exc, "catalog")


def test_unknown_table_through_the_cli_keeps_the_value_error_exit(
    stage5_small_file, capsys
):
    """The CLI still reports a bad choice as a user error (exit 1)."""
    rc = main(["read", "table", str(stage5_small_file), "no_such_table"])
    assert rc == 1
