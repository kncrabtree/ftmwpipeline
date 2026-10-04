"""The ``FinalPeak`` per-line fit fields on real 2638 data, on every interface.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Final products: per-line fit fields.
Reuses the session-scoped ``stage5_reviewed_2638`` build (never rebuilt); every
test that mutates copies it into ``tmp_path`` first. The table carries no
arrays, so the cross-interface comparison is on the JSON envelope; there is
nothing to compare through ``.npy``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
from pathlib import Path

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Absent, Pipeline, to_jsonable
from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline._internal.stage6_impl import (
    _final_products_is_stale,
    refresh_persisted_final_products_impl,
)
from ftmwpipeline.cli.main import main
from ftmwpipeline.core.data_structures import FinalPeak
from ftmwpipeline.file_manager import (
    PipelineCorruptionError,
    PipelineFileNotFoundError,
)
from ftmwpipeline.fitting.validation import feature_fwhm
from ftmwpipeline.io.stage6_review_serialization import (
    FINAL_PEAK_FIELDS_VERSION,
    load_stage6_review_from_file,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

NEW_FIELDS = (
    "decay_time_us",
    "decay_time_error_us",
    "shape",
    "fwhm_mhz",
    "detection_index",
    "fit_window_mhz",
)


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _cli(argv, capsys):
    rc = main(argv)
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    return json.loads(cap.out)


def _jsonable(obj):
    return json.loads(json.dumps(to_jsonable(obj)))


def _make_legacy(path) -> None:
    """Rewrite the stored table the way a pre-fields build wrote it: no
    ``peak_fields`` stamp and none of the new keys in any peak record."""
    with h5py.File(path, "a") as h5f:
        fp = h5f["stage6_review/final_products"]
        if "peak_fields" in fp.attrs:
            del fp.attrs["peak_fields"]
        data = json.loads(fp.attrs["data"])
        for row in data["peaks"]:
            for name in NEW_FIELDS:
                row.pop(name, None)
                row.pop(f"{name}__status", None)
        fp.attrs["data"] = json.dumps(data)


def _stamp(path):
    with h5py.File(path, "r") as h5f:
        stamp = h5f["stage6_review/final_products"].attrs.get("peak_fields")
    return None if stamp is None else int(stamp)


@pytest.fixture
def stamped(stage5_reviewed_2638, tmp_path):
    """A private copy whose table was written by the current build."""
    path = tmp_path / "stamped.ftmw"
    shutil.copy(stage5_reviewed_2638, path)
    assert _stamp(path) == FINAL_PEAK_FIELDS_VERSION
    return str(path)


@pytest.fixture
def legacy(stamped, tmp_path):
    """The same file with its stored table downgraded to the old format."""
    path = tmp_path / "legacy.ftmw"
    shutil.copy(stamped, path)
    _make_legacy(path)
    assert _stamp(path) is None
    return str(path)


# ---------------------------------------------------------------------------
# The fields on a table built by the current code
# ---------------------------------------------------------------------------


def test_every_line_carries_every_field_as_a_value_or_absent(stamped):
    fp = ftmw.get_final_products(stamped)
    assert fp is not None and fp.peaks
    for p in fp.peaks:
        for name in NEW_FIELDS:
            v = getattr(p, name)
            assert v is not None
            if isinstance(v, float):
                assert math.isfinite(v), (name, v)
        assert p.detection_index is Absent.UNDEFINED or p.detection_index >= 0


def test_fwhm_is_exactly_the_feature_fwhm_call_with_stage5_acquisition(stamped):
    acq = ftmw.read_metadata(stamped)["stage5.acquisition_us"]
    assert acq > 0
    fp = ftmw.get_final_products(stamped)
    checked = 0
    for p in fp.peaks:
        if isinstance(p.decay_time_us, Absent):
            assert isinstance(p.fwhm_mhz, Absent)
            continue
        assert p.fwhm_mhz == feature_fwhm(p.decay_time_us, acq, shape=p.shape)
        checked += 1
    assert checked > 0


def test_fit_fields_match_the_stage5_record_of_each_lines_window(stamped):
    fit = ftmw.load_fit(stamped)
    by_window = {int(w.window_id): w for w in fit.window_fits}
    fp = ftmw.get_final_products(stamped)
    seen_fixed = seen_free = False
    for p in fp.peaks:
        wf = by_window.get(p.window_id)
        if wf is None:
            assert all(getattr(p, n) is Absent.UNDEFINED for n in NEW_FIELDS)
            continue
        tau = wf.shared_parameters["tau_us"]
        assert p.decay_time_us == float(tau["value"])
        assert p.shape == str(getattr(wf.shape, "value", wf.shape))
        if tau.get("fitted") is False:
            assert p.decay_time_error_us is Absent.UNDEFINED
            seen_fixed = True
        elif not isinstance(p.decay_time_error_us, Absent):
            assert p.decay_time_error_us == float(tau["error"])
            seen_free = True
    assert seen_free  # the fixture has fitted windows; the check is not vacuous
    # a window with tau held fixed is a legitimate case, not a requirement
    del seen_fixed


def test_fit_window_is_the_stage5_window_in_the_calibrated_frame(stamped):
    fit = ftmw.load_fit(stamped)
    by_window = {int(w.window_id): w for w in fit.window_fits}
    fp = ftmw.get_final_products(stamped)
    eps, probe = fp.epsilon, fp.probe_freq_mhz

    def calibrated(f):
        return f if eps == 0.0 else probe + (f - probe) / (1.0 + eps)

    n = 0
    for p in fp.peaks:
        if isinstance(p.fit_window_mhz, Absent):
            continue
        lo_raw, hi_raw = sorted(by_window[p.window_id].window.freq_range)
        lo, hi = p.fit_window_mhz
        assert lo == pytest.approx(calibrated(lo_raw), abs=1e-9)
        assert hi == pytest.approx(calibrated(hi_raw), abs=1e-9)
        assert lo <= p.frequency_mhz <= hi  # same frame as frequency_mhz
        n += 1
    assert n > 0


# ---------------------------------------------------------------------------
# A table that predates the fields: rebuilt in memory, file untouched
# ---------------------------------------------------------------------------


def test_legacy_table_is_seen_as_stale(legacy, stamped):
    old = load_stage6_review_from_file(legacy).final_products
    assert old is not None
    assert all(getattr(old.peaks[0], n) is Absent.UNDEFINED for n in NEW_FIELDS)
    assert _final_products_is_stale(old, legacy) is True
    fresh = load_stage6_review_from_file(stamped).final_products
    assert _final_products_is_stale(fresh, stamped) is False


def test_legacy_table_is_rebuilt_in_memory_and_the_file_is_untouched(legacy, stamped):
    before = _md5(legacy)
    rebuilt = ftmw.get_final_products(legacy)
    assert _md5(legacy) == before
    assert _stamp(legacy) is None  # nothing was stamped by the read

    reference = ftmw.get_final_products(stamped)
    assert len(rebuilt.peaks) == len(reference.peaks)
    for got, want in zip(rebuilt.peaks, reference.peaks):
        assert got.frequency_mhz == want.frequency_mhz
        assert got.peak_uid == want.peak_uid
        for name in NEW_FIELDS:
            assert getattr(got, name) == getattr(want, name), name
    assert any(
        not isinstance(getattr(p, n), Absent) for p in rebuilt.peaks for n in NEW_FIELDS
    )


def test_legacy_read_is_identical_and_read_only_on_every_interface(legacy, capsys):
    before = _md5(legacy)
    via_api = ftmw.get_final_products(legacy)
    via_pipeline = Pipeline.open(legacy).final_products()
    envelope = _cli(["read", "get_final_products", legacy], capsys)
    assert _md5(legacy) == before

    assert via_api == via_pipeline
    envelope.pop("schema")
    expected = _jsonable(via_api)
    expected.pop("schema", None)
    assert envelope == expected
    assert _md5(legacy) == before


def test_legacy_report_table_renders_the_rebuilt_fields_without_writing(legacy):
    before = _md5(legacy)
    text = ftmw.report_table(legacy, fmt="csv")
    assert _md5(legacy) == before
    rows = list(csv.reader(ln for ln in io.StringIO(text) if not ln.startswith("#")))
    header = rows[0]
    assert header[-7:] == [
        "decay_time_us",
        "decay_time_error_us",
        "shape",
        "fwhm_mhz",
        "detection_index",
        "fit_window_low_mhz",
        "fit_window_high_mhz",
    ]
    assert rows[1][header.index("decay_time_us")] != ""


def _persist_via_set_sigma_floor(path):
    floor = ftmw.get_final_products(path).sigma_floor_khz
    ftmw.set_sigma_floor(path, floor)


def _persist_via_refresh(path):
    with atomic_write(path):
        assert refresh_persisted_final_products_impl(path) is True


def _persist_via_review_run(path):
    ftmw.review_run(path)


@pytest.mark.parametrize(
    "write",
    [_persist_via_set_sigma_floor, _persist_via_refresh, _persist_via_review_run],
    ids=["set_sigma_floor", "timebase_refresh", "review_run"],
)
def test_write_paths_persist_the_new_fields(legacy, write):
    write(legacy)
    assert _stamp(legacy) == FINAL_PEAK_FIELDS_VERSION
    stored = load_stage6_review_from_file(legacy).final_products
    assert stored.peaks == ftmw.get_final_products(legacy).peaks
    assert any(
        not isinstance(getattr(p, n), Absent) for p in stored.peaks for n in NEW_FIELDS
    )
    # once persisted, the table is current: a read is a plain read again
    assert _final_products_is_stale(stored, legacy) is False
    after = _md5(legacy)
    ftmw.get_final_products(legacy)
    assert _md5(legacy) == after


def test_refresh_of_a_current_table_writes_nothing(stamped):
    before = _md5(stamped)
    with atomic_write(stamped):
        assert refresh_persisted_final_products_impl(stamped) is False
    assert _md5(stamped) == before


# ---------------------------------------------------------------------------
# Lines of a window created during review
# ---------------------------------------------------------------------------


def _widest_gap_anchor(path):
    fit = ftmw.load_fit(path)
    ranges = sorted(tuple(map(float, w.window.freq_range)) for w in fit.window_fits)
    gaps = [(b[0] - a[1], 0.5 * (a[1] + b[0])) for a, b in zip(ranges, ranges[1:])]
    width, anchor = max(gaps)
    assert width > 0
    return anchor


def test_line_from_a_created_window_reports_that_windows_calibrated_bounds(stamped):
    anchor = _widest_gap_anchor(stamped)
    created = ftmw.review_create(stamped, anchor, frame="raw")
    ftmw.review_edit(stamped, created.window_id, add=[anchor], frame="raw")

    fp = ftmw.get_final_products(stamped)
    lines = [p for p in fp.peaks if p.window_id == created.window_id]
    assert lines, "the added line must be in the final products"
    for p in lines:
        assert p.fit_window_mhz == pytest.approx(created.freq_range_calibrated)
        lo, hi = p.fit_window_mhz
        assert lo <= p.frequency_mhz <= hi
        assert not isinstance(p.decay_time_us, Absent)
        assert not isinstance(p.shape, Absent)


# ---------------------------------------------------------------------------
# Interfaces agree; reads never write
# ---------------------------------------------------------------------------


def test_api_pipeline_and_cli_agree_and_the_read_is_byte_identical(stamped, capsys):
    before = _md5(stamped)
    via_api = ftmw.get_final_products(stamped)
    via_pipeline = Pipeline.open(stamped).final_products()
    rc = main(["read", "get_final_products", stamped, "--format", "json"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    envelope = json.loads(cap.out)
    assert _md5(stamped) == before

    assert via_api == via_pipeline
    assert envelope["schema"] == "ftmw/final_products@1"
    envelope.pop("schema")
    expected = _jsonable(via_api)
    expected.pop("schema", None)
    assert envelope == expected


def test_envelope_rows_carry_the_fields_with_absent_as_null_plus_reason(
    stamped, capsys
):
    envelope = _cli(["read", "get_final_products", stamped], capsys)
    fp = ftmw.get_final_products(stamped)
    assert len(envelope["peaks"]) == len(fp.peaks)
    for row, p in zip(envelope["peaks"], fp.peaks):
        for name in NEW_FIELDS:
            v = getattr(p, name)
            if isinstance(v, Absent):
                assert row[name] is None
                assert row[f"{name}_absent"] == v.value
            elif name == "fit_window_mhz":
                assert row[name] == list(v)
                assert f"{name}_absent" not in row
            else:
                assert row[name] == v
                assert f"{name}_absent" not in row


def test_manifest_declares_every_field_and_the_envelope_has_them(stamped, capsys):
    from ftmwpipeline import MANIFEST

    declared = set(MANIFEST.fields["FinalPeak"])
    assert set(NEW_FIELDS) <= declared
    row = _cli(["read", "get_final_products", stamped], capsys)["peaks"][0]
    assert declared <= set(row)


def test_csv_report_has_the_columns_with_empty_cells_for_held_fixed_tau(stamped):
    text = ftmw.report_table(stamped, fmt="csv")
    rows = list(csv.reader(ln for ln in io.StringIO(text) if not ln.startswith("#")))
    header = rows[0]
    fp = ftmw.get_final_products(stamped)
    assert len(rows) - 1 == len(fp.peaks)
    for col in ("decay_time_us", "decay_time_error_us", "fwhm_mhz", "shape"):
        assert col in header
    i_err = header.index("decay_time_error_us")
    for row, p in zip(rows[1:], fp.peaks):
        cell = row[i_err]
        assert (cell == "") == isinstance(p.decay_time_error_us, Absent)


# ---------------------------------------------------------------------------
# Typed errors and the no-table case
# ---------------------------------------------------------------------------


def test_final_products_before_stage6_are_none_not_a_table_of_absent(
    baseline_2638_stage5_small,
):
    path = str(baseline_2638_stage5_small)
    assert ftmw.get_final_products(path) is None
    assert Pipeline.open(path).final_products() is None


def test_missing_file_is_a_typed_error_on_every_interface(tmp_path, capsys):
    missing = tmp_path / "nope.ftmw"
    with pytest.raises(PipelineFileNotFoundError) as exc:
        ftmw.get_final_products(missing)
    assert exc.value.to_dict()["code"] == "not_found"
    with pytest.raises(PipelineFileNotFoundError):
        Pipeline.open(missing).final_products()
    rc = main(["read", "get_final_products", str(missing), "--format", "json"])
    cap = capsys.readouterr()
    assert rc == 1 and cap.out == ""
    payload = json.loads(cap.err)
    assert payload["schema"] == "ftmw/error@1" and payload["code"] == "not_found"
    assert not missing.exists()  # a failed read creates nothing


def test_corrupt_file_is_a_typed_error_and_is_not_modified(tmp_path, capsys):
    junk = tmp_path / "junk.ftmw"
    junk.write_bytes(b"not hdf5" * 100)
    before = _md5(junk)
    with pytest.raises(PipelineCorruptionError) as exc:
        ftmw.get_final_products(junk)
    assert exc.value.to_dict()["code"] == "file_corrupt"
    rc = main(["read", "get_final_products", str(junk), "--format", "json"])
    cap = capsys.readouterr()
    assert rc == 2 and json.loads(cap.err)["code"] == "file_corrupt"
    assert _md5(junk) == before


def test_finalpeak_defaults_are_absent_not_none():
    names = {f.name: f.default for f in FinalPeak.__dataclass_fields__.values()}
    for name in NEW_FIELDS:
        assert names[name] is Absent.UNDEFINED
