"""The per-line Stage 5 fit fields of ``FinalPeak`` (contract Wave 2, task 2.1).

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Final products: per-line fit fields.
Each line carries ``decay_time_us``, ``decay_time_error_us``, ``shape``,
``fwhm_mhz``, ``detection_index`` and ``fit_window_mhz`` joined from the Stage 5
fit of its window; every field uses ``Absent`` (never ``None``/``nan``), and a
stored table that predates them is detected so it can be rebuilt in memory.

These tests build a synthetic ``SpectrumFit`` and go straight to
``_build_final_products``; the real-file behaviour (rebuild-on-read, write
paths, CLI) is in
``tests/integration/test_final_peak_fit_fields_cross_interface.py``.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import math

import h5py
import numpy as np
import pytest

from ftmwpipeline import Absent, to_jsonable
from ftmwpipeline._internal import report_html_impl as html_impl
from ftmwpipeline._internal import report_impl
from ftmwpipeline._internal.stage6_impl import _build_final_products
from ftmwpipeline.contract import MANIFEST
from ftmwpipeline.core.data_structures import (
    FinalPeak,
    FittedPeak,
    FittingResult,
    Sideband,
    SpectralWindow,
    SpectrumFit,
    Stage6Review,
)
from ftmwpipeline.fitting.validation import feature_fwhm
from ftmwpipeline.io.stage6_review_serialization import (
    FINAL_PEAK_FIELDS_VERSION,
    final_products_predate_fit_fields,
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)

pytestmark = [pytest.mark.unit]

PROBE = 40960.0
ACQ_US = 12.73
NEW_FIELDS = (
    "decay_time_us",
    "decay_time_error_us",
    "shape",
    "fwhm_mhz",
    "detection_index",
    "fit_window_mhz",
)


# ---------------------------------------------------------------------------
# Synthetic fit
# ---------------------------------------------------------------------------


def _peak(freq, window_id, detection_index=0) -> FittedPeak:
    return FittedPeak(
        detection_index=detection_index,
        frequency_mhz=freq,
        amplitude=1.0,
        phase=0.1,
        frequency_error=0.0002,
        snr=50.0,
        window_id=window_id,
        origin="auto",
    )


def _window_fit(
    window_id,
    freq_range,
    *,
    tau=0.9,
    tau_error=0.02,
    tau_fitted=True,
    shape="gaussian",
) -> FittingResult:
    freqs = np.linspace(freq_range[0], freq_range[1], 8)
    window = SpectralWindow(
        None,
        freqs,
        np.zeros(8, dtype=complex),
        (float(freq_range[0]), float(freq_range[1])),
        window_id=window_id,
    )
    wf = FittingResult(success=True, window=window, window_id=window_id, shape=shape)
    wf.set_shared_parameter("tau_us", tau, tau_error, detection_indices=[0])
    wf.shared_parameters["tau_us"]["fitted"] = tau_fitted
    return wf


def _fit(window_fits, peaks, *, acquisition_us=ACQ_US) -> SpectrumFit:
    parameters = {} if acquisition_us is None else {"acquisition_us": acquisition_us}
    return SpectrumFit(
        window_fits=list(window_fits), fitted_peaks=list(peaks), parameters=parameters
    )


def _build(fit, *, epsilon=0.0, state="rb_locked"):
    return _build_final_products(
        fit,
        probe_freq_mhz=PROBE,
        sideband=Sideband.LOWER,
        calibration_state=state,
        epsilon=epsilon,
        sigma_epsilon=0.0,
        sigma_floor_khz=0.0,
        clocks_declared=False,
    )


def _calibrate(f, eps):
    return PROBE + (f - PROBE) / (1.0 + eps)


# ---------------------------------------------------------------------------
# The join from the window's fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian"])
def test_fields_are_joined_from_the_window_fit(shape):
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0), tau=0.9, tau_error=0.02, shape=shape)],
        [_peak(30000.0, 3, detection_index=7)],
    )
    p = _build(fit).peaks[0]
    assert p.decay_time_us == 0.9
    assert p.decay_time_error_us == 0.02
    assert p.shape == shape
    assert p.detection_index == 7
    assert p.fit_window_mhz == (29990.0, 30010.0)
    assert not isinstance(p.fwhm_mhz, Absent)


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian"])
def test_fwhm_is_exactly_the_feature_fwhm_call(shape):
    """fwhm_mhz == feature_fwhm(tau, acquisition_us, shape), bit for bit, with
    the record length the fit recorded (the ``stage5.acquisition_us`` value)."""
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0), tau=1.3, shape=shape)],
        [_peak(30000.0, 3)],
        acquisition_us=ACQ_US,
    )
    p = _build(fit).peaks[0]
    assert p.fwhm_mhz == feature_fwhm(1.3, ACQ_US, shape=shape)


def test_fwhm_follows_the_recorded_record_length():
    short = _build(
        _fit([_window_fit(3, (1.0, 2.0))], [_peak(1.5, 3)], acquisition_us=5.0)
    )
    long = _build(
        _fit([_window_fit(3, (1.0, 2.0))], [_peak(1.5, 3)], acquisition_us=50.0)
    )
    assert short.peaks[0].fwhm_mhz == feature_fwhm(0.9, 5.0, shape="gaussian")
    assert long.peaks[0].fwhm_mhz == feature_fwhm(0.9, 50.0, shape="gaussian")
    assert short.peaks[0].fwhm_mhz != long.peaks[0].fwhm_mhz


def test_lines_of_one_window_share_its_fit_and_keep_their_own_index():
    fit = _fit(
        [
            _window_fit(3, (29990.0, 30010.0)),
            _window_fit(4, (31000.0, 31010.0), tau=2.0),
        ],
        [
            _peak(29995.0, 3, detection_index=1),
            _peak(30005.0, 3, detection_index=2),
            _peak(31005.0, 4, detection_index=9),
        ],
    )
    a, b, c = _build(fit).peaks
    assert a.decay_time_us == b.decay_time_us == 0.9
    assert a.fit_window_mhz == b.fit_window_mhz == (29990.0, 30010.0)
    assert (a.detection_index, b.detection_index) == (1, 2)
    assert c.decay_time_us == 2.0
    assert c.fit_window_mhz == (31000.0, 31010.0)
    assert c.detection_index == 9


# ---------------------------------------------------------------------------
# Absent
# ---------------------------------------------------------------------------


def test_tau_error_is_undefined_when_tau_was_held_fixed():
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0), tau=0.9, tau_error=None, tau_fitted=False)],
        [_peak(30000.0, 3)],
    )
    p = _build(fit).peaks[0]
    assert p.decay_time_error_us is Absent.UNDEFINED
    # tau itself, and everything derived from it, is still known
    assert p.decay_time_us == 0.9
    assert p.fwhm_mhz == feature_fwhm(0.9, ACQ_US, shape="gaussian")


def test_held_fixed_tau_error_is_undefined_even_if_an_error_value_is_stored():
    """The held-fixed rule is about the fit, not about whether a number sits in
    the record: a fixed tau has no 1-sigma error."""
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0), tau_error=0.5, tau_fitted=False)],
        [_peak(30000.0, 3)],
    )
    assert _build(fit).peaks[0].decay_time_error_us is Absent.UNDEFINED


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_unusable_tau_error_is_undefined_never_nan_or_none(bad):
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0), tau_error=bad, tau_fitted=True)],
        [_peak(30000.0, 3)],
    )
    assert _build(fit).peaks[0].decay_time_error_us is Absent.UNDEFINED


@pytest.mark.parametrize("window_id", [None, 99])
def test_a_line_with_no_fit_record_has_every_field_undefined(window_id):
    """No window id, or a window id no Stage 5 fit record carries."""
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, window_id)])
    p = _build(fit).peaks[0]
    for name in NEW_FIELDS:
        assert getattr(p, name) is Absent.UNDEFINED, name


def test_a_fit_with_no_windows_at_all_gives_undefined_fields():
    p = _build(_fit([], [_peak(30000.0, 3)])).peaks[0]
    assert all(getattr(p, n) is Absent.UNDEFINED for n in NEW_FIELDS)


def test_no_line_of_a_missing_record_borrows_a_neighbours_fit():
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, 3), _peak(5.0, 8)])
    a, b = _build(fit).peaks
    assert a.decay_time_us == 0.9
    assert all(getattr(b, n) is Absent.UNDEFINED for n in NEW_FIELDS)


def test_detection_index_minus_one_is_never_reported_as_an_index():
    """Stage 5 writes -1 for a line no Stage 3 detection seeded; the contract
    forbids a sentinel as an absence marker."""
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0))],
        [_peak(30000.0, 3, detection_index=-1)],
    )
    p = _build(fit).peaks[0]
    assert p.detection_index is Absent.UNDEFINED
    assert p.decay_time_us == 0.9  # the rest of the join is unaffected


def test_unrecorded_acquisition_makes_fwhm_absent_not_a_guess():
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0))],
        [_peak(30000.0, 3)],
        acquisition_us=None,
    )
    p = _build(fit).peaks[0]
    assert isinstance(p.fwhm_mhz, Absent)
    assert p.decay_time_us == 0.9


def test_new_fields_default_to_undefined_on_a_directly_built_peak():
    base = _build(_fit([], [_peak(30000.0, 3)])).peaks[0]
    fields = {
        f.name: getattr(base, f.name)
        for f in dataclasses.fields(FinalPeak)
        if f.name not in NEW_FIELDS
    }
    bare = FinalPeak(**fields)
    assert all(getattr(bare, n) is Absent.UNDEFINED for n in NEW_FIELDS)


# ---------------------------------------------------------------------------
# Frame of fit_window_mhz
# ---------------------------------------------------------------------------


def test_fit_window_is_in_the_calibrated_frame_like_frequency():
    eps = 2.19e-6
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0))],
        [_peak(29990.0, 3), _peak(30010.0, 3)],
    )
    lo_line, hi_line = _build(fit, epsilon=eps, state="self_calibrated").peaks
    lo, hi = lo_line.fit_window_mhz
    assert lo == pytest.approx(_calibrate(29990.0, eps), abs=1e-9)
    assert hi == pytest.approx(_calibrate(30010.0, eps), abs=1e-9)
    # A line sitting on a raw window edge sits on the calibrated edge: the same
    # correction moved both.
    assert lo_line.frequency_mhz == pytest.approx(lo, abs=1e-9)
    assert hi_line.frequency_mhz == pytest.approx(hi, abs=1e-9)
    # ... and the correction is large enough to tell the frames apart.
    assert abs(lo - 29990.0) > 1e-3
    assert lo < hi


def test_fit_window_equals_raw_bounds_when_epsilon_is_zero():
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, 3)])
    assert _build(fit, epsilon=0.0).peaks[0].fit_window_mhz == (29990.0, 30010.0)


def test_calibrated_fit_window_still_contains_its_lines():
    eps = -3.0e-6
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0))],
        [_peak(29991.0, 3), _peak(30009.0, 3)],
    )
    for p in _build(fit, epsilon=eps, state="self_calibrated").peaks:
        lo, hi = p.fit_window_mhz
        assert lo <= p.frequency_mhz <= hi


def test_fit_window_is_an_ordered_pair_of_floats():
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, 3)])
    bounds = _build(fit).peaks[0].fit_window_mhz
    assert isinstance(bounds, tuple) and len(bounds) == 2
    assert all(isinstance(b, float) for b in bounds)
    assert bounds[0] < bounds[1]


# ---------------------------------------------------------------------------
# Persistence: Absent is stored losslessly
# ---------------------------------------------------------------------------


def _roundtrip(products, tmp_path):
    out = tmp_path / "review.h5"
    with h5py.File(out, "w") as h5f:
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(Stage6Review(final_products=products), grp)
    with h5py.File(out, "r") as h5f:
        loaded = load_stage6_review_from_hdf5(h5f["stage6_review"])
    assert loaded.final_products is not None
    return out, loaded.final_products


def test_present_fields_roundtrip_exactly(tmp_path):
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0), tau=0.912345678901, tau_error=0.0123)],
        [_peak(30000.0, 3, detection_index=4)],
    )
    products = _build(fit, epsilon=1.5e-6, state="self_calibrated")
    _, loaded = _roundtrip(products, tmp_path)
    assert loaded.peaks == products.peaks
    assert isinstance(loaded.peaks[0].fit_window_mhz, tuple)


def test_each_absence_reason_roundtrips_distinctly(tmp_path):
    """NOT_RUN and UNDEFINED stay different through the file, per field."""
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, 3)])
    base = _build(fit).peaks[0]
    peaks = [
        dataclasses.replace(base, **{name: Absent.NOT_RUN}) for name in NEW_FIELDS
    ] + [dataclasses.replace(base, **{name: Absent.UNDEFINED}) for name in NEW_FIELDS]
    products = dataclasses.replace(_build(fit), peaks=peaks)
    _, loaded = _roundtrip(products, tmp_path)
    for i, name in enumerate(NEW_FIELDS):
        assert getattr(loaded.peaks[i], name) is Absent.NOT_RUN, name
        assert getattr(loaded.peaks[len(NEW_FIELDS) + i], name) is Absent.UNDEFINED
    assert loaded.peaks == peaks


def test_absent_is_stored_as_a_value_plus_a_status_code(tmp_path):
    """The columnar status convention: 0 present, 1 not run, 2 undefined."""
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, 3)])
    base = _build(fit).peaks[0]
    peaks = [
        base,
        dataclasses.replace(base, fwhm_mhz=Absent.NOT_RUN),
        dataclasses.replace(base, decay_time_error_us=Absent.UNDEFINED),
    ]
    out, _ = _roundtrip(dataclasses.replace(_build(fit), peaks=peaks), tmp_path)
    with h5py.File(out, "r") as h5f:
        rows = json.loads(h5f["stage6_review/final_products"].attrs["data"])["peaks"]
    assert rows[0]["fwhm_mhz__status"] == 0
    assert rows[0]["fwhm_mhz"] == base.fwhm_mhz
    assert rows[1]["fwhm_mhz__status"] == 1 and rows[1]["fwhm_mhz"] is None
    assert rows[2]["decay_time_error_us__status"] == 2
    assert rows[2]["decay_time_error_us"] is None
    for name in NEW_FIELDS:
        assert f"{name}__status" in rows[0]


def test_saved_table_is_stamped_with_the_field_version(tmp_path):
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, 3)])
    out, _ = _roundtrip(_build(fit), tmp_path)
    with h5py.File(out, "r") as h5f:
        stamp = h5f["stage6_review/final_products"].attrs["peak_fields"]
        assert int(stamp) == FINAL_PEAK_FIELDS_VERSION
        assert not final_products_predate_fit_fields(h5f["stage6_review"])


def _stored_table(tmp_path, *, stamp="current", strip_keys=False):
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, 3)])
    out = tmp_path / "stored.h5"
    with h5py.File(out, "w") as h5f:
        grp = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(Stage6Review(final_products=_build(fit)), grp)
        fp = grp["final_products"]
        if stamp is None:
            del fp.attrs["peak_fields"]
        elif stamp != "current":
            fp.attrs["peak_fields"] = stamp
        if strip_keys:
            data = json.loads(fp.attrs["data"])
            for row in data["peaks"]:
                for name in NEW_FIELDS:
                    row.pop(name, None)
                    row.pop(f"{name}__status", None)
            fp.attrs["data"] = json.dumps(data)
    return out


def test_predate_detection_none_group_and_no_table_are_not_old(tmp_path):
    assert final_products_predate_fit_fields(None) is False
    out = tmp_path / "empty.h5"
    with h5py.File(out, "w") as h5f:
        grp = h5f.create_group("stage6_review")
        assert final_products_predate_fit_fields(grp) is False
        save_stage6_review_to_hdf5(Stage6Review(), grp)
        assert final_products_predate_fit_fields(grp) is False


def test_predate_detection_current_stamp_is_not_old(tmp_path):
    out = _stored_table(tmp_path)
    with h5py.File(out, "r") as h5f:
        assert final_products_predate_fit_fields(h5f["stage6_review"]) is False


@pytest.mark.parametrize("stamp", [None, 1])
def test_predate_detection_unstamped_or_older_stamp_is_old(tmp_path, stamp):
    out = _stored_table(tmp_path, stamp=stamp)
    with h5py.File(out, "r") as h5f:
        assert final_products_predate_fit_fields(h5f["stage6_review"]) is True


def test_predate_detection_never_writes(tmp_path):
    out = _stored_table(tmp_path, stamp=None)
    before = out.read_bytes()
    with h5py.File(out, "r") as h5f:
        final_products_predate_fit_fields(h5f["stage6_review"])
    assert out.read_bytes() == before


def test_a_legacy_table_loads_with_every_new_field_undefined(tmp_path):
    """A table written before the fields existed (no stamp, no keys) loads
    without error; nothing is invented -- the reader rebuilds it."""
    out = _stored_table(tmp_path, stamp=None, strip_keys=True)
    with h5py.File(out, "r") as h5f:
        loaded = load_stage6_review_from_hdf5(h5f["stage6_review"])
    peak = loaded.final_products.peaks[0]
    assert peak.frequency_mhz == 30000.0
    for name in NEW_FIELDS:
        assert getattr(peak, name) is Absent.UNDEFINED, name


# ---------------------------------------------------------------------------
# Wire form
# ---------------------------------------------------------------------------


def test_to_jsonable_present_fields_and_pair_as_array():
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, 3, 5)])
    row = json.loads(json.dumps(to_jsonable(_build(fit).peaks[0])))
    assert row["decay_time_us"] == 0.9
    assert row["decay_time_error_us"] == 0.02
    assert row["shape"] == "gaussian"
    assert row["detection_index"] == 5
    assert row["fit_window_mhz"] == [29990.0, 30010.0]
    assert row["fwhm_mhz"] == feature_fwhm(0.9, ACQ_US, shape="gaussian")
    for name in NEW_FIELDS:
        assert f"{name}_absent" not in row


def test_to_jsonable_absent_fields_are_null_plus_reason_sibling():
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0), tau_error=None, tau_fitted=False)],
        [_peak(30000.0, 3), _peak(5.0, None)],
    )
    fixed, orphan = (json.loads(json.dumps(to_jsonable(p))) for p in _build(fit).peaks)
    assert fixed["decay_time_error_us"] is None
    assert fixed["decay_time_error_us_absent"] == "undefined"
    for name in NEW_FIELDS:
        assert orphan[name] is None
        assert orphan[f"{name}_absent"] == "undefined"


def test_manifest_declares_the_new_fields_and_the_type_carries_them():
    declared = set(MANIFEST.fields["FinalPeak"])
    assert set(NEW_FIELDS) <= declared
    carried = {f.name for f in dataclasses.fields(FinalPeak)}
    assert declared <= carried


# ---------------------------------------------------------------------------
# CSV / HTML renderings
# ---------------------------------------------------------------------------

CSV_NEW = [
    "decay_time_us",
    "decay_time_error_us",
    "shape",
    "fwhm_mhz",
    "detection_index",
    "fit_window_low_mhz",
    "fit_window_high_mhz",
]


def _csv_for(peak):
    cols = list(report_impl._CSV_COLUMNS)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    w.writerow(report_impl._csv_row(peak, 1.0))
    buf.seek(0)
    return cols, next(csv.DictReader(buf))


def test_csv_appends_the_columns_after_the_existing_ones():
    cols = report_impl._CSV_COLUMNS
    assert cols[-len(CSV_NEW) :] == CSV_NEW
    assert len(set(cols)) == len(cols)
    assert cols[-len(CSV_NEW) - 1] == "peak_uid"  # nothing existing was moved


def test_csv_row_is_as_wide_as_the_header_and_renders_values():
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, 3, 6)])
    peak = _build(fit).peaks[0]
    cols, row = _csv_for(peak)
    assert len(report_impl._csv_row(peak, 1.0)) == len(cols)
    assert float(row["decay_time_us"]) == pytest.approx(0.9)
    assert float(row["decay_time_error_us"]) == pytest.approx(0.02)
    assert row["shape"] == "gaussian"
    assert float(row["fwhm_mhz"]) == pytest.approx(peak.fwhm_mhz, rel=1e-5)
    assert row["detection_index"] == "6"
    assert float(row["fit_window_low_mhz"]) == pytest.approx(29990.0)
    assert float(row["fit_window_high_mhz"]) == pytest.approx(30010.0)


def test_csv_renders_absent_as_an_empty_cell_like_other_missing_values():
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0), tau_error=None, tau_fitted=False)],
        [_peak(30000.0, 3), _peak(5.0, None)],
    )
    fixed, orphan = _build(fit).peaks
    _, row = _csv_for(fixed)
    assert row["decay_time_error_us"] == ""
    assert row["decay_time_us"] != ""
    _, row = _csv_for(orphan)
    assert all(row[c] == "" for c in CSV_NEW)
    # a missing value elsewhere in the row is rendered the same way
    assert row["derivation"] == ""


def test_json_report_renders_absent_as_null():
    fit = _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(5.0, None)])
    products = _build(fit)
    payload = report_impl._peak_json(products.peaks[0], 1.0, None, with_catalog=False)
    for name in NEW_FIELDS:
        assert payload[name] is None, name


def test_html_cells_show_tau_with_error_or_plain_when_fixed():
    free = _build(
        _fit([_window_fit(3, (29990.0, 30010.0))], [_peak(30000.0, 3, 2)])
    ).peaks[0]
    fixed = _build(
        _fit(
            [_window_fit(3, (29990.0, 30010.0), tau_error=None, tau_fitted=False)],
            [_peak(30000.0, 3)],
        )
    ).peaks[0]
    free_cells = html_impl._fit_field_cells(free)
    fixed_cells = html_impl._fit_field_cells(fixed)
    assert len(free_cells) == len(html_impl._FIT_FIELD_HEAD) == len(fixed_cells)
    assert "(" in free_cells[0]
    assert "(" not in fixed_cells[0] and fixed_cells[0] != ""
    assert free_cells[1] == "gaussian"
    assert float(free_cells[2]) == pytest.approx(free.fwhm_mhz * 1e3, rel=1e-3)
    assert free_cells[3] == "2"
    assert "29990" in free_cells[4] and "30010" in free_cells[4]


def test_html_cells_for_a_line_without_a_fit_record_are_empty():
    orphan = _build(_fit([], [_peak(30000.0, None)])).peaks[0]
    assert html_impl._fit_field_cells(orphan) == [""] * len(html_impl._FIT_FIELD_HEAD)


def test_html_tables_have_matching_header_and_row_widths_and_curation_last():
    fit = _fit(
        [_window_fit(3, (29990.0, 30010.0))],
        [_peak(29995.0, 3), _peak(30005.0, 3)],
    )
    products = _build(fit)

    index = html_impl._index_final_table(products)
    assert index.count("<th>") == 7 + len(html_impl._FIT_FIELD_HEAD)
    assert index.count("<td>") == 2 * index.count("<th>")
    assert "FWHM (kHz)" in index

    table = html_impl._window_peak_table(products.peaks, "mV", 1.0, window_id=3)
    n_head = table.count("<th>")
    assert table.count("<td>") == 2 * n_head
    # the curation column stays the last header, after the new columns
    heads = [h.split("</th>")[0] for h in table.split("<th>")[1:]]
    assert "cur-col-h" in heads[-1]
    assert all("cur-col-h" not in h for h in heads[:-1])
    assert heads.index("FWHM (kHz)") < len(heads) - 1


def test_nothing_in_the_fields_is_nan():
    fit = _fit(
        [
            _window_fit(3, (29990.0, 30010.0)),
            _window_fit(4, (31000.0, 31010.0), tau_error=float("nan")),
        ],
        [_peak(30000.0, 3), _peak(31005.0, 4)],
    )
    for p in _build(fit).peaks:
        for name in NEW_FIELDS:
            v = getattr(p, name)
            assert v is not None
            if isinstance(v, float):
                assert math.isfinite(v)
