"""Unit tests for the shared catalog cross-reference helper.

Covers the format-light reader (CSV / whitespace / header / comments), the one
shared tolerance/match test, and the pull-distribution tally that backs the
calibration surface.
"""

from __future__ import annotations

import math

import pytest

from ftmwpipeline._internal.catalog_xref import (
    CatalogEntry,
    build_cross_ref,
    load_cross_ref,
    match_peak,
    read_catalog,
)
from ftmwpipeline.core.data_structures import FinalPeak


def _peak(freq, sigma_f_khz=1.0) -> FinalPeak:
    return FinalPeak(
        frequency_mhz=freq,
        frequency_raw_mhz=freq,
        f_baseband_mhz=abs(40960.0 - freq),
        sigma_f_khz=sigma_f_khz,
        sigma_stat_khz=sigma_f_khz,
        sigma_eps_khz=0.0,
        sigma_floor_khz=0.0,
        amplitude=1.0e-6,
        snr=50.0,
    )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def test_read_csv_with_header(tmp_path):
    f = tmp_path / "cat.csv"
    f.write_text(
        "# my catalog\n"
        "frequency_mhz,uncertainty_khz,label\n"
        "30000.0,2.0,line-A\n"
        "39000.5,,line-B\n"
    )
    cat = read_catalog(f)
    assert len(cat) == 2
    assert cat[0] == CatalogEntry(30000.0, 2.0, "line-A")
    assert cat[1].frequency_mhz == 39000.5
    assert cat[1].sigma_khz is None
    assert cat[1].label == "line-B"


def test_read_positional_no_header(tmp_path):
    f = tmp_path / "cat.csv"
    f.write_text("30000.0,2.0,A\n39000.5,3.0,B\n")
    cat = read_catalog(f)
    assert [e.label for e in cat] == ["A", "B"]
    assert cat[0].sigma_khz == 2.0


def test_read_frequency_only_synthesizes_label(tmp_path):
    f = tmp_path / "cat.txt"
    f.write_text("30000.0\n39000.5\n")
    cat = read_catalog(f)
    assert cat[0].sigma_khz is None
    assert cat[0].label == "30000.0000"


def test_read_whitespace_label_with_spaces(tmp_path):
    f = tmp_path / "cat.txt"
    f.write_text("30000.0 2.0 J=1-0 K=0\n")
    cat = read_catalog(f)
    assert cat[0].label == "J=1-0 K=0"


def test_read_header_keyword_columns_out_of_order(tmp_path):
    f = tmp_path / "cat.csv"
    f.write_text("species,freq_MHz,sigma_kHz\nfoo,30000.0,1.5\n")
    cat = read_catalog(f)
    assert cat[0].frequency_mhz == 30000.0
    assert cat[0].sigma_khz == 1.5
    assert cat[0].label == "foo"


def test_header_freq_outranks_label_substring(tmp_path):
    # "transition_frequency_mhz" carries both a label keyword ("transition") and a
    # frequency keyword; frequency must win so the column is read as the frequency.
    f = tmp_path / "cat.csv"
    f.write_text("transition_frequency_mhz,name\n30000.0,xyz\n")
    cat = read_catalog(f)
    assert cat[0].frequency_mhz == 30000.0
    assert cat[0].label == "xyz"


def test_header_short_label_keyword_no_substring_misfire(tmp_path):
    # The two-character "id" keyword must not fire inside "midpoint"; the
    # frequency column stays correctly identified.
    f = tmp_path / "cat.csv"
    f.write_text("freq_mhz,midpoint,name\n30000.0,1.0,abc\n")
    cat = read_catalog(f)
    assert cat[0].frequency_mhz == 30000.0
    assert cat[0].label == "abc"


def test_read_empty_file(tmp_path):
    f = tmp_path / "cat.csv"
    f.write_text("# only comments\n\n")
    assert read_catalog(f) == []


def test_read_csv_uncertainty_unit_from_header(tmp_path):
    # A *_mhz uncertainty header is read as MHz and converted to kHz.
    f = tmp_path / "cat.csv"
    f.write_text("freq_mhz,unc_mhz,quantum_numbers\n27520.0037,0.0021,14 2 1\n")
    cat = read_catalog(f)
    assert cat[0].sigma_khz == pytest.approx(2.1)  # 0.0021 MHz -> 2.1 kHz
    assert cat[0].label == "14 2 1"  # quantum_numbers detected as the label


def test_read_csv_khz_header_unscaled(tmp_path):
    f = tmp_path / "cat.csv"
    f.write_text("frequency_mhz,uncertainty_khz,label\n30000.0,2.0,A\n")
    assert read_catalog(f)[0].sigma_khz == pytest.approx(2.0)


def test_read_spcat_cat_file(tmp_path):
    # Pickett/SPCAT fixed-width: FREQ F13.4, ERR F8.4 (MHz), ... QN tail.
    f = tmp_path / "pred.cat"
    f.write_text(
        "    9234.8712  0.0002 -6.9951 3   -0.0000  3  54506 304 1 0 1 1     "
        "0 0 0 1     \n"
        "   18031.6567  0.0003 -6.3667 3    1.7876  5  54506 304 2 1 2 2     "
        "1 1 1 1     \n"
    )
    cat = read_catalog(f)
    assert len(cat) == 2
    assert cat[0].frequency_mhz == pytest.approx(9234.8712)
    assert cat[0].sigma_khz == pytest.approx(0.2)  # 0.0002 MHz -> 0.2 kHz
    # Opaque label = species tag + upper/lower quantum numbers.
    assert cat[0].label == "54506: 1 0 1 1 <- 0 0 0 1"


def test_read_spcat_skips_nonnumeric_lines(tmp_path):
    f = tmp_path / "pred.cat"
    f.write_text(
        "header junk that is not a catalog row\n"
        "    9234.8712  0.0002 -6.9951 3   -0.0000  3  54506 304 1 0 1 1     "
        "0 0 0 1     \n"
    )
    cat = read_catalog(f)
    assert len(cat) == 1 and cat[0].frequency_mhz == pytest.approx(9234.8712)


# ---------------------------------------------------------------------------
# Match test
# ---------------------------------------------------------------------------


def test_match_within_tolerance():
    cat = [CatalogEntry(30000.0, 0.0, "A")]
    # 2 kHz away, sigma_f = 1 kHz, N=3 -> tol = 3 kHz -> matches.
    m = match_peak(30000.002, 1.0, cat, n_sigma=3.0)
    assert m is not None
    assert m.label == "A"
    assert m.delta_khz == pytest.approx(2.0, abs=1e-6)
    assert m.pull == pytest.approx(2.0, abs=1e-6)


def test_match_outside_tolerance_returns_none():
    cat = [CatalogEntry(30000.0, 0.0, "A")]
    # 4 kHz away, tol = 3 kHz -> no match.
    assert match_peak(30000.004, 1.0, cat, n_sigma=3.0) is None


def test_match_combines_catalog_uncertainty():
    cat = [CatalogEntry(30000.0, 4.0, "A")]
    # delta 4 kHz, combined sigma = sqrt(1+16) ~ 4.123, N=1 -> tol 4.123 -> matches.
    m = match_peak(30000.004, 1.0, cat, n_sigma=1.0)
    assert m is not None
    assert m.combined_sigma_khz == pytest.approx(math.hypot(1.0, 4.0))
    assert m.pull == pytest.approx(4.0 / math.hypot(1.0, 4.0))


def test_match_picks_nearest():
    cat = [CatalogEntry(30000.0, 0.0, "A"), CatalogEntry(30000.010, 0.0, "B")]
    m = match_peak(30000.009, 5.0, cat, n_sigma=3.0)
    assert m is not None and m.label == "B"


def test_match_zero_combined_sigma_requires_exact():
    cat = [CatalogEntry(30000.0, 0.0, "A")]
    assert match_peak(30000.001, 0.0, cat, n_sigma=3.0) is None
    m = match_peak(30000.0, 0.0, cat, n_sigma=3.0)
    assert m is not None and math.isnan(m.pull)


# ---------------------------------------------------------------------------
# Cross-ref + pull stats
# ---------------------------------------------------------------------------


def test_build_cross_ref_match_rate_and_pull():
    cat = [CatalogEntry(30000.0, 0.0, "A"), CatalogEntry(39000.0, 0.0, "B")]
    peaks = [
        _peak(30000.001),  # +1 kHz -> matches A, pull +1
        _peak(39000.0),  # exact -> matches B, pull 0
        _peak(35000.0),  # nothing nearby -> no match
    ]
    xref = build_cross_ref(peaks, cat, catalog_path="cat.csv", n_sigma=3.0)
    assert xref.n_total == 3
    assert xref.n_matched == 2
    assert xref.match_rate == pytest.approx(2 / 3)
    assert xref.matches[2] is None
    assert xref.pull_mean == pytest.approx(0.5, abs=1e-6)
    assert xref.pull_std is not None  # two finite pulls -> sample std defined


def test_load_cross_ref_none_path():
    assert load_cross_ref([_peak(30000.0)], None, 3.0) is None


def test_load_cross_ref_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_cross_ref([_peak(30000.0)], tmp_path / "nope.csv", 3.0)


def test_load_cross_ref_empty_catalog(tmp_path):
    f = tmp_path / "cat.csv"
    f.write_text("# nothing\n")
    with pytest.raises(ValueError, match="zero entries"):
        load_cross_ref([_peak(30000.0)], f, 3.0)
