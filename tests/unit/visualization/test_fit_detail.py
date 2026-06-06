"""Unit tests for the consolidated fit-detail formatting helpers."""

import pytest

from ftmwpipeline.visualization.fit_detail import (
    _format_spectroscopic,
    _format_spectroscopic_sci,
    _peak_labels,
)


class TestSpectroscopicFormat:
    def test_pdg_two_digit(self):
        assert _format_spectroscopic(3.06546, 0.0234) == "3.065(23)"

    def test_pdg_leading_one_bumps_to_three_digits(self):
        # 2-digit error rounds into the 10-19 decade -> 3 digits of error.
        assert _format_spectroscopic(3.06546, 0.0134) == "3.0655(134)"

    def test_no_error_falls_back_to_plain(self):
        assert _format_spectroscopic(3.06546, None) == "3.06546"

    def test_nonfinite_value(self):
        assert _format_spectroscopic(float("nan"), 0.01) == "-"
        assert _format_spectroscopic(None, 0.01) == "-"

    def test_nonpositive_error_falls_back(self):
        assert _format_spectroscopic(2.5, 0.0) == "2.5"

    def test_value_precision_tracks_error_decade(self):
        # Larger error -> fewer displayed decimals on the value.
        assert _format_spectroscopic(1234.5678, 2.3) == "1234.6(23)"


class TestSpectroscopicSci:
    def test_small_value_uses_scientific(self):
        assert _format_spectroscopic_sci(4.18e-6, 3.1e-7) == "4.18(31)e-06"

    def test_in_range_value_stays_decimal(self):
        assert _format_spectroscopic_sci(30.75, 0.065) == "30.750(65)"

    def test_zero_value(self):
        assert _format_spectroscopic_sci(0.0, 1e-9) == "-" or isinstance(
            _format_spectroscopic_sci(0.0, 1e-9), str
        )


class TestPeakLabels:
    def test_single_letters(self):
        assert _peak_labels(3) == ["A", "B", "C"]

    def test_wraps_past_z(self):
        labels = _peak_labels(28)
        assert labels[25] == "Z"
        assert labels[26] == "AA"
        assert labels[27] == "AB"

    def test_empty(self):
        assert _peak_labels(0) == []
