"""
Pure unit tests for FTSettings, resolve(), and argspec helpers.

No real data required; all tests are fast and free of I/O.
"""

import argparse
import pytest

from ftmwpipeline.core.settings import (
    FTSettings,
    resolve,
)
from ftmwpipeline.cli._argspec import add_settings_args, settings_from_namespace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_settings_args(p)
    return p


# ---------------------------------------------------------------------------
# is_empty / overrides
# ---------------------------------------------------------------------------

class TestIsEmptyAndOverrides:
    def test_default_instance_is_empty(self):
        assert FTSettings().is_empty()

    def test_single_field_set_not_empty(self):
        s = FTSettings(zpf=2)
        assert not s.is_empty()

    def test_overrides_returns_only_set_fields(self):
        s = FTSettings(zpf=2, expf_us=5.0)
        ov = s.overrides()
        assert ov == {"zpf": 2, "expf_us": 5.0}

    def test_overrides_empty_when_no_fields_set(self):
        assert FTSettings().overrides() == {}

    def test_trim_in_overrides(self):
        s = FTSettings(trim=(100.0, 200.0))
        assert "trim" in s.overrides()
        assert s.overrides()["trim"] == (100.0, 200.0)


# ---------------------------------------------------------------------------
# resolve() precedence
# ---------------------------------------------------------------------------

class TestResolve:
    def test_explicit_beats_persisted_beats_recommended(self):
        explicit = FTSettings(zpf=4)
        persisted = FTSettings(zpf=2, expf_us=3.0)
        recommended = FTSettings(zpf=1, expf_us=7.0, units_power=3)

        result = resolve(explicit, persisted, recommended)

        # zpf: explicit wins
        assert result.zpf == 4
        # expf_us: explicit absent → persisted wins
        assert result.expf_us == 3.0
        # units_power: only recommended set → recommended wins
        assert result.units_power == 3

    def test_hard_defaults_fill_run_critical_fields(self):
        result = resolve(None, None, None)
        assert result.zpf == 1
        assert result.expf_us == 5.0
        assert result.units_power == 6
        assert result.rdc is True

    def test_optional_fields_stay_none_when_unset(self):
        result = resolve(None, None, None)
        assert result.start_us is None
        assert result.end_us is None
        assert result.window_function is None
        assert result.trim is None

    def test_explicit_none_falls_through_to_persisted(self):
        explicit = FTSettings(zpf=None)
        persisted = FTSettings(zpf=3)
        result = resolve(explicit, persisted, None)
        assert result.zpf == 3

    def test_trim_propagated_from_explicit(self):
        explicit = FTSettings(trim=(26500.0, 40000.0))
        result = resolve(explicit, None, None)
        assert result.trim == (26500.0, 40000.0)

    def test_trim_propagated_from_persisted_when_explicit_absent(self):
        persisted = FTSettings(trim=(30000.0, 35000.0))
        result = resolve(FTSettings(), persisted, None)
        assert result.trim == (30000.0, 35000.0)

    def test_none_layers_treated_as_empty(self):
        """resolve(None, None, None) must not raise."""
        result = resolve(None, None, None)
        assert result.zpf == 1

    def test_field_by_field_precedence_exhaustive(self):
        """Every field independently follows explicit > persisted > recommended."""
        e = FTSettings(zpf=10)
        p = FTSettings(zpf=20, expf_us=2.0, start_us=1.0)
        r = FTSettings(zpf=30, expf_us=4.0, start_us=2.0, units_power=5)

        result = resolve(e, p, r)

        assert result.zpf == 10        # explicit
        assert result.expf_us == 2.0   # persisted (explicit absent)
        assert result.start_us == 1.0  # persisted (explicit absent)
        assert result.units_power == 5 # recommended (only one set)

    def test_rdc_default_true(self):
        result = resolve(None, None, None)
        assert result.rdc is True

    def test_rdc_explicit_overrides_default(self):
        explicit = FTSettings(rdc=False)
        result = resolve(explicit, None, None)
        assert result.rdc is False


# ---------------------------------------------------------------------------
# to_attrs / from_attrs round-trips
# ---------------------------------------------------------------------------

class TestToAttrsFromAttrs:
    def test_roundtrip_with_trim_set(self):
        s = FTSettings(zpf=2, expf_us=5.0, trim=(26500.0, 40000.0), units_power=6, rdc=True)
        attrs = s.to_attrs()
        restored = FTSettings.from_attrs(attrs)

        assert restored.zpf == 2
        assert restored.expf_us == 5.0
        assert restored.trim == (26500.0, 40000.0)
        assert restored.units_power == 6
        assert restored.rdc is True

    def test_roundtrip_trim_none_uses_none_marker(self):
        s = FTSettings(zpf=1)
        attrs = s.to_attrs()

        # None serialized to __None__ marker
        assert attrs["trim_min_mhz"] == "__None__"
        assert attrs["trim_max_mhz"] == "__None__"

        # and restores to None
        restored = FTSettings.from_attrs(attrs)
        assert restored.trim is None

    def test_roundtrip_window_function_none(self):
        s = FTSettings(zpf=1)
        attrs = s.to_attrs()
        assert attrs["window_function"] == "__None__"

        restored = FTSettings.from_attrs(attrs)
        assert restored.window_function is None

    def test_roundtrip_window_function_set(self):
        s = FTSettings(window_function="hann")
        attrs = s.to_attrs()
        assert attrs["window_function"] == "hann"

        restored = FTSettings.from_attrs(attrs)
        assert restored.window_function == "hann"

    def test_from_attrs_accepts_winf_alias(self):
        """Recommended-style dicts use key 'winf'; from_attrs must accept it."""
        attrs = {"winf": "blackman", "zpf": 1, "expf_us": 5.0}
        s = FTSettings.from_attrs(attrs)
        assert s.window_function == "blackman"

    def test_from_attrs_window_function_key_preferred_over_winf(self):
        """When both 'window_function' and 'winf' present, 'window_function' wins."""
        attrs = {"window_function": "hann", "winf": "blackman"}
        s = FTSettings.from_attrs(attrs)
        assert s.window_function == "hann"

    def test_from_attrs_tolerant_of_missing_keys(self):
        """Sparse dicts (e.g. old recommended records) do not raise."""
        s = FTSettings.from_attrs({"zpf": 2})
        assert s.zpf == 2
        assert s.expf_us is None
        assert s.trim is None

    def test_from_attrs_bytes_values_decoded(self):
        """HDF5 sometimes returns byte strings; from_attrs must handle them."""
        attrs = {"zpf": b"2", "expf_us": b"5.0"}
        s = FTSettings.from_attrs(attrs)
        assert s.zpf == 2
        assert s.expf_us == 5.0

    def test_roundtrip_all_fields_set(self):
        s = FTSettings(
            start_us=1.0,
            end_us=14.0,
            zpf=2,
            expf_us=5.0,
            window_function="hann",
            units_power=6,
            trim=(26500.0, 40000.0),
            rdc=True,
        )
        restored = FTSettings.from_attrs(s.to_attrs())

        assert restored.start_us == 1.0
        assert restored.end_us == 14.0
        assert restored.zpf == 2
        assert restored.expf_us == 5.0
        assert restored.window_function == "hann"
        assert restored.units_power == 6
        assert restored.trim == (26500.0, 40000.0)
        assert restored.rdc is True

    def test_trim_min_max_stored_as_floats(self):
        s = FTSettings(trim=(26500, 40000))
        attrs = s.to_attrs()
        assert isinstance(attrs["trim_min_mhz"], float)
        assert isinstance(attrs["trim_max_mhz"], float)


# ---------------------------------------------------------------------------
# Argspec: add_settings_args / settings_from_namespace
# ---------------------------------------------------------------------------

class TestArgspec:
    def test_full_parse_produces_correct_ftsettings(self):
        p = _make_parser()
        ns = p.parse_args([
            "--zpf", "2",
            "--expf_us", "5.0",
            "--trim", "26500:40000",
            "--window-function", "hann",
            "--start-us", "1.0",
        ])
        s = settings_from_namespace(ns)

        assert s.zpf == 2
        assert s.expf_us == 5.0
        assert s.trim == (26500.0, 40000.0)
        assert s.window_function == "hann"
        assert s.start_us == 1.0

        # Fields not passed must be None
        assert s.end_us is None
        assert s.units_power is None

    def test_empty_argv_yields_all_none_ftsettings(self):
        p = _make_parser()
        ns = p.parse_args([])
        s = settings_from_namespace(ns)
        assert s.is_empty()

    def test_invalid_trim_no_colon_raises_system_exit(self):
        """argparse should call error() which raises SystemExit for bad --trim."""
        p = _make_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--trim", "26500"])

    def test_trim_inverted_range_raises_system_exit(self):
        """Trim where max <= min is caught by _parse_trim and raises SystemExit."""
        p = _make_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["--trim", "40000:26500"])

    def test_zpf_parsed_as_int(self):
        p = _make_parser()
        ns = p.parse_args(["--zpf", "4"])
        s = settings_from_namespace(ns)
        assert isinstance(s.zpf, int)
        assert s.zpf == 4

    def test_expf_us_parsed_as_float(self):
        p = _make_parser()
        ns = p.parse_args(["--expf_us", "3.5"])
        s = settings_from_namespace(ns)
        assert isinstance(s.expf_us, float)
        assert s.expf_us == 3.5

    def test_units_power_parsed_as_int(self):
        p = _make_parser()
        ns = p.parse_args(["--units-power", "3"])
        s = settings_from_namespace(ns)
        assert isinstance(s.units_power, int)
        assert s.units_power == 3

    def test_start_us_end_us_parsed_as_float(self):
        p = _make_parser()
        ns = p.parse_args(["--start-us", "0.5", "--end-us", "12.0"])
        s = settings_from_namespace(ns)
        assert s.start_us == 0.5
        assert s.end_us == 12.0
