"""
Pure unit tests for FTSettings, resolve(), and argspec helpers.

The canonical FT is unconditionally unapodized, un-windowed, native-length, and
unconditionally DC-removed: there are no ``zpf`` / ``expf_us`` /
``window_function`` / ``rdc`` knobs. The settable fields are data selection
(``start_us`` / ``end_us`` / ``trim``) plus the display/scaling knob
(``units_power``).

No real data required; all tests are fast and free of I/O.
"""

import argparse

import pytest

from ftmwpipeline.cli._argspec import add_settings_args, settings_from_namespace
from ftmwpipeline.core.settings import (
    FTSettings,
    resolve,
)

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
        s = FTSettings(units_power=2)
        assert not s.is_empty()

    def test_overrides_returns_only_set_fields(self):
        s = FTSettings(start_us=1.0, units_power=3)
        ov = s.overrides()
        assert ov == {"start_us": 1.0, "units_power": 3}

    def test_overrides_empty_when_no_fields_set(self):
        assert FTSettings().overrides() == {}

    def test_trim_in_overrides(self):
        s = FTSettings(trim=(100.0, 200.0))
        assert "trim" in s.overrides()
        assert s.overrides()["trim"] == (100.0, 200.0)

    def test_no_apodization_fields(self):
        """The retired apodization knobs are gone from FTSettings."""
        s = FTSettings()
        assert not hasattr(s, "zpf")
        assert not hasattr(s, "expf_us")
        assert not hasattr(s, "window_function")

    def test_no_rdc_field(self):
        """DC removal is unconditional; there is no rdc knob on FTSettings."""
        s = FTSettings()
        assert not hasattr(s, "rdc")


# ---------------------------------------------------------------------------
# resolve() precedence
# ---------------------------------------------------------------------------


class TestResolve:
    def test_explicit_beats_persisted_beats_recommended(self):
        explicit = FTSettings(start_us=4.0)
        persisted = FTSettings(start_us=2.0, end_us=3.0)
        recommended = FTSettings(start_us=1.0, end_us=7.0, units_power=3)

        result = resolve(explicit, persisted, recommended)

        # start_us: explicit wins
        assert result.start_us == 4.0
        # end_us: explicit absent → persisted wins
        assert result.end_us == 3.0
        # units_power: only recommended set → recommended wins
        assert result.units_power == 3

    def test_hard_defaults_fill_run_critical_fields(self):
        result = resolve(None, None, None)
        assert result.units_power == 6

    def test_optional_fields_stay_none_when_unset(self):
        result = resolve(None, None, None)
        assert result.start_us is None
        assert result.end_us is None
        assert result.trim is None

    def test_explicit_none_falls_through_to_persisted(self):
        explicit = FTSettings(start_us=None)
        persisted = FTSettings(start_us=3.0)
        result = resolve(explicit, persisted, None)
        assert result.start_us == 3.0

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
        assert result.units_power == 6

    def test_field_by_field_precedence_exhaustive(self):
        """Every field independently follows explicit > persisted > recommended."""
        e = FTSettings(start_us=10.0)
        p = FTSettings(start_us=20.0, end_us=2.0)
        r = FTSettings(start_us=30.0, end_us=4.0, units_power=5)

        result = resolve(e, p, r)

        assert result.start_us == 10.0  # explicit
        assert result.end_us == 2.0  # persisted (explicit absent)
        assert result.units_power == 5  # recommended (only one set)


# ---------------------------------------------------------------------------
# to_attrs / from_attrs round-trips
# ---------------------------------------------------------------------------


class TestToAttrsFromAttrs:
    def test_roundtrip_with_trim_set(self):
        s = FTSettings(start_us=1.0, trim=(26500.0, 40000.0), units_power=6)
        attrs = s.to_attrs()
        restored = FTSettings.from_attrs(attrs)

        assert restored.start_us == 1.0
        assert restored.trim == (26500.0, 40000.0)
        assert restored.units_power == 6

    def test_roundtrip_trim_none_uses_none_marker(self):
        s = FTSettings(units_power=6)
        attrs = s.to_attrs()

        # None serialized to __None__ marker
        assert attrs["trim_min_mhz"] == "__None__"
        assert attrs["trim_max_mhz"] == "__None__"

        # and restores to None
        restored = FTSettings.from_attrs(attrs)
        assert restored.trim is None

    def test_from_attrs_ignores_legacy_apodization_keys(self):
        """Legacy records may carry retired apodization/rdc keys; ignore them."""
        attrs = {
            "winf": "blackman",
            "zpf": 2,
            "expf_us": 5.0,
            "window_function": "hann",
            "rdc": False,
            "start_us": 1.0,
        }
        s = FTSettings.from_attrs(attrs)
        assert s.start_us == 1.0
        assert not hasattr(s, "zpf")
        assert not hasattr(s, "expf_us")
        assert not hasattr(s, "window_function")
        assert not hasattr(s, "rdc")

    def test_from_attrs_tolerant_of_missing_keys(self):
        """Sparse dicts (e.g. old recommended records) do not raise."""
        s = FTSettings.from_attrs({"units_power": 6})
        assert s.units_power == 6
        assert s.trim is None

    def test_from_attrs_bytes_values_decoded(self):
        """HDF5 sometimes returns byte strings; from_attrs must handle them."""
        attrs = {"units_power": b"3", "start_us": b"1.5"}
        s = FTSettings.from_attrs(attrs)
        assert s.units_power == 3
        assert s.start_us == 1.5

    def test_roundtrip_all_fields_set(self):
        s = FTSettings(
            start_us=1.0,
            end_us=14.0,
            units_power=6,
            trim=(26500.0, 40000.0),
        )
        restored = FTSettings.from_attrs(s.to_attrs())

        assert restored.start_us == 1.0
        assert restored.end_us == 14.0
        assert restored.units_power == 6
        assert restored.trim == (26500.0, 40000.0)

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
        ns = p.parse_args(
            [
                "--trim",
                "26500:40000",
                "--start-us",
                "1.0",
                "--units-power",
                "3",
            ]
        )
        s = settings_from_namespace(ns)

        assert s.trim == (26500.0, 40000.0)
        assert s.start_us == 1.0
        assert s.units_power == 3

        # Fields not passed must be None
        assert s.end_us is None

    def test_apodization_flags_rejected(self):
        """The retired apodization flags no longer exist on the parser."""
        p = _make_parser()
        for flag in ("--zpf", "--expf_us", "--window-function"):
            with pytest.raises(SystemExit):
                p.parse_args([flag, "2"])

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
