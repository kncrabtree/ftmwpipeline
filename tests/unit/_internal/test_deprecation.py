"""Unit tests for the legacy-kwarg deprecation helpers."""

from __future__ import annotations

import warnings

import pytest

from ftmwpipeline._internal.deprecation import (
    warn_legacy_flag,
    warn_legacy_kwargs,
)


class TestWarnLegacyKwargs:
    """``warn_legacy_kwargs`` fires one warning per call when any kwarg is set."""

    def test_no_warning_when_all_none(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            warn_legacy_kwargs(
                func_name="foo",
                legacy_kwargs={"a": None, "b": None, "c": None},
                migration_hint="use settings=...",
            )

    def test_warns_when_one_kwarg_set(self) -> None:
        with pytest.warns(
            DeprecationWarning, match=r"foo: legacy per-knob kwargs \['a'\]"
        ):
            warn_legacy_kwargs(
                func_name="foo",
                legacy_kwargs={"a": 1, "b": None},
                migration_hint="use settings=...",
            )

    def test_one_warning_per_call_lists_every_kwarg(self) -> None:
        """Multiple non-None kwargs collapse into one DeprecationWarning."""
        with pytest.warns(DeprecationWarning) as record:
            warn_legacy_kwargs(
                func_name="foo",
                legacy_kwargs={"a": 1, "b": "value", "c": None, "d": 3.5},
                migration_hint="migrate",
            )
        assert len(record) == 1
        message = str(record[0].message)
        assert "'a'" in message
        assert "'b'" in message
        assert "'d'" in message
        assert "'c'" not in message  # None kwargs are not listed

    def test_kwarg_names_are_sorted_in_message(self) -> None:
        """Stable message ordering helps grep/diff workflows."""
        with pytest.warns(DeprecationWarning) as record:
            warn_legacy_kwargs(
                func_name="foo",
                legacy_kwargs={"zebra": 1, "alpha": 1, "mike": 1},
                migration_hint="migrate",
            )
        message = str(record[0].message)
        assert (
            message.index("'alpha'")
            < message.index("'mike'")
            < message.index("'zebra'")
        )

    def test_migration_hint_in_message(self) -> None:
        with pytest.warns(DeprecationWarning, match="use settings=Foo"):
            warn_legacy_kwargs(
                func_name="foo",
                legacy_kwargs={"a": 1},
                migration_hint="use settings=Foo or preset='name'",
            )

    def test_false_value_counts_as_supplied(self) -> None:
        """``False`` / ``0`` / empty containers are intentional settings, not unset."""
        with pytest.warns(DeprecationWarning, match=r"\['flag'\]"):
            warn_legacy_kwargs(
                func_name="foo",
                legacy_kwargs={"flag": False, "other": None},
                migration_hint="migrate",
            )

    def test_empty_kwargs_dict(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_legacy_kwargs(
                func_name="foo",
                legacy_kwargs={},
                migration_hint="migrate",
            )


class TestWarnLegacyFlag:
    """``warn_legacy_flag`` is for boolean shims like ``from_saved_params=True``."""

    def test_warns_with_flag_name_and_hint(self) -> None:
        with pytest.warns(DeprecationWarning) as record:
            warn_legacy_flag(
                func_name="estimate_noise",
                flag_name="from_saved_params",
                migration_hint="drop the flag",
            )
        assert len(record) == 1
        message = str(record[0].message)
        assert "estimate_noise" in message
        assert "'from_saved_params'" in message
        assert "drop the flag" in message
