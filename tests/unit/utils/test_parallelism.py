"""Unit tests for the worker-pool size resolver."""

from __future__ import annotations

import os

import pytest

from ftmwpipeline.utils.parallelism import ENV_MAX_WORKERS, resolve_worker_count


@pytest.fixture(autouse=True)
def _deterministic_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ``cpu_count`` and clear the env var so the default is deterministic."""
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.delenv(ENV_MAX_WORKERS, raising=False)


def test_default_is_cpu_count_minus_two() -> None:
    assert resolve_worker_count() == 6


def test_explicit_beats_default() -> None:
    assert resolve_worker_count(3) == 3


def test_explicit_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MAX_WORKERS, "4")
    assert resolve_worker_count(3) == 3


def test_env_beats_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MAX_WORKERS, "4")
    assert resolve_worker_count() == 4


def test_override_beats_explicit() -> None:
    assert resolve_worker_count(3, override=5) == 5


def test_override_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MAX_WORKERS, "4")
    assert resolve_worker_count(override=5) == 5


def test_explicit_clamped_to_at_least_one() -> None:
    assert resolve_worker_count(0) == 1
    assert resolve_worker_count(-3) == 1


def test_override_clamped_to_at_least_one() -> None:
    assert resolve_worker_count(override=0) == 1


def test_env_clamped_to_at_least_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MAX_WORKERS, "0")
    assert resolve_worker_count() == 1


def test_malformed_env_falls_through_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_MAX_WORKERS, "abc")
    assert resolve_worker_count() == 6


def test_blank_env_falls_through_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MAX_WORKERS, "   ")
    assert resolve_worker_count() == 6


def test_default_clamped_when_cpu_count_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert resolve_worker_count() == 1


def test_default_when_cpu_count_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    # (2 or 2) - 2 == 0, clamped to 1.
    assert resolve_worker_count() == 1
