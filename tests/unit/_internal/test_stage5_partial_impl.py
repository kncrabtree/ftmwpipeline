"""The resume decision of a Stage 5 partial fit (Wave 5.2).

``dev-docs/CONTRACT_STRATEGY.md`` §Stage 5 partial fits: a fit resumes a partial
fit only when its requested settings, the values it consumes from other stages
and ``ANALYSIS_EPOCH`` equal the partial fit's; otherwise it starts over and
says why (``restart_requested``, ``settings_changed``,
``incomplete_provenance``, ``thaw_refit``), and never resumes on a guess.

The decision is taken on a small hand-built file here; the fits that follow it
are in ``tests/integration/test_stage5_resume.py``.
"""

from __future__ import annotations

import enum
from types import SimpleNamespace
from typing import Any, Dict

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal.stage5_partial_impl import (
    _canonical,
    decide_resume,
    plan_identity,
)
from ftmwpipeline.contract import FIT_RESTART_REASONS
from ftmwpipeline.fitting.plan_execution import PartialWalk
from ftmwpipeline.io.stage5_partial_serialization import (
    encode_stage5_partial,
    write_stage5_partial,
)
from tests.unit.io.test_stage5_partial_serialization import _outcome

pytestmark = pytest.mark.unit

_PROVENANCE: Dict[str, str] = {
    "analysis_epoch": "5.0",
    "settings": '{"shape": "lorentzian"}',
    "consumed": '{"tau0_us": 3.0}',
    "context": '{"digest": "abc"}',
}
_WALK = {"phase": "initial", "accepted_thaw": False, "walk_mode": "dag", "n_windows": 4}


def _write(path, *, walk=None, provenance=None, window_id=3):
    out, record = _outcome(window_id)
    partial = PartialWalk(
        "initial", [window_id], {window_id: out}, {window_id: record}, False, "dag", 4
    )
    encoded, refused = encode_stage5_partial(partial)
    assert not refused
    stored: Dict[str, Any] = {
        **(provenance or _PROVENANCE),
        "walk": _WALK if walk is None else walk,
    }
    with h5py.File(path, "w") as h5f:
        write_stage5_partial(h5f, encoded, stored)
    return path


def _decide(path, *, restart=False, provenance=None, window_ids=(0, 1, 2, 3)):
    return decide_resume(
        str(path),
        provenance or _PROVENANCE,
        restart=restart,
        window_ids=list(window_ids),
    )


def test_the_vocabulary_is_the_four_reasons():
    assert FIT_RESTART_REASONS == (
        "restart_requested",
        "settings_changed",
        "incomplete_provenance",
        "thaw_refit",
    )


@pytest.mark.parametrize("restart", [False, True])
def test_nothing_to_resume_is_a_fresh_fit_with_no_reason(tmp_path, restart):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass
    decision = _decide(path, restart=restart)
    assert decision.carried is None and decision.restart_reason is None


def test_matching_provenance_resumes(tmp_path):
    decision = _decide(_write(tmp_path / "p.h5"))
    assert decision.restart_reason is None
    assert decision.carried is not None and decision.carried.order == [3]


def test_a_restart_request_wins(tmp_path):
    decision = _decide(_write(tmp_path / "p.h5"), restart=True)
    assert decision.carried is None
    assert decision.restart_reason == "restart_requested"


@pytest.mark.parametrize("key", ["analysis_epoch", "settings", "consumed", "context"])
def test_any_difference_in_what_the_fit_depends_on_starts_over(tmp_path, key):
    path = _write(tmp_path / "p.h5")
    changed = {**_PROVENANCE, key: _PROVENANCE[key] + " "}
    decision = _decide(path, provenance=changed)
    assert decision.carried is None
    assert decision.restart_reason == "settings_changed"


def test_an_accepted_thaw_starts_over_when_everything_else_matches(tmp_path):
    path = _write(tmp_path / "p.h5", walk={**_WALK, "accepted_thaw": True})
    decision = _decide(path)
    assert decision.carried is None and decision.restart_reason == "thaw_refit"


def test_a_difference_is_reported_before_a_thaw(tmp_path):
    path = _write(tmp_path / "p.h5", walk={**_WALK, "accepted_thaw": True})
    changed = {**_PROVENANCE, "settings": "{}"}
    assert _decide(path, provenance=changed).restart_reason == "settings_changed"


@pytest.mark.parametrize("key", ["analysis_epoch", "settings", "consumed", "context"])
def test_a_missing_provenance_entry_is_incomplete(tmp_path, key):
    provenance = {k: v for k, v in _PROVENANCE.items() if k != key}
    path = _write(tmp_path / "p.h5", provenance=provenance)
    decision = _decide(path)
    assert decision.carried is None
    assert decision.restart_reason == "incomplete_provenance"


@pytest.mark.parametrize(
    "walk", [{}, {"accepted_thaw": "no"}, {"accepted_thaw": None}, "walk"]
)
def test_a_malformed_walk_record_is_incomplete(tmp_path, walk):
    path = _write(tmp_path / "p.h5", walk=walk)
    assert _decide(path).restart_reason == "incomplete_provenance"


def test_a_window_the_plan_does_not_have_is_incomplete(tmp_path):
    path = _write(tmp_path / "p.h5", window_id=3)
    decision = _decide(path, window_ids=(0, 1, 2))
    assert decision.carried is None
    assert decision.restart_reason == "incomplete_provenance"


def test_unreadable_windows_are_incomplete(tmp_path):
    path = _write(tmp_path / "p.h5")
    with h5py.File(path, "a") as h5f:
        del h5f["stage5_partial/windows"]
    assert _decide(path).restart_reason == "incomplete_provenance"


# ---- the comparison is of values ------------------------------------------------------------


class _Color(enum.Enum):
    RED = "red"


def test_equal_values_compare_equal_whatever_their_spelling():
    assert _canonical({"a": 3, "b": [1, 2]}) == _canonical({"b": [1.0, 2.0], "a": 3.0})
    assert _canonical(np.array([1.0, 2.0])) == _canonical([1, 2])
    assert _canonical((1, 2)) == _canonical([1, 2])
    assert _canonical(_Color.RED) == _canonical("red")
    assert _canonical(np.float64(2.5)) == _canonical(2.5)
    assert _canonical(float("nan")) == _canonical(float("nan"))


def test_different_values_compare_different():
    assert _canonical({"a": 3}) != _canonical({"a": 3.0000001})
    assert _canonical({"a": 3}) != _canonical({"b": 3})
    assert _canonical(True) != _canonical(1)
    assert _canonical([1, 2]) != _canonical([2, 1])
    assert _canonical(None) != _canonical(0)


def _plan(rng=(10.0, 20.0), revision=0):
    window = SimpleNamespace(
        window_id=0,
        freq_range=rng,
        free_peak_indices=[1, 2],
        fixed_contributors=[],
        batch=0,
    )
    return SimpleNamespace(
        plan_revision=revision,
        windows=[window],
        dependency_edges=[],
        topological_order=[0],
        parameters={"min_freeze_snr": 3.0},
    )


def test_the_plan_identity_follows_the_plan():
    base = plan_identity(_plan())
    assert base == plan_identity(_plan())
    assert base["n_windows"] == 1
    assert plan_identity(_plan(rng=(10.0, 21.0)))["digest"] != base["digest"]
    assert plan_identity(_plan(revision=1))["digest"] != base["digest"]


@pytest.mark.parametrize(
    "exc", [IndexError("x"), RuntimeError("x"), AttributeError("x"), KeyError("x")]
)
@pytest.mark.parametrize(
    "reader", ["read_stage5_partial_windows", "read_stage5_partial_provenance"]
)
def test_any_error_reading_the_partial_fit_is_incomplete(
    tmp_path, monkeypatch, exc, reader
):
    """Backstop: whatever reading or decoding a partial fit raises (not only the
    codec's error), the fit starts over -- it never resumes, and never fails."""
    import ftmwpipeline._internal.stage5_partial_impl as impl

    def boom(*_a, **_k):
        raise exc

    path = _write(tmp_path / "p.h5")
    monkeypatch.setattr(impl, reader, boom)
    decision = _decide(path)
    assert decision.carried is None
    assert decision.restart_reason == "incomplete_provenance"


# ---- the clocks stale-declaration note sees a partial fit ---------------------------


@pytest.mark.parametrize(
    "groups, expected",
    [((), False), (("stage5_fitting",), True), (("stage5_partial",), True)],
    ids=["no_fit", "complete_fit", "partial_fit"],
)
def test_stage5_fit_present_counts_a_partial_fit(groups, expected, tmp_path):
    """A partial fit keeps the clock declaration it started with (its resume
    compares it), so ``clocks set`` after a cancelled fit gets the
    ``settings unset stage5.spur.clocks`` note too."""
    from ftmwpipeline._internal.clocks_impl import stage5_fit_present

    path = tmp_path / "f.ftmw"
    with h5py.File(path, "w") as h5f:
        for name in groups:
            h5f.create_group(name)
    assert stage5_fit_present(str(path)) is expected
