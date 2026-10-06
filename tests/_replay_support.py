"""The Stage 6 write invariant as a reusable check.

Design G1 (``scratch/replay-engine-design.md`` §1.3): after every successful
Stage 6 write, the persisted curated state equals, bit for bit, the reference
replay (:func:`~ftmwpipeline._internal.replay_reference.replay_full`) of the
persisted decision log under the persisted review parameters.
:func:`assert_persisted_is_reference` is that check for one file;
:func:`check_every_write` wraps the write path's one persist
(``_finish_batch``) so a test asserts it after each of its writes, sessions and
staged previews included. Test-only: the production write path never calls it.
A write computes its state incrementally (it refits only the windows whose
keys changed), so the check is independent of it; it does not pin the
replay's semantics, which both share (see
:mod:`ftmwpipeline._internal.replay_reference`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Union

import pytest

from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.replay_reference import (
    differing_parts,
    persisted_state_digest,
    replay_full,
    state_digest,
)
from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file


def assert_persisted_is_reference(path: Union[str, Path]) -> None:
    """The state persisted in *path* is the reference replay of its own log
    under its own recorded review parameters, bit for bit.

    Reads through the package's file opener, so inside a write's transaction
    it checks the working copy that write is about to commit."""
    review = load_stage6_review_from_file(str(path))
    reference = replay_full(path, review.decision_log, review.review_params)
    assert state_digest(reference) == persisted_state_digest(path), (
        "the persisted state is not the reference replay of its log; differing "
        f"parts: {differing_parts(reference, path)}"
    )


class WriteChecks:
    """What :func:`check_every_write` installed: the paths it has checked, in
    write order, and a switch a test that deliberately leaves the reference
    behind (a monkeypatched fit, say) turns off."""

    def __init__(self) -> None:
        self.checked: List[str] = []
        self.enabled = True


def check_every_write(monkeypatch: pytest.MonkeyPatch) -> WriteChecks:
    """Assert :func:`assert_persisted_is_reference` right after every persist
    of the test (``_finish_batch``), inside the write's transaction."""
    checks = WriteChecks()
    persist = s6._finish_batch

    def persist_then_check(curated: Any, path: str) -> None:
        persist(curated, path)
        if checks.enabled:
            assert_persisted_is_reference(path)
            checks.checked.append(path)

    monkeypatch.setattr(s6, "_finish_batch", persist_then_check)
    return checks
