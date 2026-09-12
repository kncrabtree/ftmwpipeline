"""``review run`` announces itself on the log stream like every other stage:
a start line and a completion summary, so a caller tailing the ``ftmwpipeline``
logger can tell Stage 6 ran (and finished) rather than watching silence."""

from __future__ import annotations

import logging

import pytest

from ftmwpipeline._internal.stage6_impl import review_run_impl

pytestmark = [pytest.mark.integration]


def test_review_run_logs_a_start_line_and_a_completion_summary(
    stage5_small_file, caplog
):
    with caplog.at_level(logging.INFO, logger="ftmwpipeline"):
        result = review_run_impl(stage5_small_file)
    recs = [r for r in caplog.records if r.name.endswith("stage6_impl")]
    msgs = [r.getMessage() for r in recs]
    start = [m for m in msgs if m.startswith("Stage 6 review: routing attention for")]
    done = [m for m in msgs if m.startswith("Saved Stage 6 review to")]
    assert len(start) == 1 and len(done) == 1
    assert f"for {result.n_windows} windows in" in start[0]
    assert f"{result.n_windows} windows, {result.n_attention} need attention" in done[0]
    assert msgs.index(start[0]) < msgs.index(done[0])
    assert all(
        r.levelno == logging.INFO for r in recs if r.getMessage() in start + done
    )
