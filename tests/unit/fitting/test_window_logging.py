"""The per-window log records of the Stage 5 fit walk.

Every fitted window logs one INFO detail line under one template, then the
``window n/total`` progress line (both rendered from its ``WindowProgress``
event, in the parent). A slow window ADDITIONALLY logs a WARNING under its own
template (its ``slow_window`` warning event), so a consumer filtering by level
and one matching on ``record.msg`` agree on which records mean "this window was
slow" -- neither has to re-derive the threshold from the rendered text.
"""

from __future__ import annotations

import logging

import pytest

from ftmwpipeline._internal.events import detached_scope
from ftmwpipeline._internal.progress import WINDOW_LOG_PREFIX
from ftmwpipeline.contract import Stage
from ftmwpipeline.fitting import plan_execution as pe

pytestmark = [pytest.mark.unit]


def _records(caplog):
    return [r for r in caplog.records if r.name == pe.logger.name]


def _report(wid, freq_range, *, n_peaks, reduced_chi2, elapsed_s):
    """Report one finished window the way the walks do (index 1 of 1)."""
    pe._report_window(
        detached_scope(Stage.FIT, verb="fit run"),
        pe._WindowReport(wid, freq_range, n_peaks, reduced_chi2, elapsed_s),
        phase="initial",
        walk_round=0,
        index=1,
        total=1,
    )


def test_fast_window_logs_one_info_detail_line(caplog):
    with caplog.at_level(logging.INFO, logger=pe.logger.name):
        _report(7, (100.0, 101.0), n_peaks=3, reduced_chi2=1.25, elapsed_s=0.4)
    recs = _records(caplog)
    assert [(r.levelno, r.msg) for r in recs] == [
        (logging.INFO, pe.WINDOW_DETAIL_LOG_TEMPLATE),
        (logging.INFO, WINDOW_LOG_PREFIX),
    ]
    assert recs[1].args == (1, 1)
    assert recs[0].getMessage() == "w7 [100.0-101.0 MHz]: 3 peaks, chi2r=1.25, 0.4s"


def test_slow_window_keeps_the_detail_line_and_adds_a_warning(caplog):
    slow = pe.SLOW_WINDOW_WARNING_S + 1.0
    with caplog.at_level(logging.INFO, logger=pe.logger.name):
        _report(7, (100.0, 101.0), n_peaks=3, reduced_chi2=1.25, elapsed_s=slow)
    recs = _records(caplog)
    assert [(r.levelno, r.msg) for r in recs] == [
        (logging.INFO, pe.WINDOW_DETAIL_LOG_TEMPLATE),
        (logging.INFO, WINDOW_LOG_PREFIX),
        (logging.WARNING, pe.SLOW_WINDOW_LOG_TEMPLATE),
    ]
    # The detail line is the same record as for a fast window (same level,
    # same template); the slow flag is a separate record of its own shape.
    assert recs[2].getMessage() == (
        f"slow window w7 [100.0-101.0 MHz]: {slow:.1f}s "
        f"(over {pe.SLOW_WINDOW_WARNING_S:.0f}s)"
    )
    assert recs[2].args[0] == 7


def test_threshold_is_exclusive(caplog):
    with caplog.at_level(logging.INFO, logger=pe.logger.name):
        _report(
            1,
            (0.0, 1.0),
            n_peaks=0,
            reduced_chi2=1.0,
            elapsed_s=pe.SLOW_WINDOW_WARNING_S,
        )
    assert [r.levelno for r in _records(caplog)] == [logging.INFO, logging.INFO]


def test_dropped_window_logs_the_dropped_line_and_never_slow(caplog):
    with caplog.at_level(logging.INFO, logger=pe.logger.name):
        _report(
            4,
            (10.0, 11.0),
            n_peaks=None,
            reduced_chi2=None,
            elapsed_s=pe.SLOW_WINDOW_WARNING_S + 5.0,
        )
    recs = _records(caplog)
    assert [r.levelno for r in recs] == [logging.INFO, logging.INFO]
    assert recs[0].getMessage() == "w4 [10.0-11.0 MHz]: dropped (cascaded to empty)"
    assert recs[1].msg == WINDOW_LOG_PREFIX


def test_templates_are_distinct_and_the_progress_counter_is_untouched():
    assert pe.WINDOW_DETAIL_LOG_TEMPLATE != pe.SLOW_WINDOW_LOG_TEMPLATE
    assert pe.SLOW_WINDOW_LOG_TEMPLATE.startswith("slow window w%d")
    # The bar-driving progress record keeps its template (progress.py matches
    # on it) and neither per-window record can be mistaken for it.
    assert not pe.WINDOW_DETAIL_LOG_TEMPLATE.startswith(WINDOW_LOG_PREFIX)
    assert not pe.SLOW_WINDOW_LOG_TEMPLATE.startswith(WINDOW_LOG_PREFIX)
