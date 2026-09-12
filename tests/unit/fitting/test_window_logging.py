"""The per-window log records of the Stage 5 fit walk.

Every fitted window logs one INFO detail line under one template. A slow
window ADDITIONALLY logs a WARNING under its own template, so a consumer
filtering by level and one matching on ``record.msg`` agree on which records
mean "this window was slow" -- neither has to re-derive the threshold from the
rendered text.
"""

from __future__ import annotations

import logging

import pytest

from ftmwpipeline.fitting import plan_execution as pe

pytestmark = [pytest.mark.unit]


def _records(caplog):
    return [r for r in caplog.records if r.name == pe.logger.name]


def test_fast_window_logs_one_info_detail_line(caplog):
    with caplog.at_level(logging.INFO, logger=pe.logger.name):
        pe._log_window_outcome(
            7, (100.0, 101.0), n_peaks=3, reduced_chi2=1.25, elapsed_s=0.4
        )
    recs = _records(caplog)
    assert [(r.levelno, r.msg) for r in recs] == [
        (logging.INFO, pe.WINDOW_DETAIL_LOG_TEMPLATE)
    ]
    assert recs[0].getMessage() == "w7 [100.0-101.0 MHz]: 3 peaks, chi2r=1.25, 0.4s"


def test_slow_window_keeps_the_detail_line_and_adds_a_warning(caplog):
    slow = pe.SLOW_WINDOW_WARNING_S + 1.0
    with caplog.at_level(logging.INFO, logger=pe.logger.name):
        pe._log_window_outcome(
            7, (100.0, 101.0), n_peaks=3, reduced_chi2=1.25, elapsed_s=slow
        )
    recs = _records(caplog)
    assert [(r.levelno, r.msg) for r in recs] == [
        (logging.INFO, pe.WINDOW_DETAIL_LOG_TEMPLATE),
        (logging.WARNING, pe.SLOW_WINDOW_LOG_TEMPLATE),
    ]
    # The detail line is the same record as for a fast window (same level,
    # same template); the slow flag is a separate record of its own shape.
    assert recs[1].getMessage() == (
        f"slow window w7 [100.0-101.0 MHz]: {slow:.1f}s "
        f"(over {pe.SLOW_WINDOW_WARNING_S:.0f}s)"
    )
    assert recs[1].args[0] == 7


def test_threshold_is_exclusive(caplog):
    with caplog.at_level(logging.INFO, logger=pe.logger.name):
        pe._log_window_outcome(
            1,
            (0.0, 1.0),
            n_peaks=0,
            reduced_chi2=1.0,
            elapsed_s=pe.SLOW_WINDOW_WARNING_S,
        )
    assert [r.levelno for r in _records(caplog)] == [logging.INFO]


def test_templates_are_distinct_and_the_progress_counter_is_untouched():
    assert pe.WINDOW_DETAIL_LOG_TEMPLATE != pe.SLOW_WINDOW_LOG_TEMPLATE
    assert pe.SLOW_WINDOW_LOG_TEMPLATE.startswith("slow window w%d")
    # The bar-driving progress record keeps its template (progress.py matches
    # on it) and neither per-window record can be mistaken for it.
    from ftmwpipeline._internal.progress import WINDOW_LOG_PREFIX

    assert not pe.WINDOW_DETAIL_LOG_TEMPLATE.startswith(WINDOW_LOG_PREFIX)
    assert not pe.SLOW_WINDOW_LOG_TEMPLATE.startswith(WINDOW_LOG_PREFIX)
