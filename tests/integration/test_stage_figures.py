"""Smoke test for the stage-page figure harness.

Runs ``docs/source/figures/generate.py`` against the checked-in ``2638``
fixture and asserts the stage figures render to disk without error. The
harness is the committed source of the figures embedded in the Stage 0-4 pages;
this guards it against API drift.

Marked ``slow``: it builds a full Stage 0-4 pipeline (a few seconds). Run the
default fast suite with ``-m "not slow"`` to skip it. Figures are written to a
temporary directory, never the committed tree.
"""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_GENERATE = _ROOT / "docs" / "source" / "figures" / "generate.py"

_spec = importlib.util.spec_from_file_location("stage_figures_generate", _GENERATE)
assert _spec is not None and _spec.loader is not None
generate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def test_harness_renders_all_stage_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(generate, "FIG_DIR", tmp_path)
    generate.make_figures()

    for name in (
        "stage0_start_detection.png",
        "stage1_canonical_ft.png",
        "stage2_noise.png",
        "stage2b_tau_distribution.png",
        "stage2b_tau_decay_examples.png",
        "stage2b_tau_heatmap_zoom.png",
        "stage3_peaks.png",
        "stage4_windows.png",
    ):
        out = tmp_path / name
        assert out.is_file() and out.stat().st_size > 0
