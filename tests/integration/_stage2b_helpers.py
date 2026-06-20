"""Test-only helpers for Stage 2b ``auto_recommend``.

The production default is ``auto_recommend=True`` so the Stage 5
resolver's *recommended* layer fires on every fresh Stage 2b run.
On the 2638 fixture the 3-way classifier pass adds ~50s per
``calibrate_tau`` / ``calibrate_tau_G`` call. Tests that exercise
those entry points for an unrelated reason (cross-interface τ
identity, Gaussian fit-peaks plumbing, etc.) can opt out via the
helpers here to keep the suite fast.
"""

from __future__ import annotations

from pathlib import Path


def skip_auto_recommend_settings():
    """``TauCalibrationSettings`` with ``auto_recommend=False`` set.

    Pass via ``settings=`` to ``Pipeline.calibrate_tau`` /
    ``Pipeline.calibrate_tau(shape="gaussian")`` / ``ftmw.calibrate_tau`` /
    ``ftmw.calibrate_tau(shape="gaussian")``. Every other field falls through to
    the resolver's hard defaults.
    """
    from ftmwpipeline.core.tau_calibration_settings import TauCalibrationSettings

    s = TauCalibrationSettings()
    s.recommendation.auto_recommend = False
    return s


def skip_auto_recommend_preset_yaml(dest_dir: Path) -> Path:
    """Write the minimal opt-out preset YAML and return its path.

    Use with the CLI's ``--preset`` flag. The YAML carries only the
    ``recommendation.auto_recommend = false`` override under the
    ``stage2b:`` block.
    """
    p = dest_dir / "skip_auto_recommend.yaml"
    p.write_text("stage2b:\n" "  recommendation:\n" "    auto_recommend: false\n")
    return p
