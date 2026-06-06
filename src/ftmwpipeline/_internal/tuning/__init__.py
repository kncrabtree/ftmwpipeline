"""Companion parameter-tuning surface.

A knob-registry-driven sweep engine that lets a user scan a single pipeline
parameter across a grid of values, see the effect as a table (and, where a
plot adapter is registered, a figure), get a best-effort recommendation, and
learn how to apply the chosen value. The engine
(:mod:`ftmwpipeline._internal.tuning.engine`) is knob-agnostic; everything
parameter-specific lives as data in the registry
(:mod:`ftmwpipeline._internal.tuning.registry`).

This package is the single implementation behind the CLI ``scan`` meta-object
and the ``Pipeline`` / functional-API ``scan_run`` / ``scan_list`` wrappers, per
the repo's dual-interface rule. It also backs the ``settings`` meta-object
(:mod:`settings_inspection`, :mod:`settings_mutation`). See
``dev-docs/planning/companion-tuning-tools.md``.
"""

from .engine import BatchItem, SweepResult, SweepRow, run_scan, run_scan_batch
from .registry import KnobSpec, get_knob, list_knobs
from .settings_inspection import SettingRow, resolve_settings_view
from .settings_mutation import (
    ExportResult,
    SetResult,
    export_settings,
    set_setting,
)

__all__ = [
    "KnobSpec",
    "get_knob",
    "list_knobs",
    "SweepResult",
    "SweepRow",
    "run_scan",
    "BatchItem",
    "run_scan_batch",
    "SettingRow",
    "resolve_settings_view",
    "SetResult",
    "ExportResult",
    "set_setting",
    "export_settings",
]
