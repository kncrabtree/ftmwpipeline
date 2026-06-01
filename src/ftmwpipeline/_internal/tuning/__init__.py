"""Companion parameter-tuning surface.

A knob-registry-driven sweep engine that lets a user scan a single pipeline
parameter across a grid of values, see the effect as a table (and, where a
plot adapter is registered, a figure), get a best-effort recommendation, and
learn how to apply the chosen value. The engine
(:mod:`ftmwpipeline._internal.tuning.engine`) is knob-agnostic; everything
parameter-specific lives as data in the registry
(:mod:`ftmwpipeline._internal.tuning.registry`).

This package is the single implementation behind the CLI ``tune`` namespace and
the ``Pipeline`` / functional-API ``tune_scan`` / ``tune_list`` wrappers, per
the repo's dual-interface rule. See
``dev-docs/planning/companion-tuning-tools.md``.
"""

from .registry import KnobSpec, get_knob, list_knobs
from .engine import SweepResult, SweepRow, run_scan

__all__ = [
    "KnobSpec",
    "get_knob",
    "list_knobs",
    "SweepResult",
    "SweepRow",
    "run_scan",
]
