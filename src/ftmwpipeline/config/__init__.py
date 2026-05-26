"""
Configuration management for FTMW pipeline.

This module exposes the legacy ``PipelineConfig`` placeholder. The canonical
settings dataclasses now live in :mod:`ftmwpipeline.core.settings` (Stage 1)
and :mod:`ftmwpipeline.core.stage_fit_settings` (Stage 5); future stages will
follow that pattern. ``PipelineConfig`` here is unused by the production
pipeline and is retained for the dead-code-cleanup follow-up.
"""

from .pipeline_config import PipelineConfig, load_config, save_config

__all__ = [
    "PipelineConfig",
    "load_config",
    "save_config",
]
