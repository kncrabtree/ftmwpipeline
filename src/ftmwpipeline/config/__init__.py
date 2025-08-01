"""
Configuration management for FTMW pipeline.

This module provides:
- Pipeline configuration management
- Default parameter settings
- Algorithm selection and tuning
"""

from .pipeline_config import PipelineConfig, load_config, save_config
from .default_settings import DEFAULT_SETTINGS, get_default_parameters

__all__ = [
    "PipelineConfig",
    "load_config",
    "save_config",
    "DEFAULT_SETTINGS",
    "get_default_parameters",
]