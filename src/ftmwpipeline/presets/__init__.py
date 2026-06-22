"""
Bundled pipeline presets.

A preset is a YAML file describing partial settings for one or more pipeline
stages (top-level ``stage2`` / ``stage2b`` / ``stage3`` / ``stage4`` /
``stage5`` blocks); each stage's loader reads only its own block. The package
ships one preset, ``defaults`` -- every knob at its package-wide default, as a
copy-and-edit template. A preset may also be a user-authored YAML on disk
(referenced by path). See each stage settings module's ``load_preset`` (e.g.
:func:`ftmwpipeline.core.stage_fit_settings.load_preset`) for the resolution
rules, and ``docs/source/settings_and_presets.rst`` for the YAML format.
"""
