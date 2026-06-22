"""
Canonical Stage 2 noise-estimation settings.

``NoiseSettings`` is the single source of truth for the Stage 2 parameters
across every surface:

* the public API signatures (``Pipeline.estimate_noise`` /
  ``ftmwpipeline.api.estimate_noise``),
* the CLI ``--preset`` flag plus the existing per-knob flags,
* the resolution chain ``explicit > persisted > preset > recommended >
  hard default``,
* the persisted canonical record in ``processing_parameters/stage2_noise``,
* the YAML preset interchange format.

Every field is ``Optional`` with ``None`` meaning *unset* (fall through the
resolution chain). A *resolved* instance (produced by :func:`resolve`) has
every field filled with a hard default if no layer supplied a value.

Stage 2 has a single estimator (the high-pass, region-aware scatter MAD), so
the knobs sit directly on ``NoiseSettings`` -- a flat dataclass rather than the
sub-block layout the multi-group stage settings
(:class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`,
:class:`~ftmwpipeline.core.tau_calibration_settings.TauCalibrationSettings`)
use. The fields map 1:1 to attrs on the ``processing_parameters/stage2_noise``
group and to keys under the YAML ``stage2:`` block.

The *recommended* layer of :func:`resolve` is reserved but unused for
Stage 2 today -- Stage 2 has no upstream feeder. The layer is kept in
the signature so a future cross-stage recommender (e.g. Stage 1's
T_active-driven smoothing-window suggestion) can land without API churn.

The ``_HARD_DEFAULTS`` dict mirrors the ``estimate_noise_scatter`` kernel's
signature defaults in :mod:`ftmwpipeline.preprocessing.noise_estimation`; those
defaults are the readable canonical source and these must track them.

This module is dependency-free within the package (stdlib + PyYAML for
preset interchange) so it can be imported from ``core`` without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union, cast

import yaml  # type: ignore[import-untyped]

from .knob_metadata import knob_field

# Mirrors the marker used by io.fid_serialization for optional HDF5 attrs.
_NONE = "__None__"


@dataclass
class NoiseSettings:
    """Stage 2 scatter-estimator settings (see module docstring).

    The fields mirror the ``estimate_noise_scatter`` kernel signature.
    """

    window_mhz: Optional[float] = knob_field(
        help="Width of the per-region scatter-MAD window (scale over which "
        "sigma(f) is constant).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(40.0, 60.0, 80.0, 120.0, 160.0),
        cli=True,
        argtype=float,
    )
    pedestal_mhz: Optional[float] = knob_field(
        help="High-pass running-median width isolating the smooth leakage " "pedestal.",
        tier="primary",
        inst_sensitivity="Y",
        grid=(10.0, 20.0, 40.0, 80.0),
        cli=True,
        argtype=float,
    )
    line_k: Optional[float] = knob_field(
        help="Robust-sigma multiple above which a bin is flagged a line " "(excluded).",
        inst_sensitivity="maybe",
        grid=(4.0, 6.0, 8.0, 12.0),
        cli=True,
        argtype=float,
    )
    n_iter: Optional[int] = knob_field(
        help="Self-mask refinement iterations of the scatter estimator.",
        grid=(1, 2, 3, 5),
        cli=True,
        argtype=int,
    )
    region_aware: Optional[bool] = knob_field(
        help="Use the region-aware Rician correction (else a fixed mid-regime "
        "factor).",
        inst_sensitivity="maybe",
        grid=(False, True),
        cli=True,
        is_flag=True,
    )
    smoothing_mhz: Optional[float] = knob_field(
        help="Broad lower-envelope median sigma smoothing width (0 disables).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(0.0, 400.0, 800.0, 1200.0),
        cli=True,
        argtype=float,
    )
    smoothing_percentile: Optional[float] = knob_field(
        help="Percentile of the broad sigma smoothing (50=median; "
        "lower=lower-envelope).",
        inst_sensitivity="maybe",
        grid=(25.0, 50.0, 75.0),
        cli=True,
        argtype=float,
    )
    convolve_mhz: Optional[float] = knob_field(
        help="Gaussian sigma (MHz) of the second, step-removing smoothing pass "
        "(0=off).",
        inst_sensitivity="maybe",
        grid=(0.0, 100.0, 200.0, 400.0),
        cli=True,
        argtype=float,
    )

    def is_empty(self) -> bool:
        """True if no field is set."""
        return all(getattr(self, f.name) is None for f in fields(self))


# Hard defaults. These mirror the ``estimate_noise_scatter`` kernel's signature
# defaults in ``preprocessing/noise_estimation.py``. Kept as inline literals
# (rather than imported from ``preprocessing/``) to keep ``core`` dependency-free
# from ``preprocessing``; the kernel defaults are the readable canonical source
# and these must track them.
_HARD_DEFAULTS: Dict[str, Any] = {
    "window_mhz": 80.0,
    "pedestal_mhz": 20.0,
    "line_k": 8.0,
    "n_iter": 3,
    "region_aware": True,
    "smoothing_mhz": 800.0,
    "smoothing_percentile": 50.0,
    "convolve_mhz": 200.0,
}


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
def _first_set_field(name: str, *layers: Optional["NoiseSettings"]) -> Any:
    for layer in layers:
        if layer is None:
            continue
        value = getattr(layer, name)
        if value is not None:
            return value
    return None


def resolve(
    explicit: Optional[NoiseSettings] = None,
    preset: Optional[NoiseSettings] = None,
    persisted: Optional[NoiseSettings] = None,
    recommended: Optional[NoiseSettings] = None,
) -> NoiseSettings:
    """Merge the four layers by precedence into a resolved ``NoiseSettings``.

    Per-field precedence: ``explicit > persisted > preset > recommended``,
    then any remaining ``None`` field falls back to the matching value in
    :data:`_HARD_DEFAULTS`. A value persisted in the ``.ftmw`` outranks a
    ``.yml`` preset, so the preset only seeds fields the file has not fixed
    and a shared experiment reproduces from the file alone. The
    ``recommended`` layer is reserved for a future upstream recommender;
    Stage 2 call sites currently pass ``None`` there.
    """
    layers = (explicit, persisted, preset, recommended)
    merged = NoiseSettings()
    for f in fields(NoiseSettings):
        value = _first_set_field(f.name, *layers)
        if value is None:
            value = _HARD_DEFAULTS.get(f.name)
        setattr(merged, f.name, value)
    return merged


# ---------------------------------------------------------------------------
# Dict <-> dataclass round-trip
# ---------------------------------------------------------------------------
def _encode_value(value: Any) -> Any:
    if value is None:
        return _NONE
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and value == _NONE:
        return None
    return value


def to_attrs(settings: NoiseSettings) -> Dict[str, Any]:
    """Flat attrs dict (one key per field).

    Unset fields encode as the ``__None__`` sentinel string.
    """
    return {f.name: _encode_value(getattr(settings, f.name)) for f in fields(settings)}


def from_attrs(attrs: Dict[str, Any]) -> NoiseSettings:
    """Inverse of :func:`to_attrs`. Unknown keys are ignored; missing keys
    stay ``None``."""
    valid = {f.name for f in fields(NoiseSettings)}
    kwargs = {key: _decode_value(value) for key, value in attrs.items() if key in valid}
    return NoiseSettings(**kwargs)


# ---------------------------------------------------------------------------
# YAML interchange
# ---------------------------------------------------------------------------
def to_yaml_dict(settings: NoiseSettings) -> Dict[str, Any]:
    """Sparse dict suitable for ``yaml.safe_dump`` (omits ``None`` fields)."""
    out: Dict[str, Any] = {}
    for f in fields(settings):
        value = getattr(settings, f.name)
        if value is not None:
            out[f.name] = value
    return out


def from_yaml_dict(data: Optional[Mapping[str, Any]]) -> NoiseSettings:
    if data is None:
        return NoiseSettings()
    if not isinstance(data, dict):
        raise ValueError(f"preset YAML root must be a mapping; got {type(data)}")
    valid_names = {f.name for f in fields(NoiseSettings)}
    allowed = valid_names | {"name", "description"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"unknown stage2 fields in preset: {sorted(unknown)} "
            f"(valid: {sorted(valid_names)})"
        )
    kwargs = {key: data[key] for key in valid_names if key in data}
    return NoiseSettings(**kwargs)


def from_yaml(source: Union[str, Path]) -> NoiseSettings:
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    ):
        text = Path(source).read_text()
    else:
        text = str(source)
    data = yaml.safe_load(text)
    return from_yaml_dict(data)


def _looks_like_path(name_or_path: Union[str, Path]) -> bool:
    if isinstance(name_or_path, Path):
        return True
    s = str(name_or_path)
    return ("/" in s) or ("\\" in s) or s.endswith((".yaml", ".yml"))


def load_preset(name_or_path: Union[str, Path]) -> NoiseSettings:
    """Load a Stage 2 preset by bare name or by filesystem path.

    Bare names resolve against the packaged ``ftmwpipeline.presets``
    resources (e.g. ``"defaults"`` ->
    ``ftmwpipeline/presets/defaults.yaml``); paths load directly.
    Preset YAML wraps the Stage 2 settings inside a top-level ``stage2:``
    block (alongside optional ``stage2b:`` / ``stage5:`` blocks for other
    stages).

    The ``name:`` and ``description:`` metadata fields are accepted but
    ignored by the settings parser -- they're documentation for the preset
    author.

    Returns an empty :class:`NoiseSettings` (no fields set) when the
    preset carries no ``stage2:`` block, so a Stage-5-only or
    Stage-2b-only preset loads cleanly without producing spurious Stage 2
    overrides.

    Parameters
    ----------
    name_or_path :
        Bare preset name (no extension) or a path to a YAML file.

    Returns
    -------
    NoiseSettings
        The parsed preset; unset fields stay ``None`` so the resolver can
        fall through to higher-precedence layers.

    Raises
    ------
    FileNotFoundError
        If a bare name does not match any packaged preset, or the
        supplied path does not exist.
    """
    if _looks_like_path(name_or_path):
        path = Path(name_or_path)
        if not path.exists():
            raise FileNotFoundError(f"preset file not found: {path}")
        text = path.read_text()
    else:
        from importlib.resources import files

        candidate = files("ftmwpipeline.presets") / f"{name_or_path}.yaml"
        if not candidate.is_file():
            available = sorted(
                p.name[:-5]
                for p in files("ftmwpipeline.presets").iterdir()
                if p.name.endswith(".yaml")
            )
            raise FileNotFoundError(
                f"no packaged preset named {name_or_path!r}; " f"available: {available}"
            )
        text = candidate.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(
            f"preset YAML root must be a mapping; got {type(data)} from "
            f"{name_or_path}"
        )
    inner = data.get("stage2")
    if inner is None:
        return NoiseSettings()
    if not isinstance(inner, dict):
        raise ValueError(
            f"preset 'stage2' block must be a mapping; got {type(inner)} "
            f"from {name_or_path}"
        )
    block = dict(inner)
    for meta in ("name", "description"):
        if meta in data and meta not in block:
            block[meta] = data[meta]
    return from_yaml_dict(block)


def to_yaml(settings: NoiseSettings) -> str:
    """Serialize to a YAML string (sparse; omits unset fields)."""
    text: Any = yaml.safe_dump(
        to_yaml_dict(settings), sort_keys=False, default_flow_style=False
    )
    return cast(str, text)


__all__ = [
    "NoiseSettings",
    "resolve",
    "to_attrs",
    "from_attrs",
    "to_yaml",
    "from_yaml",
    "to_yaml_dict",
    "from_yaml_dict",
    "load_preset",
]
