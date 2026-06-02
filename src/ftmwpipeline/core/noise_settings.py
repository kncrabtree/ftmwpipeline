"""
Canonical Stage 2 noise-estimation settings.

``NoiseSettings`` is the single source of truth for the Stage 2 parameters
across every surface:

* the public API signatures (``Pipeline.estimate_noise`` /
  ``ftmwpipeline.api.estimate_noise``),
* the CLI ``--preset`` flag plus the existing per-knob flags,
* the resolution chain ``explicit > preset > persisted > recommended >
  hard default``,
* the persisted canonical record in ``processing_parameters/stage2_noise``,
* the YAML preset interchange format.

The dataclass mirrors :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`
and :class:`~ftmwpipeline.core.tau_calibration_settings.TauCalibrationSettings`:
every field is ``Optional`` with ``None`` meaning *unset* (fall through the
resolution chain). A *resolved* instance (produced by :func:`resolve`) has
every field filled with a hard default if no layer supplied a value.

The dataclass is structured into sub-dataclasses grouping the knobs by what
they configure. The adaptive estimator uses ``binning``, ``skewness``,
``smoothing``, and ``skirt_exclusion``; the scatter (high-pass) estimator uses
``scatter``. Both sub-block families coexist on one instance -- the ``method``
argument selects which the kernel consumes. The grouping maps 1:1 to HDF5
subgroups under ``processing_parameters/stage2_noise`` so each sub-block is
independently inspectable.

The *recommended* layer of :func:`resolve` is reserved but unused for
Stage 2 today -- Stage 2 has no upstream feeder. The layer is kept in
the signature so a future cross-stage recommender (e.g. Stage 1's
T_active-driven smoothing-window suggestion) can land without API churn.

The ``_HARD_DEFAULTS`` nested dict mirrors the module-level constants
(``DEFAULT_SMOOTHING_MHZ``, ``ABS_MIN_BIN_SIZE``, ``SUBDIVISION_THRESHOLD``,
``STRONG_PEAK_SNR``, ``SKIRT_EXCLUSION_K``, ``MAX_SKIRT_EXCLUSION_MHZ``) and
function-signature defaults (``skew_target``, ``inc``, ``min_bin_fraction``,
``min_noise_fraction``) in :mod:`ftmwpipeline.preprocessing.noise_estimation`.
Those constants are still imported by the kernel as its parameter defaults;
once every consumer reads from a resolved ``NoiseSettings``, the constants
become docstring-only and can be removed.

This module is dependency-free within the package (stdlib + PyYAML for
preset interchange) so it can be imported from ``core`` without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union, cast

import yaml  # type: ignore[import-untyped]

# Mirrors the marker used by io.fid_serialization for optional HDF5 attrs.
_NONE = "__None__"


# ---------------------------------------------------------------------------
# Sub-dataclasses (one per HDF5 subgroup / YAML block)
# ---------------------------------------------------------------------------
@dataclass
class BinningSubSettings:
    """MAD/median recursive subdivision knobs."""

    subdivision_threshold: Optional[float] = None
    abs_min_bin_size: Optional[int] = None
    min_bin_fraction: Optional[float] = None
    min_noise_fraction: Optional[float] = None


@dataclass
class SkewnessSubSettings:
    """Per-bin Rayleigh-target trim (drives the noise mask inside each bin)."""

    skew_target: Optional[float] = None
    inc: Optional[float] = None


@dataclass
class SmoothingSubSettings:
    """Moving-mean σ_x reconstruction."""

    smoothing_window_mhz: Optional[float] = None


@dataclass
class SkirtExclusionSubSettings:
    """Lorentzian-skirt mask refinement around strong lines."""

    strong_peak_snr: Optional[float] = None
    skirt_exclusion_k: Optional[float] = None
    max_skirt_exclusion_mhz: Optional[float] = None


@dataclass
class ScatterSubSettings:
    """High-pass, region-aware scatter-MAD estimator knobs.

    Mirrors the ``estimate_noise_scatter`` kernel signature; the
    ``method="scatter"`` path consumes these (the adaptive sub-blocks above
    are ignored, and vice versa).
    """

    window_mhz: Optional[float] = None
    pedestal_mhz: Optional[float] = None
    line_k: Optional[float] = None
    n_iter: Optional[int] = None
    region_aware: Optional[bool] = None
    smoothing_mhz: Optional[float] = None
    smoothing_percentile: Optional[float] = None
    convolve_mhz: Optional[float] = None


@dataclass
class NoiseSettings:
    """Stage 2 noise-estimation settings (see module docstring)."""

    binning: BinningSubSettings = field(default_factory=BinningSubSettings)
    skewness: SkewnessSubSettings = field(default_factory=SkewnessSubSettings)
    smoothing: SmoothingSubSettings = field(default_factory=SmoothingSubSettings)
    skirt_exclusion: SkirtExclusionSubSettings = field(
        default_factory=SkirtExclusionSubSettings
    )
    scatter: ScatterSubSettings = field(default_factory=ScatterSubSettings)

    def is_empty(self) -> bool:
        """True if no field is set across any sub-dataclass."""
        for sub_name in _SUB_NAMES:
            sub = getattr(self, sub_name)
            if any(getattr(sub, f.name) is not None for f in fields(sub)):
                return False
        return True


# Sub-dataclass field names on NoiseSettings, in HDF5/YAML order.
_SUB_NAMES = ("binning", "skewness", "smoothing", "skirt_exclusion", "scatter")


# Hard defaults per sub-dataclass. These mirror the module-level constants
# and function-signature defaults in ``preprocessing/noise_estimation.py``.
# Kept as inline literals (rather than imported from ``preprocessing/``) to
# keep ``core`` dependency-free from ``preprocessing``; the noise-estimator
# constants are the readable canonical source and these must track them.
_HARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "binning": {
        "subdivision_threshold": 0.08,
        "abs_min_bin_size": 300,
        "min_bin_fraction": 1 / 64,
        "min_noise_fraction": 2 / 3,
    },
    "skewness": {
        "skew_target": 0.631,
        "inc": 0.01,
    },
    "smoothing": {
        "smoothing_window_mhz": 300.0,
    },
    "skirt_exclusion": {
        "strong_peak_snr": 20.0,
        "skirt_exclusion_k": 1.5,
        "max_skirt_exclusion_mhz": 500.0,
    },
    "scatter": {
        "window_mhz": 80.0,
        "pedestal_mhz": 20.0,
        "line_k": 8.0,
        "n_iter": 3,
        "region_aware": True,
        "smoothing_mhz": 800.0,
        "smoothing_percentile": 50.0,
        "convolve_mhz": 200.0,
    },
}


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
def _first_set_field(name: str, *layers: Any) -> Any:
    for layer in layers:
        if layer is None:
            continue
        value = getattr(layer, name)
        if value is not None:
            return value
    return None


def _resolve_sub(
    sub_name: str,
    *layers: Optional["NoiseSettings"],
) -> Any:
    sub_layers = [getattr(s, sub_name) for s in layers if s is not None]
    template = getattr(NoiseSettings(), sub_name)
    merged = type(template)()
    for f in fields(template):
        value = _first_set_field(f.name, *sub_layers)
        if value is None:
            value = _HARD_DEFAULTS.get(sub_name, {}).get(f.name)
        setattr(merged, f.name, value)
    return merged


def resolve(
    explicit: Optional[NoiseSettings] = None,
    preset: Optional[NoiseSettings] = None,
    persisted: Optional[NoiseSettings] = None,
    recommended: Optional[NoiseSettings] = None,
) -> NoiseSettings:
    """Merge the four layers by precedence into a resolved ``NoiseSettings``.

    Per-field precedence: ``explicit > preset > persisted > recommended``,
    then any remaining ``None`` field falls back to the matching value in
    :data:`_HARD_DEFAULTS`. The ``recommended`` layer is reserved for a
    future upstream recommender; Stage 2 call sites currently pass
    ``None`` there.
    """
    layers = (explicit, preset, persisted, recommended)
    merged = NoiseSettings()
    for sub_name in _SUB_NAMES:
        setattr(merged, sub_name, _resolve_sub(sub_name, *layers))
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


def _sub_to_attrs(sub: Any) -> Dict[str, Any]:
    return {f.name: _encode_value(getattr(sub, f.name)) for f in fields(sub)}


def _sub_from_attrs(cls: type, attrs: Dict[str, Any]) -> Any:
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in attrs:
            continue
        kwargs[f.name] = _decode_value(attrs[f.name])
    return cls(**kwargs)


def to_attrs(settings: NoiseSettings) -> Dict[str, Any]:
    """Nested attrs dict (one top-level key per sub-dataclass).

    Sub-dataclass values use ``__None__`` for unset fields.
    """
    out: Dict[str, Any] = {}
    for sub_name in _SUB_NAMES:
        out[sub_name] = _sub_to_attrs(getattr(settings, sub_name))
    return out


def from_attrs(attrs: Dict[str, Any]) -> NoiseSettings:
    settings = NoiseSettings()
    for sub_name in _SUB_NAMES:
        sub_attrs = attrs.get(sub_name, {})
        if not isinstance(sub_attrs, dict):
            raise ValueError(
                f"sub-block {sub_name!r} must be a mapping; got {type(sub_attrs)}"
            )
        template = getattr(NoiseSettings(), sub_name)
        setattr(settings, sub_name, _sub_from_attrs(type(template), sub_attrs))
    return settings


# ---------------------------------------------------------------------------
# YAML interchange
# ---------------------------------------------------------------------------
def _yaml_sub_to_mapping(sub: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in fields(sub):
        value = getattr(sub, f.name)
        if value is None:
            continue
        out[f.name] = value
    return out


def to_yaml_dict(settings: NoiseSettings) -> Dict[str, Any]:
    """Sparse nested dict suitable for ``yaml.safe_dump`` (omits ``None``)."""
    out: Dict[str, Any] = {}
    for sub_name in _SUB_NAMES:
        sub_dict = _yaml_sub_to_mapping(getattr(settings, sub_name))
        if sub_dict:
            out[sub_name] = sub_dict
    return out


def from_yaml_dict(data: Optional[Mapping[str, Any]]) -> NoiseSettings:
    if data is None:
        return NoiseSettings()
    if not isinstance(data, dict):
        raise ValueError(f"preset YAML root must be a mapping; got {type(data)}")
    settings = NoiseSettings()
    known_subs = set(_SUB_NAMES)
    for sub_name in _SUB_NAMES:
        if sub_name not in data:
            continue
        block = data[sub_name]
        if not isinstance(block, dict):
            raise ValueError(
                f"preset block {sub_name!r} must be a mapping; got {type(block)}"
            )
        template = getattr(NoiseSettings(), sub_name)
        valid_names = {f.name for f in fields(template)}
        unknown = set(block) - valid_names
        if unknown:
            raise ValueError(
                f"unknown {sub_name!r} fields in preset: {sorted(unknown)} "
                f"(valid: {sorted(valid_names)})"
            )
        setattr(settings, sub_name, type(template)(**block))
    allowed_top = known_subs | {"name", "description"}
    extra_top = set(data) - allowed_top
    if extra_top:
        raise ValueError(
            f"unknown top-level preset keys: {sorted(extra_top)} "
            f"(allowed: {sorted(allowed_top)})"
        )
    return settings


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
    resources (e.g. ``"instrument_bc_2638"`` ->
    ``ftmwpipeline/presets/instrument_bc_2638.yaml``); paths load directly.
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
                f"no packaged preset named {name_or_path!r}; "
                f"available: {available}"
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
    "BinningSubSettings",
    "SkewnessSubSettings",
    "SmoothingSubSettings",
    "SkirtExclusionSubSettings",
    "ScatterSubSettings",
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
