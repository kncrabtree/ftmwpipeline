"""
Canonical Stage 4 window-planning settings.

``WindowPlanningSettings`` is the single source of truth for the Stage 4
parameters across every surface:

* the public API signatures (``Pipeline.assign_windows`` /
  ``ftmwpipeline.api.assign_windows``),
* the CLI ``--preset`` flag plus the existing per-knob flags,
* the resolution chain ``explicit > persisted > preset > recommended >
  hard default``,
* the persisted canonical record in ``processing_parameters/stage4_windows``,
* the YAML preset interchange format.

The dataclass mirrors the architectural template established by Stages 5,
2b, 2, and 3: every field is ``Optional`` with ``None`` meaning *unset*
(fall through the resolution chain). A *resolved* instance (produced by
:func:`resolve`) has every field filled with a hard default if no layer
supplied a value.

The dataclass is structured into four sub-dataclasses grouping the knobs
by what they configure: ``coherence``, ``clustering``, ``contributor``,
and ``leakage``. The grouping maps 1:1 to HDF5 subgroups under
``processing_parameters/stage4_windows`` so each sub-block is
independently inspectable.

The *recommended* layer of :func:`resolve` is reserved but unused for
Stage 4 today -- Stage 4 has no automatic upstream recommender. The slot
is kept in the signature so a future cross-stage recommender (e.g. a
Stage 2b τ_maj-driven ``tau_us`` suggestion) can land without API churn.

The ``_HARD_DEFAULTS`` nested dict mirrors the ``DEFAULT_*`` constants in
:mod:`ftmwpipeline.preprocessing.window_planning` and
:mod:`ftmwpipeline.preprocessing.edge_coherence`. Those constants are
still imported by the kernel as its parameter defaults; once every
consumer reads from a resolved ``WindowPlanningSettings``, the constants
become docstring-only and can be removed.

``leakage.tau_us`` is allowed to remain ``None`` after resolution: the
hard default is intentionally ``None`` (boxcar / undamped limit -- the
analytic leakage envelope reduces to ``2/(2π·Δf·T)``). A future
auto-feeder from the persisted Stage 2b ``τ_maj`` would fill this slot
via the resolver's *recommended* layer; until then, users override it
explicitly via ``settings.leakage.tau_us = ...``.

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
class CoherenceSubSettings:
    """Rolling edge-coherence statistic knobs (drive the spatial partition).

    ``edge_m`` and ``trim_m`` are the band widths used by the rolling-scan
    and trim-refinement coherence statistics; ``edge_threshold`` is the
    ``S_coh`` cutoff ``T_edge`` above which a frequency interval is
    considered leakage-touched.
    """

    edge_m: Optional[int] = None
    trim_m: Optional[int] = None
    edge_threshold: Optional[float] = None


@dataclass
class ClusteringSubSettings:
    """Window-extent decisions.

    ``max_window_width_mhz`` is the width cap above which a window is
    classified as HARD and gets a split proposal; ``min_window_half_width_mhz``
    is the minimum half-width of a window built around an isolated weak line.
    ``max_peaks_per_window`` is the per-window promoted-peak cap; ``0`` (the
    default) disables it so a window is bounded only by ``max_window_width_mhz``.
    Bounding by width alone keeps the Stage 5 AICc-with-``n_eff`` gate's effective
    sample size large enough to self-regulate K on dense clusters; a fragmenting
    peak cap starved it and drove both under- and over-fit. The width cap (and the
    width-bounded strong-cluster merge) is what prevents a dense ultra-high-SNR
    spectrum from collapsing into one GHz-scale mega-window. A positive value
    restores an explicit cap and tracks the Stage 5 ``conservative.max_peaks``.
    """

    max_window_width_mhz: Optional[float] = None
    min_window_half_width_mhz: Optional[float] = None
    max_peaks_per_window: Optional[int] = None


@dataclass
class ContributorSubSettings:
    """Fixed-contributor freeze/attach decisions.

    ``min_freeze_snr`` is the freeze-eligibility SNR cutoff (a fixed
    contributor below this is flagged as a thaw-and-re-fit candidate
    rather than safely frozen); ``magnitude_attachment_threshold`` is the
    analytic-skirt-magnitude attachment rule (in units of σ_c on the
    target window) governing which strong promoted peaks are attached to
    a window's ``fixed_contributors``.
    """

    min_freeze_snr: Optional[float] = None
    magnitude_attachment_threshold: Optional[float] = None


@dataclass
class LeakageSubSettings:
    """Leakage-envelope parameter.

    ``tau_us`` is the assumed shared time-domain decay constant used by
    the analytic finite-T leakage envelope. ``None`` (the hard default)
    means undamped/boxcar -- the envelope reduces to ``2/(2π·Δf·T)``. A
    future auto-feeder may populate this from the persisted Stage 2b
    ``τ_maj`` via the resolver's recommended layer.
    """

    tau_us: Optional[float] = None


@dataclass
class WindowPlanningSettings:
    """Stage 4 window-planning settings (see module docstring)."""

    coherence: CoherenceSubSettings = field(default_factory=CoherenceSubSettings)
    clustering: ClusteringSubSettings = field(default_factory=ClusteringSubSettings)
    contributor: ContributorSubSettings = field(default_factory=ContributorSubSettings)
    leakage: LeakageSubSettings = field(default_factory=LeakageSubSettings)

    def is_empty(self) -> bool:
        """True if no field is set across any sub-dataclass."""
        for sub_name in _SUB_NAMES:
            sub = getattr(self, sub_name)
            if any(getattr(sub, f.name) is not None for f in fields(sub)):
                return False
        return True


# Sub-dataclass field names on WindowPlanningSettings, in HDF5/YAML order.
_SUB_NAMES = ("coherence", "clustering", "contributor", "leakage")


# Hard defaults per sub-dataclass. These mirror the ``DEFAULT_*`` constants
# in ``preprocessing/window_planning.py`` and
# ``preprocessing/edge_coherence.py``. Kept as inline literals (rather than
# imported from ``preprocessing/``) to keep ``core`` dependency-free from
# ``preprocessing``; the kernel module's constants are the readable canonical
# source and these must track them.
_HARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "coherence": {
        "edge_m": 64,
        "trim_m": 32,
        "edge_threshold": 8.0,
    },
    "clustering": {
        "max_window_width_mhz": 40.0,
        "min_window_half_width_mhz": 2.0,
        "max_peaks_per_window": 0,
    },
    "contributor": {
        "min_freeze_snr": 50.0,
        "magnitude_attachment_threshold": 0.1,
    },
    "leakage": {
        # ``tau_us`` legitimately stays None (boxcar / undamped limit).
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
    *layers: Optional["WindowPlanningSettings"],
) -> Any:
    sub_layers = [getattr(s, sub_name) for s in layers if s is not None]
    template = getattr(WindowPlanningSettings(), sub_name)
    merged = type(template)()
    for f in fields(template):
        value = _first_set_field(f.name, *sub_layers)
        if value is None:
            value = _HARD_DEFAULTS.get(sub_name, {}).get(f.name)
        setattr(merged, f.name, value)
    return merged


def resolve(
    explicit: Optional[WindowPlanningSettings] = None,
    preset: Optional[WindowPlanningSettings] = None,
    persisted: Optional[WindowPlanningSettings] = None,
    recommended: Optional[WindowPlanningSettings] = None,
) -> WindowPlanningSettings:
    """Merge the four layers by precedence into a resolved ``WindowPlanningSettings``.

    Per-field precedence: ``explicit > persisted > preset > recommended``,
    then any remaining ``None`` field falls back to the matching value in
    :data:`_HARD_DEFAULTS`. A value persisted in the ``.ftmw`` outranks a
    ``.yml`` preset, so the preset only seeds fields the file has not fixed
    and a shared experiment reproduces from the file alone. The
    ``recommended`` layer is reserved for a future upstream recommender;
    Stage 4 call sites currently pass ``None`` there.
    """
    layers = (explicit, persisted, preset, recommended)
    merged = WindowPlanningSettings()
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


def to_attrs(settings: WindowPlanningSettings) -> Dict[str, Any]:
    """Nested attrs dict (one top-level key per sub-dataclass).

    Sub-dataclass values use ``__None__`` for unset fields.
    """
    out: Dict[str, Any] = {}
    for sub_name in _SUB_NAMES:
        out[sub_name] = _sub_to_attrs(getattr(settings, sub_name))
    return out


def from_attrs(attrs: Dict[str, Any]) -> WindowPlanningSettings:
    settings = WindowPlanningSettings()
    for sub_name in _SUB_NAMES:
        sub_attrs = attrs.get(sub_name, {})
        if not isinstance(sub_attrs, dict):
            raise ValueError(
                f"sub-block {sub_name!r} must be a mapping; got {type(sub_attrs)}"
            )
        template = getattr(WindowPlanningSettings(), sub_name)
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


def to_yaml_dict(settings: WindowPlanningSettings) -> Dict[str, Any]:
    """Sparse nested dict suitable for ``yaml.safe_dump`` (omits ``None``)."""
    out: Dict[str, Any] = {}
    for sub_name in _SUB_NAMES:
        sub_dict = _yaml_sub_to_mapping(getattr(settings, sub_name))
        if sub_dict:
            out[sub_name] = sub_dict
    return out


def from_yaml_dict(data: Optional[Mapping[str, Any]]) -> WindowPlanningSettings:
    if data is None:
        return WindowPlanningSettings()
    if not isinstance(data, dict):
        raise ValueError(f"preset YAML root must be a mapping; got {type(data)}")
    settings = WindowPlanningSettings()
    known_subs = set(_SUB_NAMES)
    for sub_name in _SUB_NAMES:
        if sub_name not in data:
            continue
        block = data[sub_name]
        if not isinstance(block, dict):
            raise ValueError(
                f"preset block {sub_name!r} must be a mapping; got {type(block)}"
            )
        template = getattr(WindowPlanningSettings(), sub_name)
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


def from_yaml(source: Union[str, Path]) -> WindowPlanningSettings:
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


def load_preset(name_or_path: Union[str, Path]) -> WindowPlanningSettings:
    """Load a Stage 4 preset by bare name or by filesystem path.

    Bare names resolve against the packaged ``ftmwpipeline.presets``
    resources (e.g. ``"instrument_bc_2638"`` ->
    ``ftmwpipeline/presets/instrument_bc_2638.yaml``); paths load directly.
    Preset YAML wraps the Stage 4 settings inside a top-level ``stage4:``
    block (alongside optional ``stage2:`` / ``stage2b:`` / ``stage3:`` /
    ``stage5:`` blocks for other stages).

    The ``name:`` and ``description:`` metadata fields are accepted but
    ignored by the settings parser -- they're documentation for the preset
    author.

    Returns an empty :class:`WindowPlanningSettings` (no fields set) when
    the preset carries no ``stage4:`` block, so a stage-spanning preset
    that omits Stage 4 loads cleanly without producing spurious overrides.

    Parameters
    ----------
    name_or_path :
        Bare preset name (no extension) or a path to a YAML file.

    Returns
    -------
    WindowPlanningSettings
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
    inner = data.get("stage4")
    if inner is None:
        return WindowPlanningSettings()
    if not isinstance(inner, dict):
        raise ValueError(
            f"preset 'stage4' block must be a mapping; got {type(inner)} "
            f"from {name_or_path}"
        )
    block = dict(inner)
    for meta in ("name", "description"):
        if meta in data and meta not in block:
            block[meta] = data[meta]
    return from_yaml_dict(block)


def to_yaml(settings: WindowPlanningSettings) -> str:
    """Serialize to a YAML string (sparse; omits unset fields)."""
    text: Any = yaml.safe_dump(
        to_yaml_dict(settings), sort_keys=False, default_flow_style=False
    )
    return cast(str, text)


__all__ = [
    "CoherenceSubSettings",
    "ClusteringSubSettings",
    "ContributorSubSettings",
    "LeakageSubSettings",
    "WindowPlanningSettings",
    "resolve",
    "to_attrs",
    "from_attrs",
    "to_yaml",
    "from_yaml",
    "to_yaml_dict",
    "from_yaml_dict",
    "load_preset",
]
