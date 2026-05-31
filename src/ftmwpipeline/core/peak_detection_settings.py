"""
Canonical Stage 3 peak-detection settings.

``PeakDetectionSettings`` is the single source of truth for the Stage 3
parameters across every surface:

* the public API signatures (``Pipeline.detect_peaks`` /
  ``ftmwpipeline.api.detect_peaks``),
* the CLI ``--preset`` flag plus the existing per-knob flags,
* the resolution chain ``explicit > preset > persisted > recommended >
  hard default``,
* the persisted canonical record in ``processing_parameters/stage3_peaks``,
* the YAML preset interchange format.

The dataclass mirrors :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`,
:class:`~ftmwpipeline.core.tau_calibration_settings.TauCalibrationSettings`,
and :class:`~ftmwpipeline.core.noise_settings.NoiseSettings`: every field
is ``Optional`` with ``None`` meaning *unset* (fall through the resolution
chain). A *resolved* instance (produced by :func:`resolve`) has every
field filled with a hard default if no layer supplied a value.

The dataclass is structured into four sub-dataclasses grouping the knobs
by what they configure: ``promotion``, ``savgol``, ``primary_pass``, and
``gap_pass``. The grouping maps 1:1 to HDF5 subgroups under
``processing_parameters/stage3_peaks`` so each sub-block is independently
inspectable.

The *recommended* layer of :func:`resolve` is reserved but unused for
Stage 3 today -- the Stage 2b ``τ_maj`` value the gap pass uses for its
matched-filter ``tau_basis_us`` is a *runtime value*, not a settings knob,
so Stage 3's impl reads it directly from the persisted
``stage2b_tau_calibration`` block rather than routing it through the
resolver. The layer is kept in the signature so a future cross-stage
recommender can land without API churn.

The ``_HARD_DEFAULTS`` nested dict mirrors the module-level constants in
:mod:`ftmwpipeline.preprocessing.peak_detection`
(``DEFAULT_MIN_SNR``, ``DEFAULT_INTERNAL_MIN_SNR``,
``DEFAULT_WEAK_MEDIUM_SNR``, ``DEFAULT_MEDIUM_STRONG_SNR``) and in
:mod:`ftmwpipeline._internal.stage3_impl`
(``DEFAULT_PRIMARY_WINDOW``, ``GAP_MASK_EDGE_THRESHOLD``,
``_DETECTION_ZPF``, ``_GAP_ACTIVE_ZPF``, ``_SG_FWHM_COVERAGE``,
``_SG_MIN_WINDOW``), plus the hardcoded ``sg_window=11`` and
``sg_order=3`` defaults inside ``detect_peaks_impl``. Those constants
are still imported by the kernel and orchestrator as their parameter
defaults; once every consumer reads from a resolved
``PeakDetectionSettings``, the constants become docstring-only.

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
class PromotionSubSettings:
    """SNR cutoffs governing detection-floor and user-grid promotion.

    ``min_snr`` is the user-grid promotion cutoff (which peaks move to
    Stage 4); ``internal_min_snr`` caps the floor at which detection
    actually runs on the internal zpf=1 grids (effective floor is
    ``min(internal_min_snr, min_snr)``). ``weak_medium_snr`` and
    ``medium_strong_snr`` are the classification bin edges.
    """

    min_snr: Optional[float] = None
    internal_min_snr: Optional[float] = None
    weak_medium_snr: Optional[float] = None
    medium_strong_snr: Optional[float] = None


@dataclass
class SavgolSubSettings:
    """Savitzky-Golay second-derivative apex localiser knobs.

    ``sg_window`` is the primary-pass fixed window (odd, > sg_order).
    The gap-pass window is computed at runtime from the actual grid
    spacing and the line FWHM via ``sg_fwhm_coverage * FWHM / freq_step``
    rounded up to odd, floored at ``sg_min_window``.
    """

    sg_window: Optional[int] = None
    sg_order: Optional[int] = None
    sg_fwhm_coverage: Optional[float] = None
    sg_min_window: Optional[int] = None


@dataclass
class PrimaryPassSubSettings:
    """Primary-pass apodization + zpf knobs.

    The primary pass runs at zpf=``detection_zpf`` on a strongly-windowed
    spectrum (``primary_window``) to suppress truncation sidelobes;
    ``min_exclusion_mhz`` is the half-width around every primary detection
    that the gap pass excludes from its mask. ``primary_leakage_floor_k``
    scales the continuous leakage-aware detection floor ``k·(S_coh/√M)·σ``
    added to the primary-pass threshold so that a strong line's coherent
    skirt ripple is not re-detected as weak lines (the primary pass has no
    hard leakage mask -- a hard mask would delete the strong lines that
    generate the coherence). ``0`` disables the floor.
    """

    primary_window: Optional[str] = None
    min_exclusion_mhz: Optional[float] = None
    detection_zpf: Optional[int] = None
    primary_leakage_floor_k: Optional[float] = None


@dataclass
class GapPassSubSettings:
    """Matched-filter gap-pass knobs.

    ``run_gap_pass`` enables/disables the pass. ``gap_active_zpf`` is the
    zero-padding factor for the active-region rfft (chosen so the Lorentzian
    FWHM lands at ~3 bins on the resulting grid; SavGol's operating range).
    ``gap_mask_edge_threshold`` is the de-ramped coherent-leakage map cutoff
    above which gap-pass detections are dropped as sidelobes.

    ``tau_basis_us`` (the matched filter's exponential time constant) is
    *not* a Stage 3 knob -- it is the upstream-feeder value the Stage 3
    impl picks up from the persisted Stage 2b ``τ_maj`` (or falls back to
    the Stage 1 user apodization).
    """

    run_gap_pass: Optional[bool] = None
    gap_active_zpf: Optional[int] = None
    gap_mask_edge_threshold: Optional[float] = None


@dataclass
class PeakDetectionSettings:
    """Stage 3 peak-detection settings (see module docstring)."""

    promotion: PromotionSubSettings = field(default_factory=PromotionSubSettings)
    savgol: SavgolSubSettings = field(default_factory=SavgolSubSettings)
    primary_pass: PrimaryPassSubSettings = field(
        default_factory=PrimaryPassSubSettings
    )
    gap_pass: GapPassSubSettings = field(default_factory=GapPassSubSettings)

    def is_empty(self) -> bool:
        """True if no field is set across any sub-dataclass."""
        for sub_name in _SUB_NAMES:
            sub = getattr(self, sub_name)
            if any(getattr(sub, f.name) is not None for f in fields(sub)):
                return False
        return True


# Sub-dataclass field names on PeakDetectionSettings, in HDF5/YAML order.
_SUB_NAMES = ("promotion", "savgol", "primary_pass", "gap_pass")


# Hard defaults per sub-dataclass. These mirror the module-level constants
# in ``preprocessing/peak_detection.py`` and ``_internal/stage3_impl.py``.
# Kept as inline literals (rather than imported from those modules) to
# keep ``core`` dependency-free from ``preprocessing`` / ``_internal``;
# the kernel modules' constants are the readable canonical source and
# these must track them.
_HARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "promotion": {
        "min_snr": 3.0,
        "internal_min_snr": 2.0,
        "weak_medium_snr": 10.0,
        "medium_strong_snr": 50.0,
    },
    "savgol": {
        "sg_window": 11,
        "sg_order": 3,
        "sg_fwhm_coverage": 4.0,
        "sg_min_window": 5,
    },
    "primary_pass": {
        "primary_window": "blackmanharris",
        "min_exclusion_mhz": 0.0,
        "detection_zpf": 1,
        "primary_leakage_floor_k": 1.0,
    },
    "gap_pass": {
        "run_gap_pass": True,
        "gap_active_zpf": 2,
        "gap_mask_edge_threshold": 8.0,
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
    *layers: Optional["PeakDetectionSettings"],
) -> Any:
    sub_layers = [getattr(s, sub_name) for s in layers if s is not None]
    template = getattr(PeakDetectionSettings(), sub_name)
    merged = type(template)()
    for f in fields(template):
        value = _first_set_field(f.name, *sub_layers)
        if value is None:
            value = _HARD_DEFAULTS.get(sub_name, {}).get(f.name)
        setattr(merged, f.name, value)
    return merged


def resolve(
    explicit: Optional[PeakDetectionSettings] = None,
    preset: Optional[PeakDetectionSettings] = None,
    persisted: Optional[PeakDetectionSettings] = None,
    recommended: Optional[PeakDetectionSettings] = None,
) -> PeakDetectionSettings:
    """Merge the four layers by precedence into a resolved ``PeakDetectionSettings``.

    Per-field precedence: ``explicit > preset > persisted > recommended``,
    then any remaining ``None`` field falls back to the matching value in
    :data:`_HARD_DEFAULTS`. The ``recommended`` layer is reserved for a
    future upstream recommender; Stage 3 call sites currently pass
    ``None`` there.
    """
    layers = (explicit, preset, persisted, recommended)
    merged = PeakDetectionSettings()
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


def to_attrs(settings: PeakDetectionSettings) -> Dict[str, Any]:
    """Nested attrs dict (one top-level key per sub-dataclass).

    Sub-dataclass values use ``__None__`` for unset fields.
    """
    out: Dict[str, Any] = {}
    for sub_name in _SUB_NAMES:
        out[sub_name] = _sub_to_attrs(getattr(settings, sub_name))
    return out


def from_attrs(attrs: Dict[str, Any]) -> PeakDetectionSettings:
    settings = PeakDetectionSettings()
    for sub_name in _SUB_NAMES:
        sub_attrs = attrs.get(sub_name, {})
        if not isinstance(sub_attrs, dict):
            raise ValueError(
                f"sub-block {sub_name!r} must be a mapping; got {type(sub_attrs)}"
            )
        template = getattr(PeakDetectionSettings(), sub_name)
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


def to_yaml_dict(settings: PeakDetectionSettings) -> Dict[str, Any]:
    """Sparse nested dict suitable for ``yaml.safe_dump`` (omits ``None``)."""
    out: Dict[str, Any] = {}
    for sub_name in _SUB_NAMES:
        sub_dict = _yaml_sub_to_mapping(getattr(settings, sub_name))
        if sub_dict:
            out[sub_name] = sub_dict
    return out


def from_yaml_dict(data: Optional[Mapping[str, Any]]) -> PeakDetectionSettings:
    if data is None:
        return PeakDetectionSettings()
    if not isinstance(data, dict):
        raise ValueError(f"preset YAML root must be a mapping; got {type(data)}")
    settings = PeakDetectionSettings()
    known_subs = set(_SUB_NAMES)
    for sub_name in _SUB_NAMES:
        if sub_name not in data:
            continue
        block = data[sub_name]
        if not isinstance(block, dict):
            raise ValueError(
                f"preset block {sub_name!r} must be a mapping; got {type(block)}"
            )
        template = getattr(PeakDetectionSettings(), sub_name)
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


def from_yaml(source: Union[str, Path]) -> PeakDetectionSettings:
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


def load_preset(name_or_path: Union[str, Path]) -> PeakDetectionSettings:
    """Load a Stage 3 preset by bare name or by filesystem path.

    Bare names resolve against the packaged ``ftmwpipeline.presets``
    resources (e.g. ``"instrument_bc_2638"`` ->
    ``ftmwpipeline/presets/instrument_bc_2638.yaml``); paths load directly.
    Preset YAML wraps the Stage 3 settings inside a top-level ``stage3:``
    block (alongside optional ``stage2:`` / ``stage2b:`` / ``stage5:``
    blocks for other stages).

    The ``name:`` and ``description:`` metadata fields are accepted but
    ignored by the settings parser -- they're documentation for the preset
    author.

    Returns an empty :class:`PeakDetectionSettings` (no fields set) when
    the preset carries no ``stage3:`` block, so a stage-spanning preset
    that omits Stage 3 loads cleanly without producing spurious overrides.

    Parameters
    ----------
    name_or_path :
        Bare preset name (no extension) or a path to a YAML file.

    Returns
    -------
    PeakDetectionSettings
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
    inner = data.get("stage3")
    if inner is None:
        return PeakDetectionSettings()
    if not isinstance(inner, dict):
        raise ValueError(
            f"preset 'stage3' block must be a mapping; got {type(inner)} "
            f"from {name_or_path}"
        )
    block = dict(inner)
    for meta in ("name", "description"):
        if meta in data and meta not in block:
            block[meta] = data[meta]
    return from_yaml_dict(block)


def to_yaml(settings: PeakDetectionSettings) -> str:
    """Serialize to a YAML string (sparse; omits unset fields)."""
    text: Any = yaml.safe_dump(
        to_yaml_dict(settings), sort_keys=False, default_flow_style=False
    )
    return cast(str, text)


__all__ = [
    "PromotionSubSettings",
    "SavgolSubSettings",
    "PrimaryPassSubSettings",
    "GapPassSubSettings",
    "PeakDetectionSettings",
    "resolve",
    "to_attrs",
    "from_attrs",
    "to_yaml",
    "from_yaml",
    "to_yaml_dict",
    "from_yaml_dict",
    "load_preset",
]
