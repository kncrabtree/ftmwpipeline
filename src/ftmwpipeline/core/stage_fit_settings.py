"""
Canonical Stage 5 fit settings.

``StageFitSettings`` is the single source of truth for the Stage 5 fitting
parameters across every surface:

* the public API signatures (``Pipeline.fit_peaks`` / ``api.fit_peaks``),
* the CLI ``--preset`` flag plus the existing per-knob flags,
* the resolution chain ``explicit > persisted > preset > recommended >
  hard default``,
* the persisted canonical record in ``processing_parameters/stage5_fit``,
* the YAML preset interchange format.

Every field is ``Optional`` with ``None`` meaning *unset* (fall through the
resolution chain). A *resolved* instance (produced by :func:`resolve`) has
every field filled with the hard default if no layer supplied a value.

The dataclass is structured into one top-level setting (``shape``) plus six
sub-dataclasses grouping the knobs by what they configure
(``tau``, ``seeder``, ``conservative``, ``penalties``, ``rescue``, ``thaw``).
The grouping maps 1:1 to HDF5 subgroups under
``processing_parameters/stage5_fit`` so each sub-block is independently
inspectable.

The ``_HARD_DEFAULTS`` nested dict mirrors the ``DEFAULT_*`` constants in the
fitting modules. Those constants are still imported by the fitting functions
as their parameter defaults; once every consumer reads from a resolved
``StageFitSettings``, the constants become docstring-only and can be removed.

This module is dependency-free within the package (stdlib + the local
``__None__`` HDF5 marker convention shared with ``io.fid_serialization``,
plus PyYAML for the preset interchange) so it can be imported from ``core``
without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union, cast

import yaml  # type: ignore[import-untyped]

from .peak_shape import PeakShape

# Mirrors the marker used by io.fid_serialization for optional HDF5 attrs.
_NONE = "__None__"


# ---------------------------------------------------------------------------
# Shape spec (discriminated union scaffold)
# ---------------------------------------------------------------------------
@dataclass
class ShapeSpec:
    """Per-line envelope shape discriminator.

    ``kind`` is the :class:`PeakShape` enum -- ``LORENTZIAN`` /
    ``GAUSSIAN`` today, with ``VOIGT`` anticipated. Shape-specific parameter
    blocks (e.g. ``voigt_params: Optional[VoigtParams]``) attach here when
    Voigt support lands; no API churn elsewhere is required.
    """

    kind: PeakShape = PeakShape.LORENTZIAN

    @classmethod
    def coerce(cls, value: Any) -> Optional["ShapeSpec"]:
        """Coerce a shape-like value into a :class:`ShapeSpec` (or ``None``).

        Accepts the dataclass itself, a :class:`PeakShape`, a string member
        (``"gaussian"``), a mapping like ``{"kind": "gaussian"}``, or ``None``.
        Other inputs raise ``ValueError``.
        """
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, PeakShape):
            return cls(kind=value)
        if isinstance(value, str):
            return cls(kind=PeakShape.coerce(value))
        if isinstance(value, Mapping):
            if "kind" not in value:
                raise ValueError(
                    f"ShapeSpec mapping must carry a 'kind' key; got {value!r}"
                )
            return cls(kind=PeakShape.coerce(value["kind"]))
        raise ValueError(f"cannot coerce {value!r} to ShapeSpec")


# ---------------------------------------------------------------------------
# Sub-dataclasses (one per HDF5 subgroup / YAML block)
# ---------------------------------------------------------------------------
@dataclass
class TauSubSettings:
    """Stage 5 τ handling (initial guess, bounds, free-vs-fixed, anchoring)."""

    tau0_us: Optional[float] = None
    fit_tau: Optional[bool] = None
    max_decay_factor: Optional[float] = None
    fit_tau_min_snr: Optional[float] = None
    tau_penalty_lambda: Optional[float] = None
    tau_penalty_n_sigma: Optional[float] = None
    tau_maj_override_us: Optional[float] = None
    sigma_tau_override_us: Optional[float] = None
    per_band_tau: Optional[bool] = None


@dataclass
class SeederSubSettings:
    """Conservative-fit blend-aware seeder thresholds."""

    seeder_rchi2: Optional[float] = None
    seeder_straddle_factor: Optional[float] = None
    seeder_max_k: Optional[int] = None


@dataclass
class ConservativeSubSettings:
    """Add-one-peak loop gates and per-call caps."""

    significance: Optional[float] = None
    max_peaks: Optional[int] = None
    patience: Optional[int] = None
    min_separation_factor: Optional[float] = None
    min_pair_separation_factor: Optional[float] = None
    min_pair_separation_resolution_factor: Optional[float] = None
    n_eff_kind: Optional[str] = None
    weak_window_snr_threshold: Optional[float] = None
    max_nfev: Optional[int] = None


@dataclass
class PenaltySubSettings:
    """Phase / amplitude soft-penalty weights."""

    phase_penalty_lambda: Optional[float] = None
    phase_penalty_cutoff_fwhm: Optional[float] = None
    amp_penalty_lambda: Optional[float] = None
    amp_max_headroom: Optional[float] = None


@dataclass
class RescueSubSettings:
    """Residual-rescue B-loop knobs."""

    max_rounds: Optional[int] = None
    snr_threshold: Optional[float] = None
    prominence_threshold: Optional[float] = None
    cleanup_significance: Optional[float] = None
    merge_separation_factor: Optional[float] = None
    structural_merge_factor: Optional[float] = None
    overfit_amp_ratio_band: Optional[float] = None
    overfit_amp_ratio_threshold: Optional[float] = None


@dataclass
class ThawSubSettings:
    """Local-thaw + structural-replan orchestration."""

    max_thaw_rounds: Optional[int] = None
    max_replan_rounds: Optional[int] = None
    residual_edge_threshold: Optional[float] = None
    residual_edge_m: Optional[int] = None


@dataclass
class BaselineSubSettings:
    """Leakage-wing complex-baseline nuisance term knobs.

    An evidence-triggered low-order complex baseline ``B(u) = Σ_{k≤p}
    (a_k + i b_k)(u/u_s)^k`` added to a window's fit to absorb the coherent
    residual a neighbouring strong line's mismodeled leakage skirt leaves
    behind. Fires only where ``residual_edge_coherence`` exceeds
    ``edge_threshold`` (a dedicated threshold well below the thaw default of
    8.0); fit jointly with the free lines so its flexibility is priced into
    the reported per-line uncertainties. See
    ``dev-docs/planning/stage5-leakage-wing-baseline.md``.
    """

    enabled: Optional[bool] = None
    order: Optional[int] = None
    edge_threshold: Optional[float] = None


@dataclass
class SpurSubSettings:
    """Clock/LO-spur detection + masking knobs.

    A spur is a persistent CW tone (clock harmonic): a single-bin delta no
    finite-T line shape can represent. The gate is
    ``integer-MHz ∧ (frequency-domain narrow ∨ Stage 2b flat/saturated)``;
    detected spurs are dropped from peak nomination and excluded from the
    residual / chi-squared. See ``dev-docs/planning/stage5-spur-masking.md``.
    """

    enabled: Optional[bool] = None
    integer_tol_mhz: Optional[float] = None
    narrowness_ratio: Optional[float] = None
    snr_threshold: Optional[float] = None
    mask_half_width_bins: Optional[int] = None
    use_stft_catalogue: Optional[bool] = None


@dataclass
class StageFitSettings:
    """Stage 5 fit settings (see module docstring)."""

    shape: Optional[ShapeSpec] = None
    tau: TauSubSettings = field(default_factory=TauSubSettings)
    seeder: SeederSubSettings = field(default_factory=SeederSubSettings)
    conservative: ConservativeSubSettings = field(
        default_factory=ConservativeSubSettings
    )
    penalties: PenaltySubSettings = field(default_factory=PenaltySubSettings)
    rescue: RescueSubSettings = field(default_factory=RescueSubSettings)
    thaw: ThawSubSettings = field(default_factory=ThawSubSettings)
    spur: SpurSubSettings = field(default_factory=SpurSubSettings)
    baseline: BaselineSubSettings = field(default_factory=BaselineSubSettings)

    def is_empty(self) -> bool:
        """True if no field is set across any sub-dataclass."""
        if self.shape is not None:
            return False
        for sub_name in _SUB_NAMES:
            sub = getattr(self, sub_name)
            if any(getattr(sub, f.name) is not None for f in fields(sub)):
                return False
        return True


# Sub-dataclass field names on StageFitSettings, in HDF5/YAML order.
_SUB_NAMES = (
    "tau",
    "seeder",
    "conservative",
    "penalties",
    "rescue",
    "thaw",
    "spur",
    "baseline",
)


# Hard defaults per sub-dataclass. These mirror the ``DEFAULT_*`` constants
# in ``fitting/window_fit.py``, ``fitting/residual_rescue.py``,
# ``fitting/plan_execution.py``, ``fitting/validation.py`` and
# ``_internal/stage5_impl.py``. Kept as inline literals (rather than imported
# from fitting/) to keep ``core`` dependency-free from ``fitting``; the
# fitting modules' constants are the readable canonical source and these
# must track them.
_HARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "shape": {"kind": PeakShape.LORENTZIAN},
    "tau": {
        "max_decay_factor": 5.0,
        # The free-τ floor: τ is freed above max(this, weak_window_snr_threshold).
        # Defaults to the weak-window floor (10) so behaviour is unchanged until
        # tuned upward; see fitting.window_fit.DEFAULT_FIT_TAU_MIN_SNR.
        "fit_tau_min_snr": 10.0,
        "tau_penalty_lambda": 50.0,
        "tau_penalty_n_sigma": 5.0,
        "per_band_tau": True,
        # ``tau0_us`` / ``fit_tau`` / overrides legitimately stay None
        # (tau0_us derives at runtime from Stage 2b / expf_us / T_active/3;
        # fit_tau defaults to True inside the impl; overrides are unset by
        # design until a user supplies the atomic pair).
    },
    "seeder": {
        "seeder_rchi2": 1.5,
        "seeder_straddle_factor": 1.0,
        "seeder_max_k": 3,
    },
    "conservative": {
        "significance": 0.05,
        "max_peaks": 8,
        "patience": 1,
        "min_separation_factor": 1.0,
        "min_pair_separation_factor": 0.5,
        "min_pair_separation_resolution_factor": 1.0,
        "n_eff_kind": "perplexity_log1p_snr",
        "weak_window_snr_threshold": 10.0,
        "max_nfev": 2000,
    },
    "penalties": {
        "phase_penalty_lambda": 100.0,
        "phase_penalty_cutoff_fwhm": 2.0,
        "amp_penalty_lambda": 10.0,
        "amp_max_headroom": 3.0,
    },
    "rescue": {
        "max_rounds": 5,
        "snr_threshold": 2.5,
        "prominence_threshold": 2.0,
        "cleanup_significance": 0.05,
        "merge_separation_factor": 0.5,
        "structural_merge_factor": 0.5,
        "overfit_amp_ratio_band": 1.5,
        "overfit_amp_ratio_threshold": 6.0,
    },
    "thaw": {
        "max_thaw_rounds": 2,
        "max_replan_rounds": 2,
        "residual_edge_threshold": 8.0,
        "residual_edge_m": 32,
    },
    "spur": {
        # Spur masking defaults on: the gate is integer-MHz-anchored and
        # validated zero real-line false-positive on 2638. These mirror the
        # ``DEFAULT_*`` constants in ``fitting/spur_detection.py``.
        "enabled": True,
        "integer_tol_mhz": 0.04,
        "narrowness_ratio": 0.30,
        "snr_threshold": 5.0,
        "mask_half_width_bins": 2,
        "use_stft_catalogue": True,
    },
    "baseline": {
        # Leakage-wing baseline defaults on: the trigger fires only on a
        # coherent wing residual (edge-coh > 3.5, well above its ~0.9 null on
        # clean / narrow / low-SNR windows) and was validated zero-harmful on
        # 2638. ``const`` order is the load-bearing guardrail (too smooth to
        # mimic a narrow line). Mirrors ``DEFAULT_BASELINE_*`` in
        # ``fitting/plan_execution.py``.
        "enabled": True,
        "order": 0,
        "edge_threshold": 3.5,
    },
}


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
def _first_set_field(name: str, *layers: Any) -> Any:
    """Walk layers left-to-right, returning the first non-``None`` field value."""
    for layer in layers:
        if layer is None:
            continue
        value = getattr(layer, name)
        if value is not None:
            return value
    return None


def _resolve_sub(
    sub_name: str,
    *layers: Optional[StageFitSettings],
) -> Any:
    """Per-sub-dataclass field-merge with hard-default fallback."""
    sub_layers = [getattr(s, sub_name) for s in layers if s is not None]
    if not sub_layers:
        sub_layers = []
    template = getattr(StageFitSettings(), sub_name)
    merged = type(template)()
    for f in fields(template):
        value = _first_set_field(f.name, *sub_layers)
        if value is None:
            value = _HARD_DEFAULTS.get(sub_name, {}).get(f.name)
        setattr(merged, f.name, value)
    return merged


def resolve(
    explicit: Optional[StageFitSettings] = None,
    preset: Optional[StageFitSettings] = None,
    persisted: Optional[StageFitSettings] = None,
    recommended: Optional[StageFitSettings] = None,
) -> StageFitSettings:
    """Merge the four layers by precedence into a resolved ``StageFitSettings``.

    Per-field precedence: ``explicit > persisted > preset > recommended``,
    then any remaining ``None`` field falls back to the matching value in
    :data:`_HARD_DEFAULTS`. A value persisted in the ``.ftmw`` outranks a
    ``.yml`` preset, so the preset only seeds fields the file has not fixed
    and a shared experiment reproduces from the file alone. The shape
    discriminator is resolved separately: the first non-``None`` ``ShapeSpec``
    across the layers wins, then the ``LORENTZIAN`` hard default.
    """
    layers = (explicit, persisted, preset, recommended)
    shape_resolved: Optional[ShapeSpec] = None
    for layer in layers:
        if layer is not None and layer.shape is not None:
            shape_resolved = layer.shape
            break
    if shape_resolved is None:
        shape_resolved = ShapeSpec(kind=_HARD_DEFAULTS["shape"]["kind"])
    merged = StageFitSettings(shape=shape_resolved)
    for sub_name in _SUB_NAMES:
        setattr(merged, sub_name, _resolve_sub(sub_name, *layers))
    return merged


# ---------------------------------------------------------------------------
# Dict <-> dataclass round-trip (drives both HDF5 and YAML serialization)
# ---------------------------------------------------------------------------
def _encode_value(value: Any) -> Any:
    """Encode a field value for the attrs/dict form (``None`` -> ``__None__``)."""
    if value is None:
        return _NONE
    if isinstance(value, PeakShape):
        return value.value
    return value


def _decode_value(value: Any) -> Any:
    """Inverse of :func:`_encode_value`."""
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


def to_attrs(settings: StageFitSettings) -> Dict[str, Any]:
    """Nested attrs dict (one top-level key per sub-dataclass + ``shape``).

    The shape is encoded as a nested dict ``{"kind": "<value>"}`` (or the
    ``__None__`` sentinel when unset) so future shape-specific parameter
    blocks can attach inside the same subgroup. Sub-dataclass values use
    ``__None__`` for unset fields.
    """
    out: Dict[str, Any] = {}
    if settings.shape is None:
        out["shape"] = _NONE
    else:
        out["shape"] = {"kind": settings.shape.kind.value}
    for sub_name in _SUB_NAMES:
        out[sub_name] = _sub_to_attrs(getattr(settings, sub_name))
    return out


def from_attrs(attrs: Dict[str, Any]) -> StageFitSettings:
    """Inverse of :func:`to_attrs` (tolerant of missing sub-blocks)."""
    shape_raw = attrs.get("shape")
    shape_spec: Optional[ShapeSpec]
    if shape_raw is None or (isinstance(shape_raw, str) and shape_raw == _NONE):
        shape_spec = None
    elif isinstance(shape_raw, dict):
        shape_spec = ShapeSpec.coerce(shape_raw)
    elif isinstance(shape_raw, (str, PeakShape)):
        shape_spec = ShapeSpec.coerce(shape_raw)
    else:
        raise ValueError(f"cannot decode shape attrs from {shape_raw!r}")
    settings = StageFitSettings(shape=shape_spec)
    for sub_name in _SUB_NAMES:
        sub_attrs = attrs.get(sub_name, {})
        if not isinstance(sub_attrs, dict):
            raise ValueError(
                f"sub-block {sub_name!r} must be a mapping; got {type(sub_attrs)}"
            )
        template = getattr(StageFitSettings(), sub_name)
        setattr(settings, sub_name, _sub_from_attrs(type(template), sub_attrs))
    return settings


# ---------------------------------------------------------------------------
# YAML interchange
# ---------------------------------------------------------------------------
def _yaml_sub_to_mapping(sub: Any) -> Dict[str, Any]:
    """YAML view: drop ``None`` fields entirely (presets are sparse)."""
    out: Dict[str, Any] = {}
    for f in fields(sub):
        value = getattr(sub, f.name)
        if value is None:
            continue
        out[f.name] = value
    return out


def to_yaml_dict(settings: StageFitSettings) -> Dict[str, Any]:
    """Sparse nested dict suitable for ``yaml.safe_dump`` (omits ``None``).

    Sparseness lets a preset author write only the fields they want to
    override. Round-trip through :func:`from_yaml_dict` reproduces the
    same dataclass (unset fields stay ``None``).
    """
    out: Dict[str, Any] = {}
    if settings.shape is not None:
        out["shape"] = settings.shape.kind.value
    for sub_name in _SUB_NAMES:
        sub_dict = _yaml_sub_to_mapping(getattr(settings, sub_name))
        if sub_dict:
            out[sub_name] = sub_dict
    return out


def from_yaml_dict(data: Optional[Dict[str, Any]]) -> StageFitSettings:
    """Build a :class:`StageFitSettings` from a YAML-shaped mapping.

    Accepts a ``shape:`` shorthand string (``shape: gaussian``) at the top
    level. Each sub-dataclass block is a mapping of field name -> value;
    unknown keys raise ``ValueError`` so typos surface loudly.
    """
    if data is None:
        return StageFitSettings()
    if not isinstance(data, dict):
        raise ValueError(f"preset YAML root must be a mapping; got {type(data)}")
    settings = StageFitSettings()
    if "shape" in data:
        settings.shape = ShapeSpec.coerce(data["shape"])
    known_subs = set(_SUB_NAMES)
    for sub_name in _SUB_NAMES:
        if sub_name not in data:
            continue
        block = data[sub_name]
        if not isinstance(block, dict):
            raise ValueError(
                f"preset block {sub_name!r} must be a mapping; got {type(block)}"
            )
        template = getattr(StageFitSettings(), sub_name)
        valid_names = {f.name for f in fields(template)}
        unknown = set(block) - valid_names
        if unknown:
            raise ValueError(
                f"unknown {sub_name!r} fields in preset: {sorted(unknown)} "
                f"(valid: {sorted(valid_names)})"
            )
        setattr(settings, sub_name, type(template)(**block))
    # Reject top-level keys that are neither 'shape' nor a known sub-block,
    # except for the preset metadata keys 'name' and 'description' which
    # presets may carry for documentation but the settings parser ignores.
    allowed_top = known_subs | {"shape", "name", "description"}
    extra_top = set(data) - allowed_top
    if extra_top:
        raise ValueError(
            f"unknown top-level preset keys: {sorted(extra_top)} "
            f"(allowed: {sorted(allowed_top)})"
        )
    return settings


def from_yaml(source: Union[str, Path]) -> StageFitSettings:
    """Load a :class:`StageFitSettings` from a YAML file path or text."""
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    ):
        text = Path(source).read_text()
    else:
        text = str(source)
    data = yaml.safe_load(text)
    return from_yaml_dict(data)


def _looks_like_path(name_or_path: Union[str, Path]) -> bool:
    """Heuristic: does ``name_or_path`` reference a file rather than a bare name?

    A bare preset name is a single identifier (e.g. ``instrument_bc_2638``)
    that resolves against the packaged ``ftmwpipeline.presets`` resources.
    Anything else -- a path with separators, a string ending in ``.yaml``,
    or an absolute path -- gets loaded directly.
    """
    if isinstance(name_or_path, Path):
        return True
    s = str(name_or_path)
    return ("/" in s) or ("\\" in s) or s.endswith((".yaml", ".yml"))


def load_preset(name_or_path: Union[str, Path]) -> StageFitSettings:
    """Load a Stage 5 preset by bare name or by filesystem path.

    Bare names resolve against the packaged ``ftmwpipeline.presets``
    resources (e.g. ``"instrument_bc_2638"`` ->
    ``ftmwpipeline/presets/instrument_bc_2638.yaml``); paths load
    directly. Preset YAML may wrap the Stage 5 settings inside a
    top-level ``stage5:`` block (the new convention, allowing parallel
    ``stage2b:`` / ``stage2:`` blocks for other stages), a legacy
    ``fit:`` block (accepted for back-compat with presets written before
    the per-stage wrapper landed), or carry the settings flat at the top
    level; all three forms parse identically. ``stage5:`` and ``fit:``
    must not both appear in the same file.

    The ``name:`` and ``description:`` metadata fields are accepted but
    ignored by the settings parser -- they're documentation for the
    preset author. Sibling stage blocks (``stage2b:``, etc.) are
    ignored here; they belong to other stages' settings loaders.

    Parameters
    ----------
    name_or_path :
        Bare preset name (no extension) or a path to a YAML file.

    Returns
    -------
    StageFitSettings
        The parsed preset; unset fields stay ``None`` so the resolver
        can fall through to higher-precedence layers.

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
        # Bare name -> packaged resource
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
    # Per-stage top-level blocks are the current convention; ``fit:`` is
    # the legacy spelling kept as a back-compat shim for presets written
    # before the per-stage wrapper landed (see
    # ``dev-docs/planning/settings-backfill.md`` § "Back-compat shims").
    has_fit = "fit" in data and isinstance(data["fit"], dict)
    has_stage5 = "stage5" in data and isinstance(data["stage5"], dict)
    if has_fit and has_stage5:
        raise ValueError(
            f"preset {name_or_path!r} carries both 'fit:' (legacy) and "
            f"'stage5:' (current) wrappers; pick one"
        )
    inner_block: Optional[Dict[str, Any]] = None
    if has_stage5:
        inner_block = dict(data["stage5"])
    elif has_fit:
        import warnings as _warnings

        _warnings.warn(
            f"preset {name_or_path!r}: top-level 'fit:' wrapper is "
            "deprecated; rename it to 'stage5:' (per-stage block "
            "convention -- see dev-docs/planning/settings-backfill.md "
            "back-compat shim #1)",
            DeprecationWarning,
            stacklevel=2,
        )
        inner_block = dict(data["fit"])
    if inner_block is not None:
        for meta in ("name", "description"):
            if meta in data and meta not in inner_block:
                inner_block[meta] = data[meta]
        return from_yaml_dict(inner_block)
    # No per-stage wrapper: treat the document root as Stage 5 settings,
    # but strip out sibling stage blocks (``stage2b:`` etc.) so they
    # don't trip ``from_yaml_dict``'s unknown-key rejection.
    flat = {
        k: v
        for k, v in data.items()
        if k not in ("stage2", "stage2b", "stage3", "stage4")
    }
    return from_yaml_dict(flat)


def to_yaml(settings: StageFitSettings) -> str:
    """Serialize to a YAML string (sparse; omits unset fields)."""
    text: Any = yaml.safe_dump(
        to_yaml_dict(settings), sort_keys=False, default_flow_style=False
    )
    return cast(str, text)


__all__ = [
    "ShapeSpec",
    "TauSubSettings",
    "SeederSubSettings",
    "ConservativeSubSettings",
    "PenaltySubSettings",
    "RescueSubSettings",
    "ThawSubSettings",
    "BaselineSubSettings",
    "StageFitSettings",
    "resolve",
    "to_attrs",
    "from_attrs",
    "to_yaml",
    "from_yaml",
    "to_yaml_dict",
    "from_yaml_dict",
    "load_preset",
]
