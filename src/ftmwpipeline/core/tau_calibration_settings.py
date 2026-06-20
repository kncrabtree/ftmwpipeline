"""
Canonical Stage 2b τ calibration settings.

``TauCalibrationSettings`` is the single source of truth for the Stage 2b
parameters across every surface:

* the public API signatures (``Pipeline.calibrate_tau`` /
  ``Pipeline.calibrate_tau_G`` / ``Pipeline.recommend_shape`` and the
  matching ``ftmwpipeline.api`` functions),
* the CLI ``--preset`` flag plus the existing per-knob flags,
* the resolution chain ``explicit > persisted > preset > recommended >
  hard default``,
* the persisted canonical record in ``processing_parameters/stage2b_tau``,
* the YAML preset interchange format.

The dataclass mirrors the Stage 5 :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`
template: every field is ``Optional`` with ``None`` meaning *unset* (fall
through the resolution chain). A *resolved* instance (produced by
:func:`resolve`) has every field filled with a hard default if no layer
supplied a value.

The dataclass is structured into six sub-dataclasses grouping the knobs by
what they configure: ``stft``, ``polish``, ``aggregation``, ``band``,
``gaussian``, and ``recommendation``. The grouping maps 1:1 to HDF5
subgroups under ``processing_parameters/stage2b_tau`` so each sub-block is
independently inspectable. ``gaussian`` and ``recommendation`` overlap in
three fields by design (``snr_min`` / bounds / seeds); each consumer reads
from its own block so the Gaussian-twin τ calibration and the 3-way
shape-recommendation hook can evolve independent operating points.

The *recommended* layer of :func:`resolve` is reserved but currently
unused for Stage 2b -- Stage 2b is the originator of recommendations
(it produces the ``recommended_shape`` attr the Stage 5 resolver
consumes), not a downstream consumer of any upstream recommendation. The
layer is kept in the signature so a future upstream recommender (e.g. a
Stage 2 σ-driven ``snr_min`` suggestion) can land without API churn.

The ``_HARD_DEFAULTS`` nested dict mirrors the ``DEFAULT_*`` constants in
:mod:`ftmwpipeline.fitting.tau_calibration`. Those constants are still
imported by the kernel functions as their parameter defaults; once every
consumer reads from a resolved ``TauCalibrationSettings``, the constants
become docstring-only and can be removed.

This module is dependency-free within the package (stdlib + PyYAML for
preset interchange) so it can be imported from ``core`` without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import yaml  # type: ignore[import-untyped]

from .knob_metadata import knob_field

# Mirrors the marker used by io.fid_serialization for optional HDF5 attrs.
_NONE = "__None__"


# ---------------------------------------------------------------------------
# Sub-dataclasses (one per HDF5 subgroup / YAML block)
# ---------------------------------------------------------------------------
@dataclass
class StftSubSettings:
    """Sliding-active-window STFT knobs (shared by every Stage 2b consumer)."""

    n_seg: Optional[int] = knob_field(
        help="Number of non-overlapping STFT frames (window = T_full / n_seg).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(6, 8, 10, 14, 20),
        cli=True,
        argtype=int,
    )
    t_sigma: Optional[float] = knob_field(
        help="Above-threshold SNR gate for per-frame signal detection "
        "(contributor floor).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(3.0, 4.0, 5.0, 6.0, 8.0),
        cli=True,
        argtype=float,
    )
    tau_max_us: Optional[float] = knob_field(
        help="Hard upper clip on recovered tau (saturation -> spur candidate); "
        "unset -> derived.",
        inst_sensitivity="maybe",
        grid=(20.0, 40.0, 80.0),
        cli=True,
        argtype=float,
    )
    tau_max_factor: Optional[float] = knob_field(
        help="tau_max as a multiple of the full-record duration when "
        "tau_max_us is unset.",
        inst_sensitivity="maybe",
        grid=(3.0, 5.0, 8.0, 12.0),
    )
    rss_gate_factor: Optional[float] = knob_field(
        help="Bad-fit gate strength (relative-or-absolute residual hybrid).",
        inst_sensitivity="maybe",
        grid=(3.0, 5.0, 8.0, 12.0),
        cli=True,
        argtype=float,
    )
    relative_gate_fraction: Optional[float] = knob_field(
        help="Relative-RSS fraction below which a per-frame fit is accepted.",
        inst_sensitivity="maybe",
        grid=(0.02, 0.05, 0.10, 0.20),
    )
    sigma_x_full: Optional[float] = None
    sigma_time: Optional[float] = knob_field(
        help="Time-domain sigma_t override; default measures from the FID "
        "active-region tail.",
        cli=True,
        argtype=float,
    )


@dataclass
class PolishSubSettings:
    """Pure-exp polish step (consumed by ``calibrate_tau`` only)."""

    polish: Optional[bool] = None
    polish_n_iter: Optional[int] = knob_field(
        help="Gauss-Newton polish iterations per eligible contributor.",
        inst_sensitivity="N",
        grid=(1, 2, 3),
    )
    polish_top_n: Optional[int] = knob_field(
        help="Polish only the top-N contributors by SNR (unset -> all).",
        inst_sensitivity="N",
        grid=(200, 500, 1000, 2000),
    )
    polish_snr_cap: Optional[float] = knob_field(
        help="SNR above which the Gauss-Newton polish is skipped (avoid "
        "over-correction).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(5.0, 7.0, 9.0, 12.0),
    )
    polish_noise_debias: Optional[bool] = knob_field(
        help="Apply Rician-unbiased magnitude on high-SNR frames (removes "
        "residual bias).",
        inst_sensitivity="Y",
        grid=(False, True),
    )


@dataclass
class AggregationSubSettings:
    """Majority-vote + acceptance pre-conditions (shared by both τ twins)."""

    min_contributors: Optional[int] = knob_field(
        help="Minimum contributor count for the calibration to pass "
        "preconditions.",
        inst_sensitivity="N",
        grid=(100, 200, 400, 800),
        cli=True,
        argtype=int,
    )
    sigma_tau_fraction_max: Optional[float] = knob_field(
        help="Max sigma_tau/tau_maj for the calibration to pass preconditions.",
        inst_sensitivity="N",
        grid=(0.10, 0.20, 0.30),
        cli=True,
        argtype=float,
    )
    bimodality_dominant_fraction: Optional[float] = knob_field(
        help="Dominant-mode fraction above which a bimodal histogram still "
        "passes.",
        inst_sensitivity="N",
        grid=(0.6, 0.7, 0.8),
        cli=True,
        argtype=float,
    )
    sigma_tau_floor_us: Optional[float] = knob_field(
        help="Floor on the reported sigma_tau (guards against over-tight "
        "spreads).",
        inst_sensitivity="maybe",
        grid=(0.0, 0.5, 1.0),
    )
    spur_cluster_multiplier: Optional[float] = knob_field(
        help="Scale on the spur-cluster width (wider -> more bins flagged as "
        "spurs).",
        inst_sensitivity="maybe",
        grid=(1.0, 1.5, 2.0),
    )


@dataclass
class BandSubSettings:
    """Per-band majority routing (shared by both τ twins)."""

    compute_band_majorities: Optional[bool] = knob_field(
        help="Compute per-band tau majorities (the tau-vs-frequency band "
        "steps).",
        inst_sensitivity="Y",
        grid=(False, True),
    )
    band_edges_mhz: Optional[Tuple[float, ...]] = None
    band_labels: Optional[Tuple[str, ...]] = None
    min_contributors_per_band: Optional[int] = knob_field(
        help="Min contributors for a band to use its own tau majority (else "
        "band-wide).",
        tier="primary",
        inst_sensitivity="Y",
        grid=(25, 50, 100, 200),
    )


@dataclass
class GaussianSubSettings:
    """``calibrate_tau_G``-only knobs.

    ``min_contributors`` is distinct from
    :attr:`AggregationSubSettings.min_contributors` -- the Gaussian twin
    has a smaller hard default (50 vs 200) because the eligible Gaussian
    pool is naturally smaller after the Δχ²ᵣ filter. The two live on
    different sub-blocks to avoid the name collision.
    """

    snr_min: Optional[float] = knob_field(
        help="Gaussian tau_G: per-bin SNR floor for a contributor to enter "
        "the fit.",
        tier="primary",
        inst_sensitivity="Y",
        grid=(10.0, 15.0, 20.0, 30.0),
        cli=True,
        argtype=float,
    )
    tau_G_bound_lo: Optional[float] = knob_field(
        help="Gaussian tau_G lower fit bound (us).",
        inst_sensitivity="maybe",
        grid=(0.2, 0.5, 1.0),
        cli=True,
        argtype=float,
        flag="--tau-g-bound-lo",
    )
    tau_G_bound_hi: Optional[float] = knob_field(
        help="Gaussian tau_G upper fit bound (us).",
        inst_sensitivity="maybe",
        grid=(50.0, 100.0, 200.0),
        cli=True,
        argtype=float,
        flag="--tau-g-bound-hi",
    )
    tau_G_seeds: Optional[Tuple[float, ...]] = None
    delta_chi2r_min: Optional[float] = knob_field(
        help="Min chi2r improvement of the Gaussian over the exp fit to count "
        "a bin.",
        inst_sensitivity="maybe",
        grid=(0.5, 1.0, 2.0),
        cli=True,
        argtype=float,
    )
    tau_G_upper_fraction: Optional[float] = knob_field(
        help="Fraction of the tau_G bound above which a fit is treated as "
        "railed.",
        inst_sensitivity="maybe",
        grid=(0.5, 0.7, 0.9),
        cli=True,
        argtype=float,
        flag="--tau-g-upper-fraction",
    )
    min_contributors: Optional[int] = knob_field(
        help="Minimum Gaussian-eligible contributor count for tau_G "
        "preconditions.",
        inst_sensitivity="maybe",
        grid=(25, 50, 100),
    )


@dataclass
class RecommendationSubSettings:
    """``recommend_shape``-only knobs plus the calibrate_tau auto-run flag.

    Overlaps with :class:`GaussianSubSettings` in three fields
    (``snr_min`` / ``tau_bound_lo`` / ``tau_bound_hi`` / ``tau_G_seeds``)
    by design: the shape-recommendation hook and the production τ_G
    calibration are conceptually independent and may legitimately ship
    with different operating points (the recommender's contributor pool
    can be wider or narrower than the calibration's).

    ``auto_recommend`` controls whether :func:`calibrate_tau` /
    :func:`calibrate_tau_G` invoke :func:`recommend_shape` automatically
    after the primary calibration writes; default ``True`` so the
    Stage 5 resolver's *recommended* layer fires on every fresh Stage 2b
    run without a second explicit user step.
    """

    snr_min: Optional[float] = knob_field(
        help="Shape vote: per-bin SNR floor for a contributor to vote.",
        inst_sensitivity="maybe",
        grid=(10.0, 15.0, 20.0, 30.0),
    )
    tau_bound_lo: Optional[float] = knob_field(
        help="Shape vote: lower tau fit bound shared by the per-bin model "
        "fits (us).",
        inst_sensitivity="maybe",
        grid=(0.2, 0.5, 1.0),
    )
    tau_bound_hi: Optional[float] = knob_field(
        help="Shape vote: upper tau fit bound shared by the per-bin model "
        "fits (us).",
        inst_sensitivity="maybe",
        grid=(50.0, 100.0, 200.0),
    )
    tau_G_seeds: Optional[Tuple[float, ...]] = None
    pure_margin_threshold: Optional[float] = knob_field(
        help="Min SNR-weighted vote margin for a pure shape to win (else "
        "'none').",
        tier="primary",
        inst_sensitivity="maybe",
        grid=(0.05, 0.10, 0.15, 0.20),
    )
    auto_recommend: Optional[bool] = None


@dataclass
class TauCalibrationSettings:
    """Stage 2b τ calibration settings (see module docstring)."""

    stft: StftSubSettings = field(default_factory=StftSubSettings)
    polish: PolishSubSettings = field(default_factory=PolishSubSettings)
    aggregation: AggregationSubSettings = field(default_factory=AggregationSubSettings)
    band: BandSubSettings = field(default_factory=BandSubSettings)
    gaussian: GaussianSubSettings = field(default_factory=GaussianSubSettings)
    recommendation: RecommendationSubSettings = field(
        default_factory=RecommendationSubSettings
    )

    def is_empty(self) -> bool:
        """True if no field is set across any sub-dataclass."""
        for sub_name in _SUB_NAMES:
            sub = getattr(self, sub_name)
            if any(getattr(sub, f.name) is not None for f in fields(sub)):
                return False
        return True


# Sub-dataclass field names on TauCalibrationSettings, in HDF5/YAML order.
_SUB_NAMES = (
    "stft",
    "polish",
    "aggregation",
    "band",
    "gaussian",
    "recommendation",
)


# Hard defaults per sub-dataclass. These mirror the ``DEFAULT_*`` constants
# in ``fitting/tau_calibration.py``. Kept as inline literals (rather than
# imported from ``fitting/``) to keep ``core`` dependency-free from
# ``fitting``; the fitting module's constants are the readable canonical
# source and these must track them.
_HARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "stft": {
        "n_seg": 10,
        "t_sigma": 5.0,
        "tau_max_factor": 5.0,  # tau_max = factor * T_full when tau_max_us unset
        "rss_gate_factor": 5.0,
        "relative_gate_fraction": 0.05,
        # ``tau_max_us``, ``sigma_x_full``, and ``sigma_time`` legitimately
        # stay None: tau_max_us derives at runtime from T_full * tau_max_factor;
        # sigma_x_full is an opt-in spectral-noise override; sigma_time is
        # measured from the FID tail when unset.
    },
    "polish": {
        "polish": True,
        "polish_n_iter": 1,
        "polish_snr_cap": 9.0,
        "polish_noise_debias": False,
        # ``polish_top_n`` legitimately stays None (polish every contributor).
    },
    "aggregation": {
        "min_contributors": 200,
        "sigma_tau_fraction_max": 0.20,
        "bimodality_dominant_fraction": 0.70,
        "sigma_tau_floor_us": 0.5,
        "spur_cluster_multiplier": 1.0,
    },
    "band": {
        "compute_band_majorities": True,
        "min_contributors_per_band": 50,
        # ``band_edges_mhz`` legitimately stays None (default arithmetic
        # three-way split based on the Stage 1 trim range); ``band_labels``
        # legitimately stays None (defaults to ("low", "mid", "high") inside
        # ``compute_band_majorities``).
    },
    "gaussian": {
        "snr_min": 20.0,
        "tau_G_bound_lo": 0.5,
        "tau_G_bound_hi": 100.0,
        "tau_G_seeds": (100.0, 50.0, 20.0, 10.0, 5.0, 3.0),
        "delta_chi2r_min": 1.0,
        "tau_G_upper_fraction": 0.7,
        "min_contributors": 50,
    },
    "recommendation": {
        "snr_min": 20.0,
        "tau_bound_lo": 0.5,
        "tau_bound_hi": 100.0,
        "tau_G_seeds": (100.0, 50.0, 20.0, 10.0, 5.0, 3.0),
        "pure_margin_threshold": 0.10,
        "auto_recommend": True,
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
    *layers: Optional["TauCalibrationSettings"],
) -> Any:
    """Per-sub-dataclass field-merge with hard-default fallback."""
    sub_layers = [getattr(s, sub_name) for s in layers if s is not None]
    template = getattr(TauCalibrationSettings(), sub_name)
    merged = type(template)()
    for f in fields(template):
        value = _first_set_field(f.name, *sub_layers)
        if value is None:
            value = _HARD_DEFAULTS.get(sub_name, {}).get(f.name)
        setattr(merged, f.name, value)
    return merged


def resolve(
    explicit: Optional[TauCalibrationSettings] = None,
    preset: Optional[TauCalibrationSettings] = None,
    persisted: Optional[TauCalibrationSettings] = None,
    recommended: Optional[TauCalibrationSettings] = None,
) -> TauCalibrationSettings:
    """Merge the four layers by precedence into a resolved ``TauCalibrationSettings``.

    Per-field precedence: ``explicit > persisted > preset > recommended``,
    then any remaining ``None`` field falls back to the matching value in
    :data:`_HARD_DEFAULTS`. A value persisted in the ``.ftmw`` outranks a
    ``.yml`` preset, so the preset only seeds fields the file has not fixed
    and a shared experiment reproduces from the file alone. The
    ``recommended`` layer is reserved for a future upstream recommender; it
    is currently always passed ``None`` by Stage 2b call sites, and the slot
    is kept here so the resolver shape stays uniform with Stage 5's.
    """
    layers = (explicit, persisted, preset, recommended)
    merged = TauCalibrationSettings()
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
    if isinstance(value, tuple):
        return list(value)
    return value


def _decode_value(value: Any, field_name: str) -> Any:
    """Inverse of :func:`_encode_value`, restoring tuple-typed fields."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and value == _NONE:
        return None
    if field_name in _TUPLE_FIELDS and value is not None:
        # HDF5 reads tuples back as numpy arrays; coerce to plain tuples.
        return tuple(_coerce_tuple_element(field_name, v) for v in value)
    return value


# Fields that must round-trip as tuples (not lists / arrays). Mirror the
# typed declarations on the sub-dataclasses above.
_TUPLE_FIELDS = {"tau_G_seeds", "band_edges_mhz", "band_labels"}


def _coerce_tuple_element(field_name: str, value: Any) -> Any:
    if field_name == "band_labels":
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)
    return float(value)


def _sub_to_attrs(sub: Any) -> Dict[str, Any]:
    return {f.name: _encode_value(getattr(sub, f.name)) for f in fields(sub)}


def _sub_from_attrs(cls: type, attrs: Dict[str, Any]) -> Any:
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in attrs:
            continue
        kwargs[f.name] = _decode_value(attrs[f.name], f.name)
    return cls(**kwargs)


def to_attrs(settings: TauCalibrationSettings) -> Dict[str, Any]:
    """Nested attrs dict (one top-level key per sub-dataclass).

    Sub-dataclass values use ``__None__`` for unset fields; tuples are
    encoded as lists so HDF5 can persist them as 1-D arrays.
    """
    out: Dict[str, Any] = {}
    for sub_name in _SUB_NAMES:
        out[sub_name] = _sub_to_attrs(getattr(settings, sub_name))
    return out


def from_attrs(attrs: Dict[str, Any]) -> TauCalibrationSettings:
    """Inverse of :func:`to_attrs` (tolerant of missing sub-blocks)."""
    settings = TauCalibrationSettings()
    for sub_name in _SUB_NAMES:
        sub_attrs = attrs.get(sub_name, {})
        if not isinstance(sub_attrs, dict):
            raise ValueError(
                f"sub-block {sub_name!r} must be a mapping; got {type(sub_attrs)}"
            )
        template = getattr(TauCalibrationSettings(), sub_name)
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
        if isinstance(value, tuple):
            value = list(value)
        out[f.name] = value
    return out


def to_yaml_dict(settings: TauCalibrationSettings) -> Dict[str, Any]:
    """Sparse nested dict suitable for ``yaml.safe_dump`` (omits ``None``)."""
    out: Dict[str, Any] = {}
    for sub_name in _SUB_NAMES:
        sub_dict = _yaml_sub_to_mapping(getattr(settings, sub_name))
        if sub_dict:
            out[sub_name] = sub_dict
    return out


def from_yaml_dict(data: Optional[Mapping[str, Any]]) -> TauCalibrationSettings:
    """Build a :class:`TauCalibrationSettings` from a YAML-shaped mapping.

    Each sub-dataclass block is a mapping of field name -> value; unknown
    keys raise ``ValueError`` so typos surface loudly. Preset-metadata
    keys ``name`` and ``description`` at the top level are accepted but
    ignored.
    """
    if data is None:
        return TauCalibrationSettings()
    if not isinstance(data, dict):
        raise ValueError(f"preset YAML root must be a mapping; got {type(data)}")
    settings = TauCalibrationSettings()
    known_subs = set(_SUB_NAMES)
    for sub_name in _SUB_NAMES:
        if sub_name not in data:
            continue
        block = data[sub_name]
        if not isinstance(block, dict):
            raise ValueError(
                f"preset block {sub_name!r} must be a mapping; got {type(block)}"
            )
        template = getattr(TauCalibrationSettings(), sub_name)
        valid_names = {f.name for f in fields(template)}
        unknown = set(block) - valid_names
        if unknown:
            raise ValueError(
                f"unknown {sub_name!r} fields in preset: {sorted(unknown)} "
                f"(valid: {sorted(valid_names)})"
            )
        kwargs: Dict[str, Any] = {}
        for key, value in block.items():
            if key in _TUPLE_FIELDS and value is not None:
                kwargs[key] = tuple(_coerce_tuple_element(key, v) for v in value)
            else:
                kwargs[key] = value
        setattr(settings, sub_name, type(template)(**kwargs))
    allowed_top = known_subs | {"name", "description"}
    extra_top = set(data) - allowed_top
    if extra_top:
        raise ValueError(
            f"unknown top-level preset keys: {sorted(extra_top)} "
            f"(allowed: {sorted(allowed_top)})"
        )
    return settings


def from_yaml(source: Union[str, Path]) -> TauCalibrationSettings:
    """Load a :class:`TauCalibrationSettings` from a YAML file path or text."""
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    ):
        text = Path(source).read_text()
    else:
        text = str(source)
    data = yaml.safe_load(text)
    return from_yaml_dict(data)


def _looks_like_path(name_or_path: Union[str, Path]) -> bool:
    """Heuristic: does ``name_or_path`` reference a file rather than a bare name?"""
    if isinstance(name_or_path, Path):
        return True
    s = str(name_or_path)
    return ("/" in s) or ("\\" in s) or s.endswith((".yaml", ".yml"))


def load_preset(name_or_path: Union[str, Path]) -> TauCalibrationSettings:
    """Load a Stage 2b preset by bare name or by filesystem path.

    Bare names resolve against the packaged ``ftmwpipeline.presets``
    resources (e.g. ``"instrument_bc_2638"`` ->
    ``ftmwpipeline/presets/instrument_bc_2638.yaml``); paths load directly.
    Preset YAML wraps the Stage 2b settings inside a top-level ``stage2b:``
    block (alongside an optional ``stage5:`` block for Stage 5 settings).

    The ``name:`` and ``description:`` metadata fields are accepted but
    ignored by the settings parser -- they're documentation for the preset
    author.

    Returns an empty :class:`TauCalibrationSettings` (no fields set) when
    the preset carries no ``stage2b:`` block, so a Stage-5-only preset
    loads cleanly without producing spurious Stage 2b overrides.

    Parameters
    ----------
    name_or_path :
        Bare preset name (no extension) or a path to a YAML file.

    Returns
    -------
    TauCalibrationSettings
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
    inner = data.get("stage2b")
    if inner is None:
        return TauCalibrationSettings()
    if not isinstance(inner, dict):
        raise ValueError(
            f"preset 'stage2b' block must be a mapping; got {type(inner)} "
            f"from {name_or_path}"
        )
    block = dict(inner)
    for meta in ("name", "description"):
        if meta in data and meta not in block:
            block[meta] = data[meta]
    return from_yaml_dict(block)


def to_yaml(settings: TauCalibrationSettings) -> str:
    """Serialize to a YAML string (sparse; omits unset fields)."""
    text: Any = yaml.safe_dump(
        to_yaml_dict(settings), sort_keys=False, default_flow_style=False
    )
    return cast(str, text)


__all__ = [
    "StftSubSettings",
    "PolishSubSettings",
    "AggregationSubSettings",
    "BandSubSettings",
    "GaussianSubSettings",
    "RecommendationSubSettings",
    "TauCalibrationSettings",
    "resolve",
    "to_attrs",
    "from_attrs",
    "to_yaml",
    "from_yaml",
    "to_yaml_dict",
    "from_yaml_dict",
    "load_preset",
]
