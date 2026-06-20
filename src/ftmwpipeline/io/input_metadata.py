"""Shared acquisition-metadata resolution for the generic data loaders.

The generic CSV and native-HDF5 loaders accept acquisition metadata from three
layers and resolve them by a single precedence rule::

    explicit call parameter  >  sidecar file  >  embedded in the data file  >  default

A *sidecar* is a small JSON or YAML file next to the data source carrying the
acquisition parameters and, optionally, the chirp window and clock
declarations -- the no-code home for metadata that a CSV (or any format that
cannot embed it) does not carry.  This module owns sidecar discovery/parsing
and the per-field merge so both generic loaders behave identically.

The resolved clock declarations and chirp window are returned untouched for the
loader to attach to ``fid.metadata`` (``clock_sources`` / ``chirp_window``),
where the Stage 0 import path persists them exactly as it does for the
loader-injected forms.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .data_loaders.base import LoaderError

# Acquisition fields a sidecar or explicit layer may set, with their defaults.
# ``spacing_us`` has no default: it is required and resolves to ``None`` when
# unset so the loader can raise a targeted error. ``probe_freq_mhz`` defaults to
# ``0`` MHz -- a direct-sampling instrument whose baseband *is* the molecular
# frequency -- so it need not be supplied.
_DEFAULTS: Dict[str, Any] = {
    "spacing_us": None,
    "probe_freq_mhz": 0.0,
    "sideband": "upper",
    "shots": 1,
}

# Keys a sidecar file may contain (acquisition fields plus the two structured
# blocks).  Anything else is a typo the user wants to hear about.
_ALLOWED_SIDECAR_KEYS = set(_DEFAULTS) | {"chirp_window", "clock_sources"}

# Suffixes tried, in order, for sidecar auto-discovery next to the source.
_SIDECAR_SUFFIXES = (".ftmwmeta.json", ".ftmwmeta.yaml", ".ftmwmeta.yml")


@dataclass
class ResolvedInputMetadata:
    """Acquisition metadata resolved across the explicit/sidecar/embedded layers."""

    spacing_us: float
    probe_freq_mhz: float
    sideband: str
    shots: int
    chirp_window: Optional[Dict[str, Any]]
    clock_sources: Optional[List[Dict[str, Any]]]

    @property
    def spacing_s(self) -> float:
        """Sample period in seconds (the unit :class:`FID` stores)."""
        return self.spacing_us * 1e-6


def find_sidecar(
    source_path: Union[str, Path], explicit_path: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """Return the sidecar path to use, or ``None``.

    An explicit ``--metadata`` path wins and must exist.  Otherwise the
    adjacent ``<source>.ftmwmeta.{json,yaml,yml}`` files are tried in order.
    """
    if explicit_path is not None:
        p = Path(explicit_path)
        if not p.is_file():
            raise LoaderError(f"Metadata sidecar not found: {p}")
        return p
    source_path = Path(source_path)
    for suffix in _SIDECAR_SUFFIXES:
        candidate = source_path.with_name(source_path.name + suffix)
        if candidate.is_file():
            return candidate
    return None


def load_sidecar(path: Path) -> Dict[str, Any]:
    """Parse a sidecar JSON/YAML file into a validated metadata dict.

    The format is chosen by extension (``.json`` -> JSON, ``.yaml``/``.yml`` ->
    YAML).  Unknown top-level keys raise :class:`LoaderError` so typos surface
    rather than being silently ignored.
    """
    suffix = path.suffix.lower()
    try:
        text = path.read_text()
    except OSError as exc:
        raise LoaderError(f"Could not read metadata sidecar {path}: {exc}") from exc

    try:
        if suffix == ".json":
            data = json.loads(text)
        elif suffix in (".yaml", ".yml"):
            import yaml  # type: ignore[import-untyped]

            data = yaml.safe_load(text)
        else:
            raise LoaderError(
                f"Unsupported metadata sidecar extension '{suffix}' "
                f"({path.name}); use .json, .yaml, or .yml"
            )
    except (json.JSONDecodeError, ValueError) as exc:
        raise LoaderError(f"Could not parse metadata sidecar {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise LoaderError(
            f"Metadata sidecar {path.name} must contain a mapping at top level; "
            f"got {type(data).__name__}"
        )

    unknown = set(data) - _ALLOWED_SIDECAR_KEYS
    if unknown:
        raise LoaderError(
            f"Unknown key(s) {sorted(unknown)} in metadata sidecar {path.name} "
            f"(valid keys: {sorted(_ALLOWED_SIDECAR_KEYS)})"
        )
    return data


def resolve_input_metadata(
    *,
    explicit: Optional[Dict[str, Any]] = None,
    sidecar: Optional[Dict[str, Any]] = None,
    embedded: Optional[Dict[str, Any]] = None,
) -> ResolvedInputMetadata:
    """Merge the three metadata layers by precedence and validate the result.

    Each layer is a mapping that may set any subset of the acquisition fields
    plus ``chirp_window`` / ``clock_sources``.  For every field the first layer
    (explicit, then sidecar, then embedded) that supplies a non-``None`` value
    wins; otherwise the built-in default applies.  ``spacing_us`` is required and
    raises :class:`LoaderError` when no layer supplies it; ``probe_freq_mhz``
    defaults to ``0`` MHz (a direct-sampling instrument) when unset.
    """
    layers = [layer or {} for layer in (explicit, sidecar, embedded)]

    def pick(key: str) -> Any:
        for layer in layers:
            value = layer.get(key)
            if value is not None:
                return value
        return _DEFAULTS.get(key)

    resolved: Dict[str, Any] = {key: pick(key) for key in _DEFAULTS}

    if resolved["spacing_us"] is None:
        raise LoaderError(
            "Missing required acquisition metadata: spacing_us. Supply it as a "
            "load parameter (e.g. --spacing_us) or in a --metadata sidecar."
        )

    spacing_us = float(resolved["spacing_us"])
    probe_freq_mhz = float(resolved["probe_freq_mhz"])
    if spacing_us <= 0:
        raise LoaderError(f"spacing_us must be positive; got {spacing_us}")
    if probe_freq_mhz < 0:
        raise LoaderError(f"probe_freq_mhz must be non-negative; got {probe_freq_mhz}")

    sideband = str(resolved["sideband"]).lower()
    if sideband not in ("upper", "lower"):
        raise LoaderError(
            f"sideband must be 'upper' or 'lower'; got {resolved['sideband']!r}"
        )

    shots = int(resolved["shots"])
    if shots <= 0:
        raise LoaderError(f"shots must be positive; got {shots}")

    return ResolvedInputMetadata(
        spacing_us=spacing_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
        shots=shots,
        chirp_window=pick("chirp_window"),
        clock_sources=pick("clock_sources"),
    )


def explicit_layer_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the explicit precedence layer from generic-loader kwargs.

    The acquisition fields pass through unchanged; the ``chirp_*`` kwargs are
    folded into a single ``chirp_window`` block so the resolver sees one shape.
    Only keys with non-``None`` values are emitted.
    """
    explicit: Dict[str, Any] = {
        key: kwargs[key]
        for key in ("spacing_us", "probe_freq_mhz", "sideband", "shots")
        if kwargs.get(key) is not None
    }
    if kwargs.get("chirp_end_us") is not None:
        chirp: Dict[str, Any] = {"chirp_end_us": float(kwargs["chirp_end_us"])}
        if kwargs.get("chirp_start_us") is not None:
            chirp["chirp_start_us"] = float(kwargs["chirp_start_us"])
        if kwargs.get("start_margin_us") is not None:
            chirp["start_margin_us"] = float(kwargs["start_margin_us"])
        explicit["chirp_window"] = chirp
    return explicit


def build_fid_metadata(resolved: ResolvedInputMetadata) -> Dict[str, Any]:
    """Build the ``fid.metadata`` payload (clock + chirp blocks) from a resolution.

    Only the keys Stage 0 consumes are emitted, and only when present, so an
    import without clocks or a chirp window leaves no spurious metadata.
    """
    metadata: Dict[str, Any] = {}
    if resolved.clock_sources is not None:
        metadata["clock_sources"] = resolved.clock_sources
    if resolved.chirp_window is not None:
        metadata["chirp_window"] = resolved.chirp_window
    return metadata
