"""
Canonical FT processing settings.

``FTSettings`` is the single source of truth for the Stage 1 FT processing
parameters across every surface:

* the public API signatures (``Pipeline.compute_ft`` / ``api.compute_ft``),
* the CLI flags (generated from field metadata -- see
  :mod:`ftmwpipeline.cli._argspec`),
* the resolution chain ``explicit override > persisted user settings >
  import-time recommended``,
* the persisted canonical record in ``processing_parameters/ft_processing``.

Every field is ``Optional`` with ``None`` meaning *unset* (fall through the
resolution chain). A *resolved* instance (produced by :func:`resolve`) has the
run-critical ``units_power`` field filled with a hard default if no layer
supplied it; ``start_us`` / ``end_us`` / ``trim`` may legitimately stay ``None``
(meaning no windowing / no trim).

The canonical FT is unconditionally unapodized, un-windowed, and native-length:
there are no ``expf_us`` / ``window_function`` / ``zpf`` knobs. Apodization
trades resolution and biases the line shape, and zero-padding interpolates bins
and corrupts the Stage 2/5 noise and chi-squared statistics; the robust
per-window fit is the intended alternative. DC removal (subtracting the
active-region mean before the transform) is likewise unconditional, so there is
no ``rdc`` knob. ``start_us`` / ``end_us`` (active region) and ``trim``
(analysis band) are data *selection*, not weighting, and are retained.

This module is intentionally dependency-free within the package (only stdlib +
the local ``__None__`` HDF5 marker convention shared with
``io.fid_serialization``) so it can be imported from ``core`` without cycles.
"""

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Optional, Tuple

from .knob_metadata import knob_field

# Mirrors the marker used by io.fid_serialization for optional HDF5 attrs.
_NONE = "__None__"

# Hard fallbacks for the fields that must be concrete to run FID.preprocess.
# Other fields fall back to None (a legitimate "absent" value).
_HARD_DEFAULTS: Dict[str, Any] = {
    "units_power": 6,
}

# Canonical HDF5 location of the persisted (user-chosen) settings.
FT_PROCESSING_PATH = "processing_parameters/ft_processing"
# Import-time recommendations written by Stage 0.
RECOMMENDED_PATH = "stage0_fid_data/recommended_processing"


def _parse_trim(text: str) -> Tuple[float, float]:
    """Parse a ``"min:max"`` MHz trim string into a ``(min, max)`` tuple."""
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Trim must be 'min:max' in MHz, got {text!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if hi <= lo:
        raise ValueError(f"Trim max must exceed min, got {text!r}")
    return (lo, hi)


def cli_field(
    *,
    help: str,
    flag: Optional[str] = None,
    argtype: Optional[Callable[[str], Any]] = None,
    metavar: Optional[str] = None,
    is_flag: bool = False,
) -> Any:
    """Declare an ``FTSettings`` field that is also exposed as a CLI option.

    Thin Stage 1 spelling of the stage-agnostic
    :func:`ftmwpipeline.core.knob_metadata.knob_field` (``cli=True``): it
    predates ``knob_field`` and is kept so the ``FTSettings`` declarations read
    naturally. The default is always ``None`` (the unset sentinel); concrete
    values come from the resolution chain, never from the dataclass default, so
    the precedence order stays intact.

    Parameters
    ----------
    help:
        ``argparse`` help string (single source -- not duplicated in the CLI).
    flag:
        Explicit long flag (e.g. ``"--expf_us"`` to preserve a legacy name).
        If omitted, derived as ``"--" + name.replace("_", "-")``.
    argtype:
        Callable passed to ``argparse``'s ``type=`` (coercion lives in
        metadata, never inferred from the annotation -- robust under
        ``from __future__ import annotations`` and Python 3.9).
    metavar:
        Optional ``argparse`` metavar.
    is_flag:
        Tri-state boolean rendered with ``BooleanOptionalAction``
        (``--x`` / ``--no-x``).
    """
    return knob_field(
        help=help,
        cli=True,
        flag=flag,
        argtype=argtype,
        metavar=metavar,
        is_flag=is_flag,
    )


@dataclass
class FTSettings:
    """Stage 1 FT processing settings (see module docstring).

    ``trim`` is the canonical frequency analysis range (MHz) and is persisted
    alongside the other FT settings (D7 decision: trim lives inside
    ``ft_processing``).
    """

    start_us: Optional[float] = cli_field(
        argtype=float,
        help="FID window start time in microseconds (earlier points zeroed)",
    )
    end_us: Optional[float] = cli_field(
        argtype=float,
        help="FID window end time in microseconds (later points zeroed)",
    )
    units_power: Optional[int] = cli_field(
        argtype=int,
        help="Spectrum scaling as power of 10 "
        "(default: persisted/recommended, else 6)",
    )
    trim: Optional[Tuple[float, float]] = cli_field(
        argtype=_parse_trim,
        metavar="MIN:MAX",
        help="Frequency analysis range to keep as 'min:max' in MHz "
        "(e.g. 26500:40000); persisted as canonical and binding downstream",
    )

    # -- introspection -------------------------------------------------------

    def is_empty(self) -> bool:
        """True if no field is set (a pure "use whatever is persisted" call)."""
        return all(getattr(self, f.name) is None for f in fields(self))

    def overrides(self) -> Dict[str, Any]:
        """The explicitly-set fields only (used to detect override intent)."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    # -- mapping to FID.preprocess ------------------------------------------

    def to_preprocess_kwargs(self) -> Dict[str, Any]:
        """Kwargs for ``FID.preprocess`` (``trim`` is applied post-FFT, not here).

        Should be called on a *resolved* instance; the four run-critical fields
        are guaranteed concrete after :func:`resolve`.
        """
        return {
            "start_us": self.start_us,
            "end_us": self.end_us,
            "units_power": self.units_power,
        }

    # -- HDF5 (de)serialization for the canonical ft_processing record ------

    def to_attrs(self) -> Dict[str, Any]:
        """Flat attribute dict for ``processing_parameters/ft_processing``.

        ``None`` -> the ``__None__`` marker; ``trim`` is split into
        ``trim_min_mhz`` / ``trim_max_mhz`` scalar attrs.
        """
        attrs: Dict[str, Any] = {}
        for name in (
            "start_us",
            "end_us",
            "units_power",
        ):
            value = getattr(self, name)
            attrs[name] = _NONE if value is None else value
        if self.trim is None:
            attrs["trim_min_mhz"] = _NONE
            attrs["trim_max_mhz"] = _NONE
        else:
            attrs["trim_min_mhz"] = float(self.trim[0])
            attrs["trim_max_mhz"] = float(self.trim[1])
        return attrs

    @classmethod
    def from_attrs(cls, attrs: Dict[str, Any]) -> "FTSettings":
        """Inverse of :meth:`to_attrs` (tolerant of missing/legacy keys).

        Legacy records may carry the retired apodization keys (``zpf`` /
        ``expf_us`` / ``window_function`` / ``winf``) or the retired ``rdc``
        toggle; they are silently ignored here. Callers that recompute the
        canonical FT from a legacy file warn about the dropped keys at open time.
        """

        def _opt(key: str) -> Any:
            if key not in attrs:
                return None
            value = attrs[key]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if isinstance(value, str) and value == _NONE:
                return None
            return value

        trim_lo = _opt("trim_min_mhz")
        trim_hi = _opt("trim_max_mhz")
        trim = (
            (float(trim_lo), float(trim_hi))
            if trim_lo is not None and trim_hi is not None
            else None
        )
        units = _opt("units_power")
        return cls(
            start_us=_coerce_float(_opt("start_us")),
            end_us=_coerce_float(_opt("end_us")),
            units_power=int(units) if units is not None else None,
            trim=trim,
        )


def _coerce_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _first_set(name: str, *layers: FTSettings) -> Any:
    for layer in layers:
        value = getattr(layer, name)
        if value is not None:
            return value
    return None


def resolve(
    explicit: Optional[FTSettings],
    persisted: Optional[FTSettings],
    recommended: Optional[FTSettings],
) -> FTSettings:
    """Merge the three layers by precedence into a resolved ``FTSettings``.

    Precedence per field: ``explicit > persisted > recommended``, then the
    run-critical ``units_power`` field falls back to :data:`_HARD_DEFAULTS` if
    still unset. The result is what Stage 1 computes/persists and what every
    downstream stage operates on.
    """
    empty = FTSettings()
    e = explicit or empty
    p = persisted or empty
    r = recommended or empty
    merged = FTSettings()
    for f in fields(FTSettings):
        setattr(merged, f.name, _first_set(f.name, e, p, r))
    for name, default in _HARD_DEFAULTS.items():
        if getattr(merged, name) is None:
            setattr(merged, name, default)
    return merged
