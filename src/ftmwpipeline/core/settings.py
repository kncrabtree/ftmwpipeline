"""
Standard FT processing settings.

``FTSettings`` is the single source of truth for the Stage 1 FT processing
parameters across every surface:

* the public API signatures (``Pipeline.compute_ft`` / ``api.compute_ft``),
* the CLI flags (generated from field metadata -- see
  :mod:`ftmwpipeline.cli._argspec`),
* the resolution chain ``explicit override > persisted user settings >
  import-time recommended``,
* the persisted record in ``processing_parameters/ft_processing``.

Every field is ``Optional`` with ``None`` meaning *unset* (fall through the
resolution chain). A *resolved* instance (produced by :func:`resolve`) has the
run-critical ``units_power`` field filled with a hard default if no layer
supplied it; ``start_us`` / ``end_us`` / ``trim`` may legitimately stay ``None``
(meaning no windowing / no trim) until :meth:`FTSettings.with_effective_window`
makes the active region concrete.

Once Stage 1 has persisted a record at the current
:data:`FT_PROCESSING_FIELD_SET_VERSION`, that record is authoritative: it holds
the concrete active region Stage 1 ran with (``start_us`` 0.0 when there was no
windowing, ``end_us`` the FID duration when unset) and its ``trim``, where an
unset trim means "no trim" and never falls through to the recommended layer.
The recommended layer then no longer takes part in resolution (see
``stage1_impl.resolve_ft_settings_h5``). An older record keeps the
``explicit > persisted > recommended`` fall-through it was written under.

The FT is unconditionally unapodized, un-windowed, and native-length:
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

# Standard HDF5 location of the persisted (user-chosen) settings.
FT_PROCESSING_PATH = "processing_parameters/ft_processing"
#: Field-set version of the ``ft_processing`` record Stage 1 writes (see
#: :mod:`ftmwpipeline.io.provenance`). Stamped beside the :meth:`FTSettings.to_attrs`
#: fields, never inside them, so it takes no part in the settings comparison.
#:
#: Version 2: the record is authoritative. ``start_us`` / ``end_us`` are always
#: the concrete active region Stage 1 ran with, and an unset ``trim`` means "no
#: trim" rather than "fall through to the recommended layer". A record below 2
#: (or with no version) is pre-provenance and resolves as it always did.
FT_PROCESSING_FIELD_SET_VERSION = 2
# Import-time recommendations written by Stage 0.
RECOMMENDED_PATH = "stage0_fid_data/recommended_processing"


def _parse_trim(text: str) -> Tuple[float, float]:
    """Parse a ``"min:max"`` MHz trim string into a ``(min, max)`` tuple.

    Raises :class:`~ftmwpipeline.file_manager.BadSettingError` (a
    ``ValueError``) for a string that is not ``min:max`` with ``max > min``.
    """
    # Lazy: file_manager imports from core.
    from ..file_manager import BadSettingError

    expected = "'min:max' in MHz with max > min"
    parts = text.split(":")
    if len(parts) != 2:
        raise BadSettingError(
            "stage1.trim",
            expected,
            text,
            message=f"Trim must be 'min:max' in MHz, got {text!r}",
        )
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        raise BadSettingError(
            "stage1.trim",
            expected,
            text,
            message=f"Trim must be 'min:max' in MHz, got {text!r}",
        ) from None
    if hi <= lo:
        raise BadSettingError(
            "stage1.trim",
            expected,
            text,
            message=f"Trim max must exceed min, got {text!r}",
        )
    return (lo, hi)


def cli_field(
    *,
    help: str,
    flag: Optional[str] = None,
    argtype: Optional[Callable[[str], Any]] = None,
    metavar: Optional[str] = None,
    is_flag: bool = False,
    units: Optional[str] = None,
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
    units:
        Physical units of the value, where the declaration states them.
    """
    return knob_field(
        help=help,
        cli=True,
        flag=flag,
        argtype=argtype,
        metavar=metavar,
        is_flag=is_flag,
        units=units,
    )


@dataclass
class FTSettings:
    """Stage 1 FT processing settings (see module docstring).

    ``trim`` is the standard frequency analysis range (MHz) and is persisted
    alongside the other FT settings (D7 decision: trim lives inside
    ``ft_processing``).
    """

    start_us: Optional[float] = cli_field(
        argtype=float,
        units="us",
        help="FID window start time in microseconds (earlier points zeroed)",
    )
    end_us: Optional[float] = cli_field(
        argtype=float,
        units="us",
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
        units="MHz",
        help="Frequency analysis range to keep as 'min:max' in MHz "
        "(e.g. 26500:40000); persisted and binding downstream",
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

    def with_effective_window(self, fid_duration_us: float) -> "FTSettings":
        """A copy with the active region made concrete for an FID of this length.

        An unset ``start_us`` is the start of the record (0.0) and an unset
        ``end_us`` is its end (``fid_duration_us``): exactly the samples an unset
        bound selects in ``FID.preprocess`` and
        :func:`~ftmwpipeline.fitting.active_ft.active_region_bounds`, so the
        concrete window selects the same samples and every result is unchanged.
        ``trim`` and ``units_power`` are copied as they are.
        """
        return FTSettings(
            start_us=0.0 if self.start_us is None else float(self.start_us),
            end_us=(
                float(fid_duration_us) if self.end_us is None else float(self.end_us)
            ),
            units_power=self.units_power,
            trim=self.trim,
        )

    def active_window_us(self) -> Tuple[float, float]:
        """The concrete ``(start_us, end_us)`` active region.

        Defined only on an instance whose window has been made concrete by
        :meth:`with_effective_window` (every settings object Stage 1 hands
        downstream is); anything else is a programming error, not an unset
        window to guess at.
        """
        if self.start_us is None or self.end_us is None:
            raise ValueError(
                "FT settings have no concrete active region; resolve them "
                "through FTSettings.with_effective_window first"
            )
        return float(self.start_us), float(self.end_us)

    # -- HDF5 (de)serialization for the persisted ft_processing record ------

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
        FT from a legacy file warn about the dropped keys at open time.
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
    still unset. A caller holding an authoritative (current-version) persisted
    record passes ``recommended=None``, so nothing falls through it. The
    result is what Stage 1 computes/persists and what every downstream stage
    operates on.
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
