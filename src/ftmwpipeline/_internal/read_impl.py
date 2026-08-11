"""Read-only access to persisted pipeline data (shared implementation core).

The stage loaders (``load_peaks`` / ``load_windows`` / ``load_fit``) rebuild the
complete persisted record -- every audit step, thaw event, rescue round, doublet
alternative, fixed contributor, and covariance block -- because that is what a
curator editing the record needs. A consumer that only wants a few columns pays
for all of it: the cost is h5py's per-item Python overhead, paid thousands of
times, plus a JSON parse per window, not the handful of floats it keeps.

This module is the narrow counterpart. It exposes the persisted stage artifacts
as **tables of bulk columns**: each requested column is one whole-dataset (or
one attribute) read, nothing is recomputed, no object graph is reconstructed,
and nothing else in the file is touched. It is strictly read-only -- it opens
the ``.ftmw`` in ``"r"`` mode and never writes.

The column layouts themselves live next to the writers that define them, in
``io/peak_serialization.py``, ``io/window_serialization.py``, and
``io/fitting_serialization.py``, so a change to what a stage persists cannot
silently drift from what this surface reads.

Tables (see :data:`READ_TABLES`). Stage 2b's four are mirrored under a
``tau_g_`` prefix for the Gaussian twin (``tau run --gaussian``), which lives in
its own group and may coexist with the primary calibration:

``tau_bands``
    Per-band decay times, with their uncertainties and frequency ranges -- the
    per-window anchors Stage 5 may consume.
``tau_thirds``
    The low/mid/high split, with a median decay time per third: the
    does-tau-drift-with-frequency diagnostic.
``tau_contributors``
    One row per STFT bin that survived the gates and voted on the decay time.
``tau_spurs``
    One row per excluded spur cluster. The ragged member-bin lists stay with
    the full loader.
``peaks``
    Stage 3 detected-peak list, including the derived ``promoted`` flag.
``windows``
    Stage 4 planned-window bounds, batch, and contributor counts.
``window_free_peaks`` / ``window_contributors``
    The window plan's ragged per-window sets, in long form: which Stage 3 peaks
    a window fits freely, and which it holds fixed from a neighbor.
``fit_peaks``
    Stage 5 fitted peaks, one row per peak, ordered by molecular frequency.
    Carries the owning window's line ``shape`` as a per-peak column so no join
    is needed.
``fit_windows``
    Stage 5 per-window fit scalars (bounds, tau, cost, quality).
``fit_audit`` / ``fit_doublets``
    The per-window decision record: every candidate the conservative add loop
    tried and how it ruled, and every doublet alternative it weighed.
``fit_thaw`` / ``fit_replans`` / ``fit_rescues``
    The plan-level histories: contributors released back to free, window
    boundaries redrawn mid-fit, and residual re-searches.

Two caveats on "cheap". First, not every table is here because its loader was
slow: Stages 4 and 5 fan out over hundreds of window groups, so the narrow read
is a large win there, while Stage 2b and Stage 3 persist a single group and
already load in milliseconds. Those entries exist so that *every* persisted
artifact is reachable through one surface, with CSV export, rather than only
the ones that happened to be expensive.

Second, the event-log tables (``fit_audit``, ``fit_doublets``, ``fit_thaw``,
``fit_replans``, ``fit_rescues``) read JSON-encoded records, because that is how
the fit persists its narrative. Reaching them means parsing; there is no
narrower path. They are still far below a full ``load_fit``, but they are not
the whole-dataset reads the rest of this surface is.

Stages 1 and 2 have no table: the canonical FT is recomputed from the FID on
demand rather than persisted (only its settings are stored, and those are
scalars), and the Stage 2 noise model is a reconstruction over the persisted
bins, not a column a consumer can read off.

:func:`read_metadata_impl` covers the scalars -- provenance, the FID
acquisition, the canonical FT window (including ``ft.acquisition_us``, from
which the Fourier resolution element ``1 / T`` follows), the start-detection
sweep outcome, per-stage counts, and the timebase calibration -- all from group
attributes, without deserializing any stage artifact.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from ..file_manager import check_format_compatibility
from ..io._hdf5_helpers import ColumnSpec, load_json_attr
from ..io.fitting_serialization import (
    FIT_AUDIT_COLUMN_SPECS,
    FIT_DOUBLET_COLUMN_SPECS,
    FIT_PEAK_COLUMN_SPECS,
    FIT_REPLAN_COLUMN_SPECS,
    FIT_RESCUE_COLUMN_SPECS,
    FIT_THAW_COLUMN_SPECS,
    FIT_WINDOW_COLUMN_SPECS,
    read_fit_audit_columns,
    read_fit_doublet_columns,
    read_fit_peak_columns,
    read_fit_replan_columns,
    read_fit_rescue_columns,
    read_fit_scalars,
    read_fit_thaw_columns,
    read_fit_window_columns,
)
from ..io.peak_serialization import (
    PEAK_COLUMN_SPECS,
    read_peak_columns,
    read_peak_scalars,
)
from ..io.tau_calibration_serialization import (
    GAUSSIAN_GROUP_PATH,
    GROUP_PATH,
    TAU_BAND_COLUMN_SPECS,
    TAU_CONTRIBUTOR_COLUMN_SPECS,
    TAU_SPUR_COLUMN_SPECS,
    TAU_THIRD_COLUMN_SPECS,
    read_tau_band_columns,
    read_tau_contributor_columns,
    read_tau_scalars,
    read_tau_spur_columns,
    read_tau_third_columns,
)
from ..io.window_serialization import (
    WINDOW_CONTRIBUTOR_COLUMN_SPECS,
    WINDOW_FREE_PEAK_COLUMN_SPECS,
    WINDOW_PLAN_COLUMN_SPECS,
    read_window_contributor_columns,
    read_window_free_peak_columns,
    read_window_plan_columns,
    read_window_plan_scalars,
)
from .shared_utils import active_acquisition_us, fold_settings_blob

__all__ = [
    "READ_TABLES",
    "VALID_READ_FORMATS",
    "read_table_impl",
    "read_tables_impl",
    "read_metadata_impl",
    "format_table_impl",
    "format_metadata_impl",
    "write_text_impl",
    "normalize_table_name",
]

#: Output formats :func:`format_table_impl` renders.
VALID_READ_FORMATS: Tuple[str, ...] = ("csv", "tsv", "json")

_Reader = Callable[[h5py.Group, Optional[Sequence[str]]], Dict[str, np.ndarray]]


@dataclass(frozen=True)
class _TableSpec:
    """Where one readable table lives and how to read it."""

    group: str
    specs: Dict[str, ColumnSpec]
    reader: _Reader
    #: Where the row count lives, for the cheap listing: an attribute name on
    #: ``group``, or ``"subgroup/attr"`` when the stage records it one level in.
    #: Empty when the stage records no count for this table -- the listing then
    #: reports the table as available with an unknown length.
    count_path: str
    #: What to run when the stage is missing, quoted in the error.
    hint: str


#: The primary decay-time calibration group, named for symmetry with
#: :data:`GAUSSIAN_GROUP_PATH` -- a bare ``GROUP_PATH`` says nothing here.
TAU_GROUP_PATH = GROUP_PATH

_TAU_HINT = "calibrate_tau() / 'tau run'"
_TAU_G_HINT = "calibrate_tau(gaussian=True) / 'tau run --gaussian'"

#: The four tables each decay-time calibration exposes, as
#: ``(suffix, specs, reader, count_path)``. The Lorentzian/Voigt calibration and
#: the Gaussian twin have identical layouts in different groups, so the registry
#: is generated for both rather than written twice.
_TAU_TABLES = (
    ("bands", TAU_BAND_COLUMN_SPECS, read_tau_band_columns, "band_majorities/n_bands"),
    ("thirds", TAU_THIRD_COLUMN_SPECS, read_tau_third_columns, ""),
    (
        "contributors",
        TAU_CONTRIBUTOR_COLUMN_SPECS,
        read_tau_contributor_columns,
        "scalars/n_contributors",
    ),
    ("spurs", TAU_SPUR_COLUMN_SPECS, read_tau_spur_columns, "spur_clusters/n_clusters"),
)

_WINDOWS_HINT = "assign_windows() / 'windows run'"
_FIT_HINT = "fit_peaks() / 'fit run'"

_TABLE_SPECS: Dict[str, _TableSpec] = {}
for _prefix, _group, _hint in (
    ("tau", TAU_GROUP_PATH, _TAU_HINT),
    ("tau_g", GAUSSIAN_GROUP_PATH, _TAU_G_HINT),
):
    for _suffix, _specs, _reader, _count in _TAU_TABLES:
        _TABLE_SPECS[f"{_prefix}_{_suffix}"] = _TableSpec(
            group=_group,
            specs=_specs,
            reader=_reader,
            count_path=_count,
            hint=_hint,
        )

_TABLE_SPECS.update(
    {
        "peaks": _TableSpec(
            group="stage3_peaks",
            specs=PEAK_COLUMN_SPECS,
            reader=read_peak_columns,
            count_path="n_peaks",
            hint="detect_peaks() / 'peaks run'",
        ),
        "windows": _TableSpec(
            group="stage4_windows",
            specs=WINDOW_PLAN_COLUMN_SPECS,
            reader=read_window_plan_columns,
            count_path="n_windows",
            hint=_WINDOWS_HINT,
        ),
        "window_free_peaks": _TableSpec(
            group="stage4_windows",
            specs=WINDOW_FREE_PEAK_COLUMN_SPECS,
            reader=read_window_free_peak_columns,
            count_path="",
            hint=_WINDOWS_HINT,
        ),
        "window_contributors": _TableSpec(
            group="stage4_windows",
            specs=WINDOW_CONTRIBUTOR_COLUMN_SPECS,
            reader=read_window_contributor_columns,
            count_path="",
            hint=_WINDOWS_HINT,
        ),
        "fit_peaks": _TableSpec(
            group="stage5_fitting",
            specs=FIT_PEAK_COLUMN_SPECS,
            reader=read_fit_peak_columns,
            count_path="n_fitted_peaks",
            hint=_FIT_HINT,
        ),
        "fit_windows": _TableSpec(
            group="stage5_fitting",
            specs=FIT_WINDOW_COLUMN_SPECS,
            reader=read_fit_window_columns,
            count_path="n_windows",
            hint=_FIT_HINT,
        ),
        "fit_audit": _TableSpec(
            group="stage5_fitting",
            specs=FIT_AUDIT_COLUMN_SPECS,
            reader=read_fit_audit_columns,
            count_path="",
            hint=_FIT_HINT,
        ),
        "fit_doublets": _TableSpec(
            group="stage5_fitting",
            specs=FIT_DOUBLET_COLUMN_SPECS,
            reader=read_fit_doublet_columns,
            count_path="",
            hint=_FIT_HINT,
        ),
        "fit_thaw": _TableSpec(
            group="stage5_fitting",
            specs=FIT_THAW_COLUMN_SPECS,
            reader=read_fit_thaw_columns,
            count_path="",
            hint=_FIT_HINT,
        ),
        "fit_replans": _TableSpec(
            group="stage5_fitting",
            specs=FIT_REPLAN_COLUMN_SPECS,
            reader=read_fit_replan_columns,
            count_path="",
            hint=_FIT_HINT,
        ),
        "fit_rescues": _TableSpec(
            group="stage5_fitting",
            specs=FIT_RESCUE_COLUMN_SPECS,
            reader=read_fit_rescue_columns,
            count_path="",
            hint=_FIT_HINT,
        ),
    }
)

#: The readable table names, in a sensible reading order.
READ_TABLES: Tuple[str, ...] = tuple(_TABLE_SPECS)


def normalize_table_name(table: str) -> str:
    """Canonicalize a table name (``fit-peaks`` and ``fit_peaks`` are one table).

    Raises ``ValueError`` naming the valid tables when *table* is not one.
    """
    name = str(table).strip().lower().replace("-", "_")
    if name not in _TABLE_SPECS:
        raise ValueError(
            f"unknown table {table!r}; available tables are {list(READ_TABLES)}"
        )
    return name


def _open(file_path: Union[str, Path]) -> h5py.File:
    """Open the pipeline file read-only, gating on the format-version stamp.

    Deliberately *not* the full :func:`open_pipeline_file` validation: that also
    requires the provenance record, which a read-only column tap has no business
    demanding. What it does share is the compatibility gate -- a file written by
    a future MAJOR format version must be refused, not misread. Every interface
    goes through this one function, so the CLI and the Python API accept and
    reject exactly the same files.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline file not found: {path}\n\n"
            f"To create a new pipeline:\n"
            f"  ftmwpipeline data import {path} path/to/data/"
        )
    h5f = h5py.File(path, "r")
    try:
        check_format_compatibility(path, h5f)
    except BaseException:
        h5f.close()
        raise
    return h5f


def read_table_impl(
    file_path: Union[str, Path],
    table: str,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read one persisted table as bulk columns, without recomputing anything.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` file.
    table :
        One of :data:`READ_TABLES` (hyphens and underscores are equivalent).
    columns :
        Column names to read; ``None`` reads every column of the table. An
        unknown name raises ``ValueError`` listing what is available.

    Returns
    -------
    dict
        ``{column_name: numpy array}``, all of equal length, in the requested
        order. Sentinel conventions (NaN for an absent float, ``-1`` for an
        absent id, tri-state flags) are documented on each table's column-spec
        mapping in the ``io`` serializers.

    Raises
    ------
    ValueError
        If the table name or a column name is unknown, if the stage that
        produces the table has not been run, or if the persisted group is
        malformed.
    """
    name = normalize_table_name(table)
    spec = _TABLE_SPECS[name]
    with _open(file_path) as h5f:
        if spec.group not in h5f:
            raise ValueError(
                f"No {spec.group} data found in {file_path}; table {name!r} is "
                f"unavailable. Run {spec.hint} first."
            )
        return spec.reader(h5f[spec.group], columns)


def _row_count(stage_group: h5py.Group, count_path: str) -> Optional[int]:
    """The row count a stage group records, or ``None`` when it records none.

    ``count_path`` is an attribute name, or ``"subgroup/attr"`` for the stages
    that keep their counts one level in, or empty when the stage records no
    count for this table. Missing at any step reads as unknown rather than as
    an error: the listing is informational, and a table with an unknown count is
    still readable.
    """
    if not count_path:
        return None
    subgroup, _, attr = count_path.rpartition("/")
    group: Any = stage_group
    if subgroup:
        if subgroup not in stage_group:
            return None
        group = stage_group[subgroup]
    raw = group.attrs.get(attr)
    return None if raw is None else int(raw)


def read_tables_impl(file_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """List the readable tables and what each one holds in this file.

    Reads only group attributes, so it is cheap on any file.

    Returns
    -------
    dict
        ``{table_name: {"available": bool, "n_rows": int or None,
        "columns": [...], "group": str}}`` for every table in
        :data:`READ_TABLES`. ``available`` is False (and ``n_rows`` ``None``)
        when the producing stage has not been run; ``columns`` is the canonical
        column list either way, since a column absent from an older file still
        reads back as its documented fill value.
    """
    out: Dict[str, Dict[str, Any]] = {}
    with _open(file_path) as h5f:
        for name, spec in _TABLE_SPECS.items():
            available = spec.group in h5f
            n_rows: Optional[int] = None
            if available:
                n_rows = _row_count(h5f[spec.group], spec.count_path)
            out[name] = {
                "available": available,
                "n_rows": n_rows,
                "columns": list(spec.specs),
                "group": spec.group,
            }
    return out


# ---------------------------------------------------------------------------
# Cheap top-level scalars
# ---------------------------------------------------------------------------


def _decode(value: Any) -> Any:
    """h5py attribute -> a plain Python scalar."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _copy_attrs(
    h5f: h5py.File, group: str, prefix: str, names: Sequence[str], out: Dict[str, Any]
) -> None:
    """Copy the named attributes of *group* into *out* under ``prefix.``."""
    if group not in h5f:
        return
    attrs = h5f[group].attrs
    for name in names:
        if name in attrs:
            out[f"{prefix}.{name}"] = _decode(attrs[name])


_TIMEBASE_ATTRS = (
    "epsilon",
    "sigma_epsilon",
    "kappa_sys",
    "preconditions_passed",
    "lattice_g_mhz",
    "n_detected",
    "n_used",
    "start_us",
    "end_us",
    "span_us",
    "creation_time",
)

_FID_ATTRS = (
    "n_points",
    "duration_us",
    "probe_freq_mhz",
    "sideband",
    "shots",
    "spacing_seconds",
)

_SOURCE_ATTRS = ("source_path", "format_name", "import_timestamp", "source_hash")

#: Canonical Stage 1 data-selection settings, as persisted by ``ft run``. They
#: are what every later stage analyzes, so ``ft.acquisition_us`` -- and with it
#: the Fourier resolution element -- is fixed here, not at Stage 5.
_FT_ATTRS = (
    "start_us",
    "end_us",
    "trim_min_mhz",
    "trim_max_mhz",
    "units_power",
)

#: Fields of the start-detection sweep record worth reporting: what the sweep
#: found, not the knobs it was given (those are ``settings show``'s job).
#: ``guard_margin_us`` is the exception -- ``start_us = chirp_end_us + margin``,
#: so without it the stamped start time cannot be checked against the record.
_START_RECORD_FIELDS = (
    "chirp_end_us",
    "chirp_detected",
    "guard_margin_us",
    "floor",
    "plateau",
    "resolved_band_min_mhz",
    "resolved_band_max_mhz",
)

#: JSON sentinel the start-detection record uses for an unset optional field.
_NONE_SENTINEL = "__None__"


def _read_ft_window(h5f: h5py.File, out: Dict[str, Any]) -> None:
    """Report the canonical FT window, including its derived active length."""
    group = "processing_parameters/ft_processing"
    if group not in h5f:
        return
    # Older records stored the settings bundle as one JSON `parameters` blob
    # rather than as individual attributes, and Stage 1 reads both shapes. This
    # view has to read both the same way, or a blob-only file reports no
    # `ft.units_power` here while the display transform reads one from the blob
    # -- the same question answered two ways by two readers.
    attrs = fold_settings_blob(dict(h5f[group].attrs))
    for name in _FT_ATTRS:
        if name in attrs:
            out[f"ft.{name}"] = _decode(attrs[name])
    duration = out.get("fid.duration_us")
    if duration is not None:
        # The one derived value here, computed by the same helper Stages 3-5
        # use, so a consumer's resolution element cannot disagree with the fit's.
        out["ft.acquisition_us"] = active_acquisition_us(
            float(duration), out.get("ft.start_us"), out.get("ft.end_us")
        )
    _alias_section(out, "ft", "stage1")


def _alias_section(out: Dict[str, Any], canonical: str, synonym: str) -> None:
    """Emit every ``canonical.`` key a second time under ``synonym.``.

    The CLI declares the semantic object name and its ``stageN`` spelling fully
    interchangeable (``ft`` / ``stage1``, see ``cli.utils.add_stage_object``),
    and ``settings show`` names the same persisted Stage 1 knobs
    ``stage1.units_power`` where this view names them ``ft.units_power``. Rather
    than make a consumer know which surface it is talking to, both spellings
    resolve here to the same value.
    """
    for key in [k for k in out if k.startswith(f"{canonical}.")]:
        out[f"{synonym}.{key.split('.', 1)[1]}"] = out[key]


def _read_start_record(h5f: h5py.File, out: Dict[str, Any]) -> None:
    """Report the start-detection sweep outcome, when the sweep has been run."""
    if "stage0_fid_data" not in h5f:
        return
    record = load_json_attr(
        h5f["stage0_fid_data"], "recommended_start_detection", None, label="stage0"
    )
    if not isinstance(record, dict):
        return
    for name in _START_RECORD_FIELDS:
        if name not in record:
            continue
        value = record[name]
        if value == _NONE_SENTINEL:
            value = None
        out[f"start.{name}"] = value


def _read_tau_scalars(
    h5f: h5py.File, group: str, prefix: str, out: Dict[str, Any]
) -> None:
    """Report one decay-time calibration's scalars under ``prefix.``."""
    if group not in h5f:
        return
    for key, value in read_tau_scalars(h5f[group]).items():
        out[f"{prefix}.{key}"] = value


def read_metadata_impl(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Read the cheap top-level scalars a lightweight consumer needs.

    Group attributes only -- no stage artifact is deserialized and no window is
    traversed, so this is fast regardless of how much the file holds. Keys are
    dotted and a section is simply absent when its stage has not been run, so a
    consumer reads with ``.get()`` rather than branching on stage completion:

    ``file.`` / ``source.``
        Format version, the stage-completion list, and the import provenance.
    ``fid.``
        The raw acquisition: point count, record length, probe frequency,
        sideband, shot count, sample spacing.
    ``start.``
        The start-detection sweep outcome, present once ``start run`` has
        stamped one: where the chirp collapsed, whether it was found at all,
        the guard margin added past it, and the band the sweep integrated over.
    ``ft.`` (equivalently ``stage1.``)
        The canonical Stage 1 data selection -- the analyzed window and the
        frequency trim -- plus the derived ``ft.acquisition_us``, the active
        record length every later stage's resolution element ``1 / T`` follows
        from. It is fixed here, at Stage 1, so it is readable long before a fit
        exists. Every key in this section is emitted under both prefixes,
        because the CLI treats ``ft`` and ``stage1`` as interchangeable object
        names and ``settings show`` spells these same knobs ``stage1.``.

        ``stage5.acquisition_us`` is what the fit recorded. The two agree
        whenever the fit ran on the canonical Stage 1 window -- the usual case
        -- but editing the processing settings between runs separates them, and
        a fitted decay time must be paired against the window the fit measured
        it over. Note also that ``1 / T`` is the resolution element, not a line
        width: the FWHM of the finite-``T`` line shape is
        :func:`~ftmwpipeline.fitting.validation.feature_fwhm`, a factor of order
        two wider at typical decay times.
    ``tau.`` / ``tau_g.``
        The Stage 2b decay-time calibration and its Gaussian twin: the fitted
        decay time with its uncertainty, the contributor and spur-bin counts,
        and the line-shape vote.
    ``stage3.`` / ``stage4.`` / ``stage5.``
        Per-stage counts and creation times, plus the fit's ``acquisition_us``.
    ``timebase.``
        The digitizer-clock scale error and its uncertainty.

    ``file.completed_stages`` is a list of stage keys; every other value is a
    scalar (``str`` / ``int`` / ``float`` / ``bool``) or ``None``.
    """
    out: Dict[str, Any] = {}
    with _open(file_path) as h5f:
        out["file.path"] = str(Path(file_path))
        version = h5f.attrs.get("ftmw_format_version")
        out["file.format_version"] = _decode(version) if version is not None else None
        created = h5f.attrs.get("created_with_ftmwpipeline")
        out["file.created_with"] = _decode(created) if created is not None else None
        out["file.completed_stages"] = sorted(
            load_json_attr(h5f["pipeline_stages"], "completed_stages", [])
            if "pipeline_stages" in h5f
            else []
        )

        _copy_attrs(h5f, "source_metadata", "source", _SOURCE_ATTRS, out)
        _copy_attrs(h5f, "stage0_fid_data/acquisition", "fid", _FID_ATTRS, out)
        _read_start_record(h5f, out)
        _read_ft_window(h5f, out)
        _copy_attrs(h5f, "timebase_calibration", "timebase", _TIMEBASE_ATTRS, out)

        _read_tau_scalars(h5f, TAU_GROUP_PATH, "tau", out)
        _read_tau_scalars(h5f, GAUSSIAN_GROUP_PATH, "tau_g", out)
        if "stage3_peaks" in h5f:
            for key, value in read_peak_scalars(h5f["stage3_peaks"]).items():
                out[f"stage3.{key}"] = value
        if "stage4_windows" in h5f:
            for key, value in read_window_plan_scalars(h5f["stage4_windows"]).items():
                out[f"stage4.{key}"] = value
        if "stage5_fitting" in h5f:
            for key, value in read_fit_scalars(h5f["stage5_fitting"]).items():
                out[f"stage5.{key}"] = value
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    """One table cell as text.

    Floats use ``repr`` (shortest round-trip) rather than a fixed precision:
    this surface dumps the persisted values themselves, so a reader must get
    back exactly what is on disk. The presentation-formatted, calibrated line
    list is ``report table``'s job, not this one.
    """
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return ";".join(_cell(v) for v in value)
    return str(value)


def _json_value(value: Any) -> Any:
    """A JSON-safe view of a cell value (non-finite floats -> ``None``)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _rows(table: Dict[str, np.ndarray]) -> int:
    lengths = {len(v) for v in table.values()}
    if len(lengths) > 1:
        raise ValueError(f"table columns have mismatched lengths: {lengths}")
    return lengths.pop() if lengths else 0


def format_table_impl(table: Dict[str, np.ndarray], fmt: str = "csv") -> str:
    """Render a column table as delimited text or JSON.

    ``csv`` / ``tsv`` emit a header row then one row per record; ``json`` emits
    a list of objects (non-finite floats become ``null`` so the output is
    strict-valid JSON).
    """
    if fmt not in VALID_READ_FORMATS:
        raise ValueError(
            f"unknown format {fmt!r}; expected one of {list(VALID_READ_FORMATS)}"
        )
    names = list(table)
    n = _rows(table)

    if fmt == "json":
        records = [
            {name: _json_value(table[name][i]) for name in names} for i in range(n)
        ]
        return json.dumps(records, indent=2) + "\n"

    buf = io.StringIO()
    writer = csv.writer(
        buf, delimiter="\t" if fmt == "tsv" else ",", lineterminator="\n"
    )
    writer.writerow(names)
    for i in range(n):
        writer.writerow([_cell(table[name][i]) for name in names])
    return buf.getvalue()


def format_metadata_impl(metadata: Dict[str, Any], fmt: str = "csv") -> str:
    """Render the flat metadata mapping as ``key,value`` rows or JSON."""
    if fmt not in VALID_READ_FORMATS:
        raise ValueError(
            f"unknown format {fmt!r}; expected one of {list(VALID_READ_FORMATS)}"
        )
    if fmt == "json":
        safe = {key: _json_value(value) for key, value in metadata.items()}
        return json.dumps(safe, indent=2) + "\n"
    buf = io.StringIO()
    writer = csv.writer(
        buf, delimiter="\t" if fmt == "tsv" else ",", lineterminator="\n"
    )
    writer.writerow(["key", "value"])
    for key, value in metadata.items():
        writer.writerow([key, _cell(value)])
    return buf.getvalue()


def write_text_impl(text: str, output: Union[str, Path]) -> Path:
    """Write rendered text to *output*, creating parent directories."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
