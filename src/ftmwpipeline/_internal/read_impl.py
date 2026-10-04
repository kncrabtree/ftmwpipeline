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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

from ..contract import (
    FID_SAMPLES_SCHEMA,
    WINDOW_STATUS_SCHEMA,
    Absent,
    WindowStatusRow,
)
from ..core.absent import STATUS_NOT_RUN, STATUS_PRESENT, STATUS_UNDEFINED
from ..core.settings_framework import NONE as NONE_SENTINEL
from ..file_manager import (
    BadSettingError,
    PipelineCorruptionError,
    PipelineFileNotFoundError,
    PipelineStageTracker,
    StageDependencyError,
    check_format_compatibility,
    is_transient_open_error,
)
from ..io._hdf5_helpers import (
    REQUIRED,
    ColumnSpec,
    load_json_attr,
    resolve_column_selection,
)
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
    read_fit_window_quality_recorded,
)
from ..io.peak_serialization import (
    PEAK_COLUMN_SPECS,
    _promotion_cutoff,
    read_peak_columns,
    read_peak_scalars,
)
from ..io.stage5_partial_serialization import read_stage5_partial_counts
from ..io.stage6_review_serialization import (
    fit_declares_clocks,
    read_created_window_bounds,
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
from ..serialize import STATUS_SUFFIX, with_status_columns
from .absence_rules import (
    clock_lattice_or_absent,
    degenerate_edge_coherence,
    degenerate_f_test,
    degenerate_internal_snr,
    degenerate_orth_evidence,
    degenerate_stage3_snr,
    float_or_absent,
    int_or_absent,
    knockout_absence,
)
from .atomic import exists as pipeline_exists
from .atomic import h5open
from .shared_utils import active_acquisition_us

__all__ = [
    "READ_TABLES",
    "VALID_READ_FORMATS",
    "read_table_impl",
    "read_tables_impl",
    "read_metadata_impl",
    "window_status_impl",
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

#: Columns that can be absent, each with a ``<column>__status`` companion.
_WINDOW_STATUS_ABSENT_CAPABLE = ("n_fitted_peaks", "live")

#: Layout of the ``window_status`` table (the status columns are ``uint8``).
WINDOW_STATUS_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "window_id": ("i8", REQUIRED),
    "freq_min_mhz": ("f8", REQUIRED),
    "freq_max_mhz": ("f8", REQUIRED),
    "created": ("bool", REQUIRED),
    "n_fitted_peaks": ("i8", 0),
    "n_fitted_peaks__status": ("u1", 0),
    "live": ("bool", False),
    "live__status": ("u1", 0),
}

_WINDOW_STATUS_DTYPES: Dict[str, Any] = {
    "window_id": np.int64,
    "freq_min_mhz": np.float64,
    "freq_max_mhz": np.float64,
    "created": np.bool_,
    "n_fitted_peaks": np.int64,
    "live": np.bool_,
}

_FIT_HINT = "fit_peaks() / 'fit run'"

#: The bare CLI verb that produces each table group, named in the refusal.
_COMMAND_BY_GROUP: Dict[str, str] = {
    TAU_GROUP_PATH: "tau run",
    GAUSSIAN_GROUP_PATH: "tau run --gaussian",
    "stage3_peaks": "peaks run",
    "stage4_windows": "windows run",
    "stage5_fitting": "fit run",
}

# ---------------------------------------------------------------------------
# Status columns. The io readers return the stored encodings (NaN / -1 / ''
# fills) untouched; this layer adds a ``<column>__status`` companion to each
# absent-capable column, derived at read time with the shared rules in
# ``absence_rules`` so a quantity reports the same status here as on
# ``FinalPeak``. The value column keeps its stored fill, with two exceptions:
# a ``fit_audit`` separation reject stored placeholder ``f_statistic`` 0.0 and
# ``p_value`` 1.0 for a test that never ran, and those read as ``nan``; and a
# degenerate statistic an earlier writer stored as an ordinary number (the
# ``absence_rules.degenerate_*`` predicates) reads as the ``nan`` a current
# writer stores, with status ``UNDEFINED``, so old and new files read alike.
# ---------------------------------------------------------------------------

_StatusDeriver = Callable[[h5py.Group, Dict[str, np.ndarray]], Dict[str, np.ndarray]]


def _status_specs(
    specs: Dict[str, ColumnSpec], absent_capable: Sequence[str]
) -> Dict[str, ColumnSpec]:
    """*specs* with a ``("u1", 0)`` ``<column>__status`` entry after each column."""
    out: Dict[str, ColumnSpec] = {}
    for name, spec in specs.items():
        out[name] = spec
        if name in absent_capable:
            out[name + STATUS_SUFFIX] = ("u1", 0)
    return out


def _status_of(items: Sequence[Any]) -> np.ndarray:
    """The ``uint8`` status codes of a list of values-or-:class:`Absent`."""
    return np.fromiter(
        (i.status if isinstance(i, Absent) else STATUS_PRESENT for i in items),
        dtype=np.uint8,
        count=len(items),
    )


def _floats(values: np.ndarray, not_run: Optional[np.ndarray] = None) -> np.ndarray:
    """Status of a float column: non-finite is ``UNDEFINED``, *not_run* rows ``NOT_RUN``."""
    return _status_of(
        [
            Absent.NOT_RUN if not_run is not None and not_run[i] else float_or_absent(v)
            for i, v in enumerate(values)
        ]
    )


def _nan_not_run(values: np.ndarray) -> np.ndarray:
    """Status of a float column whose NaN means the quantity was never computed."""
    return _status_of(
        [Absent.NOT_RUN if math.isnan(float(v)) else float_or_absent(v) for v in values]
    )


def _ints(
    values: np.ndarray, absent: Absent = Absent.NOT_RUN, sentinel: int = -1
) -> np.ndarray:
    return _status_of(
        [int_or_absent(v, sentinel=sentinel, absent=absent) for v in values]
    )


# -- fit_peaks ---------------------------------------------------------------

_FIT_PEAK_ABSENT_CAPABLE = (
    "detection_index",
    "frequency_error",
    "amplitude_error",
    "phase",
    "phase_error",
    "decay_rate",
    "decay_rate_error",
    "snr",
    "chi_squared",
    "clock_lattice",
    "derivation",
    "peak_uid",
    "knockout_delta_chi2",
    "knockout_expected_delta_chi2",
    "knockout_supported",
    "knockout_p_value",
    "knockout_n_eff",
    "knockout_aicc_delta",
    "unresolved_spread_mhz",
)

_FIT_PEAK_NEEDS = ("knockout_supported", "knockout_delta_chi2")

#: The knockout block whose columns are all ``NOT_RUN`` when the test did not run.
_KNOCKOUT_COLUMNS = (
    "knockout_delta_chi2",
    "knockout_expected_delta_chi2",
    "chi_squared",
    "knockout_p_value",
    "knockout_n_eff",
    "knockout_aicc_delta",
)


def _fit_peak_status(
    h5_group: h5py.Group, raw: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    ko = [
        knockout_absence(s, d)
        for s, d in zip(raw["knockout_supported"], raw["knockout_delta_chi2"])
    ]
    no_knockout = np.array([k is not None for k in ko], dtype=bool)
    out: Dict[str, np.ndarray] = {}
    for col in _FIT_PEAK_ABSENT_CAPABLE:
        if col not in raw:
            continue
        values = raw[col]
        if col == "knockout_supported":
            out[col] = _status_of(ko)
        elif col in _KNOCKOUT_COLUMNS:
            out[col] = _floats(values, no_knockout)
        elif col == "detection_index":
            out[col] = _ints(values, Absent.UNDEFINED)
        elif col in ("derivation", "peak_uid"):
            out[col] = _ints(values)
        elif col == "unresolved_spread_mhz":
            out[col] = _nan_not_run(values)
        elif col == "clock_lattice":
            declared = fit_declares_clocks(h5_group.file)
            out[col] = _status_of(
                [clock_lattice_or_absent(v, declared=declared) for v in values]
            )
        else:
            out[col] = _floats(values)
    return out


# -- fit_windows -------------------------------------------------------------

_FIT_WINDOW_ABSENT_CAPABLE = (
    "freq_min",
    "freq_max",
    "tau_us",
    "tau_error",
    "tau_fitted",
    "aic",
    "reduced_chi2",
    "edge_coherence_low",
    "edge_coherence_high",
)


_EDGE_COHERENCE_COLUMNS = ("edge_coherence_low", "edge_coherence_high")


def _fit_window_status(
    h5_group: h5py.Group, raw: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    edges = [c for c in _EDGE_COHERENCE_COLUMNS if c in raw]
    recorded = read_fit_window_quality_recorded(h5_group, edges) if edges else {}
    for col in _FIT_WINDOW_ABSENT_CAPABLE:
        if col not in raw:
            continue
        values = raw[col]
        if col in edges:
            # NaN is UNDEFINED when the fit computed the edge (it was
            # degenerate: an empty residual or no positive noise; read-time
            # masking turns an earlier writer's 0.0 into that NaN) and NOT_RUN
            # when it never evaluated the window.
            out[col] = _status_of(
                [
                    (
                        (Absent.UNDEFINED if recorded[col][i] else Absent.NOT_RUN)
                        if math.isnan(float(v))
                        else float_or_absent(v)
                    )
                    for i, v in enumerate(values)
                ]
            )
        elif col in ("freq_min", "freq_max"):
            out[col] = _nan_not_run(values)
        elif col == "tau_fitted":
            out[col] = _ints(values)
        elif col == "tau_us":
            # A non-positive decay time is as undefined as a missing one
            # (FinalPeak.decay_time_us).
            out[col] = _floats(np.where(values > 0, values, np.nan))
        else:
            out[col] = _floats(values)
    return out


def _mask_degenerate_edges(raw: Dict[str, np.ndarray]) -> None:
    """Read an earlier writer's 0.0 for an undefined edge coherence as ``nan``."""
    for col in _EDGE_COHERENCE_COLUMNS:
        if col in raw:
            raw[col] = np.where(degenerate_edge_coherence(raw[col]), np.nan, raw[col])


# -- fit_audit ---------------------------------------------------------------

_FIT_AUDIT_ABSENT_CAPABLE = (
    "chi2_after",
    "f_statistic",
    "p_value",
    "aic_after",
    "n_eff",
    "aicc_delta",
)

_FIT_AUDIT_NEEDS = (
    "decision",
    "separation_ok",
    "n_eff",
    "f_statistic",
    "p_value",
    "chi2_before",
    "chi2_after",
)


def _audit_masks(raw: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """``(decision, separation_gate_never_ran)`` for the audit rows."""
    decision = np.asarray([str(d) for d in raw["decision"]], dtype=object)
    sep_reject = (~np.asarray(raw["separation_ok"], dtype=bool)) & np.isnan(
        np.asarray(raw["n_eff"], dtype=float)
    )
    return decision, sep_reject


def _fit_audit_status(
    h5_group: h5py.Group, raw: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    decision, sep_reject = _audit_masks(raw)
    is_seed = decision == "seed"
    is_spur = decision == "spur-drop"
    is_ko = decision == "knockout-null"
    out: Dict[str, np.ndarray] = {}
    for col in _FIT_AUDIT_ABSENT_CAPABLE:
        if col not in raw:
            continue
        values = raw[col]
        if col in ("n_eff", "aicc_delta"):
            not_run = is_seed | is_spur | sep_reject
        elif col == "f_statistic":
            not_run = is_spur | sep_reject | is_ko
        elif col == "p_value":
            not_run = is_spur | sep_reject
        else:  # chi2_after, aic_after
            not_run = is_spur
        out[col] = _floats(values, not_run)
    return out


def _mask_unrun_gates(raw: Dict[str, np.ndarray]) -> None:
    """Replace the 0.0 / 1.0 stored for F and p where no F-test value exists with NaN.

    A separation reject stores them for a test that never ran (status
    ``NOT_RUN``). An earlier writer also stored them for a degenerate test (no
    residual degrees of freedom, a non-positive chi-squared), which a current
    writer stores as ``nan`` (status ``UNDEFINED``); see
    :func:`~ftmwpipeline._internal.absence_rules.degenerate_f_test`. The
    ``knockout-null`` step's p comes from the reversed knockout test and is
    left as stored.
    """
    decision, sep_reject = _audit_masks(raw)
    degenerate = (
        degenerate_f_test(
            raw["f_statistic"], raw["p_value"], raw["chi2_before"], raw["chi2_after"]
        )
        & ~sep_reject
        & (decision != "spur-drop")
        & (decision != "knockout-null")
    )
    for col in ("f_statistic", "p_value"):
        if col in raw:
            raw[col] = np.where(sep_reject | degenerate, np.nan, raw[col])


# -- fit_doublets ------------------------------------------------------------

_FIT_DOUBLET_ABSENT_CAPABLE = (
    "chi2r_merged",
    "delta_chi2_raw",
    "delta_aicc",
    "merged_frequency_mhz",
    "merged_amplitude",
    "merged_phase",
    "merged_tau_us",
    "orth_evidence_delta_chi2",
)


def _fit_doublet_status(
    h5_group: h5py.Group, raw: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    return {col: _floats(raw[col]) for col in _FIT_DOUBLET_ABSENT_CAPABLE if col in raw}


def _mask_untested_orth_evidence(raw: Dict[str, np.ndarray]) -> None:
    """Read an earlier writer's 0.0 orthogonal evidence from an untested pair as ``nan``."""
    col = "orth_evidence_delta_chi2"
    if col in raw:
        raw[col] = np.where(
            degenerate_orth_evidence(raw[col], raw["support_bins"]), np.nan, raw[col]
        )


# -- peaks -------------------------------------------------------------------

_PEAK_ABSENT_CAPABLE = (
    "index",
    "snr",
    "noise_std_local",
    "classification",
    "promoted",
    "internal_snr",
    "internal_frequency",
    "leakage_pedestal",
)


def _peak_status(
    h5_group: h5py.Group, raw: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for col in _PEAK_ABSENT_CAPABLE:
        if col not in raw:
            continue
        values = raw[col]
        if col == "index":
            out[col] = _ints(values)
        elif col == "classification":
            out[col] = _status_of(
                [Absent.NOT_RUN if str(v) == "" else v for v in values]
            )
        elif col == "promoted":
            # Promotion is derived against the recorded cutoff; a file without
            # one predates the record, so promotion was never decided.
            code = (
                STATUS_NOT_RUN
                if _promotion_cutoff(h5_group) is None
                else STATUS_PRESENT
            )
            out[col] = np.full(len(values), code, dtype=np.uint8)
        elif col in ("snr", "noise_std_local"):
            out[col] = _floats(values)
        elif col == "internal_snr":
            undefined = degenerate_internal_snr(values, raw["internal_frequency"])
            out[col] = np.where(
                undefined, STATUS_UNDEFINED, _nan_not_run(values)
            ).astype(np.uint8)
        else:
            out[col] = _nan_not_run(values)
    return out


def _mask_degenerate_snr(raw: Dict[str, np.ndarray]) -> None:
    """Read an earlier writer's 0.0 for an undefined Stage 3 SNR as ``nan``.

    ``snr`` is undefined where the stored local noise is not positive;
    ``internal_snr`` where the internal pass contributed the peak and stored no
    value (see :mod:`~ftmwpipeline._internal.absence_rules`).
    """
    if "snr" in raw:
        raw["snr"] = np.where(
            degenerate_stage3_snr(raw["noise_std_local"]), np.nan, raw["snr"]
        )
    if "internal_snr" in raw:
        raw["internal_snr"] = np.where(
            degenerate_internal_snr(raw["internal_snr"], raw["internal_frequency"]),
            np.nan,
            raw["internal_snr"],
        )


@dataclass(frozen=True)
class _StatusLayout:
    """How one table gains its status columns."""

    name: str
    specs: Dict[str, ColumnSpec]
    io_reader: Callable[..., Dict[str, np.ndarray]]
    absent_capable: Tuple[str, ...]
    derive: _StatusDeriver
    #: Raw columns the derivation needs whatever is requested.
    needs: Tuple[str, ...] = ()
    #: Subgroup holding the stored datasets, to tell a synthesized fill from a
    #: stored one (``None``: the stage group itself).
    dataset_group: Optional[str] = None
    #: Columns computed rather than stored (never a synthesized fill).
    derived: Tuple[str, ...] = ()
    #: JSON event logs report synthesized rows through ``record_row`` instead.
    json_log: bool = False
    #: Post-process the raw columns (value masking) before they are returned.
    adjust: Optional[Callable[[Dict[str, np.ndarray]], None]] = None
    #: Columns whose own rule decides a synthesized fill too. The knockout
    #: rule takes precedence (spec): a line whose test ran but whose file
    #: predates the p-value column reads UNDEFINED, as on ``FinalPeak``.
    rule_over_synthesized: Tuple[str, ...] = ()


def _status_reader(layout: _StatusLayout) -> _Reader:
    extended = _status_specs(layout.specs, layout.absent_capable)

    def reader(
        h5_group: h5py.Group, columns: Optional[Sequence[str]]
    ) -> Dict[str, np.ndarray]:
        requested = resolve_column_selection(columns, list(extended), table=layout.name)
        wanted_status = [
            c[: -len(STATUS_SUFFIX)] for c in requested if c.endswith(STATUS_SUFFIX)
        ]
        base = [c for c in requested if not c.endswith(STATUS_SUFFIX)]
        to_read = list(dict.fromkeys([*base, *wanted_status]))
        if wanted_status or layout.adjust is not None:
            to_read = list(dict.fromkeys([*to_read, *layout.needs]))
        flags: Dict[str, List[bool]] = {c: [] for c in to_read}
        if layout.json_log:
            raw = layout.io_reader(h5_group, to_read, synthesized=flags)
        else:
            raw = layout.io_reader(h5_group, to_read)
        raw = dict(raw)
        n = len(next(iter(raw.values()))) if raw else 0
        if layout.adjust is not None:
            layout.adjust(raw)
        status: Dict[str, np.ndarray] = {}
        if wanted_status:
            status = layout.derive(h5_group, raw)
            for col in wanted_status:
                if col in layout.rule_over_synthesized:
                    continue
                synth = _synthesized(h5_group, layout, col, flags, n)
                status[col] = np.where(synth, STATUS_NOT_RUN, status[col]).astype(
                    np.uint8
                )
        out: Dict[str, np.ndarray] = {}
        for c in requested:
            if c.endswith(STATUS_SUFFIX):
                out[c] = status[c[: -len(STATUS_SUFFIX)]]
            else:
                out[c] = raw[c]
        return out

    return reader


def _synthesized(
    h5_group: h5py.Group,
    layout: _StatusLayout,
    col: str,
    flags: Dict[str, List[bool]],
    n: int,
) -> np.ndarray:
    """Per-row flags: was this column's value a fill the reader synthesized?"""
    if layout.json_log:
        return np.asarray(flags.get(col, []), dtype=bool).reshape(n)
    if col in layout.derived:
        return np.zeros(n, dtype=bool)
    holder = h5_group[layout.dataset_group] if layout.dataset_group else h5_group
    return np.full(n, col not in holder, dtype=bool)


_STATUS_LAYOUTS: Dict[str, _StatusLayout] = {
    layout.name: layout
    for layout in (
        _StatusLayout(
            "fit_peaks",
            FIT_PEAK_COLUMN_SPECS,
            read_fit_peak_columns,
            _FIT_PEAK_ABSENT_CAPABLE,
            _fit_peak_status,
            needs=_FIT_PEAK_NEEDS,
            dataset_group="peaks",
            derived=("shape",),
            rule_over_synthesized=_KNOCKOUT_COLUMNS,
        ),
        _StatusLayout(
            "fit_windows",
            FIT_WINDOW_COLUMN_SPECS,
            read_fit_window_columns,
            _FIT_WINDOW_ABSENT_CAPABLE,
            _fit_window_status,
            dataset_group="windows",
            derived=("n_peaks",),
            adjust=_mask_degenerate_edges,
        ),
        _StatusLayout(
            "fit_audit",
            FIT_AUDIT_COLUMN_SPECS,
            read_fit_audit_columns,
            _FIT_AUDIT_ABSENT_CAPABLE,
            _fit_audit_status,
            needs=_FIT_AUDIT_NEEDS,
            json_log=True,
            adjust=_mask_unrun_gates,
        ),
        _StatusLayout(
            "fit_doublets",
            FIT_DOUBLET_COLUMN_SPECS,
            read_fit_doublet_columns,
            _FIT_DOUBLET_ABSENT_CAPABLE,
            _fit_doublet_status,
            needs=("support_bins",),
            json_log=True,
            adjust=_mask_untested_orth_evidence,
        ),
        _StatusLayout(
            "peaks",
            PEAK_COLUMN_SPECS,
            read_peak_columns,
            _PEAK_ABSENT_CAPABLE,
            _peak_status,
            needs=("noise_std_local", "internal_frequency"),
            derived=("promoted",),
            adjust=_mask_degenerate_snr,
        ),
    )
}

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
            specs=_status_specs(
                PEAK_COLUMN_SPECS, _STATUS_LAYOUTS["peaks"].absent_capable
            ),
            reader=_status_reader(_STATUS_LAYOUTS["peaks"]),
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
            specs=_status_specs(
                FIT_PEAK_COLUMN_SPECS, _STATUS_LAYOUTS["fit_peaks"].absent_capable
            ),
            reader=_status_reader(_STATUS_LAYOUTS["fit_peaks"]),
            count_path="n_fitted_peaks",
            hint=_FIT_HINT,
        ),
        "fit_windows": _TableSpec(
            group="stage5_fitting",
            specs=_status_specs(
                FIT_WINDOW_COLUMN_SPECS, _STATUS_LAYOUTS["fit_windows"].absent_capable
            ),
            reader=_status_reader(_STATUS_LAYOUTS["fit_windows"]),
            count_path="n_windows",
            hint=_FIT_HINT,
        ),
        "fit_audit": _TableSpec(
            group="stage5_fitting",
            specs=_status_specs(
                FIT_AUDIT_COLUMN_SPECS, _STATUS_LAYOUTS["fit_audit"].absent_capable
            ),
            reader=_status_reader(_STATUS_LAYOUTS["fit_audit"]),
            count_path="",
            hint=_FIT_HINT,
        ),
        "fit_doublets": _TableSpec(
            group="stage5_fitting",
            specs=_status_specs(
                FIT_DOUBLET_COLUMN_SPECS, _STATUS_LAYOUTS["fit_doublets"].absent_capable
            ),
            reader=_status_reader(_STATUS_LAYOUTS["fit_doublets"]),
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
        "window_status": _TableSpec(
            group="stage4_windows",
            specs=WINDOW_STATUS_COLUMN_SPECS,
            reader=lambda group, columns: _window_status_columns(group.file, columns),
            count_path="",
            hint=_WINDOWS_HINT,
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
        raise BadSettingError(
            "table",
            f"one of: {', '.join(READ_TABLES)}",
            table,
            message=f"unknown table {table!r}; available tables are "
            f"{list(READ_TABLES)}",
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
    if not pipeline_exists(path):
        raise PipelineFileNotFoundError(
            path,
            message=(
                f"Pipeline file not found: {path}\n\n"
                f"To create a new pipeline:\n"
                f"  ftmwpipeline data import {path} path/to/data/"
            ),
        )
    try:
        h5f = h5open(path, "r")
    except OSError as exc:
        if is_transient_open_error(exc):
            raise
        # Exists but is not an openable HDF5 file: ``file_corrupt``.
        raise PipelineCorruptionError(
            path,
            f"HDF5 error: {exc}",
            message=f"Failed to open pipeline file {path}: {exc}",
        ) from exc
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
        The ``fit_peaks``, ``fit_windows``, ``fit_audit``, ``fit_doublets``
        and ``peaks`` tables also carry a ``uint8`` ``<column>__status`` column
        for each absent-capable column (``0`` present, ``1`` not run, ``2``
        undefined); the value column keeps its stored fill, except that a
        ``fit_audit`` separation reject's placeholder ``f_statistic`` /
        ``p_value`` (no test ran) read as ``nan``, and so does a degenerate
        statistic an earlier release stored as a number (an F-test without
        degrees of freedom, an untested doublet's orthogonal evidence, an
        undefined residual edge coherence, a Stage 3 SNR without positive
        noise), with status ``2``.

    Raises
    ------
    StageDependencyError
        If the stage that produces the table has not been run (code
        ``stage_not_run``, ``command`` the CLI verb that produces it). Also a
        ``ValueError``.
    ValueError
        If the table name or a column name is unknown, or if the persisted
        group is malformed.
    """
    name = normalize_table_name(table)
    spec = _TABLE_SPECS[name]
    with _open(file_path) as h5f:
        if spec.group not in h5f:
            raise StageDependencyError(
                f"read_table {name}",
                [spec.group],
                Path(file_path),
                command=_COMMAND_BY_GROUP[spec.group],
                message=(
                    f"No {spec.group} data found in {file_path}; table {name!r} "
                    f"is unavailable. Run {spec.hint} first."
                ),
            )
        return spec.reader(h5f[spec.group], columns)


# ---------------------------------------------------------------------------
# Window status: the Stage 4 plan, the Stage 6 created windows and the Stage 5
# fit's per-window line counts, joined on ``window_id``.
# ---------------------------------------------------------------------------


def _window_status_rows(h5f: h5py.File) -> List[WindowStatusRow]:
    """Build the window-status rows from an open file (raises before Stage 4).

    One row per effective window id: every Stage 4 plan window, with a
    Stage-6-created window of the same id replacing the plan row (the
    narrow-gap widening case, as in the effective plan) and flagged
    ``created``; created windows with fresh ids are appended. Rows ascend by
    frequency, then ``window_id``.
    """
    if "stage4_windows" not in h5f:
        raise StageDependencyError(
            "read window_status",
            ["stage4_windows"],
            Path(str(h5f.filename)),
            command="windows run",
            message=(
                f"No Stage 4 window plan found in {h5f.filename}; table "
                f"'window_status' is unavailable. Run {_WINDOWS_HINT} first."
            ),
        )
    plan = read_window_plan_columns(
        h5f["stage4_windows"], ["window_id", "freq_min", "freq_max"]
    )
    bounds: Dict[int, Tuple[float, float]] = {
        int(i): (float(lo), float(hi))
        for i, lo, hi in zip(plan["window_id"], plan["freq_min"], plan["freq_max"])
    }
    created_ids = set()
    review = h5f["stage6_review"] if "stage6_review" in h5f else None
    for wid, lo, hi in read_created_window_bounds(review):
        bounds[wid] = (lo, hi)
        created_ids.add(wid)

    fitted: Optional[Dict[int, int]] = None
    # A partial fit (a cancelled Stage 5) reports only the windows it kept;
    # every other window stays NOT_RUN.
    partial_only = False
    if "stage5_fitting" in h5f:
        fit = read_fit_window_columns(h5f["stage5_fitting"], ["window_id", "n_peaks"])
        fitted = {int(i): int(n) for i, n in zip(fit["window_id"], fit["n_peaks"])}
    else:
        fitted = read_stage5_partial_counts(h5f)
        partial_only = fitted is not None

    def row(wid: int) -> WindowStatusRow:
        # The one place a window's fit state can be NOT_RUN.
        n: Union[int, Absent] = Absent.NOT_RUN
        live: Union[bool, Absent] = Absent.NOT_RUN
        if fitted is not None and (not partial_only or wid in fitted):
            n = fitted.get(wid, 0)
            live = n > 0
        lo, hi = bounds[wid]
        return WindowStatusRow(
            window_id=wid,
            freq_min_mhz=lo,
            freq_max_mhz=hi,
            created=wid in created_ids,
            n_fitted_peaks=n,
            live=live,
        )

    return [row(w) for w in sorted(bounds, key=lambda w: (bounds[w][0], w))]


def _window_status_columns(
    h5f: h5py.File, columns: Optional[Sequence[str]] = None
) -> Dict[str, np.ndarray]:
    """The columnar ``window_status`` table, derived from the rows.

    Each absent-capable column gains its ``<column>__status`` companion.
    """
    rows = _window_status_rows(h5f)
    table = with_status_columns(
        {
            name: [getattr(r, name) for r in rows]
            for name in (
                "window_id",
                "freq_min_mhz",
                "freq_max_mhz",
                "created",
                "n_fitted_peaks",
                "live",
            )
        },
        absent_capable=_WINDOW_STATUS_ABSENT_CAPABLE,
        fills={"n_fitted_peaks": 0, "live": False},
        dtypes=_WINDOW_STATUS_DTYPES,
    )
    requested = resolve_column_selection(
        columns, list(WINDOW_STATUS_COLUMN_SPECS), table="window_status"
    )
    return {name: table[name] for name in requested}


def window_status_impl(file_path: Union[str, Path]) -> Dict[str, Any]:
    """The ``ftmw/window_status@1`` payload: ``{"schema", "windows": [row, ...]}``.

    One :class:`~ftmwpipeline.contract.WindowStatusRow` per window. A window is
    live when the Stage 5 fit holds at least one fitted line in it. Before
    Stage 5 each row's ``n_fitted_peaks`` and ``live`` are ``Absent.NOT_RUN``.
    Read-only. The columnar form is the ``window_status`` table of
    :func:`read_table_impl`.

    Raises
    ------
    StageDependencyError
        Stage 4 has not been run (``command`` is ``"windows run"``).
    """
    with _open(file_path) as h5f:
        rows = _window_status_rows(h5f)
    return {"schema": WINDOW_STATUS_SCHEMA, "windows": rows}


def _row_count(stage_group: h5py.Group, count_path: str) -> Optional[int]:
    """The row count a stage group records, or ``None`` when it records none.

    ``count_path`` is an attribute name, or ``"subgroup/attr"`` for the stages
    that keep their counts one level in, or empty when the stage records no
    count for this table. Missing at any step reads as unknown (``None``); the
    caller then computes the count from the table itself.
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


def _table_row_count(name: str, spec: _TableSpec, stage_group: h5py.Group) -> Any:
    """The row count of an available table: recorded, else computed.

    A table whose stage records no count attribute is read through its own
    reader (one column), so the count is always a real ``int``. A table that
    cannot be read at all (a malformed or pre-1.0 layout) has a count that is
    ``Absent.UNDEFINED``.
    """
    recorded = _row_count(stage_group, spec.count_path)
    if recorded is not None:
        return recorded
    try:
        column = next(iter(spec.specs))
        return int(len(next(iter(spec.reader(stage_group, [column]).values()))))
    except ValueError:
        return Absent.UNDEFINED


def read_tables_impl(file_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """List the readable tables and what each one holds in this file.

    Reads group attributes where the stage records a row count, and counts the
    rows of the few tables that do not (the event logs and the ragged window
    sets) from the table itself.

    Returns
    -------
    dict
        ``{table_name: {"available": bool, "n_rows": int or Absent,
        "columns": [...], "group": str}}`` for every table in
        :data:`READ_TABLES`. ``available`` is False (and ``n_rows``
        ``Absent.NOT_RUN``) when the producing stage has not been run. For an
        available table ``n_rows`` is always an ``int`` (``Absent.UNDEFINED``
        only if the stored table cannot be read). ``columns`` is the canonical
        column list either way, including every ``<column>__status`` companion;
        a column absent from an older file still reads back as its documented
        fill value, with status ``NOT_RUN``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    with _open(file_path) as h5f:
        for name, spec in _TABLE_SPECS.items():
            available = spec.group in h5f
            n_rows: Any = Absent.NOT_RUN
            if available:
                n_rows = _table_row_count(name, spec, h5f[spec.group])
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
    """h5py attribute -> a plain Python scalar (the unset sentinel -> ``None``)."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str) and value == NONE_SENTINEL:
        return None
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


def _read_ft_window(h5f: h5py.File, out: Dict[str, Any]) -> None:
    """Report the canonical FT window, including its derived active length."""
    from .stage1_impl import resolve_ft_settings_h5

    group = "processing_parameters/ft_processing"
    if group not in h5f:
        return
    # Older records stored the settings bundle as one JSON `parameters` blob
    # rather than as individual attributes, and Stage 1 reads both shapes. This
    # view has to read both the same way, or a blob-only file reports no
    # `ft.units_power` here while the display transform reads one from the blob
    # -- the same question answered two ways by two readers.
    #
    # Decoding goes through the record's own codec, not a generic attr copy: an
    # unset bound or trim is persisted as the ``__None__`` sentinel, and only
    # ``FTSettings`` knows to read it back as ``None``. These are the canonical
    # Stage 1 data-selection settings every later stage analyzes, so
    # ``ft.acquisition_us`` -- and with it the Fourier resolution element -- is
    # fixed here, not at Stage 5.
    #
    # The values are the Stage 1 resolver's, the one every stage reads: an
    # authoritative record as written (a concrete window), and an older record
    # with the recommended fall-through it was written under.
    settings = resolve_ft_settings_h5(h5f)
    trim_lo, trim_hi = settings.trim if settings.trim is not None else (None, None)
    out["ft.start_us"] = settings.start_us
    out["ft.end_us"] = settings.end_us
    out["ft.trim_min_mhz"] = trim_lo
    out["ft.trim_max_mhz"] = trim_hi
    out["ft.units_power"] = settings.units_power
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


def fid_samples_impl(file_path: Union[str, Path]) -> Dict[str, Any]:
    """The stored Stage 0 FID samples: one dataset read, no pipeline load.

    Returns ``{"schema": "ftmw/fid_samples@1", "samples": float64 1-D array,
    "stored_dtype": str}``. Values equal the stored ones, in stored order; a
    narrower stored dtype is promoted losslessly. Raises
    ``StageDependencyError`` (``command`` ``"data import"``) when Stage 0 has
    not been imported into the file.
    """
    path = Path(file_path)
    h5f = _open(path)
    try:
        group = h5f.get("stage0_fid_data")
        if group is None or "time_series_data" not in group:
            raise StageDependencyError(
                "fid_samples",
                ["stage0_fid_data"],
                path,
                command="data import",
            )
        dataset = group["time_series_data"]
        stored_dtype = str(dataset.dtype)
        raw = dataset[...]
    finally:
        h5f.close()
    samples = np.array(raw, dtype=np.float64, copy=True).reshape(-1)
    return {
        "schema": FID_SAMPLES_SCHEMA,
        "samples": samples,
        "stored_dtype": stored_dtype,
    }


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
        if value == NONE_SENTINEL:
            value = None
        out[f"start.{name}"] = value
    if record.get("chirp_detected") is False and "start.chirp_end_us" in out:
        # No chirp found: the stored 0.0 is not a measured end time.
        out["start.chirp_end_us"] = Absent.UNDEFINED


def _read_tau_scalars(
    h5f: h5py.File, group: str, prefix: str, out: Dict[str, Any]
) -> None:
    """Report one decay-time calibration's scalars under ``prefix.``."""
    if group not in h5f:
        return
    for key, value in read_tau_scalars(h5f[group]).items():
        out[f"{prefix}.{key}"] = value


def _nonfinite_to_absent(out: Dict[str, Any], prefix: str) -> None:
    """Turn every non-finite float under ``prefix.`` into ``Absent.UNDEFINED``.

    A calibration scalar that was computed but has no finite value (a decay
    time with zero contributors, a GMM fit on too few points) is ``UNDEFINED``.
    """
    for key in [k for k in out if k.startswith(f"{prefix}.")]:
        value = out[key]
        if isinstance(value, float) and not math.isfinite(value):
            out[key] = Absent.UNDEFINED


def _tau_section_absence(out: Dict[str, Any], prefix: str) -> None:
    """Absence for one decay-time calibration section already copied into *out*.

    The vote's ``recommended_shape`` is stored as the unset sentinel when the
    vote had no winner: the vote ran and chose nothing, so ``UNDEFINED``. A
    file that recorded no such attribute keeps the key omitted (stage-section
    omission rule).
    """
    _nonfinite_to_absent(out, prefix)
    key = f"{prefix}.recommended_shape"
    if key in out and out[key] is None:
        out[key] = Absent.UNDEFINED


def _timebase_absence(out: Dict[str, Any]) -> None:
    """Absence for the ``timebase.`` section already copied into *out*.

    With no usable tones (``n_used == 0``) the stored scale error is ``0.0`` and
    its uncertainty ``inf``: neither is a measurement, so both are
    ``UNDEFINED`` here (the stored record keeps its encoding). A
    ``lattice_g_mhz`` of ``0.0`` means no locked lattice was found.
    """
    n_used = out.get("timebase.n_used")
    if n_used is not None and int(n_used) == 0:
        for key in ("timebase.epsilon", "timebase.sigma_epsilon"):
            if key in out:
                out[key] = Absent.UNDEFINED
    lattice = out.get("timebase.lattice_g_mhz")
    if lattice is not None and not (math.isfinite(float(lattice)) and lattice > 0.0):
        out["timebase.lattice_g_mhz"] = Absent.UNDEFINED
    _nonfinite_to_absent(out, "timebase")


def _stage5_acquisition_absence(value: Any) -> Union[float, Absent]:
    """``stage5.acquisition_us`` as a contract value.

    Mirrors ``stage6_impl._recorded_acquisition_us``: missing is ``NOT_RUN``; a
    value that is not a positive finite number is ``UNDEFINED``.
    """
    if value is None:
        return Absent.NOT_RUN
    f = float_or_absent(value)
    if isinstance(f, Absent) or f <= 0.0:
        return Absent.UNDEFINED
    return f


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

    ``file.completed_stages`` is a list of canonical stage names
    (:class:`~ftmwpipeline.Stage` values) in rerun order; every other value is a
    scalar (``str`` / ``int`` / ``float`` / ``bool``), an
    :class:`~ftmwpipeline.contract.Absent`, or -- for the ``ft.`` settings
    echoes of an unset bound or trim -- ``None``.

    **Absence.** A key of a stage that has not run is omitted. A key that is
    present without a value is ``Absent``: ``NOT_RUN`` when the file predates
    the record (``file.format_version``, ``file.created_with``, ``source.*``,
    ``stage5.acquisition_us``, the stage 3 SNR cutoffs, and any count,
    creation time, plan revision or shape attribute missing from a
    ``stage3.`` / ``stage4.`` / ``stage5.`` group); ``UNDEFINED`` when it
    was computed and has no value (``start.chirp_end_us`` when no chirp was
    found, ``timebase.epsilon`` / ``sigma_epsilon`` with no usable tones,
    ``timebase.lattice_g_mhz`` with no locked lattice, a decay time with zero
    contributors, any other non-finite calibration scalar).
    """
    out: Dict[str, Any] = {}
    with _open(file_path) as h5f:
        out["file.path"] = str(Path(file_path))
        version = h5f.attrs.get("ftmw_format_version")
        out["file.format_version"] = (
            _decode(version) if version is not None else Absent.NOT_RUN
        )
        created = h5f.attrs.get("created_with_ftmwpipeline")
        out["file.created_with"] = (
            _decode(created) if created is not None else Absent.NOT_RUN
        )
        # Canonical stage names in rerun order, like every status payload; a
        # recorded key no stage of this version owns is left out.
        out["file.completed_stages"] = PipelineStageTracker(
            load_json_attr(h5f["pipeline_stages"], "completed_stages", [])
            if "pipeline_stages" in h5f
            else []
        ).canonical_completed_stages()

        _copy_attrs(h5f, "source_metadata", "source", _SOURCE_ATTRS, out)
        for key in [k for k in out if k.startswith("source.")]:
            if out[key] is None:  # the stored "unset" sentinel
                out[key] = Absent.NOT_RUN
        _copy_attrs(h5f, "stage0_fid_data/acquisition", "fid", _FID_ATTRS, out)
        _read_start_record(h5f, out)
        _read_ft_window(h5f, out)
        _copy_attrs(h5f, "timebase_calibration", "timebase", _TIMEBASE_ATTRS, out)
        _timebase_absence(out)

        _read_tau_scalars(h5f, TAU_GROUP_PATH, "tau", out)
        _tau_section_absence(out, "tau")
        _read_tau_scalars(h5f, GAUSSIAN_GROUP_PATH, "tau_g", out)
        _tau_section_absence(out, "tau_g")
        if "stage3_peaks" in h5f:
            for key, value in read_peak_scalars(h5f["stage3_peaks"]).items():
                # None: the group does not carry the attribute (a count, the
                # creation time, or a cutoff the file predates), so the
                # quantity was never recorded.
                out[f"stage3.{key}"] = Absent.NOT_RUN if value is None else value
        if "stage4_windows" in h5f:
            for key, value in read_window_plan_scalars(h5f["stage4_windows"]).items():
                # None: the group does not carry the attribute (never filled
                # with 0 or "unknown").
                out[f"stage4.{key}"] = Absent.NOT_RUN if value is None else value
        if "stage5_fitting" in h5f:
            for key, value in read_fit_scalars(h5f["stage5_fitting"]).items():
                if key == "acquisition_us":
                    out[f"stage5.{key}"] = _stage5_acquisition_absence(value)
                else:
                    out[f"stage5.{key}"] = Absent.NOT_RUN if value is None else value
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
    if value is None or isinstance(value, Absent):
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
    """A JSON-safe view of a cell value (non-finite floats, ``Absent`` -> ``None``)."""
    if isinstance(value, Absent):
        return None
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
        raise BadSettingError(
            "format",
            f"one of: {', '.join(VALID_READ_FORMATS)}",
            fmt,
            message=f"unknown format {fmt!r}; expected one of "
            f"{list(VALID_READ_FORMATS)}",
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
        raise BadSettingError(
            "format",
            f"one of: {', '.join(VALID_READ_FORMATS)}",
            fmt,
            message=f"unknown format {fmt!r}; expected one of "
            f"{list(VALID_READ_FORMATS)}",
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
