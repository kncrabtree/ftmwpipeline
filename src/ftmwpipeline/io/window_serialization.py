"""
Stage 4 window-plan serialization to HDF5.

The window plan is curation/coordination substrate -- like the Stage 3 peak
list, it may be **hand-edited** between Stage 4 and Stage 5 (a curator can
merge/split windows, re-class difficulty, edit the free-peak set). So it is
persisted (not recomputed on demand) in a flat, obvious layout with loud
validation, consistent with the SERIALIZATION spec's treatment of peaks.

HDF5 layout (under the caller-provided group, e.g. ``/stage4_windows``)::

    .attrs:
        n_windows, creation_time, stage_name,
        parameters          (JSON)  -- Stage 4 parameters used
        dependency_edges    (JSON)  -- list of [window_id, depends_on] pairs
        topological_order   (JSON)  -- list of window_id in fit order
        diagnostics         (JSON)  -- plan-level diagnostics
    windows/
        window_0000/
            .attrs:
                window_id, freq_min, freq_max, batch, diagnostics (JSON)
            free_peak_indices        [i8]  indices into the Stage 3 peak list
            fixed_peak_index         [i8]  ditto, for fixed contributors
            fixed_primary_window_id  [i8]
            fixed_frequency_mhz      [f8]
            fixed_freeze_eligible    [i1]  bool
            fixed_edge_free          [i1]  bool (absent in legacy files -> False)
        window_0001/ ...

Round-trip contract: ``load`` -> edit -> ``save`` -> ``load`` returns the
edited plan. A malformed group (missing required attr/dataset, mismatched
fixed-contributor column lengths) raises ``ValueError`` loudly rather than
silently dropping or guessing.
"""

import json
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np

from ..core.data_structures import (
    FitWindow,
    FixedContributor,
    WindowPlan,
)
from ._hdf5_helpers import (
    REQUIRED,
    ColumnSpec,
    load_json_attr,
    read_dataset_column,
    reset_group,
    resolve_column_selection,
    stamp_stage_header,
)

__all__ = [
    "save_window_plan_to_hdf5",
    "load_window_plan_from_hdf5",
    "WINDOW_PLAN_COLUMN_SPECS",
    "WINDOW_FREE_PEAK_COLUMN_SPECS",
    "WINDOW_CONTRIBUTOR_COLUMN_SPECS",
    "read_window_plan_columns",
    "read_window_free_peak_columns",
    "read_window_contributor_columns",
    "read_window_plan_scalars",
]

_FIXED_COLUMNS = (
    "fixed_peak_index",
    "fixed_primary_window_id",
    "fixed_frequency_mhz",
    "fixed_freeze_eligible",
)

#: Window-table columns and their on-disk dtypes. ``"str"`` is a
#: variable-length UTF-8 column. ``free_offset``/``free_count`` and
#: ``fixed_offset``/``fixed_count`` address each window's rows in the two
#: long tables.
_PLAN_WINDOW_COLUMNS: Dict[str, str] = {
    "window_id": "i8",
    "freq_min": "f8",
    "freq_max": "f8",
    "batch": "i8",
    "free_offset": "i8",
    "free_count": "i8",
    "fixed_offset": "i8",
    "fixed_count": "i8",
    "diagnostics": "str",
}

#: The fixed-contributor long table, one row per (window, contributor).
_PLAN_CONTRIBUTOR_COLUMNS: Dict[str, str] = {
    "peak_index": "i8",
    "primary_window_id": "i8",
    "frequency_mhz": "f8",
    "freeze_eligible": "i1",
    "edge_free": "i1",
}


def _plan_column(values: List[Any], dtype: str) -> np.ndarray:
    """One plan-table column, vlen-UTF-8 for the string columns."""
    if dtype == "str":
        column = np.empty(len(values), dtype=object)
        for i, value in enumerate(values):
            column[i] = value
        return column
    return np.asarray(values, dtype=dtype)


def _write_plan_table(group: h5py.Group, columns: Dict[str, np.ndarray]) -> None:
    """Write *columns* as equal-length datasets, resizing in place if present.

    Chunked with an unbounded ``maxshape`` so a rewrite resizes rather than
    deletes: HDF5 never reclaims a deleted object's space.
    """
    for name, data in columns.items():
        if name in group:
            dataset = group[name]
            if dataset.shape[0] != len(data):
                dataset.resize((len(data),))
            if len(data):
                dataset[...] = data
            continue
        dtype = (
            h5py.string_dtype(encoding="utf-8")
            if getattr(data, "dtype", None) == object
            else None
        )
        group.create_dataset(
            name,
            data=data,
            dtype=dtype,
            maxshape=(None,),
            chunks=(max(len(data), 1),),
        )


def save_window_plan_to_hdf5(plan: WindowPlan, h5_group: h5py.Group) -> None:
    """Write a :class:`WindowPlan` to an HDF5 group.

    One row per window in ``windows/``, one row per (window, free peak) in
    ``free_peaks/``, and one row per (window, fixed contributor) in
    ``contributors/``. Windows are stored ascending by ``window_id``, and
    each window's rows in the two long tables are contiguous and in its own
    list order, addressed by ``free_offset``/``free_count`` and
    ``fixed_offset``/``fixed_count``.

    Parameters
    ----------
    plan : WindowPlan
        The plan to persist.
    h5_group : h5py.Group
        Destination group; any existing window-plan content is overwritten.
    """
    reset_group(h5_group)

    stamp_stage_header(h5_group, "stage4_windows", n_windows=plan.n_windows)
    h5_group.attrs["parameters"] = json.dumps(plan.parameters, default=str)
    h5_group.attrs["dependency_edges"] = json.dumps(
        [list(e) for e in plan.dependency_edges]
    )
    h5_group.attrs["topological_order"] = json.dumps(list(plan.topological_order))
    h5_group.attrs["diagnostics"] = json.dumps(plan.diagnostics, default=str)

    windows = sorted(plan.windows, key=lambda w: int(w.window_id))
    rows: Dict[str, List[Any]] = {name: [] for name in _PLAN_WINDOW_COLUMNS}
    free_indices: List[int] = []
    contributors: List[FixedContributor] = []
    free_offset = 0
    fixed_offset = 0

    for w in windows:
        fc = list(w.fixed_contributors)
        rows["window_id"].append(int(w.window_id))
        rows["freq_min"].append(float(w.freq_range[0]))
        rows["freq_max"].append(float(w.freq_range[1]))
        rows["batch"].append(int(w.batch))
        rows["free_offset"].append(free_offset)
        rows["free_count"].append(len(w.free_peak_indices))
        rows["fixed_offset"].append(fixed_offset)
        rows["fixed_count"].append(len(fc))
        rows["diagnostics"].append(json.dumps(w.diagnostics, default=str))
        free_indices.extend(int(x) for x in w.free_peak_indices)
        contributors.extend(fc)
        free_offset += len(w.free_peak_indices)
        fixed_offset += len(fc)

    _write_plan_table(
        h5_group.require_group("windows"),
        {
            name: _plan_column(rows[name], dtype)
            for name, dtype in _PLAN_WINDOW_COLUMNS.items()
        },
    )
    _write_plan_table(
        h5_group.require_group("free_peaks"),
        {"peak_index": np.asarray(free_indices, dtype="i8")},
    )
    _write_plan_table(
        h5_group.require_group("contributors"),
        {
            "peak_index": np.asarray([c.peak_index for c in contributors], dtype="i8"),
            "primary_window_id": np.asarray(
                [c.primary_window_id for c in contributors], dtype="i8"
            ),
            "frequency_mhz": np.asarray(
                [c.frequency_mhz for c in contributors], dtype="f8"
            ),
            "freeze_eligible": np.asarray(
                [1 if c.freeze_eligible else 0 for c in contributors], dtype="i1"
            ),
            "edge_free": np.asarray(
                [1 if c.edge_free else 0 for c in contributors], dtype="i1"
            ),
        },
    )


#: What a reader is told when it opens a plan written in the pre-1.0 layout.
LEGACY_PLAN_LAYOUT_MESSAGE = (
    "this file's Stage 4 window plan uses the pre-1.0 per-window layout "
    "(stage4_windows/windows/window_NNNN/), which this version cannot read. "
    "Re-run 'windows run' to rebuild the plan in the current layout; "
    "anything downstream of it (the Stage 5 fit, Stage 6 curation) must be "
    "re-run too."
)


def _plan_windows_group(h5_group: h5py.Group) -> h5py.Group:
    if "windows" not in h5_group:
        raise ValueError("stage4_windows group missing required 'windows' table")
    windows = h5_group["windows"]
    if "window_id" not in windows:
        # The flat layout always has that column. Its absence is either a
        # corrupt table or a plan from before the layout changed, where
        # `windows` held one subgroup per window. Both end in a refusal;
        # only one of them tells the reader what to do about it.
        if any(isinstance(windows.get(name), h5py.Group) for name in windows):
            raise ValueError(LEGACY_PLAN_LAYOUT_MESSAGE)
        raise ValueError("stage4_windows windows table missing column 'window_id'")
    return windows


def _plan_decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_window_plan_from_hdf5(h5_group: h5py.Group) -> WindowPlan:
    """Load a :class:`WindowPlan` from an HDF5 group, validating loudly.

    Reads the three tables whole -- a handful of dataset reads whatever the
    window count -- and slices each window's free-peak indices and fixed
    contributors out of the two long tables by offset and count.

    Raises
    ------
    ValueError
        If a table or a required column is missing, the columns within a
        table have mismatched lengths, or a window's slice of either long
        table does not lie inside it.
    """
    windows_group = _plan_windows_group(h5_group)

    parameters = load_json_attr(h5_group, "parameters", {})
    diagnostics = load_json_attr(h5_group, "diagnostics", {})
    raw_edges = load_json_attr(h5_group, "dependency_edges", [])
    dependency_edges: List[tuple] = [tuple(e) for e in raw_edges]
    topological_order: List[int] = [
        int(x) for x in load_json_attr(h5_group, "topological_order", [])
    ]

    missing = [c for c in _PLAN_WINDOW_COLUMNS if c not in windows_group]
    if missing:
        raise ValueError(f"stage4_windows windows table missing column(s): {missing}")
    columns = {c: windows_group[c][:] for c in _PLAN_WINDOW_COLUMNS}
    lengths = {c: len(v) for c, v in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"stage4_windows windows table columns have mismatched lengths: {lengths}"
        )

    if "free_peaks" not in h5_group:
        raise ValueError("stage4_windows group missing required 'free_peaks' table")
    if "peak_index" not in h5_group["free_peaks"]:
        raise ValueError("stage4_windows free_peaks table missing column 'peak_index'")
    free_indices = np.asarray(h5_group["free_peaks"]["peak_index"][:], dtype="i8")

    if "contributors" not in h5_group:
        raise ValueError("stage4_windows group missing required 'contributors' table")
    contributors_group = h5_group["contributors"]
    # `edge_free` is a late addition. The layout is fixed, but a column added
    # after a plan was written may still be absent from it, and False -- "not
    # known to be edge-free" -- is the honest default rather than a refusal.
    # This matches the column spec the bulk reader resolves it through.
    required = [c for c in _PLAN_CONTRIBUTOR_COLUMNS if c != "edge_free"]
    missing = [c for c in required if c not in contributors_group]
    if missing:
        raise ValueError(
            f"stage4_windows contributors table missing column(s): {missing}"
        )
    contributor_columns = {c: contributors_group[c][:] for c in required}
    n_contributor_rows = len(contributor_columns["peak_index"])
    contributor_columns["edge_free"] = (
        contributors_group["edge_free"][:]
        if "edge_free" in contributors_group
        else np.zeros(n_contributor_rows, dtype="i1")
    )
    contributor_lengths = {c: len(v) for c, v in contributor_columns.items()}
    if len(set(contributor_lengths.values())) != 1:
        raise ValueError(
            f"stage4_windows contributors table columns have mismatched lengths: "
            f"{contributor_lengths}"
        )
    n_contributors = next(iter(contributor_lengths.values()))

    windows: List[FitWindow] = []
    for row in range(len(columns["window_id"])):
        wid = int(columns["window_id"][row])
        free_start = int(columns["free_offset"][row])
        free_stop = free_start + int(columns["free_count"][row])
        if free_start < 0 or free_stop > free_indices.size:
            raise ValueError(
                f"window {wid} free-peak slice [{free_start}, {free_stop}) does "
                f"not lie inside the {free_indices.size}-row free_peaks table"
            )
        fixed_start = int(columns["fixed_offset"][row])
        fixed_stop = fixed_start + int(columns["fixed_count"][row])
        if fixed_start < 0 or fixed_stop > n_contributors:
            raise ValueError(
                f"window {wid} contributor slice [{fixed_start}, {fixed_stop}) does "
                f"not lie inside the {n_contributors}-row contributors table"
            )

        raw_diagnostics = _plan_decode(columns["diagnostics"][row])
        try:
            window_diagnostics = json.loads(raw_diagnostics) if raw_diagnostics else {}
        except (json.JSONDecodeError, TypeError):
            window_diagnostics = {}

        windows.append(
            FitWindow(
                window_id=wid,
                freq_range=(
                    float(columns["freq_min"][row]),
                    float(columns["freq_max"][row]),
                ),
                free_peak_indices=[int(x) for x in free_indices[free_start:free_stop]],
                fixed_contributors=[
                    FixedContributor(
                        peak_index=int(contributor_columns["peak_index"][i]),
                        primary_window_id=int(
                            contributor_columns["primary_window_id"][i]
                        ),
                        frequency_mhz=float(contributor_columns["frequency_mhz"][i]),
                        freeze_eligible=bool(
                            int(contributor_columns["freeze_eligible"][i])
                        ),
                        edge_free=bool(int(contributor_columns["edge_free"][i])),
                    )
                    for i in range(fixed_start, fixed_stop)
                ],
                batch=int(columns["batch"][row]),
                diagnostics=window_diagnostics,
            )
        )

    windows.sort(key=lambda w: w.window_id)
    return WindowPlan(
        windows=windows,
        dependency_edges=dependency_edges,
        topological_order=topological_order,
        parameters=parameters,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Read-only bulk column access (no WindowPlan reconstruction)
# ---------------------------------------------------------------------------
#
# The full loader materializes every window's free-peak index list, its
# fixed-contributor objects, and its JSON diagnostics blob. A bounds-only
# consumer wants two floats per window; these readers give it that without
# touching the rest. See the note in ``fitting_serialization`` for the shared
# rationale and the column-spec convention.

#: Per-planned-window read columns, in canonical order. All are window-group
#: attributes except the two counts, which are dataset lengths (a shape read,
#: not a data read).
WINDOW_PLAN_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "window_id": ("i8", REQUIRED),
    "freq_min": ("f8", REQUIRED),
    "freq_max": ("f8", REQUIRED),
    "batch": ("i8", REQUIRED),
    "n_free_peaks": ("i8", REQUIRED),
    "n_fixed_contributors": ("i8", REQUIRED),
}

_WINDOW_PLAN_DERIVED = ("n_free_peaks", "n_fixed_contributors")

#: Window-table column holding each derived count. Under the flat layout a
#: window's row count in a long table IS a stored column, so these are reads
#: rather than dataset-length probes.
_DERIVED_COUNT_COLUMN = {
    "n_free_peaks": "free_count",
    "n_fixed_contributors": "fixed_count",
}


def read_window_plan_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read per-window plan columns from a ``stage4_windows`` group in bulk.

    One whole-dataset read per requested column against the flat window
    table; the free-peak indices, the fixed-contributor columns and the
    per-window JSON diagnostics are never read. Rows are ordered by ascending
    ``window_id``, matching :attr:`WindowPlan.windows`.

    See :data:`WINDOW_PLAN_COLUMN_SPECS` for the available columns.
    """
    requested = resolve_column_selection(
        columns, list(WINDOW_PLAN_COLUMN_SPECS), table="windows"
    )
    windows_group = _plan_windows_group(h5_group)
    n_rows = int(np.asarray(windows_group["window_id"]).shape[0])

    # window_id always read: it defines the row order.
    stored = [c for c in requested if c not in _WINDOW_PLAN_DERIVED]
    to_read = list(dict.fromkeys(["window_id", *stored]))

    built: Dict[str, np.ndarray] = {}
    for col in to_read:
        built[col] = read_dataset_column(
            windows_group,
            col,
            WINDOW_PLAN_COLUMN_SPECS[col],
            n_rows,
            where="stage4_windows windows",
        )
    for col in _WINDOW_PLAN_DERIVED:
        if col not in requested:
            continue
        # A window's count is the length of its slice of the long table.
        source = _DERIVED_COUNT_COLUMN[col]
        if source not in windows_group:
            raise ValueError(f"stage4_windows windows table missing column {source!r}")
        built[col] = np.asarray(windows_group[source][:], dtype="i8")

    order = np.argsort(built["window_id"], kind="stable")
    return {c: np.asarray(built[c])[order] for c in requested}


#: One row per (window, free peak): which Stage 3 peaks a window fits freely.
#: A window's free-peak set is a ragged per-window list on disk, so it reaches
#: the read surface in long form -- ``window_id`` alongside the index -- which
#: is also the shape a join against the ``peaks`` table wants.
WINDOW_FREE_PEAK_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "window_id": ("i8", REQUIRED),
    "peak_index": ("i8", REQUIRED),
}

#: One row per (window, fixed contributor): a peak whose parameters a window
#: holds at values another window fitted, so its wings are modeled rather than
#: absorbed. Long form, for the same reason as the free-peak table.
WINDOW_CONTRIBUTOR_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "window_id": ("i8", REQUIRED),
    "peak_index": ("i8", REQUIRED),
    "primary_window_id": ("i8", REQUIRED),
    "frequency_mhz": ("f8", REQUIRED),
    "freeze_eligible": ("bool", REQUIRED),
    "edge_free": ("bool", False),
}

#: Retained for reference: the pre-1.0 per-window datasets each contributor
#: column used to live in. The flat ``contributors`` table stores them under
#: their column names directly.
_CONTRIBUTOR_DATASETS = {
    "peak_index": "fixed_peak_index",
    "primary_window_id": "fixed_primary_window_id",
    "frequency_mhz": "fixed_frequency_mhz",
    "freeze_eligible": "fixed_freeze_eligible",
    "edge_free": "fixed_edge_free",
}


def _read_long_window_table(
    h5_group: h5py.Group,
    specs: Dict[str, ColumnSpec],
    columns: Optional[Sequence[str]],
    *,
    table: str,
    subgroup: str,
    count_column: str,
) -> Dict[str, np.ndarray]:
    """Read one of the two long tables, tagged by owning ``window_id``.

    The table is already in long form on disk and already grouped by window
    in ascending ``window_id`` order, so this is a straight column read plus
    -- when the caller wants it -- a ``window_id`` column expanded from the
    window table's per-window counts.
    """
    requested = resolve_column_selection(columns, list(specs), table=table)
    if subgroup not in h5_group:
        raise ValueError(f"stage4_windows group missing required {subgroup!r} table")
    long_group = h5_group[subgroup]
    data_cols = [c for c in requested if c != "window_id"]

    n_rows: Optional[int] = None
    built: Dict[str, np.ndarray] = {}
    for col in data_cols:
        column = read_dataset_column(
            long_group, col, specs[col], n_rows, where=f"stage4_windows {subgroup}"
        )
        if n_rows is None:
            n_rows = len(column)
        built[col] = column

    if "window_id" in requested:
        windows_group = _plan_windows_group(h5_group)
        ids = np.asarray(windows_group["window_id"][:], dtype="i8")
        counts = np.asarray(windows_group[count_column][:], dtype="i8")
        order = np.argsort(ids, kind="stable")
        built["window_id"] = np.repeat(ids[order], counts[order])
        if n_rows is not None and built["window_id"].size != n_rows:
            raise ValueError(
                f"stage4_windows {subgroup} has {n_rows} row(s) but the window "
                f"table's {count_column} sums to {built['window_id'].size}"
            )

    return {c: built[c] for c in requested}


def read_window_free_peak_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the per-window free-peak assignments in long form.

    See :data:`WINDOW_FREE_PEAK_COLUMN_SPECS` for the available columns.
    """
    return _read_long_window_table(
        h5_group,
        WINDOW_FREE_PEAK_COLUMN_SPECS,
        columns,
        table="window_free_peaks",
        subgroup="free_peaks",
        count_column="free_count",
    )


def read_window_contributor_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the per-window fixed contributors in long form.

    See :data:`WINDOW_CONTRIBUTOR_COLUMN_SPECS` for the available columns.
    """
    return _read_long_window_table(
        h5_group,
        WINDOW_CONTRIBUTOR_COLUMN_SPECS,
        columns,
        table="window_contributors",
        subgroup="contributors",
        count_column="fixed_count",
    )


def read_window_plan_scalars(h5_group: h5py.Group) -> Dict[str, Any]:
    """Read the cheap plan-level scalars from a ``stage4_windows`` group.

    Group attributes only -- no window traversal. The batch count is not
    reported here because it is a property of the per-window ``batch`` column;
    read the ``windows`` table for it.
    """
    creation = h5_group.attrs.get("creation_time", "unknown")
    if isinstance(creation, bytes):
        creation = creation.decode("utf-8")
    edges = load_json_attr(h5_group, "dependency_edges", [], label="stage4_windows")
    return {
        "n_windows": int(h5_group.attrs.get("n_windows", 0)),
        "n_dependency_edges": len(edges),
        "creation_time": str(creation),
    }
