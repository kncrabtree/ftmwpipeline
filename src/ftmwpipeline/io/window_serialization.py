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
    build_columns,
    load_json_attr,
    read_attr_value,
    read_dataset_column,
    reset_group,
    resolve_column_selection,
    stack_columns,
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


def save_window_plan_to_hdf5(plan: WindowPlan, h5_group: h5py.Group) -> None:
    """Write a :class:`WindowPlan` to an HDF5 group.

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

    windows_group = h5_group.create_group("windows")
    for w in plan.windows:
        wg = windows_group.create_group(f"window_{w.window_id:04d}")
        wg.attrs["window_id"] = int(w.window_id)
        wg.attrs["freq_min"] = float(w.freq_range[0])
        wg.attrs["freq_max"] = float(w.freq_range[1])
        wg.attrs["batch"] = int(w.batch)
        wg.attrs["diagnostics"] = json.dumps(w.diagnostics, default=str)

        wg.create_dataset(
            "free_peak_indices",
            data=np.asarray(w.free_peak_indices, dtype="i8"),
        )
        fc = w.fixed_contributors
        wg.create_dataset(
            "fixed_peak_index",
            data=np.asarray([c.peak_index for c in fc], dtype="i8"),
        )
        wg.create_dataset(
            "fixed_primary_window_id",
            data=np.asarray([c.primary_window_id for c in fc], dtype="i8"),
        )
        wg.create_dataset(
            "fixed_frequency_mhz",
            data=np.asarray([c.frequency_mhz for c in fc], dtype="f8"),
        )
        wg.create_dataset(
            "fixed_freeze_eligible",
            data=np.asarray([c.freeze_eligible for c in fc], dtype="i1"),
        )
        # Edge-free flag. Written unconditionally; legacy files predating it
        # load with the back-compat default False (see load_window_plan).
        wg.create_dataset(
            "fixed_edge_free",
            data=np.asarray([c.edge_free for c in fc], dtype="i1"),
        )


def load_window_plan_from_hdf5(h5_group: h5py.Group) -> WindowPlan:
    """Load a :class:`WindowPlan` from an HDF5 group, validating loudly.

    Raises
    ------
    ValueError
        If the ``windows`` subgroup is missing, a window subgroup lacks a
        required attribute/dataset, or the fixed-contributor columns have
        mismatched lengths.
    """
    if "windows" not in h5_group:
        raise ValueError("stage4_windows group missing required 'windows' subgroup")

    parameters = load_json_attr(h5_group, "parameters", {})
    diagnostics = load_json_attr(h5_group, "diagnostics", {})
    raw_edges = load_json_attr(h5_group, "dependency_edges", [])
    dependency_edges: List[tuple] = [tuple(e) for e in raw_edges]
    topological_order: List[int] = [
        int(x) for x in load_json_attr(h5_group, "topological_order", [])
    ]

    windows_group = h5_group["windows"]
    windows: List[FitWindow] = []
    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        for attr in ("window_id", "freq_min", "freq_max", "batch"):
            if attr not in wg.attrs:
                raise ValueError(f"window {name!r} missing required attribute {attr!r}")

        for col in ("free_peak_indices", *_FIXED_COLUMNS):
            if col not in wg:
                raise ValueError(f"window {name!r} missing required dataset {col!r}")
        fixed_lengths = {c: len(wg[c]) for c in _FIXED_COLUMNS}
        if len(set(fixed_lengths.values())) != 1:
            raise ValueError(
                f"window {name!r} fixed-contributor columns have mismatched "
                f"lengths: {fixed_lengths}"
            )

        fixed_idx = wg["fixed_peak_index"][:]
        fixed_pw = wg["fixed_primary_window_id"][:]
        fixed_f = wg["fixed_frequency_mhz"][:]
        fixed_fe = wg["fixed_freeze_eligible"][:]
        # Optional column: legacy plans predate the edge-free attachment and
        # carry no such dataset, so default it to all-False.
        if "fixed_edge_free" in wg:
            fixed_ef = wg["fixed_edge_free"][:]
            if len(fixed_ef) != len(fixed_idx):
                raise ValueError(
                    f"window {name!r} fixed_edge_free length {len(fixed_ef)} "
                    f"!= fixed_peak_index length {len(fixed_idx)}"
                )
        else:
            fixed_ef = np.zeros(len(fixed_idx), dtype="i1")
        fixed_contributors = [
            FixedContributor(
                peak_index=int(fixed_idx[i]),
                primary_window_id=int(fixed_pw[i]),
                frequency_mhz=float(fixed_f[i]),
                freeze_eligible=bool(fixed_fe[i]),
                edge_free=bool(fixed_ef[i]),
            )
            for i in range(len(fixed_idx))
        ]

        win_diag = load_json_attr(wg, "diagnostics", {})

        windows.append(
            FitWindow(
                window_id=int(wg.attrs["window_id"]),
                freq_range=(float(wg.attrs["freq_min"]), float(wg.attrs["freq_max"])),
                free_peak_indices=[int(x) for x in wg["free_peak_indices"][:]],
                fixed_contributors=fixed_contributors,
                batch=int(wg.attrs["batch"]),
                diagnostics=win_diag,
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

#: Dataset whose length backs each derived count column.
_COUNT_SOURCE = {
    "n_free_peaks": "free_peak_indices",
    "n_fixed_contributors": "fixed_peak_index",
}


def read_window_plan_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read per-window plan columns from a ``stage4_windows`` group in bulk.

    One attribute read per window per requested column; the free-peak indices,
    the fixed-contributor columns, and the per-window JSON diagnostics are never
    read. Rows are ordered by ascending ``window_id``, matching
    :attr:`WindowPlan.windows`.

    See :data:`WINDOW_PLAN_COLUMN_SPECS` for the available columns.
    """
    requested = resolve_column_selection(
        columns, list(WINDOW_PLAN_COLUMN_SPECS), table="windows"
    )
    if "windows" not in h5_group:
        raise ValueError("stage4_windows group missing required 'windows' subgroup")
    windows_group = h5_group["windows"]

    # window_id always read: it defines the row order.
    attr_cols = [c for c in requested if c not in _WINDOW_PLAN_DERIVED]
    to_read = list(dict.fromkeys(["window_id", *attr_cols]))
    derived = [c for c in _WINDOW_PLAN_DERIVED if c in requested]
    rows: Dict[str, List[Any]] = {c: [] for c in (*to_read, *derived)}

    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        where = f"window {name!r}"
        for col in to_read:
            rows[col].append(
                read_attr_value(wg, col, WINDOW_PLAN_COLUMN_SPECS[col], where=where)
            )
        for col in derived:
            name_of = _COUNT_SOURCE[col]
            try:
                dataset = wg[name_of]
            except KeyError:
                raise ValueError(
                    f"{where} missing required dataset {name_of!r}"
                ) from None
            rows[col].append(int(dataset.shape[0]))

    keep = [*to_read, *derived]
    built = build_columns(rows, WINDOW_PLAN_COLUMN_SPECS, keep)
    order = np.argsort(built["window_id"], kind="stable")
    return {c: built[c][order] for c in requested}


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

#: On-disk dataset backing each contributor column (the writer prefixes them).
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
    anchor: str,
    dataset_names: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """Concatenate a per-window ragged column set into one long table.

    *anchor* is the dataset whose length gives each window's row count; it is
    read whether or not the caller asked for it, since without it there is no
    way to know how many rows a window contributes. Windows are visited in
    ascending ``window_id``, so the result is grouped by window.
    """
    requested = resolve_column_selection(columns, list(specs), table=table)
    if "windows" not in h5_group:
        raise ValueError("stage4_windows group missing required 'windows' subgroup")
    windows_group = h5_group["windows"]

    data_cols = [c for c in requested if c != "window_id"]
    to_read = list(dict.fromkeys([anchor, *data_cols]))
    chunks: Dict[str, List[np.ndarray]] = {c: [] for c in (*to_read, "window_id")}

    want_window_id = "window_id" in requested
    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        where = f"window {name!r}"
        n: Optional[int] = None
        for col in to_read:
            column = read_dataset_column(
                wg,
                col,
                specs[col],
                n,
                where=where,
                dataset=dataset_names.get(col, col),
            )
            if n is None:
                n = len(column)
            chunks[col].append(column)
        if want_window_id:
            assert n is not None  # the anchor column is always read first
            window_id = read_attr_value(
                wg, "window_id", WINDOW_PLAN_COLUMN_SPECS["window_id"], where=where
            )
            chunks["window_id"].append(np.full(n, window_id, dtype="i8"))

    keep = [*to_read, *(["window_id"] if want_window_id else [])]
    built = stack_columns(chunks, specs, keep)
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
        anchor="peak_index",
        dataset_names={"peak_index": "free_peak_indices"},
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
        anchor="peak_index",
        dataset_names=_CONTRIBUTOR_DATASETS,
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
