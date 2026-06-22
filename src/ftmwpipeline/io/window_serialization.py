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
from typing import List

import h5py
import numpy as np

from ..core.data_structures import (
    FitWindow,
    FixedContributor,
    WindowPlan,
)
from ._hdf5_helpers import load_json_attr, reset_group, stamp_stage_header

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
