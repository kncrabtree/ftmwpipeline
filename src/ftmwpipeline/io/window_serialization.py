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
                window_id, freq_min, freq_max, difficulty ("easy"|"hard"),
                batch, split_proposal (NaN if none), needs_joint_treatment,
                diagnostics (JSON)
            free_peak_indices        [i8]  indices into the Stage 3 peak list
            fixed_peak_index         [i8]  ditto, for fixed contributors
            fixed_primary_window_id  [i8]
            fixed_frequency_mhz      [f8]
            fixed_freeze_eligible    [i1]  bool
        window_0001/ ...

Round-trip contract: ``load`` -> edit -> ``save`` -> ``load`` returns the
edited plan. A malformed group (missing required attr/dataset, mismatched
fixed-contributor column lengths, unknown difficulty label) raises
``ValueError`` loudly rather than silently dropping or guessing.
"""

from datetime import datetime
import json
from typing import Any, Dict, List

import h5py
import numpy as np

from ..core.data_structures import (
    FitWindow,
    FixedContributor,
    WindowDifficulty,
    WindowPlan,
)

_VALID_DIFFICULTY = {d.value for d in WindowDifficulty}
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
    for key in list(h5_group.keys()):
        del h5_group[key]

    h5_group.attrs["n_windows"] = plan.n_windows
    h5_group.attrs["creation_time"] = datetime.now().isoformat()
    h5_group.attrs["stage_name"] = "stage4_windows"
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
        wg.attrs["difficulty"] = w.difficulty.value
        wg.attrs["batch"] = int(w.batch)
        wg.attrs["split_proposal"] = (
            float("nan") if w.split_proposal is None else float(w.split_proposal)
        )
        wg.attrs["needs_joint_treatment"] = bool(w.needs_joint_treatment)
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


def _load_json_attr(h5_group: h5py.Group, name: str, default: Any) -> Any:
    """Parse a JSON-encoded attribute, falling back to ``default`` if absent."""
    raw = h5_group.attrs.get(name)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"stage4_windows attribute {name!r} is not valid JSON"
        ) from exc


def load_window_plan_from_hdf5(h5_group: h5py.Group) -> WindowPlan:
    """Load a :class:`WindowPlan` from an HDF5 group, validating loudly.

    Raises
    ------
    ValueError
        If the ``windows`` subgroup is missing, a window subgroup lacks a
        required attribute/dataset, the fixed-contributor columns have
        mismatched lengths, or a difficulty label is not ``easy``/``hard``.
    """
    if "windows" not in h5_group:
        raise ValueError("stage4_windows group missing required 'windows' subgroup")

    parameters = _load_json_attr(h5_group, "parameters", {})
    diagnostics = _load_json_attr(h5_group, "diagnostics", {})
    raw_edges = _load_json_attr(h5_group, "dependency_edges", [])
    dependency_edges: List[tuple] = [tuple(e) for e in raw_edges]
    topological_order: List[int] = [
        int(x) for x in _load_json_attr(h5_group, "topological_order", [])
    ]

    windows_group = h5_group["windows"]
    windows: List[FitWindow] = []
    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        for attr in ("window_id", "freq_min", "freq_max", "difficulty", "batch"):
            if attr not in wg.attrs:
                raise ValueError(f"window {name!r} missing required attribute {attr!r}")
        difficulty_label = wg.attrs["difficulty"]
        if isinstance(difficulty_label, bytes):
            difficulty_label = difficulty_label.decode("utf-8")
        if difficulty_label not in _VALID_DIFFICULTY:
            raise ValueError(
                f"window {name!r} has invalid difficulty {difficulty_label!r}; "
                f"expected one of {sorted(_VALID_DIFFICULTY)}"
            )

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
        fixed_contributors = [
            FixedContributor(
                peak_index=int(fixed_idx[i]),
                primary_window_id=int(fixed_pw[i]),
                frequency_mhz=float(fixed_f[i]),
                freeze_eligible=bool(fixed_fe[i]),
            )
            for i in range(len(fixed_idx))
        ]

        split_raw = wg.attrs.get("split_proposal", float("nan"))
        split_proposal = (
            None
            if split_raw is None or np.isnan(float(split_raw))
            else float(split_raw)
        )
        win_diag = _load_json_attr(wg, "diagnostics", {})

        windows.append(
            FitWindow(
                window_id=int(wg.attrs["window_id"]),
                freq_range=(float(wg.attrs["freq_min"]), float(wg.attrs["freq_max"])),
                free_peak_indices=[int(x) for x in wg["free_peak_indices"][:]],
                fixed_contributors=fixed_contributors,
                difficulty=WindowDifficulty(difficulty_label),
                batch=int(wg.attrs["batch"]),
                split_proposal=split_proposal,
                needs_joint_treatment=bool(
                    wg.attrs.get("needs_joint_treatment", False)
                ),
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
