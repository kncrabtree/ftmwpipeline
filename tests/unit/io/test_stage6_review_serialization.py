"""
The replay-engine stamps in ``/stage6_review``: a decision's ``serial``, the
review's ``next_serial`` high-water mark, ``engine_version`` and recorded
``review_params``, and the
``peak_uid`` check that tells a fit Stage 6 can address from one it cannot.

A review a pre-engine build wrote carries none of the stamps; it must read
back as exactly that (``Absent.NOT_RUN`` serial, ``engine_version`` None),
never as the running code's version -- the refuse-and-flag gate keys on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ftmwpipeline.core.absent import Absent
from ftmwpipeline.core.data_structures import (
    ENGINE_VERSION,
    DecisionLogEntry,
    ReviewParams,
    Stage6Review,
)
from ftmwpipeline.io.fitting_serialization import fit_has_peak_identity
from ftmwpipeline.io.stage6_review_serialization import (
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)


def _round_trip(tmp_path: Path, review: Stage6Review) -> Stage6Review:
    path = tmp_path / "review.h5"
    with h5py.File(path, "w") as h5f:
        save_stage6_review_to_hdf5(review, h5f.create_group("stage6_review"))
    with h5py.File(path, "r") as h5f:
        return load_stage6_review_from_hdf5(h5f["stage6_review"])


def test_serials_and_the_stamps_round_trip(tmp_path):
    review = Stage6Review(
        decision_log=[
            DecisionLogEntry(0, 3, 100.0, "accept", serial=0),
            # Positions and serials are independent: an undo left a gap.
            DecisionLogEntry(1, 4, 200.0, "add", serial=5),
        ],
        next_serial=6,
    )
    loaded = _round_trip(tmp_path, review)
    assert [(e.order_index, e.serial) for e in loaded.decision_log] == [(0, 0), (1, 5)]
    assert loaded.next_serial == 6
    assert loaded.engine_version == ENGINE_VERSION
    # Never stored: a read computes it.
    assert loaded.refit_required is None


def test_a_review_without_stamps_reads_as_pre_engine(tmp_path):
    path = tmp_path / "review.h5"
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(
            Stage6Review(
                decision_log=[DecisionLogEntry(0, 3, 100.0, "accept", serial=0)],
                next_serial=1,
            ),
            group,
        )
        # Make it what a pre-engine build wrote.
        del group.attrs["engine_version"]
        del group.attrs["next_serial"]
        log = group["decision_log"]
        rows = json.loads(str(log.attrs["data"]))
        for row in rows:
            del row["serial"]
        log.attrs["data"] = json.dumps(rows)

    with h5py.File(path, "r") as h5f:
        loaded = load_stage6_review_from_hdf5(h5f["stage6_review"])
    assert loaded.engine_version is None
    assert loaded.next_serial == 0
    assert [e.serial for e in loaded.decision_log] == [Absent.NOT_RUN]


def test_a_review_with_no_engine_version_is_saved_without_one(tmp_path):
    """Saving over a group that carried a version drops it for a ``None``."""
    path = tmp_path / "review.h5"
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(Stage6Review(), group)
        assert group.attrs["engine_version"] == ENGINE_VERSION
        save_stage6_review_to_hdf5(Stage6Review(engine_version=None), group)
        assert "engine_version" not in group.attrs


def test_the_recorded_review_parameters_round_trip(tmp_path):
    """Recorded parameters read back exactly; a review that records none
    reads ``None``, and saving ``None`` over a group drops the record."""
    params = ReviewParams(
        bar=3.5, attention_candidate_evidence=12.0, kappa=0.07, noise_floor=2.5
    )
    assert _round_trip(tmp_path, Stage6Review(review_params=params)).review_params == (
        params
    )
    path = tmp_path / "review.h5"
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("stage6_review")
        save_stage6_review_to_hdf5(Stage6Review(review_params=params), group)
        assert json.loads(str(group.attrs["review_params"])) == {
            "bar": 3.5,
            "attention_candidate_evidence": 12.0,
            "kappa": 0.07,
            "noise_floor": 2.5,
        }
        save_stage6_review_to_hdf5(Stage6Review(), group)
        assert "review_params" not in group.attrs
        assert load_stage6_review_from_hdf5(group).review_params is None


def _fit_group(h5f: h5py.File, uids) -> h5py.Group:
    group = h5f.create_group("stage5_fitting")
    peaks = group.create_group("peaks")
    if uids is not None:
        peaks.create_dataset("peak_uid", data=np.asarray(uids, dtype="i8"))
    return group


def test_fit_has_peak_identity(tmp_path):
    with h5py.File(tmp_path / "fits.h5", "w") as h5f:
        # Every peak carries a uid.
        assert fit_has_peak_identity(_fit_group(h5f, [5, 9, 12]))
        del h5f["stage5_fitting"]
        # One peak does not (stored as -1): the fit predates peak identity.
        assert not fit_has_peak_identity(_fit_group(h5f, [5, -1, 12]))
        del h5f["stage5_fitting"]
        # No column at all: a fit written before the column existed.
        assert not fit_has_peak_identity(_fit_group(h5f, None))
        del h5f["stage5_fitting"]
        # No peak table: nothing to address.
        h5f.create_group("stage5_fitting")
        assert not fit_has_peak_identity(h5f["stage5_fitting"])
