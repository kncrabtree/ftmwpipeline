"""
The incremental replay engine against the full-replay reference.

A refitting Stage 6 write refits only the windows whose keys changed and keeps
every other window's persisted fit (``_engine_plan`` / ``_engine_run``). These
are the design's properties (``scratch/replay-engine-design.md`` §10.2) over
random sequences of writes:

- **P1.** After every write, the persisted state is, bit for bit, the reference
  replay (:func:`~ftmwpipeline._internal.replay_reference.replay_full`) of the
  log it leaves under the review parameters it records.
- **P2.** A permutation of a log that keeps its create order, each window's row
  order and its actions' grouping replays to the same state, positions aside.
- **P3.** A refused write leaves the file byte for byte as it was, and is
  refused before any fit.

Each step draws one write: adds, removes, inferred merges and splits (on the hub
and its dependents more often than elsewhere), bare accepts, explicit and
implied creates (two implied adds in one gap coalesce), undos of random
decision sets (orphaning sets refuse), an apply at a random log prefix,
``review run`` with and without new parameters, a bare edit (which must
refuse) and a session's preview followed by the identical apply. The engine's
own cases follow: what a write refits (an upstream edit and its undo, the
windows kept, restored by copy or recomputed), the keys left unchanged by a
write that refits nothing, a file with fits but no keys, the refusals on a
keyed file, the per-window rebuild of the final products (every status is
recomputed), a first write persisted from a staged preview, and the environment
(a changed soft input rebuilds every window, an unacknowledged epoch leaves the
writes that refit nothing available, and a file that cannot account for its
inputs still lets a write keep its fits).

The fast tier runs on the cascade fixture (655 cut to its densest hub, a
two-level graph); the slow tier on the full 655 build, with the scenarios the
random walk rarely reaches pinned on their own.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.replay_reference import (
    _state_parts,
    differing_parts,
    persisted_state_digest,
    replay_full,
    state_digest,
)
from ftmwpipeline.core.data_structures import DecisionLogEntry, FittingResult
from ftmwpipeline.file_manager import (
    AnalysisEpochMismatchError,
    BadSettingError,
    CurationConflictError,
    IncompleteProvenanceError,
    PipelineFileError,
)
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_engine_serialization import (
    STAGE6_ENGINE_GROUP,
    load_stage6_engine_state,
    save_stage6_engine_state,
)
from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.integration]

#: What a Stage 6 write refuses with: the typed errors (each a
#: ``PipelineFileError`` or a ``ValueError``). Anything else fails the test.
_REFUSALS = (ValueError, PipelineFileError)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fits(path: Path) -> Dict[int, FittingResult]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {int(wf.window_id): wf for wf in sf.window_fits if wf.window_id is not None}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clear(wf: FittingResult) -> float:
    """The in-window position furthest from every fitted peak."""
    assert wf.window is not None
    lo, hi = sorted(float(v) for v in wf.window.freq_range)
    ps = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    return max(
        (lo + (hi - lo) * t / 40 for t in range(4, 37)),
        key=lambda x: min((abs(x - q) for q in ps), default=1e9),
    )


class _Refits:
    """Counts ``refit_window_core`` calls, by window, while :attr:`on`."""

    def __init__(self) -> None:
        self.on = False
        self.calls: List[int] = []

    def start(self) -> None:
        self.calls = []
        self.on = True

    def stop(self) -> List[int]:
        self.on = False
        return list(self.calls)


@pytest.fixture
def refits(monkeypatch: pytest.MonkeyPatch) -> _Refits:
    counter = _Refits()
    core = s6.refit_window_core

    def counting(fit_ctx: Any, fit_win: Any, wf: Any, **kwargs: Any) -> Any:
        if counter.on:
            counter.calls.append(int(fit_win.window_id))
        return core(fit_ctx, fit_win, wf, **kwargs)

    monkeypatch.setattr(s6, "refit_window_core", counting)
    return counter


def _assert_reference(path: Path, refits: Optional[_Refits] = None) -> List[int]:
    """P1 for the state *path* holds; returns the refits the reference ran."""
    review = load_stage6_review_from_file(str(path))
    if refits is not None:
        refits.start()
    reference = replay_full(path, review.decision_log, review.review_params)
    ran = refits.stop() if refits is not None else []
    assert state_digest(reference) == persisted_state_digest(path), (
        "the persisted state is not the reference replay of its log; differing "
        f"parts: {differing_parts(reference, path)}"
    )
    return ran


def _engine_keys(path: Path) -> Dict[int, str]:
    with h5py.File(str(path), "r") as h5f:
        state = load_stage6_engine_state(h5f)
    return {} if state is None else {w: k.kf for w, k in state.keys.items()}


# ---------------------------------------------------------------------------
# The random walk
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Op:
    """One drawn write: the call, whether it must refuse (a bare edit), and
    whether its refusal may follow a fit (an apply at a log prefix resolves the
    file's rows against the prefix's state, computed first)."""

    name: str
    call: Callable[[], Any]
    must_refuse: bool = False
    fits_before_refusal: bool = False


class _Walk:
    """Draws writes against one file and checks P1 and P3 after each."""

    def __init__(self, path: Path, seed: int, refits: _Refits) -> None:
        self.path = path
        self.rng = random.Random(seed)
        self.refits = refits
        shared = s6._build_shared_fit_ctx(str(path))
        succs = s6._cascade_succs(shared.base_cascade_sources)
        self.hub = max(succs, key=lambda w: len(succs[w]))
        self.hubs = sorted(w for w, ds in succs.items() if len(ds) >= 3)
        self.near_hub = sorted({self.hub} | succs[self.hub])
        self.tol = s6.refit_snap_tol_mhz_impl(str(path))
        lo, hi = sorted(float(v) for v in shared.fit_ctx.trim_range)
        live = sorted(
            tuple(sorted(float(v) for v in w.freq_range))
            for w in shared.base_plan.windows
            if int(w.window_id) in shared.base_cascade_sources
        )
        # The gaps between live windows inside the band, wide enough to hold a
        # window or narrow enough to widen one: both are drawn.
        edges = [(lo, lo)] + live + [(hi, hi)]
        self.gaps = [
            (a[1], b[0]) for a, b in zip(edges, edges[1:]) if b[0] - a[1] > 0.5
        ]
        self.writes = 0
        self.skipping = 0
        self.refused: List[Tuple[str, str]] = []
        self.done: List[str] = []

    # -- what the file holds now ---------------------------------------------

    def _window(self, near_hub: float = 0.5, min_peaks: int = 0) -> Optional[int]:
        fits = _fits(self.path)
        pool = [w for w, wf in fits.items() if len(wf.fitted_peaks) >= min_peaks]
        near = [w for w in pool if w in self.near_hub]
        if near and self.rng.random() < near_hub:
            return self.rng.choice(near)
        return self.rng.choice(sorted(pool)) if pool else None

    def _gap_point(self) -> float:
        a, b = self.rng.choice(self.gaps)
        return a + (b - a) * self.rng.uniform(0.2, 0.8)

    def _log(self) -> List[DecisionLogEntry]:
        return ftmw.review_log(self.path)

    # -- the writes ----------------------------------------------------------

    def op_add(self) -> Optional[_Op]:
        w = self._window()
        if w is None:
            return None
        f = _clear(_fits(self.path)[w])
        return _Op("add", lambda: ftmw.review_edit(self.path, w, add=[f], frame="raw"))

    def op_remove(self) -> Optional[_Op]:
        w = self._window(min_peaks=1)
        if w is None:
            return None
        peak = self.rng.choice(_fits(self.path)[w].fitted_peaks)
        token: Any = (
            f"uid:{int(peak.peak_uid)}"
            if self.rng.random() < 0.5
            else float(peak.frequency_mhz)
        )
        return _Op(
            "remove",
            lambda: ftmw.review_edit(self.path, w, remove=[token], frame="raw"),
        )

    def op_merge(self) -> Optional[_Op]:
        fits = _fits(self.path)
        pairs = [
            (w, p, q)
            for w, wf in fits.items()
            for p, q in zip(
                sorted(wf.fitted_peaks, key=lambda x: x.frequency_mhz),
                sorted(wf.fitted_peaks, key=lambda x: x.frequency_mhz)[1:],
            )
            if abs(float(q.frequency_mhz) - float(p.frequency_mhz)) < 1.0
        ]
        if not pairs:
            return None
        w, p, q = self.rng.choice(pairs)
        a, b = float(p.frequency_mhz), float(q.frequency_mhz)
        return _Op(
            "merge",
            lambda: ftmw.review_edit(
                self.path, w, add=[0.5 * (a + b)], remove=[a, b], frame="raw"
            ),
        )

    def op_split(self) -> Optional[_Op]:
        w = self._window(min_peaks=1)
        if w is None:
            return None
        peak = max(_fits(self.path)[w].fitted_peaks, key=lambda p: float(p.snr or 0))
        f = float(peak.frequency_mhz) + 0.6 * self.tol
        return _Op(
            "split", lambda: ftmw.review_edit(self.path, w, add=[f], frame="raw")
        )

    def op_hub(self) -> Optional[_Op]:
        hub = self.rng.choice(self.hubs or [self.hub])
        wf = _fits(self.path)[hub]
        if wf.fitted_peaks and self.rng.random() < 0.6:
            peak = self.rng.choice(wf.fitted_peaks)
            return _Op(
                "hub remove",
                lambda: ftmw.review_edit(
                    self.path, hub, remove=[f"uid:{int(peak.peak_uid)}"], frame="raw"
                ),
            )
        f = _clear(wf)
        return _Op(
            "hub add", lambda: ftmw.review_edit(self.path, hub, add=[f], frame="raw")
        )

    def op_accept(self) -> Optional[_Op]:
        review = load_stage6_review_from_file(str(self.path))
        fits = _fits(self.path)
        lineless = [
            w
            for w, st in review.window_statuses.items()
            if w not in fits
            and any(r.kind.startswith("empty_window") for r in st.attention_reasons)
        ]
        pool = sorted(fits) + lineless * 5
        w = self.rng.choice(pool)
        return _Op("accept", lambda: ftmw.review_accept(self.path, w))

    def op_create(self) -> Optional[_Op]:
        anchor = self._gap_point()
        return _Op("create", lambda: ftmw.review_create(self.path, anchor, frame="raw"))

    def op_implied(self) -> Optional[_Op]:
        f = self._gap_point()
        actions = [{"action": "add", "window_id": None, "freq_mhz": f}]
        if self.rng.random() < 0.4:
            # A second implied add in the same gap coalesces into its window.
            actions.append(
                {"action": "add", "window_id": None, "freq_mhz": f + 3 * self.tol}
            )
        return _Op(
            "implied create",
            lambda: ftmw.review_apply(self.path, actions=actions, frame="raw"),
        )

    def op_undo(self) -> Optional[_Op]:
        log = self._log()
        if not log:
            return None
        ids = self.rng.sample([int(e.serial) for e in log], k=min(len(log), 2))
        ids = ids[: self.rng.choice([1, 1, 2])]
        return _Op("undo", lambda: ftmw.review_undo(self.path, ids))

    def op_prefix(self) -> Optional[_Op]:
        log = self._log()
        if not log:
            return None
        keep = self.rng.randint(0, len(log) - 1)
        w = self._window(near_hub=0.3)
        if w is None:
            return None
        f = _clear(_fits(self.path)[w])
        actions = [{"action": "add", "window_id": w, "freq_mhz": f}]
        return _Op(
            "apply at prefix",
            lambda: ftmw.review_apply(
                self.path, actions=actions, frame="raw", log_prefix=keep
            ),
            fits_before_refusal=True,
        )

    def op_run(self) -> Optional[_Op]:
        if self.rng.random() < 0.5:
            return _Op("review run", lambda: ftmw.review_run(self.path))
        kappa = self.rng.choice([2.0, 3.0, 5.0])
        bar = self.rng.choice([3.0, 4.0, 6.0])
        return _Op(
            "review run P'",
            lambda: ftmw.review_run(self.path, kappa=kappa, bar=bar),
        )

    def op_bare_edit(self) -> Optional[_Op]:
        w = self._window() if self.rng.random() < 0.5 else None
        return _Op(
            "bare edit",
            lambda: ftmw.review_edit(self.path, w, frame="raw"),
            must_refuse=True,
        )

    def op_session(self) -> Optional[_Op]:
        w = self._window()
        if w is None:
            return None
        f = _clear(_fits(self.path)[w])
        actions = [{"action": "add", "window_id": w, "freq_mhz": f}]

        def run() -> None:
            with Pipeline.open(str(self.path)).review_session() as session:
                session.review_preview(actions=actions, frame="raw")
                session.review_apply(actions=actions, frame="raw")

        return _Op("session preview+apply", run)

    _WEIGHTS = {
        "add": 4,
        "remove": 4,
        "merge": 1,
        "split": 1,
        "hub": 3,
        "accept": 2,
        "create": 2,
        "implied": 1,
        "undo": 3,
        "prefix": 1,
        "run": 1,
        "bare_edit": 1,
        "session": 1,
    }

    def step(self) -> None:
        names = list(self._WEIGHTS)
        op: Optional[_Op] = None
        while op is None:
            name = self.rng.choices(names, [self._WEIGHTS[n] for n in names])[0]
            op = getattr(self, f"op_{name}")()
        before = _sha(self.path)
        self.refits.start()
        try:
            op.call()
        except _REFUSALS as exc:
            fitted = self.refits.stop()
            # P3: refused before any fit, the file untouched.
            assert _sha(self.path) == before, (op.name, exc)
            if not op.fits_before_refusal:
                assert fitted == [], (op.name, exc, fitted)
            self.refused.append((op.name, type(exc).__name__))
            return
        wrote = self.refits.stop()
        assert not op.must_refuse, f"{op.name} was not refused"
        self.done.append(op.name)
        self.writes += 1
        # P1, and whether the write kept any window the reference recomputes.
        reference = _assert_reference(self.path, self.refits)
        if len(wrote) < len(reference):
            self.skipping += 1


def _walk(path: Path, seed: int, steps: int, refits: _Refits) -> _Walk:
    walk = _Walk(path, seed, refits)
    for _ in range(steps):
        walk.step()
    return walk


@pytest.mark.parametrize("seed", [11, 23])
def test_every_write_of_a_random_walk_is_the_reference(
    stage5_cascade_file, refits, seed
):
    """P1 and P3 on the cascade fixture: after every write of the walk the
    persisted state is the reference replay of its log, every refusal leaves
    the file as it was before any fit, and the engine does keep windows (some
    write refits fewer than the reference does)."""
    walk = _walk(stage5_cascade_file, seed, 12, refits)
    assert walk.writes >= 6, (walk.done, walk.refused)
    assert walk.skipping >= 1, (walk.done, walk.refused)


# ---------------------------------------------------------------------------
# P2: a valid permutation of a log replays to the same state
# ---------------------------------------------------------------------------


def _permutation(log: Sequence[DecisionLogEntry], rng: random.Random) -> List[Any]:
    """A random order of *log*'s actions that keeps the create rows in order,
    each window's rows in order, and each action's rows together."""
    groups = s6._decision_action_groups(log)
    windows = [{int(e.window_id) for e in g} for g in groups]
    creates = [any(s6._installs_window(e) for e in g) for g in groups]
    before: Dict[int, Set[int]] = {i: set() for i in range(len(groups))}
    for j in range(len(groups)):
        for i in range(j):
            if windows[i] & windows[j] or (creates[i] and creates[j]):
                before[j].add(i)
    placed: List[int] = []
    left = set(range(len(groups)))
    while left:
        ready = sorted(j for j in left if before[j] <= set(placed))
        j = rng.choice(ready)
        placed.append(j)
        left.discard(j)
    return [e for j in placed for e in groups[j]]


def _positionless_parts(state: Any) -> Dict[str, Any]:
    parts = _state_parts(state)
    log = parts["review"].pop("decision_log")
    parts["review"]["rows"] = sorted(
        ({k: v for k, v in row.items() if k != "order_index"} for row in log),
        key=lambda row: int(row["serial"]),
    )
    return parts


def test_a_permuted_log_replays_to_the_same_state(stage5_cascade_file, refits):
    """P2 (G3) on the log a random walk leaves: three random valid
    permutations replay to the state the log itself replays to, positions
    aside."""
    walk = _walk(stage5_cascade_file, 5, 10, refits)
    log = ftmw.review_log(stage5_cascade_file)
    if len({e.window_id for e in log}) < 3:
        pytest.skip(f"the walk left too small a log: {walk.done}")
    review = load_stage6_review_from_file(str(stage5_cascade_file))
    want = _positionless_parts(
        replay_full(stage5_cascade_file, log, review.review_params)
    )
    rng = random.Random(99)
    for _ in range(3):
        permuted = _permutation(log, rng)
        got = _positionless_parts(
            replay_full(stage5_cascade_file, permuted, review.review_params)
        )
        assert got == want


# ---------------------------------------------------------------------------
# The engine's keys
# ---------------------------------------------------------------------------


def _isolated(path: Path, n: int) -> List[int]:
    shared = s6._build_shared_fit_ctx(str(path))
    succs = s6._cascade_succs(shared.base_cascade_sources)
    fits = _fits(path)
    return [
        w
        for w in sorted(fits)
        if not shared.base_cascade_sources[w]
        and not succs.get(w)
        and len(fits[w].fitted_peaks) >= 1
    ][:n]


def test_an_edit_refits_its_window_and_what_it_reaches_only(
    stage5_cascade_file, refits
):
    """With the hub edited, an edit of an isolated window refits that window
    alone; an edit of a dependent replays it and refits it in the cascade; an
    undo of the isolated edit restores the window by copy. Each write is the
    reference."""
    fp = stage5_cascade_file
    shared = s6._build_shared_fit_ctx(str(fp))
    succs = s6._cascade_succs(shared.base_cascade_sources)
    hub = max(succs, key=lambda w: len(succs[w]))
    iso = _isolated(fp, 1)
    if not iso:
        pytest.skip("the cascade fixture has no isolated window")
    leaf = next(d for d in sorted(succs[hub]) if not succs.get(d))
    strongest = max(_fits(fp)[hub].fitted_peaks, key=lambda p: float(p.snr or 0))

    refits.start()
    ftmw.review_edit(fp, hub, remove=[f"uid:{strongest.peak_uid}"], frame="raw")
    assert sorted(refits.stop()) == sorted([hub, *succs[hub]])
    _assert_reference(fp)

    refits.start()
    ftmw.review_edit(fp, iso[0], add=[_clear(_fits(fp)[iso[0]])], frame="raw")
    assert refits.stop() == [iso[0]]
    _assert_reference(fp)

    refits.start()
    ftmw.review_edit(fp, leaf, add=[_clear(_fits(fp)[leaf])], frame="raw")
    assert refits.stop() == [leaf, leaf]  # its own row, then the cascade
    _assert_reference(fp)

    serial = next(e.serial for e in ftmw.review_log(fp) if e.window_id == iso[0])
    refits.start()
    ftmw.review_undo(fp, [serial])
    assert refits.stop() == []
    _assert_reference(fp)


def test_a_first_write_from_a_staged_preview_keys_under_its_baseline(
    stage5_cascade_file, refits
):
    """A session's preview on a file no write has curated keys its windows
    before any baseline exists; the apply that persists it stamps the
    baseline with the lineage those keys were computed under, so the next
    edit of another window refits that window alone."""
    fp = stage5_cascade_file
    iso = _isolated(fp, 2)
    if len(iso) < 2:
        pytest.skip("the cascade fixture has fewer than two isolated windows")
    actions = [
        {"action": "add", "window_id": iso[0], "freq_mhz": _clear(_fits(fp)[iso[0]])}
    ]
    with Pipeline.open(str(fp)).review_session() as session:
        session.review_preview(actions=actions, frame="raw")
        session.review_apply(actions=actions, frame="raw")
    _assert_reference(fp)

    refits.start()
    ftmw.review_edit(fp, iso[1], add=[_clear(_fits(fp)[iso[1]])], frame="raw")
    assert refits.stop() == [iso[1]]
    _assert_reference(fp)


def test_the_keys_describe_every_window_the_fit_holds(stage5_cascade_file):
    """``/stage6_engine`` keys every window of the curated fit, no other; a
    create adds its window, the undo of the create drops it, and undoing
    every fit-changing decision empties the store (the automatic fit is
    restored by copy)."""
    fp = stage5_cascade_file
    w = _isolated(fp, 1)[0]
    ftmw.review_edit(fp, w, add=[_clear(_fits(fp)[w])], frame="raw")
    assert set(_engine_keys(fp)) == set(_fits(fp))
    gaps = _Walk(fp, 0, _Refits()).gaps
    a, b = max(gaps, key=lambda g: g[1] - g[0])
    created = ftmw.review_create(fp, 0.5 * (a + b), frame="raw")
    assert created.window_id in _engine_keys(fp)
    assert set(_engine_keys(fp)) == set(_fits(fp))
    ftmw.review_undo(fp, [ftmw.review_log(fp)[-1].serial])
    assert created.window_id not in _engine_keys(fp)
    assert set(_engine_keys(fp)) == set(_fits(fp))
    _assert_reference(fp)
    ftmw.review_undo(fp, [e.serial for e in ftmw.review_log(fp)])
    assert _engine_keys(fp) == {}
    _assert_reference(fp)


def test_the_create_prefix_is_not_replanned(stage5_cascade_file, monkeypatch):
    """A write whose log's creates share the persisted chain reuses every
    planned window; undoing the first of two creates replans the second only."""
    fp = stage5_cascade_file
    gaps = sorted(_Walk(fp, 0, _Refits()).gaps, key=lambda g: g[0] - g[1])[:2]
    for a, b in gaps:
        ftmw.review_create(fp, 0.5 * (a + b), frame="raw")
    planned: List[float] = []
    plan = s6._plan_create

    def counting(shared: Any, overlay: Any, anchor: float, **kw: Any) -> Any:
        planned.append(anchor)
        return plan(shared, overlay, anchor, **kw)

    monkeypatch.setattr(s6, "_plan_create", counting)
    created = [e for e in ftmw.review_log(fp) if e.kind == "create_window"]
    w = created[-1].window_id
    ftmw.review_edit(fp, w, add=[_clear(_fits(fp)[w])], frame="raw")
    assert planned == []
    _assert_reference(fp)  # the reference plans every create, from scratch
    log = ftmw.review_log(fp)
    first = next(e for e in log if e.kind == "create_window")
    if any(e.window_id == first.window_id for e in log if e is not first):
        pytest.skip("the first created window carries a decision")
    planned.clear()
    ftmw.review_undo(fp, [first.serial])
    # The undo's own symbolic pass, then the write: the second create each time.
    assert planned == [float(created[-1].frequency_mhz)] * 2
    _assert_reference(fp)


def _engine_state(path: Path) -> Any:
    with h5py.File(str(path), "r") as h5f:
        return load_stage6_engine_state(h5f)


def test_undoing_an_upstream_edit_restores_what_it_reached(stage5_cascade_file, refits):
    """The hub's edit reached every dependent; with a dependent edited as
    well, undoing the hub's edit leaves that dependent unreached, so its fit
    is its own row replayed from the automatic fit (one refit, no cascade),
    while the hub and every dependent with no row are restored by copy: their
    fits are the automatic fit, bit for bit, and the write is the reference."""
    fp = stage5_cascade_file
    shared = s6._build_shared_fit_ctx(str(fp))
    succs = s6._cascade_succs(shared.base_cascade_sources)
    hub = max(succs, key=lambda w: len(succs[w]))
    leaf = next(d for d in sorted(succs[hub]) if not succs.get(d))
    strongest = max(_fits(fp)[hub].fitted_peaks, key=lambda p: float(p.snr or 0))
    ftmw.review_edit(fp, hub, remove=[f"uid:{strongest.peak_uid}"], frame="raw")
    ftmw.review_edit(fp, leaf, add=[_clear(_fits(fp)[leaf])], frame="raw")
    hub_serial = next(e.serial for e in ftmw.review_log(fp) if e.window_id == hub)
    edited = _fits(fp)

    refits.start()
    ftmw.review_undo(fp, [hub_serial])
    assert refits.stop() == [leaf]
    _assert_reference(fp)

    with h5py.File(str(fp), "r") as h5f:
        auto = load_spectrum_fit_from_hdf5(h5f[s6.STAGE5_BASELINE_GROUP])
    baseline = {int(wf.window_id): wf for wf in auto.window_fits}
    after = _fits(fp)
    restored = [w for w in {hub, *succs[hub]} if w != leaf]
    assert restored and any(
        edited[w].reduced_chi2 != baseline[w].reduced_chi2 for w in restored
    )
    for w in restored:
        assert after[w].reduced_chi2 == baseline[w].reduced_chi2, w
        assert [p.frequency_mhz for p in after[w].fitted_peaks] == [
            p.frequency_mhz for p in baseline[w].fitted_peaks
        ], w
    state = _engine_state(fp)
    assert state.keys[leaf].dirty and not state.keys[leaf].reached
    assert not any(state.keys[w].reached for w in succs[hub])


def test_a_write_that_changes_no_key_leaves_the_store_as_it_was(
    stage5_cascade_file, refits
):
    """A bare accept, ``review run`` (with and without new parameters) and the
    undo of the accept refit nothing and leave ``/stage6_engine`` -- context
    key, create chain and every window's keys -- exactly as it was."""
    fp = stage5_cascade_file
    hub, iso = _curate_some(fp)
    stored = _engine_state(fp)
    assert stored is not None and stored.keys
    refits.start()
    ftmw.review_accept(fp, iso[0])
    ftmw.review_run(fp)
    ftmw.review_run(fp, kappa=3.0, bar=4.0)
    accept = ftmw.review_log(fp)[-1]
    assert accept.kind == "accept"
    ftmw.review_undo(fp, [accept.serial])
    assert refits.stop() == []
    assert _engine_state(fp) == stored
    _assert_reference(fp)


def test_a_file_with_fits_but_no_keys_recomputes_what_it_curates(
    stage5_cascade_file, refits
):
    """A file whose store is empty (curated by a build without keys, or
    cleared) is not wrong, only unkeyed: the next write that refits
    recomputes every window it curates -- the hub and the isolated windows
    with rows, the hub's dependents through the cascade -- and keys them."""
    fp = stage5_cascade_file
    hub, iso = _curate_some(fp)
    succs = s6._cascade_succs(s6._build_shared_fit_ctx(str(fp)).base_cascade_sources)
    with h5py.File(str(fp), "a") as h5f:
        save_stage6_engine_state(h5f, None)
    assert _engine_state(fp) is None

    refits.start()
    ftmw.review_edit(fp, iso[0], add=[_clear(_fits(fp)[iso[0]]) + 0.1], frame="raw")
    assert {hub, *iso, *succs[hub]} <= set(refits.stop())
    assert set(_engine_keys(fp)) == set(_fits(fp))
    _assert_reference(fp)

    refits.start()
    ftmw.review_edit(fp, iso[1], add=[_clear(_fits(fp)[iso[1]]) + 0.1], frame="raw")
    assert refits.stop() == [iso[1]] * 2  # its two rows, nothing else
    _assert_reference(fp)


def test_a_write_to_a_keyed_file_is_refused_before_any_fit(stage5_cascade_file, refits):
    """P3 on a file the engine has keyed: a bare edit (with and without a
    window), an add outside its window, an undo that orphans a kept row, and
    both pre-engine refusals leave the file byte for byte as it was
    (``/stage6_engine`` included) and fit nothing."""
    fp = stage5_cascade_file
    hub, iso = _curate_some(fp)
    w = iso[0]
    ftmw.review_edit(fp, w, add=[_clear(_fits(fp)[w])], frame="raw")
    born = ftmw.review_log(fp)[-1]
    ftmw.review_edit(fp, w, remove=[f"uid:{born.born_uids[0]}"], frame="raw")
    other = next(x for x in _fits(fp) if x != w and x not in {hub})
    outside = _clear(_fits(fp)[other])
    wf = _fits(fp)[w]
    assert wf.window is not None
    assert not min(wf.window.freq_range) <= outside <= max(wf.window.freq_range)

    refusals: List[Tuple[str, Callable[[Path], Any], Any]] = [
        ("bare edit", lambda p: ftmw.review_edit(p, w, frame="raw"), BadSettingError),
        (
            "bare edit, no window",
            lambda p: ftmw.review_edit(p, None, frame="raw"),
            BadSettingError,
        ),
        (
            "add outside the window",
            lambda p: ftmw.review_edit(p, w, add=[outside], frame="raw"),
            CurationConflictError,
        ),
        (
            "undo that orphans",
            lambda p: ftmw.review_undo(p, [born.serial]),
            CurationConflictError,
        ),
    ]
    for label, call, error in refusals:
        before = _sha(fp)
        refits.start()
        with pytest.raises(error):
            call(fp)
        assert refits.stop() == [], label
        assert _sha(fp) == before, label

    # A review no engine of this build wrote, and a fit with no peak identity.
    def strip_engine_version(h5f: h5py.File) -> None:
        del h5f["stage6_review"].attrs["engine_version"]

    def strip_lineage(h5f: h5py.File) -> None:
        del h5f[s6.STAGE5_BASELINE_GROUP].attrs[s6.LINEAGE_ID_ATTR]

    for surgery in (strip_engine_version, strip_lineage):
        work = fp.parent / f"{surgery.__name__}.ftmw"
        shutil.copy(fp, work)
        with h5py.File(str(work), "a") as h5f:
            surgery(h5f)
        before = _sha(work)
        refits.start()
        with pytest.raises(CurationConflictError) as exc:
            ftmw.review_edit(work, w, add=[_clear(_fits(work)[w])], frame="raw")
        assert exc.value.reason == "predates_replay_engine"
        assert refits.stop() == []
        assert _sha(work) == before
        work.unlink()


def test_a_write_rebuilds_the_products_of_the_windows_it_changed_only(
    stage5_cascade_file, monkeypatch
):
    """The per-window final-products fields of a window whose fit a write
    kept are read off the file's own table, not recomputed: an edit of an
    isolated window computes them for that window alone. Every status is
    recomputed on every write, kept windows' included, whatever the
    parameters. The results are the reference's (P1)."""
    fp = stage5_cascade_file
    hub, iso = _curate_some(fp)
    fits = _fits(fp)
    lined = {w for w, wf in fits.items() if wf.fitted_peaks}
    fields: List[int] = []
    reasons: List[int] = []
    fields_core = s6._window_fit_fields
    reasons_core = s6._compute_attention_reasons

    def counting_fields(wf: Any, **kwargs: Any) -> Any:
        fields.append(int(wf.window_id))
        return fields_core(wf, **kwargs)

    def counting_reasons(wf: Any, **kwargs: Any) -> Any:
        reasons.append(int(wf.window_id))
        return reasons_core(wf, **kwargs)

    monkeypatch.setattr(s6, "_window_fit_fields", counting_fields)
    monkeypatch.setattr(s6, "_compute_attention_reasons", counting_reasons)

    ftmw.review_edit(fp, iso[0], add=[_clear(fits[iso[0]]) + 0.1], frame="raw")
    assert set(fields) & lined == {iso[0]}
    assert set(reasons) >= lined
    _assert_reference(fp)

    reasons.clear()
    ftmw.review_run(fp)
    assert set(reasons) >= lined
    _assert_reference(fp)

    reasons.clear()
    ftmw.review_run(fp, kappa=3.0, bar=4.0)
    assert set(reasons) >= lined
    _assert_reference(fp)

    fields.clear()
    reasons.clear()
    ftmw.review_edit(fp, iso[1], add=[_clear(_fits(fp)[iso[1]]) + 0.1], frame="raw")
    assert set(fields) & lined == {iso[1]}
    assert set(reasons) >= lined
    _assert_reference(fp)


def test_final_products_follow_a_changed_calibration(stage5_cascade_file):
    """A kept fit's per-line fit fields are reused only under the probe and
    ``epsilon`` they were computed with: after the timebase calibration
    changes, a write that refits nothing leaves the final products
    :func:`_rebuild_final_products` derives from the kept fits."""
    fp = stage5_cascade_file
    hub, iso = _curate_some(fp)
    ftmw.calibrate_timebase(fp, kappa_sys=3.0e-7)
    ftmw.review_accept(fp, iso[0])
    review = load_stage6_review_from_file(str(fp))
    derived = s6._rebuild_final_products(str(fp))
    assert derived is not None
    assert float(review.final_products.epsilon) == float(derived.epsilon)
    assert review.final_products.peaks == derived.peaks


# ---------------------------------------------------------------------------
# The engine's environment cases (§10.3)
# ---------------------------------------------------------------------------


def _curate_some(fp: Path) -> Tuple[int, List[int]]:
    """A hub edit and an edit of two isolated windows; returns the hub and the
    isolated windows."""
    shared = s6._build_shared_fit_ctx(str(fp))
    succs = s6._cascade_succs(shared.base_cascade_sources)
    hub = max(succs, key=lambda w: len(succs[w]))
    iso = _isolated(fp, 2)
    strongest = max(_fits(fp)[hub].fitted_peaks, key=lambda p: float(p.snr or 0))
    ftmw.review_edit(fp, hub, remove=[f"uid:{strongest.peak_uid}"], frame="raw")
    for w in iso:
        ftmw.review_edit(fp, w, add=[_clear(_fits(fp)[w])], frame="raw")
    return hub, iso


def test_a_changed_soft_input_rebuilds_every_window(stage5_cascade_file, refits):
    """Re-running the timebase calibration with another ``kappa_sys`` changes
    the analysis context (a soft Stage 5 input: Stage 5 stays). A write that
    refits nothing keeps the fits and the keys; the next write that refits
    anything rebuilds every window it curates, under the new context, and is
    the reference."""
    fp = stage5_cascade_file
    hub, iso = _curate_some(fp)
    with h5py.File(str(fp), "r") as h5f:
        stored = load_stage6_engine_state(h5f)
    assert stored is not None
    ftmw.calibrate_timebase(fp, kappa_sys=3.0e-7)
    fits = _fits(fp)

    ftmw.review_accept(fp, iso[0])
    with h5py.File(str(fp), "r") as h5f:
        kept = load_stage6_engine_state(h5f)
    assert kept is not None and kept.e_key == stored.e_key
    assert all(wf.reduced_chi2 == fits[w].reduced_chi2 for w, wf in _fits(fp).items())

    refits.start()
    ftmw.review_edit(fp, iso[0], add=[_clear(_fits(fp)[iso[0]]) + 0.1], frame="raw")
    rebuilt = set(refits.stop())
    assert {hub, *iso} <= rebuilt and len(rebuilt) > 3
    with h5py.File(str(fp), "r") as h5f:
        after = load_stage6_engine_state(h5f)
    assert after is not None and after.e_key != stored.e_key
    _assert_reference(fp)

    refits.start()
    ftmw.review_edit(fp, iso[1], add=[_clear(_fits(fp)[iso[1]])], frame="raw")
    assert refits.stop() == [iso[1], iso[1]]  # its two rows, nothing else
    _assert_reference(fp)


def test_an_unacknowledged_epoch_leaves_the_writes_that_refit_nothing(
    stage5_cascade_file, monkeypatch
):
    """Under a new ``ANALYSIS_EPOCH``: ``review run``, a bare accept and an
    undo that drops a window's only edit (restored by copy) succeed and keep
    every other fit; an edit is refused with the file untouched."""
    from ftmwpipeline.core import environment

    fp = stage5_cascade_file
    hub, iso = _curate_some(fp)
    fits = _fits(fp)
    monkeypatch.setattr(environment, "ANALYSIS_EPOCH", environment.ANALYSIS_EPOCH + 1)

    ftmw.review_run(fp, kappa=3.0)
    ftmw.review_accept(fp, iso[1])
    serial = next(e.serial for e in ftmw.review_log(fp) if e.window_id == iso[0])
    ftmw.review_undo(fp, [serial])
    after = _fits(fp)
    for w in (hub, iso[1]):
        assert after[w].reduced_chi2 == fits[w].reduced_chi2
    before = _sha(fp)
    with pytest.raises(AnalysisEpochMismatchError):
        ftmw.review_edit(fp, iso[1], add=[_clear(after[iso[1]]) + 0.1], frame="raw")
    assert _sha(fp) == before


def test_a_file_without_complete_provenance_still_keeps_windows(
    stage5_cascade_file, refits, monkeypatch
):
    """When the file cannot account for every input its stages used, the
    context key degrades to a digest of the soft inputs: a second write still
    keeps the windows it does not change."""
    from ftmwpipeline._internal import fingerprint_impl

    def incomplete(path: Any) -> Dict[str, Any]:
        raise IncompleteProvenanceError(["fit.consumed"], message="incomplete")

    monkeypatch.setattr(fingerprint_impl, "canonical_fingerprint_inputs", incomplete)
    fp = stage5_cascade_file
    hub, iso = _curate_some(fp)
    refits.start()
    ftmw.review_edit(fp, iso[0], add=[_clear(_fits(fp)[iso[0]]) + 0.1], frame="raw")
    assert refits.stop() == [iso[0], iso[0]]  # its two rows, nothing else
    _assert_reference(fp)


def test_fit_run_drops_the_engine_keys(stage5_cascade_file):
    """A fresh fit starts a new lineage: ``fit run`` drops the keys with the
    undo baseline, and the next write keys the new fit."""
    fp = stage5_cascade_file
    _curate_some(fp)
    with h5py.File(str(fp), "r") as h5f:
        assert STAGE6_ENGINE_GROUP in h5f
    ftmw.fit_peaks(str(fp))
    with h5py.File(str(fp), "r") as h5f:
        assert STAGE6_ENGINE_GROUP not in h5f
    hub, iso = _curate_some(fp)
    assert set(_engine_keys(fp)) == set(_fits(fp))
    _assert_reference(fp)


def test_an_upstream_rerun_drops_the_engine_keys(stage5_cascade_file):
    """Re-running a stage the fit depends on deletes the fit and the review,
    and the engine's keys with them."""
    fp = stage5_cascade_file
    _curate_some(fp)
    ftmw.assign_windows(fp)
    with h5py.File(str(fp), "r") as h5f:
        assert "stage6_review" not in h5f and STAGE6_ENGINE_GROUP not in h5f


# ---------------------------------------------------------------------------
# The slow tier: the full 655 build
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_every_write_of_a_random_walk_is_the_reference_on_655(
    built_655, tmp_path, refits
):
    """P1 and P3 over thirty random writes on 655: the 429 hub's 32
    dependents, the creates in dropped 332 and 373 and every other gap."""
    fp = tmp_path / "655.ftmw"
    shutil.copy(built_655, fp)
    walk = _Walk(fp, 655, refits)
    # Creates inside 332 and 373 (dropped by Stage 5) are drawn as gaps too.
    walk.gaps += [(35838.0, 35846.0), (36886.0, 36892.0)]
    for _ in range(30):
        walk.step()
    assert walk.writes >= 15, (walk.done, walk.refused)
    assert walk.skipping >= 5, (walk.done, walk.refused)


@pytest.mark.slow
def test_a_double_widening_after_its_bound_is_undone_on_655(
    built_655, tmp_path, refits
):
    """A created window A abutting window w from above, a create in the crack
    between them that widens w up to A (bounded by A), a second that widens w
    downward from the first widening's geometry, and a line added to w; then A
    is undone. Both widenings are re-derived on the replay (the first now
    reaches past where A stood), w is refit and reported, and every write --
    the undo included -- is the reference. The random walk rarely draws
    this."""
    fp = tmp_path / "655.ftmw"
    shutil.copy(built_655, fp)
    shared = s6._build_shared_fit_ctx(str(fp))
    live = sorted(
        (tuple(sorted(float(v) for v in w.freq_range)), int(w.window_id))
        for w in shared.base_plan.windows
        if int(w.window_id) in shared.base_cascade_sources
    )
    # A live window with room above it for a created window, and a live
    # window below it (so the low edge is inside the band).
    w, (lo, hi) = next(
        (wid, rng)
        for (prev, _), (rng, wid), (nxt, _) in zip(live, live[1:], live[2:])
        if nxt[0] - rng[1] >= 8.0 and rng[0] - prev[1] >= 1.0
    )
    a = ftmw.review_create(fp, hi + 3.0, frame="raw")
    if a.mode != "created" or min(a.freq_range) - hi > 0.5:
        pytest.skip("the created window does not abut the window below it")
    _assert_reference(fp)
    first = ftmw.review_create(fp, hi + 0.06, frame="raw")
    second = ftmw.review_create(fp, lo - 0.06, frame="raw")
    if (first.mode, first.window_id, second.mode, second.window_id) != (
        "widened",
        w,
        "widened",
        w,
    ):
        pytest.skip("the anchors beside the window did not widen it")
    assert max(first.freq_range) <= min(a.freq_range)
    _assert_reference(fp)
    ftmw.review_edit(fp, w, add=[_clear(_fits(fp)[w])], frame="raw")
    _assert_reference(fp)

    serial = next(
        e.serial
        for e in ftmw.review_log(fp)
        if e.kind == "create_window" and e.window_id == a.window_id
    )
    refits.start()
    result = ftmw.review_undo(fp, [serial])
    assert w in refits.stop()
    assert w in result.geometry_changed_window_ids
    assert max(_fits(fp)[w].window.freq_range) > max(first.freq_range)
    _assert_reference(fp)


def _skirt(wf: FittingResult, source: int) -> List[float]:
    """The lines of window *source* that *wf* holds frozen."""
    return sorted(
        float(e["frequency_mhz"])
        for k, e in wf.fixed_parameters.items()
        if k.startswith("frozen_peak_") and int(e["primary_window_id"]) == source
    )


@pytest.mark.slow
def test_an_edit_of_a_created_window_refreshes_the_window_beside_it_on_655(
    built_655, tmp_path, refits
):
    """A create inside dropped 332 (window 522), a nearby create that reads it
    (523), then edits of 522 and the undo of those edits. Each write is the
    reference; an edit of 522 refits 523 twice (its create row, then the
    cascade's refresh from 522's edited fit), and undoing the edits gives 523
    back the fit it had before them."""
    fp = tmp_path / "655.ftmw"
    shutil.copy(built_655, fp)
    first = ftmw.review_create(fp, 35839.957, frame="raw")
    second = ftmw.review_create(fp, 35844.73, frame="raw")
    assert (first.mode, second.mode) == ("created", "created")
    assert first.window_id in second.depends_on
    _assert_reference(fp)
    plain = _fits(fp)
    assert _skirt(plain[second.window_id], first.window_id) == []

    for k in range(2):
        add = 35839.957 if k == 0 else _clear(_fits(fp)[first.window_id])
        refits.start()
        ftmw.review_edit(fp, first.window_id, add=[add], frame="raw")
        calls = refits.stop()
        assert first.window_id in calls and calls.count(second.window_id) == 2
        _assert_reference(fp)
        # 523 reads the line 522 now holds, as a frozen skirt.
        assert _skirt(_fits(fp)[second.window_id], first.window_id)

    adds = [e.serial for e in ftmw.review_log(fp) if e.kind == "add"]
    refits.start()
    ftmw.review_undo(fp, adds)
    assert second.window_id in refits.stop()
    _assert_reference(fp)
    again = _fits(fp)
    assert _skirt(again[second.window_id], first.window_id) == []
    for w in (first.window_id, second.window_id):
        assert again[w].reduced_chi2 == plain[w].reduced_chi2
