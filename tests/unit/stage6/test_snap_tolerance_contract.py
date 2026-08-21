"""The curation snap tolerance is defined in bins and published as one read.

``REFIT_SNAP_TOL_BINS`` exists so an external tool can resolve "the peak at *f*"
exactly as the pipeline does. Since ``dev-docs/SCIENCE_STRATEGY.md``
Requirement 8 that definition is a count of active-FT bins, so the MHz value is
a property of one *file*, and the contract has two halves:

- **One definition.** No surface may carry its own copy of the tolerance. Every
  public verb defaults to ``None``, and ``None`` resolves through the single
  accessor -- a literal retyped into a signature would reproduce the drift the
  published name exists to remove.
- **One derivation.** The accessor
  (:func:`~ftmwpipeline.api.refit_snap_tol_mhz`) derives the MHz value at call
  time from the same active region the verbs consult, so a caller reading it
  and a ``review apply`` on the same file cannot disagree. An integrator that
  multiplied the bin count by an ``acquisition_us`` of its own would be the
  two-reads-can-disagree surface again.

The old absolute ``REFIT_SNAP_TOL_MHZ`` is deleted outright, with no
compatibility alias: an alias would preserve exactly the "resolve it yourself"
surface this work removes.
"""

from __future__ import annotations

import inspect
from typing import Callable, List, Tuple

import h5py
import numpy as np
import pytest

import ftmwpipeline
import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline._internal import stage6_impl
from ftmwpipeline.core.curation import REFIT_SNAP_TOL_BINS
from ftmwpipeline.file_manager import StageDependencyError

pytestmark = [pytest.mark.unit]

# Every public verb that resolves a user-supplied frequency against the file.
# ``merge``/``split`` are deliberately absent: they are not verbs on
# Pipeline/api/CLI (curation-intent inference reads them from add/remove), so
# there is no such public surface to check here. The impl-level functions
# (still gated the same way) are covered in
# test_boundary_impls_defer_and_engines_require below.
_VERBS = (
    "review_edit",
    "review_create",
    "review_accept",
)


def _public_surfaces() -> List[Tuple[str, Callable[..., object]]]:
    out: List[Tuple[str, Callable[..., object]]] = []
    for name in _VERBS:
        out.append((f"Pipeline.{name}", getattr(Pipeline, name)))
        out.append((f"api.{name}", getattr(ftmw, name)))
    return out


# ---------------------------------------------------------------------------
# One definition: nothing carries a copy of the number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,func", _public_surfaces(), ids=lambda v: v)
def test_public_default_defers_to_the_file(label, func):
    """A per-file quantity cannot have a module-level default.

    ``None`` is the only defensible default: any float here would be one
    acquisition length's answer frozen into every other file's call.
    """
    params = inspect.signature(func).parameters
    assert "snap_tol_mhz" in params, f"{label} does not expose snap_tol_mhz"
    assert (
        params["snap_tol_mhz"].default is None
    ), f"{label} carries a default tolerance instead of deferring to the file"


@pytest.mark.parametrize("label,func", _public_surfaces(), ids=lambda v: v)
def test_docstring_does_not_respell_a_value(label, func):
    """Prose is a copy too, and an absolute one is now actively wrong."""
    doc = inspect.getdoc(func) or ""
    for bad in ("default 0.05", "50 kHz)", "REFIT_SNAP_TOL_MHZ"):
        assert bad not in doc, f"{label} states an absolute tolerance: {bad!r}"


def test_boundary_impls_defer_and_engines_require():
    """Resolve once at the boundary; pass the resolved float inward.

    The engine functions take ``snap_tol_mhz`` with **no default** on purpose
    (the discipline ``gate_spurs`` adopted for ``bin_spacing_mhz`` in phase 2):
    a default further in would be an absolute constant coming back, and a batch
    could then snap two of its actions at two tolerances.
    """
    for name in (
        "refit_window_impl",
        "create_window_impl",
        "merge_peaks_impl",
        "split_peak_impl",
        "review_accept_impl",
        "review_preview_impl",
    ):
        params = inspect.signature(getattr(stage6_impl, name)).parameters
        assert params["snap_tol_mhz"].default is None, name

    for name in (
        "refit_window_core",
        "_execute_curation_batch",
        "_curation_ambiguity_warnings",
    ):
        params = inspect.signature(getattr(stage6_impl, name)).parameters
        assert (
            params["snap_tol_mhz"].default is inspect.Parameter.empty
        ), f"{name} has a default tolerance; it must take the resolved value"


def test_bin_constant_is_exported_and_the_absolute_one_is_gone():
    """The import an integrator is told to use -- and the one that is deleted.

    No deprecation alias: the only published consumer feature-detects the
    absence, and an alias would keep alive the surface where a caller resolves
    the tolerance itself.
    """
    assert ftmwpipeline.REFIT_SNAP_TOL_BINS is REFIT_SNAP_TOL_BINS
    assert "REFIT_SNAP_TOL_BINS" in ftmwpipeline.__all__
    for module in (ftmwpipeline, ftmw, stage6_impl):
        assert not hasattr(module, "REFIT_SNAP_TOL_MHZ"), module.__name__


# ---------------------------------------------------------------------------
# One derivation: the accessor, across acquisition lengths
# ---------------------------------------------------------------------------


def _synthetic_import(tmp_path, name: str, n: int, spacing_us: float):
    """A bare imported .ftmw of a chosen duration: Stage 0 and nothing else.

    Cheap on purpose. The accessor is claimed to answer on a file that has been
    through nothing but the import, so testing it behind a full pipeline build
    would test something weaker than the claim.
    """
    src = tmp_path / f"{name}_src.h5"
    sig = np.cos(2 * np.pi * np.arange(n) * 0.01)
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = spacing_us
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=sig)
    out = tmp_path / f"{name}.ftmw"
    ftmw.import_data(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


# Three materially different active regions -- ~5 us, ~12.65 us, ~40 us -- so a
# test at one acquisition length cannot pass for the reason this work item
# exists. (n * spacing_us = duration; an unset window means the whole record.)
_LENGTHS = {
    "short": (2500, 0.002, 5.0),
    "reference": (6325, 0.002, 12.65),
    "long": (20000, 0.002, 40.0),
}


@pytest.mark.parametrize("label", sorted(_LENGTHS))
def test_accessor_resolves_the_bin_count_for_this_file(tmp_path, label):
    """``REFIT_SNAP_TOL_BINS / T_active``, exactly, at every length."""
    n, spacing_us, t_active = _LENGTHS[label]
    path = _synthetic_import(tmp_path, label, n, spacing_us)

    resolved = ftmw.refit_snap_tol_mhz(path)
    assert resolved == pytest.approx(REFIT_SNAP_TOL_BINS / t_active, rel=1e-9)


def test_tolerance_is_constant_in_bins_and_varies_in_mhz(tmp_path):
    """The invariant the conversion exists to create.

    In MHz the tolerance is 8x tighter at 40 us than at 5 us -- it tracks the
    resolution, as a spectral distance must. In bins it is the same number
    everywhere, which is what makes it a definition rather than one lab's
    acquisition length.
    """
    resolved = {}
    for label, (n, spacing_us, t_active) in _LENGTHS.items():
        path = _synthetic_import(tmp_path, label, n, spacing_us)
        resolved[label] = (ftmw.refit_snap_tol_mhz(path), t_active)

    for _label, (mhz, t_active) in resolved.items():
        assert mhz * t_active == pytest.approx(REFIT_SNAP_TOL_BINS, rel=1e-9)

    assert resolved["short"][0] > resolved["reference"][0] > resolved["long"][0]
    assert resolved["short"][0] / resolved["long"][0] == pytest.approx(8.0, rel=1e-6)


def test_accessor_is_read_only(tmp_path):
    """Safe on a file someone else has open, or is about to write."""
    path = _synthetic_import(tmp_path, "readonly", 6325, 0.002)
    before = path.read_bytes()
    ftmw.refit_snap_tol_mhz(path)
    Pipeline.open(path).refit_snap_tol_mhz()
    assert path.read_bytes() == before


def test_missing_file_is_a_clean_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ftmw.refit_snap_tol_mhz(tmp_path / "nope.ftmw")


def test_no_active_region_refuses_rather_than_inventing_a_value(tmp_path):
    """The documented degrade.

    A bin-defined tolerance on a file with no spectrum has no honest MHz
    answer, so the accessor refuses instead of handing back the legacy
    absolute value the pipeline would never actually pair at. Unreachable for
    any file the importer wrote -- this fixture has to be built by hand.
    """
    path = _synthetic_import(tmp_path, "headless", 2500, 0.002)
    with h5py.File(path, "a") as f:
        del f["stage0_fid_data"]
    with pytest.raises(StageDependencyError):
        ftmw.refit_snap_tol_mhz(path)


# ---------------------------------------------------------------------------
# The CLI carries the same shape
# ---------------------------------------------------------------------------

_CLI_WIRING = {
    "edit": (
        ["review", "edit", "f.ftmw", "--window", "0", "--add", "1.0"],
        "refit_window_impl",
    ),
    "create": (["review", "create", "f.ftmw", "--at", "26622.0"], "create_window_impl"),
    "accept": (
        ["review", "accept", "f.ftmw", "--window", "0", "--candidate", "1.0"],
        "review_accept_impl",
    ),
}


@pytest.mark.parametrize("verb", sorted(_CLI_WIRING))
def test_cli_flag_reaches_the_impl(verb, monkeypatch):
    """A parsed flag is worth nothing if the handler drops it on the floor."""
    from ftmwpipeline.cli import review_commands
    from ftmwpipeline.cli.main import create_parser

    argv, impl_name = _CLI_WIRING[verb]
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        raise ValueError("stop here -- the call is all we needed to observe")

    monkeypatch.setattr(review_commands, impl_name, spy)
    args = create_parser().parse_args(argv + ["--snap-tol-mhz", "0.011"])
    assert args.func(args) == 1  # the spy's ValueError is reported, not raised
    assert seen.get("snap_tol_mhz") == pytest.approx(0.011)


@pytest.mark.parametrize("verb", sorted(_CLI_WIRING))
def test_cli_flag_defaults_to_the_file(verb):
    """The third interface defers too -- no parser-level absolute default."""
    from ftmwpipeline.cli.main import create_parser

    args = create_parser().parse_args(
        {
            "edit": ["review", "edit", "f.ftmw", "--window", "0"],
            "create": ["review", "create", "f.ftmw", "--at", "26622.0"],
            "accept": ["review", "accept", "f.ftmw", "--window", "0"],
        }[verb]
    )
    assert args.snap_tol_mhz is None
