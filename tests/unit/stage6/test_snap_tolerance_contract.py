"""The published curation snap tolerance is the one the verbs actually use.

``REFIT_SNAP_TOL_MHZ`` exists so an external tool can resolve "the peak at *f*"
exactly as the pipeline does. That is only worth anything if the constant is the
single definition every surface takes its default from -- a published name that
*happens* to agree with a set of hardcoded literals reproduces the drift it was
meant to remove, silently, the first time someone edits one and not the other.

So these tests pin identity (``is``), not equality: a literal ``0.05`` retyped
into a signature would satisfy ``==`` and fail here, which is the point.
"""

from __future__ import annotations

import inspect
from typing import Callable, List, Tuple

import pytest

import ftmwpipeline
import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.core.curation import REFIT_SNAP_TOL_MHZ

pytestmark = [pytest.mark.unit]

# Every public verb that resolves a user-supplied frequency against the file.
_VERBS = (
    "review_edit",
    "review_create",
    "review_merge",
    "review_split",
    "review_accept",
)


def _public_surfaces() -> List[Tuple[str, Callable[..., object]]]:
    out: List[Tuple[str, Callable[..., object]]] = []
    for name in _VERBS:
        out.append((f"Pipeline.{name}", getattr(Pipeline, name)))
        out.append((f"api.{name}", getattr(ftmw, name)))
    return out


@pytest.mark.parametrize("label,func", _public_surfaces(), ids=lambda v: v)
def test_public_default_is_the_published_constant(label, func):
    """No surface may carry its own copy of the tolerance."""
    params = inspect.signature(func).parameters
    assert "snap_tol_mhz" in params, f"{label} does not expose snap_tol_mhz"
    assert (
        params["snap_tol_mhz"].default is REFIT_SNAP_TOL_MHZ
    ), f"{label} defaults to a value that is not the published constant"


@pytest.mark.parametrize("label,func", _public_surfaces(), ids=lambda v: v)
def test_docstring_does_not_respell_the_value(label, func):
    """Prose is a copy too. Docstrings name the constant, not the number.

    ``50 kHz`` in a parenthetical is fine (it is a reading aid next to the
    named constant); a bare ``0.05`` as *the* stated default is the drift.
    """
    doc = inspect.getdoc(func) or ""
    assert "default 0.05" not in doc, f"{label} states the literal as its default"


def test_internal_impls_share_the_same_object():
    """The private engine defaults come from the same definition."""
    from ftmwpipeline._internal import stage6_impl

    assert stage6_impl.REFIT_SNAP_TOL_MHZ is REFIT_SNAP_TOL_MHZ
    for name in (
        "refit_window_impl",
        "create_window_impl",
        "merge_peaks_impl",
        "split_peak_impl",
        "review_accept_impl",
        "_execute_curation_batch",
    ):
        params = inspect.signature(getattr(stage6_impl, name)).parameters
        assert params["snap_tol_mhz"].default is REFIT_SNAP_TOL_MHZ, name


def test_constant_is_exported_at_top_level():
    """The import BlackQuill (and any integrator) is told to use."""
    assert ftmwpipeline.REFIT_SNAP_TOL_MHZ is REFIT_SNAP_TOL_MHZ
    assert "REFIT_SNAP_TOL_MHZ" in ftmwpipeline.__all__


_CLI_WIRING = {
    "edit": (
        ["review", "edit", "f.ftmw", "--window", "0", "--add", "1.0"],
        "refit_window_impl",
    ),
    "create": (["review", "create", "f.ftmw", "--at", "26622.0"], "create_window_impl"),
    "merge": (
        [
            "review",
            "merge",
            "f.ftmw",
            "--window",
            "0",
            "--peaks",
            "1.0",
            "--peaks",
            "2.0",
        ],
        "merge_peaks_impl",
    ),
    "split": (
        ["review", "split", "f.ftmw", "--window", "0", "--peak", "1.0"],
        "split_peak_impl",
    ),
    "accept": (
        ["review", "accept", "f.ftmw", "--window", "0", "--candidate", "1.0"],
        "review_accept_impl",
    ),
}


@pytest.mark.parametrize("verb", sorted(_CLI_WIRING))
def test_cli_flag_reaches_the_impl(verb, monkeypatch):
    """A parsed flag is worth nothing if the handler drops it on the floor.

    The parser default and the impl default agree by construction, so a
    handler that simply never forwarded the value would look correct in every
    default-path test. Passing a value that is *not* the default is what
    catches that.
    """
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


@pytest.mark.parametrize("verb", ["edit", "create", "merge", "split", "accept"])
def test_cli_flag_defaults_to_the_constant(verb):
    """The third interface carries the knob too, at the same default."""
    from ftmwpipeline.cli.main import create_parser

    args = create_parser().parse_args(
        {
            "edit": ["review", "edit", "f.ftmw", "--window", "0"],
            "create": ["review", "create", "f.ftmw", "--at", "26622.0"],
            "merge": [
                "review",
                "merge",
                "f.ftmw",
                "--window",
                "0",
                "--peaks",
                "1.0",
                "--peaks",
                "2.0",
            ],
            "split": ["review", "split", "f.ftmw", "--window", "0", "--peak", "1.0"],
            "accept": ["review", "accept", "f.ftmw", "--window", "0"],
        }[verb]
    )
    assert args.snap_tol_mhz is REFIT_SNAP_TOL_MHZ
