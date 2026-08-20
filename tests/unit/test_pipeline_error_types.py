"""``Pipeline`` methods let typed errors propagate; only the unexpected is wrapped.

Eighteen ``Pipeline`` methods used to catch ``Exception`` broadly and re-raise
everything -- a user's own ``ValueError`` from a stage's input validation, the
whole ``PipelineFileError`` family (including ``AnalysisEpochMismatchError``,
which exists precisely so a caller can route on its type), and a genuine
internal failure -- as one flat ``RuntimeError``. A caller could no longer
distinguish "you asked for something invalid" from "something broke inside".

These tests pin the new contract for a representative sample of the eighteen
sites (``Pipeline.compute_ft`` stands in for all of them, since they share one
idiom): a ``ValueError`` from a stage's own validation reaches the caller as a
``ValueError``; a ``PipelineFileError`` subclass (``AnalysisEpochMismatchError``)
reaches the caller with its own type intact; and a genuinely unexpected
exception is still wrapped in ``RuntimeError``, with the ``from e`` chain
preserved so nothing is lost.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import ftmwpipeline.pipeline as pipeline_module
from ftmwpipeline.core.environment import EnvironmentRecord
from ftmwpipeline.file_manager import AnalysisEpochMismatchError, PipelineFileError
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.unit]

_DT_US = 0.002
_N_TOTAL = 6325  # 12.65 us record, matching the P0 refusal reproduction


@pytest.fixture
def pipeline(tmp_path) -> Pipeline:
    """A ``Pipeline`` bound to a bare imported ``.ftmw`` of known duration."""
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = _DT_US
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=np.cos(2 * np.pi * np.arange(_N_TOTAL) * 0.01))
    out = tmp_path / "exp.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    return Pipeline.open(out)


def test_value_error_from_stage_validation_reaches_the_caller_typed(pipeline):
    """The motivating case: an end_us past the record raises ValueError, not
    a RuntimeError with the type flattened away."""
    with pytest.raises(ValueError, match="past the end of the recording"):
        pipeline.compute_ft(start_us=0.0, end_us=100.0)


def test_value_error_is_not_wrapped_in_runtime_error(pipeline):
    """The type must actually change, not just the message survive."""
    try:
        pipeline.compute_ft(start_us=0.0, end_us=100.0)
    except RuntimeError:
        pytest.fail(
            "compute_ft still wraps a ValueError in RuntimeError; the type "
            "was supposed to propagate unwrapped."
        )
    except ValueError:
        return
    pytest.fail("compute_ft did not raise at all; the refusal is gone.")


def test_analysis_epoch_mismatch_error_propagates_with_its_own_type(
    pipeline, monkeypatch
):
    """AnalysisEpochMismatchError -- a PipelineFileError and a ValueError --
    must reach the caller as itself, not as a bare RuntimeError."""

    def _raise(**kwargs: object) -> None:
        raise AnalysisEpochMismatchError(
            pipeline.filepath,
            EnvironmentRecord(analysis_epoch=1),
            EnvironmentRecord(analysis_epoch=2),
        )

    monkeypatch.setattr(pipeline_module, "compute_ft_impl", _raise)

    with pytest.raises(AnalysisEpochMismatchError):
        pipeline.compute_ft()

    # And it is still usable as the two ancestor types a caller may be
    # routing on instead.
    monkeypatch.setattr(pipeline_module, "compute_ft_impl", _raise)
    with pytest.raises(PipelineFileError):
        pipeline.compute_ft()
    monkeypatch.setattr(pipeline_module, "compute_ft_impl", _raise)
    with pytest.raises(ValueError):
        pipeline.compute_ft()


def test_unexpected_exception_is_still_wrapped_in_runtime_error(pipeline, monkeypatch):
    """A genuinely unexpected exception keeps the RuntimeError wrap, and the
    `from e` chain means the original is not lost."""

    class _UnexpectedInternalFailure(Exception):
        pass

    def _raise(**kwargs: object) -> None:
        raise _UnexpectedInternalFailure("boom")

    monkeypatch.setattr(pipeline_module, "compute_ft_impl", _raise)

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.compute_ft()

    assert "Failed to compute FT" in str(excinfo.value)
    assert "boom" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, _UnexpectedInternalFailure)
