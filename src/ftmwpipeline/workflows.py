"""
Convenience workflow functions for common FTMW processing tasks.

These are thin wrappers over the file-bound :class:`~ftmwpipeline.pipeline.Pipeline`
interface. They exist purely for ergonomics - they create a ``.ftmw`` pipeline
file from a raw data source and run it through the implemented stages
(import -> FT -> noise estimation). They contain no analysis logic of their
own; all behavior is delegated to ``Pipeline`` so the result is identical to
driving the pipeline directly or via the CLI.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

from .pipeline import Pipeline

logger = logging.getLogger(__name__)


def _default_ftmw_path(
    source: Union[str, Path], output_dir: Optional[Union[str, Path]]
) -> Path:
    """Derive a ``.ftmw`` file path from a data source name."""
    stem = Path(source).stem or Path(source).name
    directory = Path(output_dir) if output_dir is not None else Path.cwd()
    return directory / f"{stem}.ftmw"


def process_experiment(
    source: Union[str, Path],
    ftmw_file: Optional[Union[str, Path]] = None,
    *,
    format_name: Optional[str] = None,
    fid_index: Optional[int] = None,
    ft_params: Optional[Dict[str, Any]] = None,
    estimate_noise: bool = True,
    noise_params: Optional[Dict[str, Any]] = None,
    force: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Process a single FTMW experiment through the implemented pipeline stages.

    Creates a ``.ftmw`` pipeline file from ``source`` and runs Stage 0 (import),
    Stage 1 (FT), and optionally Stage 2 (noise estimation).

    Parameters
    ----------
    source : str or Path
        Path to the raw experimental data (file or directory).
    ftmw_file : str or Path, optional
        Destination ``.ftmw`` file. If omitted, derived from the source name
        (in ``output_dir`` if given, else the current directory).
    format_name : str, optional
        Data format name; auto-detected if omitted.
    fid_index : int, optional
        FID index for multi-FID formats (e.g. BlackChirp).
    ft_params : dict, optional
        Keyword arguments forwarded to :meth:`Pipeline.compute_ft`
        (e.g. ``{'zpf': 2, 'expf_us': 5.0, 'trim': (26500, 40000)}``).
    estimate_noise : bool, default True
        Whether to run Stage 2 noise estimation.
    noise_params : dict, optional
        Keyword arguments forwarded to :meth:`Pipeline.estimate_noise`.
    force : bool, default False
        Overwrite an existing pipeline file created from a different source.
    output_dir : str or Path, optional
        Directory for the derived ``.ftmw`` file when ``ftmw_file`` is omitted.

    Returns
    -------
    dict
        ``{'source', 'pipeline_file', 'status', 'completed_stages', 'info'}``.
    """
    if ftmw_file is None:
        ftmw_file = _default_ftmw_path(source, output_dir)

    logger.info("Processing experiment %s -> %s", source, ftmw_file)

    pipe = Pipeline.create(
        ftmw_file,
        source=source,
        format_name=format_name,
        fid_index=fid_index,
        force=force,
    )
    pipe.compute_ft(**(ft_params or {}))
    if estimate_noise:
        pipe.estimate_noise(**(noise_params or {}))

    info = pipe.info()
    return {
        "source": str(source),
        "pipeline_file": str(pipe.filepath),
        "status": "success",
        "completed_stages": info.get("completed_stages", []),
        "info": info,
    }


def batch_process_experiments(
    sources: List[Union[str, Path]],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Process multiple FTMW experiments sequentially.

    Each source is processed via :func:`process_experiment`. A failure on one
    experiment is recorded and does not abort the batch.

    Parameters
    ----------
    sources : list of str or Path
        Raw data sources to process.
    output_dir : str or Path, optional
        Directory for derived ``.ftmw`` files.
    **kwargs
        Forwarded to :func:`process_experiment`.

    Returns
    -------
    list of dict
        One result dict per source; failed entries have
        ``{'status': 'error', 'error': ...}``.
    """
    results: List[Dict[str, Any]] = []
    for source in sources:
        try:
            results.append(process_experiment(source, output_dir=output_dir, **kwargs))
        except Exception as e:  # noqa: BLE001 - report per-experiment, keep going
            logger.error("Failed to process %s: %s", source, e)
            results.append({"source": str(source), "status": "error", "error": str(e)})
    return results


def validate_installation() -> Dict[str, bool]:
    """
    Validate that the ftmwpipeline installation is working correctly.

    Returns
    -------
    dict
        Maps component name to a boolean indicating whether it is functional.
    """
    validation_results = {
        "core_imports": False,
        "dependencies": False,
        "test_data": False,
        "pipeline_creation": False,
    }

    from importlib.util import find_spec

    try:
        from . import core, preprocessing, peak_detection

        # Reference the modules so a successful import is what we assert.
        validation_results["core_imports"] = bool(
            core and preprocessing and peak_detection
        )
    except ImportError:
        pass

    validation_results["dependencies"] = all(
        find_spec(pkg) is not None for pkg in ("numpy", "scipy", "matplotlib")
    )

    try:
        from importlib.resources import files

        example = Path(str(files("ftmwpipeline"))).parent.parent / (
            "examples/blackchirp_data/2638"
        )
        validation_results["test_data"] = example.exists()
    except Exception:
        pass

    # The pipeline is file-bound: construction happens through the
    # create()/open() factory classmethods, not a bare constructor.
    validation_results["pipeline_creation"] = hasattr(Pipeline, "create") and hasattr(
        Pipeline, "open"
    )

    return validation_results
