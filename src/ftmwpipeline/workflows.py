"""
Installation validation for ftmwpipeline.

The whole-experiment workflow lives in the pipeline core: drive a raw source
through every stage with :func:`ftmwpipeline.api.run_pipeline` or
:meth:`ftmwpipeline.pipeline.Pipeline.build` (or the ``run`` CLI verb). This
module holds only :func:`validate_installation`.
"""

from pathlib import Path
from typing import Dict

from .pipeline import Pipeline


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
        from . import core, preprocessing

        # Reference the modules so a successful import is what we assert.
        validation_results["core_imports"] = bool(core and preprocessing)
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
