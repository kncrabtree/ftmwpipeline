"""Source preview: what a data source holds, found without importing it.

Backs the ``preview_source`` machine-contract accessor (spec
``dev-docs/CONTRACT_STRATEGY.md`` §Source preview). The loaders describe
themselves (:meth:`~ftmwpipeline.io.data_loaders.base.BaseLoader.preview_fids`);
this module only shapes their answer into the ``ftmw/source_preview@1``
payload. It never writes anything and never creates a ``.ftmw`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..contract import Absent
from ..io.data_loaders import preview_source as _registry_preview
from ..io.data_loaders.base import LoaderError
from ..serialize import with_status_columns

SOURCE_PREVIEW_SCHEMA = "ftmw/source_preview@1"

#: FID-table columns that can be absent (each gains a ``<column>__status``).
ABSENT_CAPABLE = ("n_points", "spacing_us", "probe_freq_mhz", "sideband", "shots")

_CHIRP_KEYS = ("chirp_start_us", "chirp_end_us", "start_margin_us")


def preview_source_impl(
    source: Union[str, Path], format_name: Optional[str] = None
) -> Dict[str, Any]:
    """Build the ``ftmw/source_preview@1`` payload for ``source``.

    Raises
    ------
    PipelineFileNotFoundError
        ``source`` does not exist.
    NotFoundError
        ``kind == "format"``: unknown ``format_name``, or none detected.
    ValueError
        The source exists but is not valid for the format (a loader refusal).
    """
    try:
        preview = _registry_preview(source, format_name)
    except LoaderError as exc:
        raise ValueError(str(exc)) from exc

    rows = preview.fids
    columns = with_status_columns(
        {
            "index": [r.index for r in rows],
            "n_points": [r.n_points for r in rows],
            "spacing_us": [r.spacing_us for r in rows],
            "probe_freq_mhz": [r.probe_freq_mhz for r in rows],
            "sideband": [r.sideband for r in rows],
            "shots": [r.shots for r in rows],
        },
        absent_capable=ABSENT_CAPABLE,
        fills={"n_points": 0, "shots": 0, "sideband": ""},
        dtypes={
            "index": "int64",
            "n_points": "int64",
            "shots": "int64",
            "sideband": "U5",
        },
    )
    # Small, human-sized table: plain lists, so the CLI needs no --output.
    fids: Dict[str, List[Any]] = {k: v.tolist() for k, v in columns.items()}

    window: Union[Dict[str, Any], Absent]
    if preview.chirp_window is None:
        window = Absent.NOT_RUN
    else:
        declared = preview.chirp_window
        window = {key: declared.get(key, Absent.NOT_RUN) for key in _CHIRP_KEYS}

    return {
        "schema": SOURCE_PREVIEW_SCHEMA,
        "source": str(source),
        "format": preview.format,
        "n_fids": len(rows),
        "fids": fids,
        "chirp_window": window,
    }
