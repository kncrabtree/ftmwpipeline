"""Source preview: what a data source holds, found without importing it.

Backs the ``preview_source`` machine-contract accessor (spec
``dev-docs/CONTRACT_STRATEGY.md`` §Source preview). The loaders describe
themselves (:meth:`~ftmwpipeline.io.data_loaders.base.BaseLoader.preview_fids`);
this module only shapes their answer into the ``ftmw/source_preview@1``
payload. It never writes anything and never creates a ``.ftmw`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..contract import SOURCE_PREVIEW_SCHEMA, Absent
from ..io.data_loaders import detect_format
from ..io.data_loaders import preview_source as _registry_preview
from ..io.data_loaders.base import LoaderError
from .stage0_impl import loader_refusal

_CHIRP_KEYS = ("chirp_start_us", "chirp_end_us", "start_margin_us")


def preview_source_impl(
    source: Union[str, Path], format_name: Optional[str] = None
) -> Dict[str, Any]:
    """Build the ``ftmw/source_preview@1`` payload for ``source``.

    ``fids`` is the FID table as a list of
    :class:`~ftmwpipeline.contract.FidPreviewRow` records, one per FID.

    Raises
    ------
    PipelineFileNotFoundError
        ``source`` does not exist.
    BadSettingError
        ``path`` ``"format"``: unknown ``format_name``, or none detected.
        ``path`` ``"source"``: the source exists but the format's loader
        refuses it (the loader's message is kept). Also a ``ValueError``.
    """
    try:
        preview = _registry_preview(source, format_name)
    except LoaderError as exc:
        resolved = format_name if format_name is not None else detect_format(source)
        raise loader_refusal(source, resolved, exc) from exc

    window: Union[Dict[str, Any], Absent]
    if preview.chirp_window is None:
        window = Absent.NOT_RUN
    elif isinstance(preview.chirp_window, Absent):
        window = preview.chirp_window  # declared but unreadable: UNDEFINED
    else:
        declared = preview.chirp_window
        window = {key: declared.get(key, Absent.NOT_RUN) for key in _CHIRP_KEYS}

    return {
        "schema": SOURCE_PREVIEW_SCHEMA,
        "source": str(source),
        "format": preview.format,
        "n_fids": len(preview.fids),
        "fids": list(preview.fids),
        "chirp_window": window,
    }
