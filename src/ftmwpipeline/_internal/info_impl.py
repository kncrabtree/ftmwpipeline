"""The analysis-environment fields of the ``get_pipeline_info`` payload.

Normative spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Missing values. The
validation report keeps its storage forms (``None``, ``{}``, ``[]``);
:func:`environment_info_fields` converts them to
:class:`~ftmwpipeline.contract.Absent` once, here, for ``Pipeline.info`` (and so
the functional API and the CLI).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ..contract import Absent

__all__ = ["environment_info_fields"]

#: The keys of ``get_pipeline_info`` that come from the file's environment stamps.
_FILE_DERIVED = (
    "format_version",
    "created_with",
    "stage_environments",
    "last_written_with",
    "environment_drift",
    "runtime_environment_drift",
    "environment_acknowledged",
)


def _environment_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """One environment record with each uncaptured (``None`` / ``''``) field ``NOT_RUN``."""
    return {
        key: Absent.NOT_RUN if value is None or value == "" else value
        for key, value in record.items()
    }


def environment_info_fields(report: Mapping[str, Any]) -> Dict[str, Any]:
    """The environment keys of the info payload, from a validation report.

    * ``format_version`` / ``created_with`` / ``last_written_with``: ``NOT_RUN``
      when the file carries no stamp.
    * ``stage_environments``: ``NOT_RUN`` when no stage was stamped (a file that
      predates the record); otherwise the per-stage records.
    * ``environment_drift`` / ``runtime_environment_drift``: ``NOT_RUN`` when
      there are no stamps to compare; otherwise a list (empty means no drift).
    * ``analysis_epoch`` and every other field an environment record did not
      capture: ``NOT_RUN``.
    * When validation could not read the stamps at all (the report carries none
      of them), every file-derived key is ``UNDEFINED``: the read was
      attempted and failed. ``current_environment`` does not depend on the file
      and is always reported.
    """
    from ..core.environment import capture_environment

    fields: Dict[str, Any]
    if "stage_environments" not in report:
        fields = {key: Absent.UNDEFINED for key in _FILE_DERIVED}
        current = report.get("current_environment")
        if current is None:
            current = capture_environment().to_dict()
        fields["current_environment"] = _environment_record(current)
        return fields

    envs = report["stage_environments"]
    stamped = bool(envs)

    def _stamp(value: Any) -> Any:
        return Absent.NOT_RUN if value is None else value

    last = report.get("last_written_with")
    current = report.get("current_environment")
    if current is None:
        current = capture_environment().to_dict()
    return {
        "format_version": _stamp(report.get("format_version")),
        "created_with": _stamp(report.get("created_with")),
        "stage_environments": (
            {k: _environment_record(v) for k, v in envs.items()}
            if stamped
            else Absent.NOT_RUN
        ),
        "last_written_with": (
            Absent.NOT_RUN if last is None else _environment_record(last)
        ),
        "environment_drift": (
            list(report.get("environment_drift") or []) if stamped else Absent.NOT_RUN
        ),
        "runtime_environment_drift": (
            list(report.get("runtime_environment_drift") or [])
            if stamped
            else Absent.NOT_RUN
        ),
        "current_environment": _environment_record(current),
        "environment_acknowledged": bool(report.get("environment_acknowledged", False)),
    }
