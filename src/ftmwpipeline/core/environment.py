"""
The analysis environment record: what produced each persisted stage.

A ``.ftmw`` claims to be reproducible -- "given the file and a compatible
package version, the analysis reproduces the same result on any machine". That
claim needs two things the file did not previously carry: a record of *which*
environment produced each artifact, and a declared notion of what "compatible"
means. This module owns both.

Why per stage rather than per file
----------------------------------
The pipeline is sequential and stateful: each stage persists an artifact that
later stages consume, and stages are run at different times -- often days apart,
across an upgrade. A single file-level stamp cannot express the situation that
actually matters, which is a file whose Stage 5 fit was produced by one version
and whose Stage 6 curation by another. So the environment is stamped per stage,
and a file whose stages disagree is detectable and reportable as such.

Stage 1 is deliberately absent from the record: it persists no artifact (the FT
is recomputed on demand from the persisted settings), so it is always the
current environment by construction.

What "compatible" means
-----------------------
Not the package version. At ``0.x`` the minor version bumps for ordinary
feature work, so keying compatibility on it would fire constantly on changes
that alter nothing numerically -- and keying on the major would never fire at
all. Instead :data:`ANALYSIS_EPOCH` is a separate integer, bumped by hand only
when a change alters numerical output. That mirrors what
``FTMW_FORMAT_VERSION`` already does for the container: a version whose meaning
this project controls, rather than one inherited from release cadence. The
package version stays in the record for humans; the epoch is what the machine
tests.

Interpreter and library versions (Python, numpy, scipy, h5py) and the BLAS
vendor are recorded but are **advisory only** -- they can shift the last bits of
a fit, but they are rarely under the user's independent control, and a
conclusion that depends on them is already fragile. They are never gated on.
"""

from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ANALYSIS_EPOCH",
    "EnvironmentRecord",
    "capture_environment",
    "describe_environment_drift",
    "gating_fields_differ",
]


ANALYSIS_EPOCH = 2
"""Declared analysis-compatibility epoch of this package.

Bump this **only** when a change alters the numerical output of a stage --
a different fitting model, a changed gate, a corrected estimator. Do not bump
it for refactors, new commands, documentation, or additive fields that leave
existing results unchanged.

It is the one machine-checkable compatibility signal: an operation that splices
a newly-computed result into an artifact produced under a different epoch is
refused (see
:func:`ftmwpipeline._internal.stage6_impl.require_splice_compatible_environment`),
because that would leave two different models inside one product. Everything
else about the environment is advisory.

Epoch history
-------------
1
    Initial epoch. Assigned when environment recording was introduced; files
    written before that carry no stamp and are treated as being of unknown
    epoch, which never blocks (see :func:`gating_fields_differ`).
2
    0.1.0b4. :func:`~ftmwpipeline.fitting.validation.feature_fwhm` solves for
    the half-maximum crossing of ``|h_T|`` instead of measuring it on a fixed
    200001-point grid spanning +/-1 MHz in absolute frequency. The width is
    exactly ``W(tau/T, shape) / T`` -- the model has only two length scales and
    frequency enters solely as ``f*T`` -- so the old grid was quantizing a
    quantity that does not depend on ``T`` with a step that does. Widths move by
    about 1e-4 relative, which can flip a peak-separation decision at a
    boundary, so fitted output is not bit-identical to epoch 1. The old grid
    also clipped silently: it returned its own 2 MHz width once the true FWHM
    outran it, below ``T = W/2`` (0.92 us for a Lorentzian at ``tau/T = 0.3``).
"""


@dataclass(frozen=True)
class EnvironmentRecord:
    """The analysis environment that produced one persisted stage.

    Attributes
    ----------
    ftmwpipeline : str
        Package version string (e.g. ``"0.1.0b3"``). For humans; not gated on.
    analysis_epoch : int or None
        :data:`ANALYSIS_EPOCH` at write time. ``None`` on a stage written
        before environment recording existed -- an unknown epoch, which is
        treated as compatible rather than assumed incompatible.
    python : str
        Interpreter version (e.g. ``"3.11.15"``).
    numpy, scipy, h5py : str
        Resolved versions of the numerical and container libraries. The project
        pins only floors (``numpy>=1.21``, ``scipy>=1.7``), so the actual
        versions span a wide range and are worth recording precisely.
    blas : str
        Runtime BLAS/LAPACK vendor and thread count (e.g.
        ``"openblas (32 threads)"``). Descriptive only, and deliberately never
        compared: it is the actual source of last-bit differences in a fit, but
        thread count legitimately varies from machine to machine, so treating a
        change as drift would cry wolf on every move between hosts.
    platform : str
        Short OS/architecture tag, for the record.
    """

    ftmwpipeline: str = ""
    analysis_epoch: Optional[int] = None
    python: str = ""
    numpy: str = ""
    scipy: str = ""
    h5py: str = ""
    blas: str = ""
    platform: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ftmwpipeline": self.ftmwpipeline,
            "analysis_epoch": self.analysis_epoch,
            "python": self.python,
            "numpy": self.numpy,
            "scipy": self.scipy,
            "h5py": self.h5py,
            "blas": self.blas,
            "platform": self.platform,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EnvironmentRecord":
        epoch = d.get("analysis_epoch")
        return cls(
            ftmwpipeline=str(d.get("ftmwpipeline", "")),
            analysis_epoch=None if epoch is None else int(epoch),
            python=str(d.get("python", "")),
            numpy=str(d.get("numpy", "")),
            scipy=str(d.get("scipy", "")),
            h5py=str(d.get("h5py", "")),
            blas=str(d.get("blas", "")),
            platform=str(d.get("platform", "")),
        )

    def summary(self) -> str:
        """One-line human summary (the form ``info`` and the reports print)."""
        epoch = "?" if self.analysis_epoch is None else str(self.analysis_epoch)
        return (
            f"ftmwpipeline {self.ftmwpipeline or '?'} (epoch {epoch}), "
            f"python {self.python or '?'}, numpy {self.numpy or '?'}, "
            f"scipy {self.scipy or '?'}"
        )


# Fields whose disagreement counts as environment drift worth telling the user
# about. ``blas`` and ``platform`` are excluded on purpose: they vary with the
# machine rather than the analysis, so comparing them would report drift every
# time a file moves between hosts. ``analysis_epoch`` is compared separately --
# it is the only field that can *block* anything.
_ADVISORY_FIELDS = ("ftmwpipeline", "python", "numpy", "scipy", "h5py")


def _blas_description() -> str:
    """Runtime BLAS vendor and thread count, or a best-effort fallback.

    ``threadpoolctl`` is a hard dependency (the Stage 5 fork pool needs it to
    re-limit an already-initialized BLAS), so the runtime query is the primary
    path. The build-time numpy config is the fallback, and its shape differs
    across the supported numpy range (``numpy.__config__.CONFIG`` on 2.x, the
    ``blas_opt_info`` dict on 1.x). Never raises: an unidentifiable BLAS is
    recorded as unknown rather than failing a stage write.
    """
    try:
        from threadpoolctl import threadpool_info

        for info in threadpool_info():
            if info.get("user_api") == "blas" or info.get("internal_api") in (
                "openblas",
                "mkl",
                "blis",
            ):
                vendor = str(info.get("internal_api", "unknown"))
                version = str(info.get("version") or "").strip()
                threads = info.get("num_threads")
                label = f"{vendor} {version}".strip()
                if threads:
                    label = f"{label} ({threads} threads)"
                return label
    except Exception:  # pragma: no cover - diagnostic only, never fatal
        pass

    try:
        import numpy as np

        cfg = getattr(np.__config__, "CONFIG", None)
        if isinstance(cfg, dict):
            blas = cfg.get("Build Dependencies", {}).get("blas", {})
            name = str(blas.get("name", "")).strip()
            version = str(blas.get("version", "")).strip()
            if name:
                return f"{name} {version}".strip()
        legacy = getattr(np.__config__, "blas_opt_info", None)
        if isinstance(legacy, dict):
            libs = legacy.get("libraries") or []
            if libs:
                return str(libs[0])
    except Exception:  # pragma: no cover - diagnostic only, never fatal
        pass

    return "unknown"


def capture_environment() -> EnvironmentRecord:
    """Snapshot the environment currently running the analysis.

    Best-effort and total: any library whose version cannot be determined is
    recorded as an empty string rather than raising, because failing to
    *describe* an environment must never fail the stage write it describes.
    """
    from .. import __version__

    def _ver(module_name: str) -> str:
        try:
            module = __import__(module_name)
            return str(getattr(module, "__version__", ""))
        except Exception:  # pragma: no cover - a missing hard dep is fatal
            return ""

    return EnvironmentRecord(
        ftmwpipeline=str(__version__),
        analysis_epoch=ANALYSIS_EPOCH,
        python=platform.python_version(),
        numpy=_ver("numpy"),
        scipy=_ver("scipy"),
        h5py=_ver("h5py"),
        blas=_blas_description(),
        platform=f"{platform.system().lower()}-{platform.machine()}",
    )


def gating_fields_differ(
    a: Optional[EnvironmentRecord], b: Optional[EnvironmentRecord]
) -> bool:
    """Whether two records disagree on the one field that can block an operation.

    Only :attr:`EnvironmentRecord.analysis_epoch` is gated on. An unknown epoch
    on either side (``None`` -- a stage written before environment recording,
    or a record that failed to load) is treated as **compatible**: refusing to
    work on a legacy file because it predates the stamp would punish users for
    an upgrade they did not choose, and the honest statement about such a file
    is "unknown", not "incompatible".
    """
    if a is None or b is None:
        return False
    if a.analysis_epoch is None or b.analysis_epoch is None:
        return False
    return int(a.analysis_epoch) != int(b.analysis_epoch)


def describe_environment_drift(
    records: Mapping[str, EnvironmentRecord],
    current: Optional[EnvironmentRecord] = None,
) -> List[str]:
    """Human-readable lines describing disagreement across *records*.

    Compares the per-stage records against each other (and against *current*
    when given), returning one line per field that disagrees. An empty list
    means the file's stages -- and the running environment, if supplied -- all
    agree on everything that matters.

    This is what makes a mixed-version file visible: the diagnostic a consumer
    actually needs is not "which version wrote this file" but "were these
    artifacts produced by the same code".
    """
    pool: Dict[str, EnvironmentRecord] = dict(records)
    if current is not None:
        pool = {**pool, "(current)": current}
    if len(pool) < 2:
        return []

    lines: List[str] = []
    for field_name in ("analysis_epoch",) + _ADVISORY_FIELDS:
        by_value: Dict[str, List[str]] = {}
        for stage, rec in pool.items():
            value = getattr(rec, field_name, None)
            if value in (None, ""):
                continue
            by_value.setdefault(str(value), []).append(stage)
        if len(by_value) > 1:
            parts = ", ".join(
                f"{value} ({', '.join(sorted(stages))})"
                for value, stages in sorted(by_value.items())
            )
            lines.append(f"{field_name}: {parts}")
    return lines
