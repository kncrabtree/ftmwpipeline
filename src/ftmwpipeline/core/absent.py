"""The contract's missing-value marker, defined where every layer can import it.

:class:`Absent` and the columnar status codes are published as
:mod:`ftmwpipeline.contract` names; they live in this dependency-free module so
the core data structures (which :mod:`ftmwpipeline.contract` itself imports)
can use ``Absent`` as a field default without an import cycle. Import them from
:mod:`ftmwpipeline.contract`.
"""

from __future__ import annotations

import enum

#: Columnar status codes (``<column>__status``, dtype ``uint8``).
STATUS_PRESENT: int = 0
STATUS_NOT_RUN: int = 1
STATUS_UNDEFINED: int = 2


class Absent(enum.Enum):
    """Why a contract field has no value.

    ``NOT_RUN``
        The stage or quantity does not exist in this file (never computed,
        never tested).
    ``UNDEFINED``
        Computed, but the quantity has no value (e.g. chi2_r with zero degrees
        of freedom, a failed K-1 refit).

    The member value is the wire spelling used in the ``"<field>_absent"``
    sibling key; :attr:`status` is the code in a ``<column>__status`` column.
    Compare by identity (``x is Absent.NOT_RUN``). Members are deliberately not
    falsy-special and not ``None``-like: test for them explicitly.
    """

    NOT_RUN = "not_run"
    UNDEFINED = "undefined"

    @property
    def status(self) -> int:
        """The ``uint8`` columnar status code (``1`` not run, ``2`` undefined)."""
        return STATUS_NOT_RUN if self is Absent.NOT_RUN else STATUS_UNDEFINED

    @classmethod
    def from_status(cls, status: int) -> "Absent":
        """The member a non-zero columnar status code stands for.

        Raises
        ------
        ValueError
            For ``0`` (present) or any code that is not a status code.
        """
        if status == STATUS_NOT_RUN:
            return cls.NOT_RUN
        if status == STATUS_UNDEFINED:
            return cls.UNDEFINED
        raise ValueError(f"{status!r} is not an absence status code")


# Published as ``ftmwpipeline.contract.Absent``; say so in reprs and docs.
Absent.__module__ = "ftmwpipeline.contract"

__all__ = ["Absent", "STATUS_PRESENT", "STATUS_NOT_RUN", "STATUS_UNDEFINED"]
