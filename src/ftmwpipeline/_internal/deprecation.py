"""Deprecation-warning helpers for the legacy per-knob kwarg surface.

The five settings dataclasses (Stages 2, 2b, 3, 4, 5) keep their legacy
per-knob kwargs alive as a back-compat shim so existing call sites do
not break when the resolver lands. ``warn_legacy_kwargs`` surfaces any
remaining legacy-form call sites at dev time so they can be migrated
to ``settings=`` / ``preset=`` before the surrounding work ships.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping


def warn_legacy_kwargs(
    *,
    func_name: str,
    legacy_kwargs: Mapping[str, Any],
    migration_hint: str,
    stacklevel: int = 3,
) -> None:
    """Emit one ``DeprecationWarning`` per call when any legacy kwarg is set.

    Parameters
    ----------
    func_name : str
        Public function name to embed in the warning message
        (e.g. ``"estimate_noise"``).
    legacy_kwargs : Mapping[str, Any]
        ``{kwarg_name: supplied_value}`` for every legacy per-knob kwarg
        on the function. Values of ``None`` count as "not supplied" and
        do not trigger the warning. The warning lists the names of every
        non-``None`` kwarg in one message — one warning per call, not
        one per kwarg.
    migration_hint : str
        Short sentence telling the caller how to migrate
        (e.g. ``"use settings=NoiseSettings(...) or preset='name'"``).
    stacklevel : int, default 3
        Forwarded to :func:`warnings.warn`. Default ``3`` skips the
        helper frame and the impl frame, surfacing the warning at the
        pipeline / api / cli layer that called the impl.
    """
    used = sorted(name for name, value in legacy_kwargs.items() if value is not None)
    if not used:
        return
    msg = (
        f"{func_name}: legacy per-knob kwargs {used} are deprecated; "
        f"{migration_hint}"
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)


def warn_legacy_flag(
    *,
    func_name: str,
    flag_name: str,
    migration_hint: str,
    stacklevel: int = 3,
) -> None:
    """Emit a ``DeprecationWarning`` for a single legacy boolean flag.

    Companion to :func:`warn_legacy_kwargs` for shim flags that are not
    per-knob kwargs (e.g. ``from_saved_params=True`` on
    ``estimate_noise``). Caller checks the flag is truthy before
    invoking this helper.
    """
    msg = (
        f"{func_name}: legacy flag {flag_name!r} is deprecated; "
        f"{migration_hint}"
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)
