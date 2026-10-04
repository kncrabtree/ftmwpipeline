"""Public tolerances for Stage 6 curation.

These are the values the pipeline itself pairs at, published so an external
tool can pair identically.  An integrator that resolved "the peak at *f*" at a
different tolerance would disagree with the file about *which* peak that is --
the disagreement this module exists to prevent -- so the constant is the single
definition every public verb's default is taken from, not a separately
maintained copy that happens to agree.

The snap tolerance is **defined in active-FT bins**, so the frequency it works
out to is a property of one file, not of this module
(``dev-docs/SCIENCE_STRATEGY.md`` Requirement 8).  Do not resolve
:data:`REFIT_SNAP_TOL_BINS` against a spacing you derived yourself: read the
MHz value the pipeline will actually pair at from
:func:`ftmwpipeline.api.refit_snap_tol_mhz`,
:meth:`ftmwpipeline.Pipeline.refit_snap_tol_mhz`, or ``ftmwpipeline review
snap-tolerance``, all of which derive it from the same active region the
curation verbs themselves consult.  One derivation, one answer.

This module is intentionally dependency-free within the package (stdlib only),
the same discipline :mod:`ftmwpipeline.core.settings` documents for itself, so
it can be imported anywhere without a cycle.  That is also why the *resolved*
value lives behind an accessor rather than here: resolving it means opening a
file.

It also holds :func:`parse_peak_token`, the ``uid:N`` addressing grammar every
``add``/``remove`` token (from any of the three interfaces, or a curation
file's ``freqs`` column) is read through -- published for the same reason as
the tolerance above: an external tool addressing a peak by identifier should
parse the same grammar the pipeline does, not reimplement the prefix.
"""

import math
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, Literal, Mapping, Optional, Union

__all__ = [
    "REFIT_SNAP_TOL_BINS",
    "CURATION_ACTION_SCHEMA",
    "CurationAction",
    "Frame",
    "PeakUidToken",
    "parse_peak_token",
]

Frame = Literal["raw", "calibrated"]
"""The frame a caller-supplied or caller-returned frequency is expressed in.

``"raw"`` is the frame the Stage 5 fit, the Stage 3 candidate ledger, and every
persisted value (the decision log, created windows) live in -- the frame
values are stored in, permanently, because ``epsilon`` is recomputable and a
stored calibrated value would silently change meaning after a timebase
re-run. ``"calibrated"`` is ``f_corr = probe + (f_raw - probe) / (1 + eps)``,
the frame the final-products table and the reports present.

Every caller-supplied frequency across the three interfaces (``add``/
``remove``, ``candidate_freq``, create anchor, a curation file's
frequencies, and each :class:`CurationAction`) takes a ``frame`` parameter of
this type. Its default is ``None``, which means ``"raw"`` -- matching the
undocumented behavior every caller already depends on -- except where noted
below. On an ``rb_locked``/``uncalibrated`` file (``epsilon == 0``) the
two frames coincide, so omitting ``frame`` is inert. On a ``self_calibrated``
file it is not: a calibrated candidate submitted as raw still resolves, and to
the *right* peak, but lands wrong by ``probe_freq * eps/(1+eps)`` -- under the
snap tolerance and over the statistical sigma, so the mistake is invisible in
the result. Omitting ``frame`` on a frequency-bearing call is therefore an
error on a ``self_calibrated`` file rather than a silent assumption; passing
``frame="raw"`` explicitly is never an error, on any file.
"""


REFIT_SNAP_TOL_BINS: float = 0.625
"""Tolerance for matching a requested frequency to an existing peak, as a
multiple of the active-FT bin spacing.

Every Stage 6 curation verb -- ``review edit`` / ``create`` / ``accept``, and
every action inside ``review apply`` -- resolves a requested ``add`` /
``remove`` / anchor / candidate frequency to the nearest fitted peak or ledger
candidate within this tolerance, and treats anything beyond it as a miss (a
hard error for ``remove``, a fresh seed at the requested frequency for
``add``). It is also the tolerance curation-intent inference reads an ``add``
against: within it of a fitted peak that is not itself being removed, the add
is applied as a split of that peak; removing >= 2 mutually-close peaks (within
this tolerance of each other) while adding one frequency in their span is
applied as a merge. ``merge``/``split`` are not verbs a caller types -- see
:func:`ftmwpipeline._internal.stage6_impl.parse_curation_file`.

**Read the resolved MHz value; do not multiply this by a spacing of your own.**
``0.625 / T_active`` is 49.4 kHz at the 12.65 us reference acquisition and
6.3 kHz at 100 us, so there is no such thing as "the" snap tolerance in MHz.
:func:`ftmwpipeline.api.refit_snap_tol_mhz` is the one read that cannot
disagree with what a ``review apply`` on that file will snap with.

Deliberately loose relative to the 0.25-bin candidate-dedup window
(``_DEDUP_TOL_BINS`` in :mod:`ftmwpipeline._internal.stage6_impl`), so two
fitted peaks can both fall inside it; that is resolved nearest-wins, and
``review apply --dry-run`` reports the multi-match as an advisory so the
ambiguity is visible before the batch runs.  Now that both are bin counts the
2.5:1 ratio between them holds at every acquisition length instead of drifting
with it, and the nearest-wins window shrinks as resolution improves.

**Bins, not MHz, and with no absolute floor** (decided 2026-08-19).  This is
the one constant in the Requirement 8 family that needs the decision spelled
out, because its previous 50 kHz absolute value was justified by *input*
precision -- "the input is a frequency a person typed off a plot or a line
list" -- and human typing does not get finer as the acquisition gets longer.
Overruled on three grounds: the plot the person reads has exactly this
resolution, so their reading precision does scale with it; an FTMW line list
is quoted to ~1 kHz, comfortably inside 6.3 kHz; and a miss is loud rather
than silently wrong (``remove`` raises and names the closest peak and its
distance; ``add`` seeds a fresh peak at the typed frequency, which is what a
genuine new line wants anyway).  A floor would also reinstate the
``max(absolute, bins)`` shape -- pinned at long acquisitions, spanning many
bins -- that Requirement 8 exists to remove.

Every public verb's ``snap_tol_mhz`` parameter defaults to the resolved value
for the file it is called on; a caller that passes its own overrides it for
that call only, in MHz.
"""

_UID_TOKEN_PREFIX = "uid:"


@dataclass(frozen=True)
class PeakUidToken:
    """A parsed ``uid:N`` addressing token.

    Names a peak by its
    :attr:`~ftmwpipeline.core.data_structures.FittedPeak.peak_uid` rather than
    by frequency. This class only carries the parsed integer -- resolving it
    to an actual peak (the fitted peak in one particular window whose
    ``peak_uid`` equals :attr:`uid`) is downstream of this module, since that
    requires the window's persisted fit, which this dependency-free module
    never opens. Frame-independent: :data:`Frame` has no bearing on a token
    that never carries a frequency of its own.
    """

    uid: int


def parse_peak_token(
    token: Union[float, str], *, path: str = "peak"
) -> Union[float, PeakUidToken]:
    """Parse one caller-supplied ``add``/``remove`` token into a frequency
    (MHz) or a :class:`PeakUidToken`.

    Grammar: a string prefixed ``"uid:"`` addresses a peak by identifier --
    ``N`` must be a non-negative integer, e.g. ``"uid:15425022"``. Anything
    else (a ``float``, or a ``str`` without that prefix) is a molecular
    frequency in MHz, exactly as every curation verb has always accepted; a
    plain numeric string (``"27549.3259"``) parses the same as the float
    ``27549.3259``, since the CLI passes strings anyway.

    Every public verb that resolves the result decides for itself whether a
    :class:`PeakUidToken` is acceptable there -- ``remove`` on ``review edit``
    and a curation file's ``remove`` row accept one; ``add`` and
    ``accept --candidate`` refuse one with their own message, since a uid
    names a peak that already exists and neither of those verbs addresses an
    existing peak.

    *path* names what the caller wrote, for the refusal's ``bad_setting``
    ``path``: the argument (``"add"``, ``"remove"``) or the curation-file
    cell (``"curation[line 3].freqs"``).

    Raises
    ------
    BadSettingError
        (``bad_setting``, a :class:`ValueError`, ``path`` *path*) When the
        token is prefixed ``"uid:"`` but what follows is not a non-negative
        integer (``"uid:"``, ``"uid:abc"``, ``"uid:-3"``, ``"uid:1.5"`` all
        raise), or when an un-prefixed token is not a valid MHz value. The
        offending token is always named in the message.
    """
    from ..file_manager import BadSettingError

    expected = 'a frequency in MHz or a peak identifier "uid:N" (N >= 0)'
    if isinstance(token, str):
        text = token.strip()
        if text.startswith(_UID_TOKEN_PREFIX):
            digits = text[len(_UID_TOKEN_PREFIX) :]
            if not digits:
                raise BadSettingError(
                    path,
                    expected,
                    token,
                    message=f"malformed peak identifier {token!r}: nothing follows "
                    f"{_UID_TOKEN_PREFIX!r}",
                )
            try:
                uid = int(digits)
            except ValueError:
                raise BadSettingError(
                    path,
                    expected,
                    token,
                    message=f"malformed peak identifier {token!r}: {digits!r} is "
                    f"not an integer",
                ) from None
            if uid < 0:
                raise BadSettingError(
                    path,
                    expected,
                    token,
                    message=f"malformed peak identifier {token!r}: uid must be "
                    f"non-negative, got {uid}",
                )
            return PeakUidToken(uid)
        try:
            return float(text)
        except ValueError:
            raise BadSettingError(
                path,
                expected,
                token,
                message=f"malformed frequency {token!r}: not a valid MHz value",
            ) from None
    return float(token)


CURATION_ACTION_SCHEMA = "ftmw/curation_action@1"
"""Schema name of :meth:`CurationAction.to_dict`'s wire form."""

_ACTION_KINDS = ("add", "remove", "accept", "create")
_FRAMES = ("raw", "calibrated")


def _bad(path: str, expected: str, value: Any, message: str) -> Exception:
    """A ``bad_setting`` refusal naming one :class:`CurationAction` field."""
    from ..file_manager import BadSettingError

    return BadSettingError(path, expected, value, message=message)


def _check_id(path: str, value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _bad(
            path,
            "a non-negative integer or null",
            value,
            f"{path} must be a non-negative integer or None, got {value!r}",
        )
    return int(value)


def _check_freq(path: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        finite = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )
    except OverflowError:  # an int too large for a float
        finite = False
    if not finite:
        raise _bad(
            path,
            "a finite number or null",
            value,
            f"{path} must be a finite number or None, got {value!r}",
        )
    return float(value)


@dataclass(frozen=True)
class CurationAction:
    """One Stage 6 curation action as data: a curation-file row, typed.

    ``review_apply`` / ``review_preview`` accept a sequence of these
    (``actions=``) in place of a curation-file path, with the same
    validation, frame handling and results: the same actions given as a file
    and as data give equal results, decision logs and files. A file's rows
    parse to the same actions (see
    :func:`ftmwpipeline._internal.stage6_impl.actions_from_curation_file`),
    and :meth:`to_row` writes one back.

    Attributes
    ----------
    action : {"add", "remove", "accept", "create"}
        The row action. ``merge`` and ``split`` are not actions: they are read
        from what an add/remove combination does to a window's peaks.
    window_id : int or None
        The window. ``None`` means "derive it from the frequency (or
        ``peak_uid``)" on ``add`` / ``remove`` -- a curation file's ``auto``
        -- and "a new window" on ``create`` (the file's ``new``); a named id
        on ``create`` pins the id the created window takes. Required on
        ``accept``.
    freq_mhz : float or None
        The action's one frequency: the peak to add or remove, or the anchor
        a ``create`` must cover. ``None`` on ``accept``, and on a ``remove``
        that names its peak by ``peak_uid``.
    peak_uid : int or None
        ``remove`` only: the peak to remove by identifier instead of by
        frequency (the file's ``uid:N``).
    candidate_mhz : float or None
        ``accept`` only: a Stage 3 ledger candidate to revive (the file's
        ``candidate=F``); ``None`` accepts the window as reviewed.
    frame : {"raw", "calibrated"} or None
        The frame of ``freq_mhz`` / ``candidate_mhz``. ``None`` takes the
        call's ``frame=``; if that is ``None`` too, a frequency is raw on a
        file whose ``epsilon`` is 0 and refused (``bad_setting``, ``path``
        ``"frame"``) on a ``self_calibrated`` file. Each action resolves its
        own frame, so one batch may mix frames; the pipeline converts every
        frequency to raw before resolving anything. Inert on an action
        without a frequency. An explicit ``frame`` that disagrees with an
        explicit call ``frame=`` is refused, as a file header that disagrees
        with it is.
    epsilon : float or None
        Only with ``frame="calibrated"``: the epsilon the calibrated
        frequency was computed under (a file's ``# epsilon:`` header). When
        given, it must match the file's current epsilon at apply time, else
        the action is refused as calibration drift, exactly as a stamped file
        is. ``None`` skips the check.

    Construction validates what the curation-file parser enforces of a row:
    one frequency (``freq_mhz``), or on ``remove`` one ``peak_uid`` in its
    place; no frequency on ``accept``; no ``peak_uid`` outside ``remove``; no
    ``candidate_mhz`` outside ``accept``; a ``window_id`` on ``accept``. A
    violation raises :class:`~ftmwpipeline.BadSettingError` whose ``path`` is
    the offending field.
    """

    __ftmw_schema__: ClassVar[str] = CURATION_ACTION_SCHEMA

    action: Literal["add", "remove", "accept", "create"]
    window_id: Optional[int] = None
    freq_mhz: Optional[float] = None
    peak_uid: Optional[int] = None
    candidate_mhz: Optional[float] = None
    frame: Optional[Frame] = None
    epsilon: Optional[float] = None

    def __post_init__(self) -> None:
        action: str = self.action
        if action not in _ACTION_KINDS:
            hint = ""
            if action in ("merge", "split"):
                hint = (
                    f"; {action!r} is not an action -- write add/remove actions, "
                    f"which are read as a {action} by what they do to the window"
                )
            raise _bad(
                "action",
                "one of: " + ", ".join(_ACTION_KINDS),
                action,
                f"unknown curation action {action!r}{hint}",
            )
        object.__setattr__(self, "window_id", _check_id("window_id", self.window_id))
        object.__setattr__(self, "peak_uid", _check_id("peak_uid", self.peak_uid))
        object.__setattr__(self, "freq_mhz", _check_freq("freq_mhz", self.freq_mhz))
        object.__setattr__(
            self, "candidate_mhz", _check_freq("candidate_mhz", self.candidate_mhz)
        )
        if self.frame is not None and self.frame not in _FRAMES:
            raise _bad(
                "frame",
                'one of: "raw", "calibrated", or null',
                self.frame,
                f'frame must be "raw", "calibrated" or None, got {self.frame!r}',
            )

        if self.epsilon is not None:
            eps = _check_freq("epsilon", self.epsilon)
            object.__setattr__(self, "epsilon", eps)
            if self.frame != "calibrated":
                raise _bad(
                    "epsilon",
                    'null unless frame is "calibrated"',
                    self.epsilon,
                    "epsilon stamps a calibrated frequency; it needs "
                    'frame="calibrated"',
                )

        if self.peak_uid is not None and action != "remove":
            raise _bad(
                "peak_uid",
                "null outside remove",
                self.peak_uid,
                f"{action} takes no peak_uid (a uid names a peak that already "
                f"exists; only remove addresses one)",
            )
        if self.candidate_mhz is not None and action != "accept":
            raise _bad(
                "candidate_mhz",
                "null outside accept",
                self.candidate_mhz,
                f"{action} takes no candidate_mhz (only accept revives a "
                f"ledger candidate)",
            )
        if action == "accept":
            if self.freq_mhz is not None:
                raise _bad(
                    "freq_mhz",
                    "null on accept",
                    self.freq_mhz,
                    "accept takes no frequency; use candidate_mhz to revive a "
                    "ledger candidate",
                )
            if self.window_id is None:
                raise _bad(
                    "window_id",
                    "a window id (required on accept)",
                    None,
                    "accept needs a window_id",
                )
        elif action == "remove":
            if self.freq_mhz is not None and self.peak_uid is not None:
                raise _bad(
                    "peak_uid",
                    "null when freq_mhz is given",
                    self.peak_uid,
                    "remove takes one peak: freq_mhz or peak_uid, not both",
                )
            if self.freq_mhz is None and self.peak_uid is None:
                raise _bad(
                    "freq_mhz",
                    "a frequency in MHz (or a peak_uid)",
                    None,
                    "remove needs exactly one peak: freq_mhz or peak_uid",
                )
        elif self.freq_mhz is None:
            what = (
                "the anchor the new window must cover"
                if action == "create"
                else ("the peak to add")
            )
            raise _bad(
                "freq_mhz",
                "a frequency in MHz",
                None,
                f"{action} needs exactly one frequency ({what})",
            )

    def to_dict(self) -> Dict[str, Any]:
        """The wire form: ``{"schema": "ftmw/curation_action@1", "action",
        "window_id", "freq_mhz", "peak_uid", "candidate_mhz", "frame",
        "epsilon"}``.

        Every key is always present; an unused field is ``None`` (JSON
        ``null``). These are request fields, so ``None`` is the plain "not
        given", not the contract's absent-result marker.
        """
        out: Dict[str, Any] = {"schema": CURATION_ACTION_SCHEMA}
        for f in fields(self):
            out[f.name] = getattr(self, f.name)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CurationAction":
        """The inverse of :meth:`to_dict`.

        ``"schema"``, when present, must be ``"ftmw/curation_action@1"``.
        ``"action"`` is required; any other field that is missing is
        ``None``. An unknown key is refused (``bad_setting`` naming the key),
        so a misspelled field is never silently dropped.
        """
        if not isinstance(data, Mapping):
            raise _bad(
                "actions",
                "a curation action object",
                data,
                f"a curation action must be an object (dict), got {data!r}",
            )
        schema = data.get("schema", CURATION_ACTION_SCHEMA)
        if schema != CURATION_ACTION_SCHEMA:
            raise _bad(
                "schema",
                f'"{CURATION_ACTION_SCHEMA}"',
                schema,
                f"not a curation action: schema {schema!r}, expected "
                f"{CURATION_ACTION_SCHEMA!r}",
            )
        names = [f.name for f in fields(cls)]
        for key in data:
            if key != "schema" and key not in names:
                raise _bad(
                    str(key),
                    "a curation action field (" + ", ".join(names) + ")",
                    data[key],
                    f"unknown curation action field {key!r}",
                )
        if "action" not in data:
            raise _bad(
                "action",
                "one of: " + ", ".join(_ACTION_KINDS),
                None,
                "a curation action needs an 'action'",
            )
        kwargs: Dict[str, Any] = {name: data.get(name) for name in names}
        return cls(**kwargs)

    def to_row(self) -> str:
        """This action as one curation-file (CSV) row, without a newline:
        ``action,window,freqs,params``.

        A ``None`` window is written ``auto`` on add/remove and ``new`` on
        create; ``peak_uid`` is written ``uid:N``; ``candidate_mhz`` is
        ``candidate=F``. Frequencies are written with ``repr`` so they parse
        back exactly. ``frame`` is not part of a row -- a file declares its
        frame once, in a ``# frame:`` header (with ``# epsilon:`` when
        calibrated) -- so a file of these rows must carry that header when
        the actions are not raw.
        """
        if self.window_id is not None:
            window = str(self.window_id)
        else:
            window = "new" if self.action == "create" else "auto"
        if self.peak_uid is not None:
            freqs = f"uid:{self.peak_uid}"
        elif self.freq_mhz is not None:
            freqs = repr(self.freq_mhz)
        else:
            freqs = ""
        params = (
            "" if self.candidate_mhz is None else f"candidate={self.candidate_mhz!r}"
        )
        return f"{self.action},{window},{freqs},{params}"
