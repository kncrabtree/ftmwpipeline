"""Catalog cross-reference: proximity-flag reported lines against a catalog.

**Annotation only, never an assignment.** The pipeline emits UNASSIGNED lines;
this echoes the nearest catalog entry's *opaque* label as a cross-check the user
can read, and exposes the frequency pull ``(f_line - f_cat) / sigma_f`` so the
user can calibrate whether the reported ``sigma_f`` budget is honest (the pull
should be ~unit-normal when it is). It never feeds back into the fit.

The match test lives here, in one place, so all three report levels (and the
pull-calibration surface) share exactly one tolerance/match definition. A line
matches the catalog when its nearest catalog entry is within
``N * sqrt(sigma_f^2 + sigma_cat^2)`` of it (``N`` = ``n_sigma``); ``sigma_cat``
is ``0`` when the catalog carries no per-line uncertainty.

The catalog reader is deliberately format-light. It accepts a CSV (or
whitespace-delimited) table of ``frequency_mhz``, an optional uncertainty (unit
read from the header), and an optional opaque label, plus the Pickett/SPCAT
``.cat`` predicted-line catalog (fixed-width frequency + MHz error + an opaque
species/quantum-number tag). Writing Pickett ``.lin`` / SPFIT emission is out of
scope by design — this surface only reads, for cross-reference.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

# Header keywords (substring match, case-folded) used to locate columns when the
# catalog file carries a header row.
_FREQ_KEYS = ("freq", "frequency", "mhz")
_UNC_KEYS = ("unc", "sigma", "err", "uncertainty")
_LABEL_KEYS = (
    "label",
    "name",
    "id",
    "species",
    "assign",
    "qn",
    "quantum",
    "transition",
)


@dataclass(frozen=True)
class CatalogEntry:
    """One catalog line: a frequency, an optional uncertainty, an opaque label.

    Attributes
    ----------
    frequency_mhz : float
        Catalog frequency (MHz).
    sigma_khz : float or None
        Per-line catalog uncertainty (kHz), or ``None`` when the catalog does
        not carry one (treated as ``0`` in the match tolerance).
    label : str
        The opaque catalog label echoed by the cross-reference. Never
        interpreted as an assignment.
    """

    frequency_mhz: float
    sigma_khz: Optional[float]
    label: str


@dataclass(frozen=True)
class CatalogMatch:
    """The nearest catalog entry to one reported line, within tolerance.

    Attributes
    ----------
    label : str
        The matched catalog entry's opaque label.
    frequency_mhz : float
        The matched catalog frequency (MHz).
    sigma_cat_khz : float
        The catalog uncertainty used in the budget (``0`` if none).
    delta_khz : float
        ``(f_line - f_cat)`` in kHz (signed).
    combined_sigma_khz : float
        ``sqrt(sigma_f^2 + sigma_cat^2)`` in kHz.
    pull : float
        ``delta_khz / combined_sigma_khz`` (``nan`` when the combined sigma is
        zero); the dimensionless residual the pull-calibration surface tallies.
    """

    label: str
    frequency_mhz: float
    sigma_cat_khz: float
    delta_khz: float
    combined_sigma_khz: float
    pull: float


@dataclass
class CatalogCrossRef:
    """The cross-reference result for one final-products table.

    ``matches`` is parallel to ``products.peaks`` (``None`` where a line has no
    catalog entry within tolerance), so every renderer can index it positionally.
    The pull summary (item-2 calibration surface) is computed over the matched
    lines' finite pulls.
    """

    catalog_path: str
    n_sigma: float
    n_catalog: int
    matches: List[Optional[CatalogMatch]]
    pull_values: List[float] = field(default_factory=list)
    pull_mean: Optional[float] = None
    pull_std: Optional[float] = None

    @property
    def n_total(self) -> int:
        return len(self.matches)

    @property
    def n_matched(self) -> int:
        return sum(1 for m in self.matches if m is not None)

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_total if self.matches else 0.0


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _is_float(token: str) -> bool:
    try:
        float(token)
    except (TypeError, ValueError):
        return False
    return True


def _split_row(row: str, comma: bool) -> List[str]:
    if comma:
        return [c.strip() for c in row.split(",")]
    return row.split()


def _unc_scale_khz(header_name: str) -> float:
    """kHz-per-unit for an uncertainty column, inferred from its header name.

    A header naming MHz (or Hz) is honoured; everything else (a bare
    ``unc`` / ``sigma`` / ``err``, or a ``khz`` token) is taken as kHz.
    """
    key = header_name.lower()
    if "mhz" in key:
        return 1.0e3
    if "khz" in key:
        return 1.0
    if "hz" in key:  # plain Hz
        return 1.0e-3
    return 1.0


def _is_label_header(key: str) -> bool:
    """True if a header names a label column, without short-substring misfires.

    Multi-character keywords match as substrings (so ``assign`` catches
    ``assignment``); the two-character keys (``id``, ``qn``) match only as whole
    tokens, so they do not fire inside ``midpoint`` / ``valid`` / ``width``.
    """
    tokens = set(re.split(r"[^a-z0-9]+", key))
    for k in _LABEL_KEYS:
        if (k in key) if len(k) >= 3 else (k in tokens):
            return True
    return False


def _resolve_columns(
    header: Sequence[str],
) -> "tuple[int, Optional[int], Optional[int], float]":
    """Map a header to ``(freq_col, unc_col, label_col, unc_scale_khz)``.

    Precedence per column: uncertainty, then frequency, then label. Frequency
    outranks label so a header carrying both (e.g. ``transition_frequency_mhz``)
    is read as the frequency column, not the label.
    """
    freq_col = 0
    unc_col: Optional[int] = None
    label_col: Optional[int] = None
    unc_scale = 1.0
    for j, name in enumerate(header):
        key = name.strip().lower()
        if unc_col is None and any(k in key for k in _UNC_KEYS):
            unc_col = j
            unc_scale = _unc_scale_khz(key)
        elif any(k in key for k in _FREQ_KEYS):
            freq_col = j
        elif label_col is None and _is_label_header(key):
            label_col = j
    return freq_col, unc_col, label_col, unc_scale


def read_catalog(path: Union[str, Path]) -> List[CatalogEntry]:
    """Read a frequency catalog from a CSV / whitespace / SPCAT ``.cat`` file.

    A ``.cat`` file is parsed as Pickett/SPCAT fixed-width (the standard
    predicted-line catalog); see :func:`_read_spcat`. Otherwise the file is read
    as a CSV (or whitespace-delimited) table: blank lines and ``#`` comments are
    skipped, and if the first data row's first field is non-numeric it is treated
    as a header. Columns are located by keyword (``freq`` / ``unc|sigma|err`` /
    ``label|name|id|…``), with the uncertainty unit taken from its header
    (``*_mhz`` → MHz, ``*_khz``/bare → kHz, ``*_hz`` → Hz); a header-less file is
    positional ``frequency_mhz``, uncertainty (kHz), opaque label. A line with
    only a frequency gets ``sigma_khz = None`` and a synthesized label.

    Returns the parsed entries (empty list for an empty file). Rows whose
    frequency does not parse are skipped.
    """
    if Path(path).suffix.lower() == ".cat":
        return _read_spcat(path)

    text = Path(path).read_text()
    rows = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not rows:
        return []

    comma = any("," in r for r in rows)
    first = _split_row(rows[0], comma)
    freq_col: int = 0
    unc_col: Optional[int] = 1
    label_col: Optional[int] = 2
    unc_scale = 1.0
    start = 0
    if first and not _is_float(first[0]):
        freq_col, unc_col, label_col, unc_scale = _resolve_columns(first)
        start = 1

    entries: List[CatalogEntry] = []
    for row in rows[start:]:
        fields = _split_row(row, comma)
        if len(fields) <= freq_col or not _is_float(fields[freq_col]):
            continue
        freq = float(fields[freq_col])
        sigma: Optional[float] = None
        if unc_col is not None and unc_col < len(fields) and _is_float(fields[unc_col]):
            sigma = float(fields[unc_col]) * unc_scale
        label = ""
        if label_col is not None and label_col < len(fields):
            # Whitespace mode: a label may carry spaces -> keep the tail joined.
            label = (
                fields[label_col] if comma else " ".join(fields[label_col:])
            ).strip()
        if not label:
            label = f"{freq:.4f}"
        entries.append(CatalogEntry(frequency_mhz=freq, sigma_khz=sigma, label=label))
    return entries


# SPCAT ``.cat`` fixed-width fields (Pickett, columns 1-indexed in the spec):
# FREQ F13.4 (MHz) | ERR F8.4 (MHz) | LGINT F8.4 | DR I2 | ELO F10.4 | GUP I3 |
# TAG I7 | QNFMT I4 | QN' 6*I2 | QN'' 6*I2. We take frequency, the MHz error,
# and build an opaque label from the species tag + the upper/lower quantum
# numbers (``tag: QN' <- QN''``).
_CAT_FREQ = slice(0, 13)
_CAT_ERR = slice(13, 21)
_CAT_TAG = slice(44, 51)
_CAT_QN_UP = slice(55, 67)
_CAT_QN_LO = slice(67, 79)


def _read_spcat(path: Union[str, Path]) -> List[CatalogEntry]:
    """Parse a Pickett/SPCAT ``.cat`` predicted-line catalog (fixed-width).

    The frequency error is in MHz (converted to kHz); the opaque label is the
    species tag and the upper/lower quantum numbers. Lines whose frequency field
    does not parse are skipped, so a stray header / footer is tolerated.
    """
    entries: List[CatalogEntry] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        freq_tok = line[_CAT_FREQ].strip()
        if not _is_float(freq_tok):
            continue
        freq = float(freq_tok)
        err_tok = line[_CAT_ERR].strip()
        # ERR is the MHz uncertainty; a merged catalog may flag a measured line
        # with a negative ERR, so use its magnitude.
        sigma_khz = abs(float(err_tok)) * 1.0e3 if _is_float(err_tok) else None
        tag = line[_CAT_TAG].strip()
        qn_up = " ".join(line[_CAT_QN_UP].split())
        qn_lo = " ".join(line[_CAT_QN_LO].split())
        if qn_up or qn_lo:
            label = f"{tag}: {qn_up} <- {qn_lo}".strip()
        elif tag:
            label = tag
        else:
            label = f"{freq:.4f}"
        entries.append(
            CatalogEntry(frequency_mhz=freq, sigma_khz=sigma_khz, label=label)
        )
    return entries


# ---------------------------------------------------------------------------
# Match test (the one shared tolerance helper)
# ---------------------------------------------------------------------------


def match_peak(
    frequency_mhz: Optional[float],
    sigma_f_khz: Optional[float],
    catalog: Sequence[CatalogEntry],
    n_sigma: float,
    *,
    cat_freqs: Optional[np.ndarray] = None,
) -> Optional[CatalogMatch]:
    """Return the nearest catalog entry within tolerance, or ``None``.

    The nearest catalog entry (by absolute frequency) is taken and accepted only
    when ``|f_line - f_cat| <= n_sigma * sqrt(sigma_f^2 + sigma_cat^2)``. Pass
    ``cat_freqs`` (a sorted array of catalog frequencies) to skip the per-call
    array build when matching many peaks against one catalog.
    """
    if frequency_mhz is None or not math.isfinite(float(frequency_mhz)) or not catalog:
        return None
    f_line = float(frequency_mhz)
    sf = (
        float(sigma_f_khz)
        if sigma_f_khz is not None and math.isfinite(float(sigma_f_khz))
        else 0.0
    )

    if cat_freqs is None:
        cat_freqs = np.asarray([e.frequency_mhz for e in catalog], dtype=float)
    # cat_freqs is sorted ascending (see build_cross_ref); searchsorted gives the
    # insertion point, and the nearest entry is one of its two neighbors.
    pos = int(np.searchsorted(cat_freqs, f_line))
    best_i = -1
    best_abs = math.inf
    for cand in (pos - 1, pos):
        if 0 <= cand < len(catalog):
            d = abs(f_line - float(cat_freqs[cand]))
            if d < best_abs:
                best_abs = d
                best_i = cand
    if best_i < 0:
        return None

    entry = catalog[best_i]
    sc = (
        float(entry.sigma_khz)
        if entry.sigma_khz is not None and math.isfinite(float(entry.sigma_khz))
        else 0.0
    )
    delta_khz = (f_line - entry.frequency_mhz) * 1.0e3
    combined = math.hypot(sf, sc)
    tol = n_sigma * combined
    if abs(delta_khz) > tol:
        return None
    pull = delta_khz / combined if combined > 0.0 else float("nan")
    return CatalogMatch(
        label=entry.label,
        frequency_mhz=entry.frequency_mhz,
        sigma_cat_khz=sc,
        delta_khz=delta_khz,
        combined_sigma_khz=combined,
        pull=pull,
    )


def build_cross_ref(
    peaks: Sequence[object],
    catalog: Sequence[CatalogEntry],
    *,
    catalog_path: str,
    n_sigma: float,
) -> CatalogCrossRef:
    """Match every peak against *catalog* and tally the pull distribution.

    *peaks* are ``FinalPeak``-like (need ``frequency_mhz`` / ``sigma_f_khz``).
    The result's ``matches`` list is parallel to *peaks*.
    """
    cat_sorted = sorted(catalog, key=lambda e: e.frequency_mhz)
    cat_freqs = np.asarray([e.frequency_mhz for e in cat_sorted], dtype=float)
    matches: List[Optional[CatalogMatch]] = []
    pulls: List[float] = []
    for p in peaks:
        m = match_peak(
            getattr(p, "frequency_mhz", None),
            getattr(p, "sigma_f_khz", None),
            cat_sorted,
            n_sigma,
            cat_freqs=cat_freqs,
        )
        matches.append(m)
        if m is not None and math.isfinite(m.pull):
            pulls.append(m.pull)

    pull_mean: Optional[float] = None
    pull_std: Optional[float] = None
    if pulls:
        arr = np.asarray(pulls, dtype=float)
        pull_mean = float(np.mean(arr))
        pull_std = float(np.std(arr, ddof=1)) if arr.size >= 2 else None
    return CatalogCrossRef(
        catalog_path=catalog_path,
        n_sigma=float(n_sigma),
        n_catalog=len(cat_sorted),
        matches=matches,
        pull_values=pulls,
        pull_mean=pull_mean,
        pull_std=pull_std,
    )


def load_cross_ref(
    peaks: Sequence[object],
    catalog_path: Optional[Union[str, Path]],
    n_sigma: float,
) -> Optional[CatalogCrossRef]:
    """Read *catalog_path* and cross-reference *peaks*; ``None`` if no path.

    The single entry point the report levels call: returns ``None`` when no
    catalog was requested (so the reports degrade to their catalog-free form),
    and raises :class:`ValueError` if the path is given but unreadable / empty.
    """
    if catalog_path is None:
        return None
    path = Path(catalog_path)
    if not path.exists():
        raise ValueError(f"catalog file not found: {catalog_path}")
    catalog = read_catalog(path)
    if not catalog:
        raise ValueError(
            f"catalog file {catalog_path} parsed to zero entries; expected a CSV "
            "of frequency_mhz[, uncertainty_khz[, label]]"
        )
    return build_cross_ref(
        peaks, catalog, catalog_path=str(catalog_path), n_sigma=n_sigma
    )
