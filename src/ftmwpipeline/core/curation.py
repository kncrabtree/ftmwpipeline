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
"""

from typing import Literal

__all__ = ["REFIT_SNAP_TOL_BINS", "Frame"]

Frame = Literal["raw", "calibrated"]
"""The frame a caller-supplied or caller-returned frequency is expressed in.

``"raw"`` is the frame the Stage 5 fit, the Stage 3 candidate ledger, and every
persisted value (the decision log, created windows) live in -- the frame
values are stored in, permanently, because ``epsilon`` is recomputable and a
stored calibrated value would silently change meaning after a timebase
re-run. ``"calibrated"`` is ``f_corr = probe + (f_raw - probe) / (1 + eps)``,
the frame the final-products table and the reports present.

Every caller-supplied frequency across the three interfaces (``add``/
``remove``, ``candidate_freq``, create anchor, and a curation file's
frequencies) takes a ``frame`` parameter of this type, default ``"raw"`` --
matching the undocumented behavior every caller already
depends on. On an ``rb_locked``/``uncalibrated`` file (``epsilon == 0``) the
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
