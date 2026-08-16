"""Public tolerances for Stage 6 curation.

These are the values the pipeline itself pairs at, published so an external
tool can pair identically.  An integrator that resolved "the peak at *f*" at a
different tolerance would disagree with the file about *which* peak that is --
the disagreement this module exists to prevent -- so the constant is the single
definition every public verb's default is taken from, not a separately
maintained copy that happens to agree.

This module is intentionally dependency-free within the package (stdlib only),
the same discipline :mod:`ftmwpipeline.core.settings` documents for itself, so
it can be imported anywhere without a cycle.
"""

__all__ = ["REFIT_SNAP_TOL_MHZ"]


REFIT_SNAP_TOL_MHZ: float = 0.05
"""Tolerance (MHz) for matching a requested frequency to an existing peak.

Every Stage 6 curation verb -- ``review edit`` / ``create`` / ``merge`` /
``split`` / ``accept``, and every action inside ``review apply`` -- resolves a
requested ``add`` / ``remove`` / anchor / candidate frequency to the nearest
fitted peak or ledger candidate within this tolerance, and treats anything
beyond it as a miss (a hard error for ``remove``, a fresh seed at the requested
frequency for ``add``).

50 kHz is deliberately loose relative to the 20 kHz candidate-dedup window: the
input is a frequency a *person* typed off a plot or a line list, so the snap has
to forgive coarse input rather than demand the fitted value.  Two fitted peaks
can therefore both fall inside it; that is resolved nearest-wins, and
``review apply --dry-run`` reports the multi-match as an advisory so the
ambiguity is visible before the batch runs.

Every public verb's ``snap_tol_mhz`` parameter defaults to this value; a caller
that passes its own overrides it for that call only.  Reading it is the
supported way to pair identically with the pipeline -- read it at call time
rather than caching it, so a future change to the value stays transparent.
"""
