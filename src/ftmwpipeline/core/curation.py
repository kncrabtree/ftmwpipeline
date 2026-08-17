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

from typing import Literal

__all__ = ["REFIT_SNAP_TOL_MHZ", "Frame"]

Frame = Literal["raw", "calibrated"]
"""The frame a caller-supplied or caller-returned frequency is expressed in.

``"raw"`` is the frame the Stage 5 fit, the Stage 3 candidate ledger, and every
persisted value (the decision log, created windows) live in -- the frame
values are stored in, permanently, because ``epsilon`` is recomputable and a
stored calibrated value would silently change meaning after a timebase
re-run. ``"calibrated"`` is ``f_corr = probe + (f_raw - probe) / (1 + eps)``,
the frame the final-products table and the reports present.

Every caller-supplied frequency across the three interfaces (``add``/
``remove``, ``candidate_freq``, merge peak sets, split peak, create anchor,
and a curation file's frequencies) takes a ``frame`` parameter of this type,
default ``"raw"`` -- matching the undocumented behavior every caller already
depends on. On an ``rb_locked``/``uncalibrated`` file (``epsilon == 0``) the
two frames coincide, so omitting ``frame`` is inert. On a ``self_calibrated``
file it is not: a calibrated candidate submitted as raw still resolves, and to
the *right* peak, but lands wrong by ``probe_freq * eps/(1+eps)`` -- under the
50 kHz snap tolerance and over the statistical sigma, so the mistake is
invisible in the result. Omitting ``frame`` on a frequency-bearing call is
therefore an error on a ``self_calibrated`` file rather than a silent
assumption; passing ``frame="raw"`` explicitly is never an error, on any file.
"""


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
