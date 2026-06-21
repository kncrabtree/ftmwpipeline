.. index::
   single: peak detection
   single: matched filter
   single: gap pass
   single: leakage-aware detection floor
   single: SNR classification
   single: promotion cutoff
   single: sub-bin position

Stage 3: Peak Detection
=======================

Peak detection finds where the lines are. It runs on the noise-referenced
spectrum from :doc:`Stage 2 <stage2_noise>` and produces a classified list of
candidate peaks for the windowing and fitting stages that follow. Its job is
**not** to be the final peak list: the project fits *unwindowed*, full-resolution
spectra, so close lines that the detector smears together are pulled apart later
by the per-window fit. Detection has two duties only — seed the analysis windows
:doc:`Stage 4 <stage4_windows>` builds, and make sure no line above the
signal-to-noise threshold is missed.

Because detection is a seeding step and not the science output, it is
deliberately allowed to be coarse where lines crowd. The cost of missing a weak
line is high (it never gets fit); the cost of an extra coarse candidate is low
(the fit resolves or discards it). Stage 3 is tuned accordingly: it detects
aggressively and leaves the final say to later stages and to the analyst.

What detection produces
-----------------------

A completed Stage 3 run persists an ordered list of peaks, each carrying:

- its **frequency** and **magnitude**, measured on the canonical active spectrum;
- its **signal-to-noise ratio** against the :doc:`Stage 2 <stage2_noise>` noise;
- an **SNR classification** — ``weak``, ``medium``, or ``strong``;
- which **pass** found it (``primary`` or ``gap``);
- a **promoted** flag marking whether it clears the promotion cutoff that moves
  peaks on to Stage 4;
- detection **provenance** (its frequency and SNR on the internal detection grid)
  for curation and diagnosis.

Every detected peak is stored, not just the promoted ones, so the promotion
threshold can be re-chosen later without re-running detection. The list is plain
enough to be hand-edited between Stage 3 and Stage 4 (see
:ref:`stage3-handedit`).

Two passes
----------

A single detector cannot do both of Stage 3's jobs well. A strong line in a
boxcar-truncated spectrum is wrapped in coherent **truncation-leakage** sidelobes
— a sinc-like ripple skirt that an unapodized detector reads as a picket fence of
spurious weak lines. Suppressing that ripple with a strong window function,
though, broadens every line and buries genuine weak lines next to strong ones.
The two requirements pull in opposite directions, so Stage 3 runs two passes on
two different spectra and merges the results.

**Primary pass — robust strong-line positions.** The primary pass transforms the
active region through a strong **Blackman-Harris** window, which all but
eliminates the truncation sidelobes, and locates peaks on the resulting magnitude
spectrum. On the reference experiment the window cuts the sidelobe-suspect
detection fraction roughly five-fold. The apodization broadens the lines and
loses some weak ones, but that is acceptable: the primary's only job is a clean,
sidelobe-free list of the strong lines, which also seeds the second pass.

**Gap pass — weak-line recovery.** The gap pass recovers the weak lines the
apodization smeared away. It transforms the active region through the line's
**matched filter** — the time-domain envelope that maximizes signal-to-noise for
a decaying line: :math:`e^{-t/\tau}` for a Lorentzian instrument,
:math:`e^{-(t/\tau)^2}` for a Gaussian one. The decay constant :math:`\tau` and
the envelope shape both come from the :doc:`Stage 2b <stage2b_tau>` calibration
(its majority decay time, or the Gaussian twin when the line-shape vote
recommends Gaussian); without Stage 2b the gap pass falls back to a default
5 µs basis. Detections within a small exclusion radius of a primary peak are
dropped as re-finds, so the gap pass contributes only *new* lines.

The matched filter is the optimal linear detector for a decaying line: it
concentrates each line's energy that an unweighted transform would spread across
the skirt, so at a fixed false-positive load the gap pass recovers more real
lines from fewer candidates. The gap pass is on by default and can be switched
off (``run_gap_pass=False`` / ``--no-gap-pass``).

The concave-down locator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both passes find positions with the same locator: a Savitzky-Golay smoothed
**second-derivative** search. It flags a peak only where the second derivative is
negative — a genuinely concave-down feature — and takes the sharpest such points.
This structural test is what lets the matched filter work: a per-bin
signal-to-noise threshold alone would accept every bin of a strong line's
monotonically-decaying skirt, but those skirt bins are not concave-down maxima,
so the locator rejects them while keeping true line centers. The smoothing window
covers about four line-widths, derived at run time from the actual grid spacing
and the calibrated line-width (floored at five bins for polynomial stability); on
the reference grid this reproduces the empirical primary-pass window of eleven
bins.

The leakage-aware detection floor
----------------------------------

Neither pass uses a hard mask to remove a strong line's leakage skirt — the
strong lines *are* what generate the coherent leakage, so a hard cutoff would
delete them. Instead each pass raises its detection threshold **continuously** by
the local coherent-leakage amplitude:

.. math::

   \text{threshold}(f) = \text{min\_snr}\cdot\sigma(f)
       + k \cdot \frac{S_\text{coh}(f)}{\sqrt{M}} \cdot \sigma(f).

The added term is the size of the coherent ripple at that frequency, measured by
a complex-domain **edge-coherence** statistic :math:`S_\text{coh}` over a band of
:math:`M` bins. A genuine line towers over this floor; a strong line's skirt
ripple — which *is* that leakage — sits right at it and is rejected. The
statistic is taken on the de-ramped spectrum so the leakage is exposed rather
than averaged away by the active-region phase ramp.

The two passes need different gains :math:`k` because they run on opposite-leakage
spectra. The Blackman-Harris primary has already suppressed most of the leakage,
so its floor is a small residual correction (``primary_leakage_floor_k = 1``). The
matched-filter gap pass retains the full leakage, so its floor carries all the
suppression and needs a stronger gain (``gap_leakage_floor_k = 3``). Both gains
are per-instrument knobs; setting one to zero disables that floor.

Scoring on the canonical spectrum
---------------------------------

The two passes only *find positions*. Every reported amplitude, signal-to-noise,
and classification is then re-measured on the single **canonical active FT** — the
unapodized, native-length spectrum that :doc:`Stage 5 <stage5_fitting>` fits — so
all peaks share one honest signal-to-noise scale and the overlay matches the
spectrum the fit sees. The apodized primary and matched-filter gap spectra are
detection scaffolding only; no apodized amplitude is ever reported.

Snapping to the canonical grid does two more things. The second-derivative
locator lands a few points off the true apex for ultra-narrow lines, so each
detection is **apex-snapped** to the nearest local maximum on the scoring
spectrum, recovering the correct peak height (without this, ultra-narrow lines
under-report their amplitude by tens of percent). Detections that snap to the
same grid point — split-strong-line artifacts, or a line found by both passes —
are then **de-duplicated**, keeping the strongest.

Detection always runs on the Stage 1 canonical spectrum, including its frequency
trim. There is no separate Stage 3 trim or zero-padding: set the analysis band by
running :doc:`Stage 1 <stage1_ft>` with the desired ``--trim`` first.

Classification and the promotion cutoff
---------------------------------------

Each peak is classified by signal-to-noise alone into ``weak``, ``medium``, or
``strong`` using two absolute boundaries (default ``10`` and ``50``). Absolute
cutoffs are used rather than percentiles because the detected-peak SNR
distribution is anchored at the detection floor with a heavy upper tail in every
regime: across the reference fixtures, spanning three orders of magnitude in line
strength, all three tiers stay populated and the fixed boundaries generalize
where percentile boundaries would brand half of a sparse spectrum's real lines
"weak." Of the three tiers, only ``strong`` drives the later stages: a strong
line is a leakage source, so :doc:`Stage 4 <stage4_windows>` uses the ``strong``
tier to anchor its window grouping and to decide which lines are attached as
fixed contributors to neighboring windows whose grid their skirts reach. The
``weak``/``medium`` distinction is curation and reporting labeling, carried for
the analyst rather than consumed by an algorithm.

Detection and promotion are separated, which is the key to detecting
aggressively without flooding the later stages:

- The user-facing ``min_snr`` (default ``3``) is the **promotion cutoff**: the
  signal-to-noise, on the canonical grid, at or above which a peak advances to
  Stage 4. The default sits at the cross-fixture knee where the candidate count
  transitions from noise-and-leakage structure to the real-line plateau.
- Detection itself runs at a fixed lower **internal floor** (``2.0``), regardless
  of the promotion cutoff. Detecting at the promotion cutoff and then re-measuring
  on the canonical grid would lose lines that genuinely clear the cutoff there;
  detecting lower recovers them, and below the internal floor there is only noise.

Every detected peak is persisted with its ``promoted`` flag derived from the
stored cutoff, so re-thresholding is free and an edited list re-derives the flag
cleanly.

.. figure:: figures/stage3_peaks.png
   :width: 95%
   :align: center

   Stage 3 detection on the example experiment. Top: the classified peaks over
   the canonical active spectrum (gray) and its per-bin noise, colored by SNR
   tier — green ``weak``, orange ``medium``, red ``strong`` — marker-styled by
   pass (circles primary, triangles gap), and filled when promoted or left open
   when below the promotion cutoff. The log magnitude axis makes the noise floor
   and the hundreds-of-times stronger lines legible together. Bottom: the
   curation panel, the signal-to-noise distribution of every detected peak with
   the promotion cutoff marked, so the threshold can be chosen against the
   visible split between the near-noise hump and the real-line tail before peaks
   move to Stage 4.

Running detection
-----------------

Stage 3 requires Stages 1 and 2 (and consumes :doc:`Stage 2b <stage2b_tau>` when
present):

.. code-block:: console

   $ ftmwpipeline peaks run exp_2638.ftmw
   $ ftmwpipeline peaks show --snr-histogram exp_2638.ftmw

The same operations on the Python interfaces:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   peaks = ftmw.detect_peaks("exp_2638.ftmw")   # all detected peaks
   ftmw.visualize_peaks("exp_2638.ftmw", show_snr_histogram=True)

   # or, object-oriented
   from ftmwpipeline import Pipeline
   pipe = Pipeline.open("exp_2638.ftmw")
   peaks = pipe.detect_peaks()
   promoted = [p for p in peaks if p.properties["promoted"]]

``detect_peaks`` returns the full candidate list (the curation substrate); Stage
4 consumes only the promoted subset. Re-running detection invalidates any Stage 4
window plan built on the old peaks.

The defaults are calibrated for the reference instrument and need no adjustment
for routine use. The behavior is controlled by the knobs below, set with per-knob
flags on ``peaks run`` (e.g. ``--min-snr``), through a preset, or via
``settings=PeakDetectionSettings(...)`` on the Python interfaces; how explicit,
preset, persisted, and recommended values resolve is described on
:doc:`settings_and_presets`.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Knob
     - Default
     - Role
   * - ``min_snr``
     - ``3``
     - Promotion cutoff: the canonical-grid signal-to-noise at or above which a
       peak advances to Stage 4. The main detection control.
   * - ``weak_medium_snr`` / ``medium_strong_snr``
     - ``10`` / ``50``
     - The SNR classification boundaries.
   * - ``run_gap_pass``
     - ``true``
     - Run the matched-filter gap pass that recovers weak lines (``--no-gap-pass``
       to skip it).
   * - ``primary_window``
     - ``blackmanharris``
     - The primary-pass apodization window (sidelobe suppression; affects
       positions only, never reported amplitude).
   * - ``min_exclusion_mhz``
     - ``0.0``
     - Half-width around each primary peak that the gap pass treats as a re-find.
   * - ``sg_window`` / ``sg_order``
     - ``11`` / ``3``
     - The primary-pass Savitzky-Golay smoothing window and polynomial order.

Advanced knobs flow through a preset or a settings bundle rather than per-flag:
the internal detection floor (``internal_min_snr``), the leakage-floor gains
(``primary_leakage_floor_k``, ``gap_leakage_floor_k``), the detection-grid
zero-padding (``detection_zpf``, ``gap_active_zpf``), the grid-aware
Savitzky-Golay rule (``sg_fwhm_coverage``, ``sg_min_window``), and the scatter
knobs for the primary pass's own apodized-domain noise estimate. These rarely
need touching.

Reading ``peaks show``
----------------------

``peaks show`` overlays the classified peaks on the canonical active spectrum on
a log magnitude axis. Adding ``--snr-histogram`` appends the curation panel of
the figure above — the signal-to-noise distribution of every detected peak with
the promotion cutoff marked — which is the view to use when choosing a promotion
threshold deliberately. By default the command opens an interactive window;
``--no-interactive`` together with ``-o`` saves a static image to the given path
instead.

.. _stage3-handedit:

The Stage 3 → Stage 4 boundary
------------------------------

Detection and window assignment are deliberately separate stages, so the
detected-peak list is a curation point: an analyst can load it, add or remove
peaks, adjust the promotion threshold, and re-save before windows are built. The
peak list is persisted as a flat set of equal-length parallel arrays plus the
detection parameters, an obvious on-disk form to edit. The loader validates the
structure loudly — a missing column, mismatched lengths, or an unknown
classification label raises an error rather than silently dropping data — so a
malformed edit fails immediately rather than corrupting the fit.

What the later stages consume
-----------------------------

- :doc:`Stage 4 <stage4_windows>` builds analysis windows from the **promoted**
  peaks, using the strong-line classification to decide which clusters need
  dedicated handling.
- :doc:`Stage 5 <stage5_fitting>` fits the lines within each window; the detected
  peaks seed its starting positions.

Limitations
-----------

- Detection deliberately smears close lines together where the primary pass's
  apodization or the grid spacing cannot resolve them. This is by design — the
  per-window fit separates them — and means the Stage 3 count is a candidate
  count, not the final line count.
- The continuous leakage floor suppresses, but does not perfectly remove, the
  ripple at the skirts of the very strongest lines; a few near-skirt candidates
  can survive into the list, where the fit and the curation pass resolve them.
- Detection runs on the canonical trim, so a line outside the Stage 1 active band
  is never seen. Set the band wide enough at Stage 1 to cover every region of
  interest.

With a classified, curated peak list in hand, :doc:`Stage 4 <stage4_windows>`
groups the promoted peaks into the disjoint analysis windows the fit runs on.
