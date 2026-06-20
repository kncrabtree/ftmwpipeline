.. index::
   single: leakage pedestal
   single: noise estimation; high signal-to-noise
   single: shot-count scaling
   single: Rician correction

Noise estimation at high signal-to-noise
========================================

This note explains why estimating the per-bin noise on a high signal-to-noise,
line-dense spectrum is harder than it looks, and on what basis the Stage 2
scatter estimator (:doc:`../stage2_noise`) can be trusted there. The figures and
the numbers quoted are regenerated from the checked-in fixtures by
``docs/source/methods/noise_snr_scaling/generate.py``; see
:ref:`noise-snr-reproducing`.

The problem: a leakage pedestal
-------------------------------

The canonical FT is deliberately raw and unapodized, which keeps the per-bin
noise statistics independent (apodization or zero-padding correlate adjacent
bins). A boxcar transform of a strong line has a slowly decaying magnitude skirt,
and the transform's own sidelobes ring across the whole record. Any one line's
wings are negligible, but on a dense, high signal-to-noise spectrum the **sum**
of the far-wings of hundreds of strong lines forms a smooth *pedestal* that
fills every quiet bin between the lines.

The natural way to estimate noise — take the *level* of the magnitude spectrum
in the gaps between lines, for instance a running median of :math:`|X|` — then
measures the pedestal, not the random noise. The taller the lines, the higher the
pedestal, so the error grows precisely on the spectra where the noise estimate
matters most.

Why it is a signal-to-noise-scaling effect
------------------------------------------

The noise and the pedestal behave oppositely with the shot count :math:`N`:

* the **random noise** is incoherent and averages down as
  :math:`\sigma \propto 1/\sqrt{N}`;
* the **pedestal** is proportional to the (coherently averaged) signal
  amplitude, which is independent of :math:`N` — so the pedestal is **constant**
  in :math:`N`.

A level estimator therefore reports
:math:`\sqrt{\sigma(N)^2 + \mathrm{pedestal}^2}`: it tracks the true noise while
the noise dominates, then plateaus once the pedestal dominates. This is the
decisive, assumption-free test — apply each estimator to a series of backup
frames at increasing shot count and look at the slope on log–log axes. A true
noise estimate must fall with slope :math:`-1/2`; a pedestal-bound estimate must
flatten.

.. figure:: noise_snr_scaling/figures/fig0_sqrtN_655.png
   :width: 80%
   :align: center

   The high signal-to-noise fixture (655). The scatter estimate falls as
   :math:`1/\sqrt{N}`; the naive level estimate is flat — it is measuring the
   constant pedestal, not the averaging-down noise.

On the highest signal-to-noise fixture the naive level estimate has a slope near
zero (dead flat), while the scatter estimate falls with a slope near
:math:`-1/2`. Across the fixtures with backup frames the scatter estimator holds
that :math:`1/\sqrt{N}` behavior:

.. figure:: noise_snr_scaling/figures/fig1_sqrtN_all.png
   :width: 80%
   :align: center

   Shot-count scaling of the scatter estimate across the multi-frame fixtures
   (per-fixture slope in the legend); all sit near the ideal :math:`-1/2`.

The scatter estimator
---------------------

The estimator separates the two contributions with a high-pass along the
frequency axis: the noise is the white, bin-to-bin *scatter* of the magnitude,
while the pedestal is its smooth part. It subtracts a broad running-median
pedestal, self-masks the lines, takes a robust scatter (a median absolute
deviation) of the residual over a sliding window, converts that to the
underlying noise with the Rician correction below, and rides the lower envelope
through line-dense bands. The step-by-step algorithm and its tunables are on the
:doc:`../stage2_noise` page; this note covers why it is correct.

The Rician correction
---------------------

The magnitude of a complex spectrum is Rician, so the scatter of :math:`|X|`
relates to the underlying per-quadrature noise :math:`\sigma_c` by a factor that
depends on the local regime: it runs from :math:`1.0` under a strong line (where
the pedestal dominates and the magnitude is locally Gaussian) to about
:math:`1.47` in a pure-noise (Rayleigh) region. A single fixed factor is
therefore wrong in a regime-dependent way — worst by ~20% exactly at the
under-line bins where the fit later evaluates its :math:`\chi^2`.

The fix needs no extra data. The dimensionless ratio
:math:`R = \mathrm{scatter}/\mathrm{pedestal}` indexes the regime, so a single
one-dimensional lookup :math:`C(R) = \sigma_c/\mathrm{scatter}` recovers the
regime-correct factor from the one spectrum in hand (both quantities are already
computed per window). :math:`C(R)` is theoretically smooth and monotone
increasing, so the table is built by simulating the Rician magnitude over a grid
of pedestal-to-noise ratios and reducing the Monte-Carlo cloud to its monotone
(isotonic) fit, which is baked into the estimator as constants. Near the Rayleigh
limit :math:`R` saturates and the raw simulation is noisy there; the monotone fit
is the physically correct curve.

.. figure:: noise_snr_scaling/figures/fig5_region_aware.png
   :width: 70%
   :align: center

   The Rician correction :math:`C(R)`: the raw Monte-Carlo samples and the
   monotone fit that is baked into the estimator. It runs from ~1.0 under strong
   lines to ~1.5 in pure-noise (Rayleigh) regions.

Validation: the overestimate is a signal-to-noise effect
--------------------------------------------------------

Comparing the naive level estimate to the scatter estimate on the same spectrum
isolates the pedestal contamination. Because each fixture retains backup frames
at increasing shot count, the comparison can be traced *within a single fixture*,
holding the line content fixed while the signal-to-noise rises — which separates
the signal-to-noise dependence from the spectrum's line density. Within every
fixture the naive over-report grows with signal-to-noise: it is near unity at low
signal-to-noise (no pedestal to speak of) and climbs to roughly a factor of six
on the highest-signal-to-noise fixture, while the scatter estimate stays on the
true floor (confirmed by its :math:`1/\sqrt{N}` slope above). The *level* a given
sample reaches also depends on its line content, so this is a family of curves,
one per fixture, rather than a single universal line.

.. figure:: noise_snr_scaling/figures/fig2_overestimate_vs_snr.png
   :width: 80%
   :align: center

   Within each fixture (one curve per fixture, traced over its backup frames),
   the naive level estimate's over-report relative to the scatter estimate grows
   with peak signal-to-noise — from near unity to roughly six on the highest.

The residual: a weak-line floor
--------------------------------

The scatter estimate's shot-count slope is not exactly :math:`-1/2` on the
densest spectra. Modelling
:math:`\mathrm{scatter}(N)^2 = c/N + f^2` separates the averaging-down noise
(:math:`c/N`) from a small **constant floor** (:math:`f`): the weak, undetected
lines that pepper a spectrum this dense, sitting in the bins counted as
line-free. The floor does not average down, so it slightly flattens the slope at
the highest shot counts. It is a small additive term — far below the
multiplicative pedestal error it replaces — and is bounded by stricter line
masking; the principled way to remove it entirely is a blank (sample-free)
acquisition series.

.. figure:: noise_snr_scaling/figures/fig3_floor_decomp.png
   :width: 70%
   :align: center

   The scatter-squared is linear in :math:`1/N`; the intercept is the additive
   weak-line floor :math:`f^2`.

Caveats
-------

* **Noise under the lines** is not measured directly — no method can, with the
  sample present. The estimator interpolates the noise floor across line
  positions, which is sound because the noise is a smooth, slowly varying
  receiver property; the definitive direct check would be a blank-FID series.
* **The weak-line floor** above is irreducible on an extremely dense spectrum
  without a blank acquisition.
* **Instrument dependence.** The window and smoothing widths are calibrated for
  the reference spectrometer; the Rician table and the
  :math:`\sigma_x = \sigma_c\sqrt{2}` conversion are analytic and not tunable.

.. _noise-snr-reproducing:

Reproducing this note
---------------------

The figures and the quoted numbers are regenerated from the checked-in fixtures
under ``examples/blackchirp_data/`` (each retains its progressive backup frames)
by::

    python docs/source/methods/noise_snr_scaling/generate.py

which writes ``figures/*.png`` and ``results.json`` beside the script (a few
seconds over all fixtures). Two fast, data-free invariants are exercised by the
unit suite — the Monte-Carlo :math:`C(R)` table matches the baked constants, and
a synthetic pedestal-plus-noise series reproduces the slope split (scatter near
:math:`-1/2`, naive near zero). The full fixture regeneration is guarded against
drift by a ``slow``-marked test that re-runs the harness and compares
``results.json`` within tolerance.
