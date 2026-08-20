.. index::
   single: clock declaration
   single: clock source
   single: clock lattice
   single: spur; lattice prior
   single: timebase calibration

Declaring Instrument Clocks
===========================

Every spectrometer carries a small set of reference clocks — the synthesizer
that drives the up- and down-conversion chain, the arbitrary-waveform generator,
the digitizer sample clock. Those clocks leave deterministic fingerprints in the
spectrum: narrow, non-decaying tones at frequencies set by clock arithmetic
(harmonics, mixing products, sample-rate images). They are *spurs*, not molecular
lines, and the :doc:`Stage 5 <stage5_fitting>` spur gate already removes the ones
it can recognize. Declaring the instrument's clocks grounds that recognition in
the known clock arithmetic instead of an empirical guess.

A clock declaration adds two pieces of prior knowledge:

- **A sharper spur prior.** The declared clock fundamentals generate a *lattice*
  of frequencies where spurs are expected. The spur gate uses the lattice both to
  know where to look and to lower its burden of proof on a tone that lands exactly
  where the clocks predict — so it catches instrumental tones an unprimed gate
  leaves behind. Any fitted line that lands on the lattice is flagged for review.
- **A self-calibrated frequency axis.** A digitizer that is *not* locked to the
  laboratory frequency standard runs at a slightly wrong rate, which stretches the
  frequency axis by a constant fractional factor. That scale error is measurable —
  prior-free — from the Rb-locked clock spurs themselves, because their true
  frequencies are known exactly. The ``timebase`` object measures it and
  :doc:`Stage 6 <stage6_review>` applies it; the estimator and its precision are
  the subject of the :doc:`timebase self-calibration methods note
  <methods/timebase_calibration>`.

Both are *additive prior knowledge*. With no declaration the pipeline behaves
exactly as it does without one (the spur gate falls back to a coarse integer-MHz
anchor, and the frequency axis is reported as measured); the results are
byte-identical. Declaring the clocks only ever adds information.

The instrument clock tree
-------------------------

The home instrument the example data come from is a useful worked example. Its
clocks are:

- an **up-conversion** synthesizer at 5760 MHz (doubled to the 11520 MHz LO,
  multiplied to the excitation band after a lower-sideband mix);
- a **down-conversion** synthesizer at 5120 MHz (multiplied to the 40960 MHz
  probe LO);
- an **arbitrary-waveform generator** clocked at 16000 MHz;
- a **digitizer** sampling at 50 GSa/s (eight interleaved 6.25 GSa/s ADCs).

The first three are referenced to a 10 MHz rubidium standard; the digitizer is
free-running. Every cross-fixture recurring spur in the example data is a direct
identity of one of these clocks — in one of two frames:

- the **baseband** frame, the frequency a tone enters the digitizer at, where the
  molecular frequency maps as :math:`f_\text{bb} = |f_\text{mol} - f_\text{probe}|`
  (a synthesizer reference at 5120 MHz, a mixing product at 2×5120, the AWG
  half-rate at 8000 MHz, an ADC image at 2×6250);
- the **molecular (RF)** frame, harmonics radiating directly into the receiver
  (5×5760, 6×5760).

All of the Rb-locked tones fall on a single comb — the **intermodulation
lattice** of the locked fundamentals, whose spacing is their greatest common
divisor. The free-running digitizer is the one source that *drifts*: its tones
wander relative to the Rb-locked comb from one acquisition to the next.

This is the structure a declaration makes available to the pipeline.

Declaring the clocks
--------------------

A declaration is a short list of clock *sources*, each a frequency in MHz, a
``locked`` flag (referenced to the laboratory standard or not), and an optional
label. The ``clocks`` object manages it:

.. code-block:: console

   # Show the current declaration
   $ ftmwpipeline clocks show exp.ftmw

   # Replace it (token: freq[:locked|free[:label]])
   $ ftmwpipeline clocks set exp.ftmw 5760:locked:upconv 5120:locked:downconv \
                                      16000:locked:awg 6250:free:digitizer

   # Append, remove by frequency, or clear
   $ ftmwpipeline clocks add exp.ftmw 8000:locked:awg-half
   $ ftmwpipeline clocks remove exp.ftmw 6250
   $ ftmwpipeline clocks clear exp.ftmw

The same operations are available on the functional API and the ``Pipeline``
object:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   ftmw.set_clock_sources(
       "exp.ftmw",
       [
           {"freq_mhz": 5760, "locked": True, "label": "upconv"},
           {"freq_mhz": 5120, "locked": True, "label": "downconv"},
           {"freq_mhz": 16000, "locked": True, "label": "awg"},
           {"freq_mhz": 6250, "locked": False, "label": "digitizer"},
       ],
   )
   ftmw.get_clock_sources("exp.ftmw")

Two conventions matter:

- **Declare fundamentals, not products.** A harmonic or mixing product is
  *generated* from the fundamentals, so the lattice produces it for free. Declare
  the 5760 MHz synthesizer, not the 11520 MHz LO it is doubled to. Frequency
  *division* is the exception — a half-rate tone (here the AWG's 8000 MHz) is not a
  generated harmonic, so if the instrument produces one it is declared as its own
  fundamental.
- **Integer-MHz fundamentals.** Rb-locked synthesizers are set to exact
  frequencies; the lattice is built from the integer greatest common divisor of
  the declared locked clocks, so declare them as integers.

Where the declaration lives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``clocks`` verbs write the **recommended** settings layer — the same layer an
importer populates from instrument metadata. In the resolution order
(:doc:`settings_and_presets`), an explicit fit-time clock argument and a
*persisted* Stage 5 spur declaration both outrank it, so the recommended
declaration is the "declare before you fit" baseline rather than an override. If a
Stage 5 fit already exists when you change the declaration, the change does not
retroactively alter that fit; rerun the fit to apply it. The CLI warns when this
is the case.

Automatic population from Blackchirp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Blackchirp experiment records its synthesis chain directly. The loader reads
the clock chain and the digitizer sample rate at import and writes the
recommended declaration, so for the example data the lattice fundamentals are
already declared before you touch the ``clocks`` verbs. The one fact the metadata
does not carry is whether the digitizer is locked, which defaults per the
instrument preset.

The spur lattice prior
----------------------

From the declared *locked* clocks the pipeline computes the lattice spacing
:math:`g` (their greatest common divisor) and predicts spurs wherever
:math:`f_\text{mol}` or :math:`f_\text{bb}` is a multiple of :math:`g`, in both
frames. For the home instrument's three locked synthesizers
(:math:`\gcd(5120, 5760, 16000) = 640` MHz) this is a few dozen in-band points — a
far tighter prior than "any integer MHz," which spans tens of thousands of
candidates. Because the locked tones are exact, the match tolerance is the
*measurement* tolerance (about one analysis bin), not a clock tolerance.

Declaring a divided clock can tighten the lattice further: the AWG's half-rate
(8000 MHz) is itself a strong tone and, declared as its own fundamental, brings the
spacing down to :math:`\gcd(5120, 5760, 8000) = 320` MHz.

Each *unlocked* clock contributes a **drifting** sub-lattice: multiples of its
fundamental, matched within a wider window (the measured drift scale). Membership
marks a candidate as *expected to drift*; it never sharpens a frequency test.

The gate uses the lattice in four ways:

- **Where to look.** With a declaration the gate's frequency sweep runs over the
  lattice points (together with the legacy integer-MHz anchors as a union, so
  off-lattice instrumental tones — confirmed by the decay probe — keep gating).
- **Burden of proof.** On a locked-lattice point the prior odds of a spur are far
  higher, so the gate relaxes its evidence bar: an on-lattice tone is gated unless
  the decay probe shows it *clearly* decays like a molecular line.
- **The drifting family.** A wandering digitizer tone defeats a fixed-frequency
  decay probe, so an unlocked-lattice match is adjudicated by a drift-tolerant
  statistic (a late-versus-early band-power ratio) instead. This is what gates the
  drifting digitizer spur the example data carry.
- **Line annotation.** Any *fitted* line that lands on the lattice is stamped with
  the matched identity (for example ``640x8 (bb)`` — the eighth multiple of a
  640 MHz lattice, in the baseband frame). The annotation is the conservative half
  of the feature: it never removes a line automatically. It appears as a marker in
  ``fit show``, as a flagged column in the :doc:`Stage 6 report <stage6_review>`
  line tables, and makes an on-lattice line a one-command review removal.

An annotated line is a *candidate* instrumental artifact, not a verdict — a real
molecular line can coincide with a lattice point. Treat the flag as a prompt to
look, especially when the same identity recurs across experiments.

Timebase self-calibration
-------------------------

A free-running digitizer samples at a rate that is slightly off its nominal value,
which stretches the baseband frequency axis by a constant fractional **scale
error** :math:`\varepsilon`. That error is measurable, prior-free, from the
Rb-locked clock spurs themselves: their true frequencies are known exactly, so
their measured offsets recover :math:`\varepsilon` directly. The ``timebase``
object runs the measurement after import and persists the result; :doc:`Stage 6
<stage6_review>` applies the correction and folds its uncertainty into the reported
per-line frequency budget.

.. code-block:: console

   # Measure epsilon from the Rb-locked spur lattice and persist it
   $ ftmwpipeline timebase run exp.ftmw

   # Print the summary and the per-tone diagnostic table
   $ ftmwpipeline timebase show exp.ftmw

.. code-block:: python

   import ftmwpipeline.api as ftmw

   result = ftmw.calibrate_timebase("exp.ftmw")
   result.epsilon, result.sigma_epsilon      # fractional scale error +/- 1 sigma

``timebase show`` prints the consensus :math:`\varepsilon` and its uncertainty, the
lattice spacing, and a per-tone table grouping the tones into those **kept** in the
fit, those **rejected** for inconsistency, and the **drift controls** (unlocked
multiples, never in the fit). The estimator — how each tone's offset is read from
its phase rather than a fitted lineshape, the precision bound that follows, and how
the correction reaches the line list — is the subject of the :doc:`timebase
self-calibration methods note <methods/timebase_calibration>`.

Reading the calibration state
------------------------------

``timebase run`` and ``timebase show`` are about the *measurement*: they only
have anything to report once a self-calibration has actually been performed.
A different, more general question — "what frame is this file's frequency
data in right now, and by how much is it corrected?" — is answered by
``timebase state`` (functional API: :func:`ftmwpipeline.api.frequency_calibration`;
``Pipeline``: :meth:`~ftmwpipeline.pipeline.Pipeline.frequency_calibration`).

This is a **derived** read, not a persisted one: it is recomputed on every call
from the clock declaration (``clocks show`` — the persisted Stage 5
declaration if a fit has run, otherwise the recommended layer an importer or
``clocks set`` wrote) and, if one exists, the live timebase calibration. It
never disagrees with what a ``frame="calibrated"`` curation call will actually
apply, because it is derived the same way that call derives its correction.
It is read-only and mutates nothing.

Because it degrades to documented defaults at every step rather than raising,
it answers on a file that has been through nothing but the FID import —
**before Stage 5 or Stage 6 have ever run.** The only error case is the file
not existing at all. This makes it the read an integrator should build
against for "is this file's frequency axis calibrated," rather than
:func:`~ftmwpipeline.api.load_timebase_calibration` (raises if no measurement
exists yet) or the Stage 6 :class:`~ftmwpipeline.core.data_structures.FinalProducts`
stamp (only exists once Stage 6 has run, and describes that past build rather
than the file's current state — it can go stale if the clock declaration or
timebase calibration changes afterward).

The returned value is a
:class:`~ftmwpipeline.core.calibration.CalibrationStamp`, carrying:

- ``state`` — one of three :data:`~ftmwpipeline.core.calibration.CalibrationState`
  strings: ``"rb_locked"`` (no unlocked clock declared; the frequency axis is
  used as acquired, :math:`\varepsilon` is a null op), ``"self_calibrated"``
  (an unlocked digitizer is declared *and* a passing timebase calibration is
  present; raw and calibrated frames actually differ), or ``"uncalibrated"``
  (an unlocked digitizer is declared but nothing usable has been measured yet
  — frequencies are reported as acquired and caveated; run ``timebase run``).
- ``epsilon`` / ``sigma_epsilon`` — the fractional scale error that will be
  applied and its 1-sigma uncertainty (both ``0.0`` unless ``state`` is
  ``"self_calibrated"``).
- ``sigma_floor_khz`` — the declared systematic accuracy floor folded into
  every peak's frequency budget (``0.0`` until one is declared).
- ``probe_freq_mhz`` / ``sideband`` — the probe/LO frequency and sideband the
  calibrated frame is defined against, read from the FID header; both are
  ``None`` only when the file carries no FID header to read them from, in
  which case no raw/calibrated frame conversion is possible yet.

The examples below are from a file with an unlocked digitizer clock declared
and a measured :math:`\varepsilon = +2.2\pm0.1` ppm persisted against it.

.. code-block:: console

   $ ftmwpipeline timebase state exp.ftmw
   Frequency calibration for: exp.ftmw
     state              : self_calibrated
                          (unlocked digitizer with a measured timebase calibration applied; raw and calibrated frames differ)
     epsilon            : +2.200 +- 0.100 ppm
     sigma floor        : 0.000 kHz
     probe frequency    : 40960.000000 MHz
     sideband           : upper

   $ ftmwpipeline timebase state exp.ftmw --format json
   {
     "state": "self_calibrated",
     "epsilon": 2.2e-06,
     "sigma_epsilon": 1e-07,
     "sigma_floor_khz": 0.0,
     "probe_freq_mhz": 40960.0,
     "sideband": "upper"
   }

.. code-block:: python

   import ftmwpipeline.api as ftmw
   from ftmwpipeline import Pipeline

   stamp = ftmw.frequency_calibration("exp.ftmw")
   # CalibrationStamp(state='self_calibrated', epsilon=2.2e-06,
   #                   sigma_epsilon=1e-07, sigma_floor_khz=0.0,
   #                   probe_freq_mhz=40960.0, sideband='upper')

   # Equivalently, bound to an open Pipeline:
   Pipeline.open("exp.ftmw").frequency_calibration() == stamp   # True

On a freshly-imported file with no clock declaration at all, ``state`` reads
``"rb_locked"`` and :math:`\varepsilon` is identically zero:

.. code-block:: console

   $ ftmwpipeline timebase state exp_fresh.ftmw
   Frequency calibration for: exp_fresh.ftmw
     state              : rb_locked
                          (no unlocked clock declared; axis absolutely calibrated as acquired (eps is a null op))
     epsilon            : +0.000 +- 0.000 ppm
     sigma floor        : 0.000 kHz
     probe frequency    : 40960.000000 MHz
     sideband           : upper

Interfaces
----------

Everything on this page is read-only with respect to the fit. The two objects
behave identically across the CLI, the functional API, and the ``Pipeline`` class:

- ``clocks`` — ``show`` / ``set`` / ``add`` / ``remove`` / ``clear``
  (``get_clock_sources`` / ``set_clock_sources`` / ``remove_clock_sources`` /
  ``clear_clock_sources``).
- ``timebase`` — ``run`` / ``show`` / ``state`` (``calibrate_timebase`` /
  ``load_timebase_calibration`` / ``frequency_calibration``).

The clock declaration is persisted with the file and travels with it; the timebase
calibration is persisted as its own record and consumed by Stage 6.
