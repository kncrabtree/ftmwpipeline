.. index::
   single: settings
   single: presets
   single: parameter resolution
   single: persistence; settings

Settings and presets
====================

Every pipeline stage past Stage 0 has knobs — adaptive-binning fractions,
edge-coherence widths, line shape, decay-time bounds, rescue SNR cutoffs.
``ftmwpipeline`` exposes those knobs through a single layered resolution
chain so you can pick the level of detail that matches your workflow:

* one-off experiments: pass keyword arguments to a stage function and
  forget about it;
* recurring instrument workflows: load a *preset* YAML by name and
  override one or two knobs from the CLI;
* programmatic sweeps: build a settings dataclass in Python and pass
  it as ``settings=``.

The same call producing the same result is the goal regardless of which
surface you use. This page walks through the mental model, the three
input surfaces, persistence behaviour, and how to write your own presets
that span multiple stages.

The mental model
----------------

A stage's parameters come from layers that are merged per field. Highest
precedence first:

1. **explicit kwargs** — ``ftmw.fit_peaks(..., max_decay_factor=3.0)``
2. **preset / settings** — a YAML preset loaded by name, or a settings
   dataclass you built in Python
3. **persisted** — what the previous run of this stage on this ``.ftmw``
   file used
4. **recommended** — an upstream-stage hint (e.g., Stage 2b's
   ``recommended_shape`` for Stage 5); empty for stages without an
   upstream feeder
5. **hard defaults** — the library's stock values

Each knob walks this chain independently. If you explicitly set
``max_decay_factor=3.0``, your value wins. Everything else falls through
one layer at a time until it hits a concrete value. The hard defaults
are guaranteed to fill any remaining gap so the resolved settings
instance is always complete.

The same template applies to every stage that exposes knobs. The five
settings dataclasses, in pipeline order:

.. list-table::
   :header-rows: 1
   :widths: 8 35 35 22

   * - Stage
     - Settings dataclass
     - Sub-blocks
     - Persisted at
   * - 1
     - :class:`~ftmwpipeline.core.settings.FTSettings`
     - flat (no sub-blocks)
     - ``processing_parameters/ft_processing``
   * - 2
     - :class:`~ftmwpipeline.core.noise_settings.NoiseSettings`
     - ``binning``, ``skewness``, ``smoothing``, ``skirt_exclusion``
     - ``processing_parameters/stage2_noise``
   * - 2b
     - :class:`~ftmwpipeline.core.tau_calibration_settings.TauCalibrationSettings`
     - ``stft``, ``polish``, ``aggregation``, ``band``, ``gaussian``,
       ``recommendation``
     - ``processing_parameters/stage2b_tau``
   * - 3
     - :class:`~ftmwpipeline.core.peak_detection_settings.PeakDetectionSettings`
     - ``promotion``, ``savgol``, ``primary_pass``, ``gap_pass``
     - ``processing_parameters/stage3_peaks``
   * - 4
     - :class:`~ftmwpipeline.core.window_planning_settings.WindowPlanningSettings`
     - ``coherence``, ``clustering``, ``contributor``, ``leakage``
     - ``processing_parameters/stage4_windows``
   * - 5
     - :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`
     - ``shape``, ``tau``, ``seeder``, ``conservative``, ``penalties``,
       ``rescue``, ``thaw``
     - ``processing_parameters/stage5_fit``

Each row is independent: you can tune Stage 2 noise binning without
touching Stage 5, override Stage 3 SNR cutoffs without re-running the
τ calibration, and so on. The persisted layer for one stage is
unrelated to the persisted layer for another.

Three ways to drive a stage
---------------------------

The three input surfaces work identically across every stage. The
examples below use Stage 5 (the fit step) because it has the most
visible knobs; substitute ``estimate_noise``, ``calibrate_tau``,
``detect_peaks``, ``assign_windows``, or ``fit_peaks`` and the same
patterns apply.

Stock defaults
~~~~~~~~~~~~~~

The simplest call uses every hard default — Lorentzian shape,
``max_decay_factor=5``, ``rescue.max_rounds=5``, and the rest of the
documented stock values:

.. code-block:: python

   import ftmwpipeline.api as ftmw
   ftmw.fit_peaks("exp.ftmw")

Equivalent at the CLI::

   ftmwpipeline fit run exp.ftmw

One-off overrides
~~~~~~~~~~~~~~~~~

Set individual knobs as keyword arguments. They beat every other layer
per field; unspecified knobs flow through the chain unchanged:

.. code-block:: python

   ftmw.fit_peaks(
       "exp.ftmw",
       shape="gaussian",
       max_decay_factor=3.0,
   )

The CLI exposes a flag for the historically-public knobs of each stage
(``--shape``, ``--max-decay-factor``, ``--max-residual-rescue-rounds``,
``--edge-m``, ``--min-snr``, …). Knobs beyond that public surface — the
instrument-tunable ones such as Stage 2's
``binning.subdivision_threshold`` or Stage 3's
``gap_pass.gap_mask_edge_threshold`` — are reachable through the Python
``settings=`` kwarg or a preset YAML.

Presets
~~~~~~~

A preset is a named bundle of knob values that brings an experiment or
instrument's recipe under version control. A preset can cover one stage
or several stages at once: a single YAML file can carry a ``stage2:``
block, a ``stage2b:`` block, a ``stage3:`` block, a ``stage4:`` block,
and a ``stage5:`` block side-by-side. Each stage's loader reads only
its own block and ignores the rest, so one preset can drive a complete
instrument-specific recipe.

``ftmwpipeline`` ships three:

* ``gaussian_default`` — clean Gaussian baseline; otherwise stock.
* ``lorentzian_legacy`` — the historical Lorentzian default, named
  explicitly for A/B comparisons.
* ``instrument_bc_2638`` — starting point for the BlackChirp 2638
  fixture: Gaussian shape, per-band τ routing on.

Use a packaged preset by bare name:

.. code-block:: python

   ftmw.fit_peaks("exp.ftmw", preset="instrument_bc_2638")

.. code-block:: shell

   ftmwpipeline fit run exp.ftmw --preset instrument_bc_2638

Or load a YAML file you wrote yourself by path:

.. code-block:: shell

   ftmwpipeline fit run exp.ftmw --preset ./my_lab_recipe.yaml

The same preset name passed to any stage CLI subcommand loads only that
stage's block:

.. code-block:: shell

   ftmwpipeline noise run    exp.ftmw --preset instrument_bc_2638
   ftmwpipeline tau run      exp.ftmw --preset instrument_bc_2638
   ftmwpipeline peaks run    exp.ftmw --preset instrument_bc_2638
   ftmwpipeline windows run  exp.ftmw --preset instrument_bc_2638
   ftmwpipeline fit run      exp.ftmw --preset instrument_bc_2638

A stage whose block is missing from the preset loads an empty
``XxxSettings`` and falls through to the next layer of the resolver —
nothing breaks.

Presets and explicit kwargs compose: kwargs win per field, so you can
adopt a preset's recipe and tweak one knob:

.. code-block:: shell

   ftmwpipeline fit run exp.ftmw \
     --preset instrument_bc_2638 \
     --max-residual-rescue-rounds 3

The Python ``settings=`` kwarg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For programmatic sweeps — comparing several knob variants from a
notebook, or building a recipe at runtime — construct the relevant
settings dataclass directly:

.. code-block:: python

   from ftmwpipeline.core.stage_fit_settings import (
       StageFitSettings, ShapeSpec
   )
   from ftmwpipeline.core.peak_shape import PeakShape

   s = StageFitSettings(shape=ShapeSpec(kind=PeakShape.GAUSSIAN))
   s.tau.max_decay_factor = 3.0
   s.rescue.max_rounds = 3
   ftmw.fit_peaks("exp.ftmw", settings=s)

Same shape for any stage:

.. code-block:: python

   from ftmwpipeline.core.noise_settings import NoiseSettings

   s = NoiseSettings()
   s.smoothing.smoothing_window_mhz = 100.0
   s.skirt_exclusion.strong_peak_snr = 25.0
   ftmw.estimate_noise("exp.ftmw", settings=s)

   from ftmwpipeline.core.peak_detection_settings import PeakDetectionSettings

   s = PeakDetectionSettings()
   s.promotion.min_snr = 4.0
   s.savgol.sg_window = 13
   ftmw.detect_peaks("exp.ftmw", settings=s)

``settings=`` and ``preset=`` are alternative ways to populate the same
layer — passing both raises ``ValueError``. (If you want both a preset
and dataclass-level overrides, use a preset name plus explicit kwargs
for the override.)

Persistence and auto-inheritance
--------------------------------

Every time a stage runs, its resolved settings are stamped into the
``.ftmw`` file under the canonical record for that stage (see the
*Persisted at* column in the table above). The next call to the same
stage on that file inherits those settings unless you override them, so
a sequence like:

.. code-block:: shell

   ftmwpipeline fit run exp.ftmw --preset instrument_bc_2638
   ftmwpipeline fit show exp.ftmw
   # ... look at the fit, decide to tighten rescue ...
   ftmwpipeline fit run exp.ftmw --max-residual-rescue-rounds 3

does what you probably expect: the second fit keeps the Gaussian shape
and the per-band τ routing from the first call's preset, and just
tightens the rescue. You don't have to re-supply ``--preset`` to keep
the prior recipe.

The persisted block is structured to be inspectable on disk::

   $ h5dump -A exp.ftmw | head -40
   /processing_parameters/stage5_fit
     @creation_time = "2026-05-27T17:42:11..."
     @preset_name = "instrument_bc_2638"
     shape/
       @kind = "gaussian"
     tau/
       @max_decay_factor = 5.0
       @per_band_tau = TRUE
       @tau_penalty_lambda = 500.0
       ...
     conservative/
       @significance = 0.05
       @max_peaks = 8
       ...
     rescue/
       @max_rounds = 5
       ...

Every sub-block of every stage is its own HDF5 group so you can grep one
block in isolation. Unset fields use the ``__None__`` sentinel string
(the same convention Stage 1's ``FTSettings`` uses).

The ``preset_name`` attribute records the bare name (or path) you
supplied to ``--preset`` for that run. The reproducibility recipe is
straightforward: the resolved values plus that name describe the run
exactly.

Writing your own preset
-----------------------

A preset is a small YAML file. Each stage's settings sit under a
top-level per-stage block (``stage2:``, ``stage2b:``, ``stage3:``,
``stage4:``, ``stage5:``); blocks compose freely so one YAML can drive
the whole pipeline:

.. code-block:: yaml

   name: my_lab_recipe
   description: |
     Whatever your lab calls this recipe. Multi-line markdown ok.

   stage2:
     smoothing:
       smoothing_window_mhz: 100.0
     skirt_exclusion:
       strong_peak_snr: 25.0

   stage2b:
     stft:
       n_seg: 10
     polish:
       polish_snr_cap: 9.0
     gaussian:
       snr_min: 20.0

   stage3:
     promotion:
       min_snr: 4.0
       weak_medium_snr: 12.0
     savgol:
       sg_window: 13
     primary_pass:
       primary_window: blackmanharris

   stage4:
     coherence:
       edge_m: 64
       edge_threshold: 8.0
     clustering:
       max_window_width_mhz: 40.0
     contributor:
       min_freeze_snr: 50.0

   stage5:
     shape: gaussian
     tau:
       max_decay_factor: 3.0
       per_band_tau: true
     conservative:
       max_peaks: 6
     rescue:
       max_rounds: 3
       snr_threshold: 3.0

A few rules:

* Each per-stage block is independent — drop the ones you don't need.
  A preset with only a ``stage5:`` block leaves Stages 2, 2b, 3, and 4
  on hard defaults; a preset with only a ``stage2:`` block leaves
  Stage 5 alone, and so on.
* Inside a stage block, each sub-block (e.g., ``stage5.tau``,
  ``stage3.promotion``) holds a flat map of ``field_name: value``. The
  fields available are listed in the *Sub-blocks* column of the table
  above and in each dataclass's docstring.
* Stage 5's ``shape`` accepts the short form (``shape: gaussian``); the
  long form (``shape: {kind: gaussian}``) also works and leaves room
  for future shape-specific parameter blocks (e.g., Voigt).
* Only set fields you care about. Anything omitted stays ``None`` so
  the resolver falls through to the next layer (probably the hard
  defaults).
* Unknown keys raise ``ValueError`` at load time, so typos surface
  immediately rather than silently doing the wrong thing.
* ``name`` and ``description`` at the top level are documentation —
  the parser preserves them but the stages don't use them.

To use a YAML you wrote, pass its path:

.. code-block:: shell

   ftmwpipeline fit run exp.ftmw --preset ./my_lab_recipe.yaml

To ship one alongside the package, drop it into
``src/ftmwpipeline/presets/`` and refer to it by bare name.

Cross-stage recommendations
---------------------------

The *recommended* layer of the resolution chain is where one stage
hands a hint to a later stage. The Stage 2b → Stage 5 path is the
canonical example: Stage 2b's τ calibration writes a
``recommended_shape`` attribute on its output group (``lorentzian``,
``gaussian``, or ``voigt``), and Stage 5's resolver reads it as the
*recommended* layer of the shape field. The attribute carries the
``__None__`` sentinel until Stage 2b's 3-way L/G/V discriminator runs;
once it does, Stage 5 picks the recommendation up automatically — one
step weaker than what you've persisted on the file, two steps weaker
than an explicit kwarg or preset. The mental model is: the library has
an opinion about the line shape, but you always get to override.

If you've persisted ``shape: gaussian`` and Stage 2b later recommends
Lorentzian, the persisted value wins (you already chose). Likewise, an
explicit ``--shape lorentzian`` always wins, regardless of the
recommendation.

The *recommended* layer of the other four stages (2, 2b, 3, 4) is
reserved but currently empty — no upstream feeder produces a hint for
those stages yet. The layer is kept in every resolver's signature so a
future cross-stage recommender (e.g., a Stage 1 ``T_active``-driven
Stage 2 smoothing-window suggestion, or a Stage 2b ``τ_maj`` feeder
into Stage 4's ``leakage.tau_us``) can land without API churn.

Where to look in the codebase
-----------------------------

Per-stage settings modules (all share the same architectural template):

* :mod:`ftmwpipeline.core.noise_settings` —
  :class:`~ftmwpipeline.core.noise_settings.NoiseSettings`, with
  :func:`~ftmwpipeline.core.noise_settings.resolve` and
  :func:`~ftmwpipeline.core.noise_settings.load_preset`.
* :mod:`ftmwpipeline.core.tau_calibration_settings` —
  :class:`~ftmwpipeline.core.tau_calibration_settings.TauCalibrationSettings`.
* :mod:`ftmwpipeline.core.peak_detection_settings` —
  :class:`~ftmwpipeline.core.peak_detection_settings.PeakDetectionSettings`.
* :mod:`ftmwpipeline.core.window_planning_settings` —
  :class:`~ftmwpipeline.core.window_planning_settings.WindowPlanningSettings`.
* :mod:`ftmwpipeline.core.stage_fit_settings` —
  :class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`.

Matching HDF5 persistence modules in ``ftmwpipeline.io``:

* ``noise_settings_serialization`` →
  ``processing_parameters/stage2_noise``
* ``tau_calibration_settings_serialization`` →
  ``processing_parameters/stage2b_tau``
* ``peak_detection_settings_serialization`` →
  ``processing_parameters/stage3_peaks``
* ``window_planning_settings_serialization`` →
  ``processing_parameters/stage4_windows``
* ``stage_fit_settings_serialization`` →
  ``processing_parameters/stage5_fit``

Other useful references:

* ``src/ftmwpipeline/presets/`` — the packaged preset YAML files.
* :mod:`ftmwpipeline.core.settings` — the Stage 1
  :class:`~ftmwpipeline.core.settings.FTSettings` precedent the
  per-stage pattern extends.
