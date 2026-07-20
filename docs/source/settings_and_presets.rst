.. index::
   single: settings
   single: presets
   single: parameter resolution
   single: persistence; settings

Settings and presets
====================

Every pipeline stage past Stage 0 has knobs — noise smoothing widths,
edge-coherence widths, line shape, decay-time bounds, rescue SNR cutoffs.
``ftmwpipeline`` exposes those knobs through a single layered resolution
chain, at the level of detail a given workflow needs:

* one-off experiments: pass keyword arguments to a stage function;
* recurring instrument workflows: load a *preset* YAML by name and
  override one or two knobs from the CLI;
* programmatic sweeps: build a settings dataclass in Python and pass
  it as ``settings=``.

All three surfaces resolve to the same settings, so the same parameters
produce the same result whichever drives the analysis.

How a stage's parameters resolve
--------------------------------

A stage's parameters come from layers that are merged per field. Highest
precedence first:

1. **explicit** — a settings dataclass passed as ``settings=`` (plus the
   handful of genuine non-knob keyword arguments a stage accepts, such as
   ``shape`` on the fit)
2. **persisted** — what the previous run of this stage on this ``.ftmw``
   file used
3. **preset** — a ``.yml`` preset loaded by name or path with ``preset=``
4. **recommended** — an upstream-stage hint (e.g., Stage 2b's
   ``recommended_shape`` for Stage 5); empty for stages without an
   upstream feeder
5. **hard defaults** — the library's stock values

Each knob walks this chain independently. A value you set in an explicit
``settings=`` bundle wins; everything else falls through one layer at a
time until it hits a concrete value, and the hard defaults are
guaranteed to fill any remaining gap so the resolved settings instance is
always complete.

A ``.yml`` preset never overrides a value persisted in the ``.ftmw``
file. The preset layer sits *below* the persisted layer: a preset only
seeds fields the file has not already fixed. This is what keeps a shared
``.ftmw`` reproducible — the file reproduces the same result on its own,
regardless of any preset a recipient happens to have loaded. To force a
value regardless of what the file holds, pass it explicitly through
``settings=``.

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
     - flat (no sub-blocks)
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
       ``rescue``, ``thaw``, ``spur``, ``baseline``,
       ``doublet_alternative``, ``peak_survival``
     - ``processing_parameters/stage5_fit``

Each row is independent: you can tune Stage 2 noise smoothing without
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

At the command line, each stage exposes a flag per knob. A flag beats
every other layer for its field; unspecified knobs flow through the chain
unchanged:

.. code-block:: shell

   ftmwpipeline fit run exp.ftmw --shape gaussian --max-decay-factor 3.0

Each flag is reconstructed into the stage's settings object before the
stage runs, so a flag and the corresponding ``settings=`` field are the
same explicit override reached by two routes.

In Python, a few knobs survive as first-class convenience keyword
arguments (the fit's ``shape`` and its τ-override pair), part of the
explicit layer as well:

.. code-block:: python

   ftmw.fit_peaks("exp.ftmw", shape="gaussian")

Every other knob is set through a ``settings=`` dataclass (below) or a
preset YAML.

Presets
~~~~~~~

A preset is a named bundle of knob values that brings an experiment or
instrument's recipe under version control. A preset can cover one stage
or several stages at once: a single YAML file can carry a ``stage2:``
block, a ``stage2b:`` block, a ``stage3:`` block, a ``stage4:`` block,
and a ``stage5:`` block side-by-side. Each stage's loader reads only
its own block and ignores the rest, so one preset can drive a complete
instrument-specific recipe.

``ftmwpipeline`` ships exactly one packaged preset, ``defaults``: every
pipeline knob written at its package-wide default value, and nothing else.
Applying it is a no-op — ``--preset defaults`` reproduces exactly what the
pipeline does with no preset at all — so its purpose is documentation. It is
the standard, copy-and-edit starting point: open it, delete the blocks you
do not care about, and change the few values you want to pin.

Use it by bare name (mostly to read it; as a recipe you would copy and edit
it first):

.. code-block:: shell

   ftmwpipeline fit run exp.ftmw --preset defaults

Or, the usual workflow, load a YAML file you wrote yourself by path:

.. code-block:: shell

   ftmwpipeline fit run exp.ftmw --preset ./my_lab_recipe.yaml

The same preset name (or path) passed to any stage CLI subcommand loads only
that stage's block:

.. code-block:: shell

   ftmwpipeline noise run    exp.ftmw --preset ./my_lab_recipe.yaml
   ftmwpipeline tau run      exp.ftmw --preset ./my_lab_recipe.yaml
   ftmwpipeline peaks run    exp.ftmw --preset ./my_lab_recipe.yaml
   ftmwpipeline windows run  exp.ftmw --preset ./my_lab_recipe.yaml
   ftmwpipeline fit run      exp.ftmw --preset ./my_lab_recipe.yaml

A stage whose block is missing from the preset loads an empty
``XxxSettings`` and falls through to the next layer of the resolver —
nothing breaks.

A preset and explicit overrides compose: the explicit value wins per
field, so you can adopt a preset's recipe and tweak one knob in the same
call. At the command line, that is a preset plus a per-knob flag:

.. code-block:: shell

   ftmwpipeline fit run exp.ftmw \
     --preset ./my_lab_recipe.yaml \
     --max-residual-rescue-rounds 3

The flag lands in the explicit layer; the preset seeds the preset layer
beneath it (and beneath anything the file has already persisted).

Preset YAML format
~~~~~~~~~~~~~~~~~~

A preset is a YAML mapping with up to five top-level stage blocks —
``stage2`` (noise), ``stage2b`` (τ calibration), ``stage3`` (peaks),
``stage4`` (windows), ``stage5`` (the fit) — plus optional ``name`` and
``description`` metadata that the loader carries but ignores. Within a
stage block, knobs are grouped into the same named sub-blocks the settings
dataclass uses (for example ``stage5`` has ``tau``, ``seeder``,
``conservative``, ``penalties``, ``rescue``, ``thaw``, ``spur``,
``baseline``, ``doublet_alternative``, ``peak_survival``; ``stage2`` is
flat, with no sub-blocks). Every field is optional: omit a knob and it
falls through the resolver; write it and it is pinned at the preset layer.
An unknown stage block, sub-block, or field name is rejected with an error
so typos surface loudly rather than silently doing nothing.

.. code-block:: yaml

   name: my_lab_recipe
   description: my instrument, tuned

   stage2:
     window_mhz: 120.0            # flat — Stage 2 has no sub-blocks
   stage3:
     promotion:
       min_snr: 4.0
   stage5:
     shape: gaussian              # see the note below before pinning this
     tau:
       max_decay_factor: 4.0

For the meaning, type, and suggested values of every knob, read the
shipped ``defaults`` preset (it lists them all at their defaults) or run
``ftmwpipeline settings show`` for a file's resolved values and
``ftmwpipeline scan list`` for the tunable knobs and their sweep grids.

Line shape and the Stage 2b recommendation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``defaults`` preset leaves ``stage5.shape`` unset for a reason. Stage 2b
runs a three-way lineshape
vote (``stage2b.recommendation.auto_recommend``, default ``true``) and
stamps the winning shape on the ``.ftmw`` as the *recommended* layer.
Stage 5 consumes that recommendation **only when ``stage5.shape`` is
unset** — because a pinned ``shape:`` sits in the higher-precedence preset
layer and silently overrides the recommendation. So:

* leave ``stage5.shape`` out to let Stage 2b choose the shape (the
  out-of-the-box behavior);
* set ``stage5.shape: lorentzian`` or ``gaussian`` to force it, knowingly
  bypassing the vote.

Declaring the instrument clock tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``stage5.spur.clocks`` block declares an instrument's clock-source
fundamentals so the spur detector can use the locked intermod *lattice*
instead of the generic integer-MHz anchor (and add a drift lane for any
unlocked source). The ``defaults`` preset ships this commented out as a
template; the Blackchirp 2638 instrument is shown as a worked example:

.. code-block:: yaml

   stage5:
     spur:
       clocks:
         - {freq_mhz: 5120, locked: true,  label: synth-downconv}
         - {freq_mhz: 5760, locked: true,  label: synth-upconv}
         - {freq_mhz: 16000, locked: true, label: awg}
         - {freq_mhz: 8000, locked: true,  label: awg-half}
         - {freq_mhz: 6250, locked: false, label: scope-adc}

Declare chain *fundamentals* only — harmonics and products derive. The same
declaration can be supplied at import from a ``clocks.csv`` sidecar instead;
see :doc:`clock_declaration`.

Constructing settings in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For programmatic sweeps (comparing several knob variants from a notebook,
or building a recipe at runtime), construct the relevant settings dataclass
directly:

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
   s.window_mhz = 120.0
   s.smoothing_mhz = 600.0
   ftmw.estimate_noise("exp.ftmw", settings=s)

   from ftmwpipeline.core.peak_detection_settings import PeakDetectionSettings

   s = PeakDetectionSettings()
   s.promotion.min_snr = 4.0
   s.savgol.sg_window = 13
   ftmw.detect_peaks("exp.ftmw", settings=s)

``settings=`` and ``preset=`` populate *different* layers, so you can
pass both in one call. A ``settings=`` bundle is the **explicit** layer
and outranks the persisted record; a ``preset=`` YAML is the **preset**
layer and is outranked by it. Reach for ``settings=`` to force values
regardless of what the file holds, and for ``preset=`` to supply a recipe
that defers to anything already chosen on the file. When both are given,
the ``settings=`` fields win per field and the preset seeds the rest:

.. code-block:: python

   s = StageFitSettings()
   s.rescue.max_rounds = 3
   ftmw.fit_peaks("exp.ftmw", preset="./my_lab_recipe.yaml", settings=s)

Persistence and auto-inheritance
--------------------------------

Every time a stage runs, its resolved settings are stamped into the
``.ftmw`` file under the persisted record for that stage (see the
*Persisted at* column in the table above). The next call to the same
stage on that file inherits those settings unless you override them, so
a sequence like:

.. code-block:: shell

   ftmwpipeline fit run exp.ftmw --preset ./my_lab_recipe.yaml
   ftmwpipeline fit show exp.ftmw
   # ... look at the fit, decide to tighten rescue ...
   ftmwpipeline fit run exp.ftmw --max-residual-rescue-rounds 3

behaves as expected: the second fit keeps the settings from the first call's
preset and only tightens the rescue; the prior recipe need not be re-supplied.

The persisted block is structured to be inspectable on disk::

   $ h5dump -A exp.ftmw | head -40
   /processing_parameters/stage5_fit
     @creation_time = "2026-05-27T17:42:11..."
     @preset_name = "my_lab_recipe"
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
     window_mhz: 120.0
     smoothing_mhz: 600.0

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
standard example: Stage 2b's τ calibration writes a
``recommended_shape`` attribute on its output group (``lorentzian``,
``gaussian``, or ``voigt``), and Stage 5's resolver reads it as the
*recommended* layer of the shape field. The attribute carries the
``__None__`` sentinel until Stage 2b's 3-way L/G/V discriminator runs;
once it does, Stage 5 picks the recommendation up automatically — one
step weaker than the file's persisted value, two steps weaker than an
explicit argument or preset. The pipeline recommends a line shape, but an
explicit or persisted choice always overrides it.

If the file has persisted ``shape: gaussian`` and Stage 2b later recommends
Lorentzian, the persisted value wins. An explicit ``--shape lorentzian``
likewise always wins, regardless of the recommendation.

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

The matching HDF5 persistence groups are the *Persisted at* column of the
settings table above.

Other useful references:

* ``src/ftmwpipeline/presets/`` — the packaged preset YAML files.
* :mod:`ftmwpipeline.core.settings` — the Stage 1
  :class:`~ftmwpipeline.core.settings.FTSettings` precedent the
  per-stage pattern extends.
