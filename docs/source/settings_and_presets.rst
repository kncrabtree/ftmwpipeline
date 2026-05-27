Settings and presets
====================

Every pipeline stage has knobs — line shape, decay-time bounds, rescue
SNR thresholds, edge-coherence widths. ``ftmwpipeline`` exposes those
knobs through a layered resolution chain so you can pick the level of
detail that matches your workflow:

* one-off experiments: pass keyword arguments to ``fit_peaks`` and
  forget about it;
* recurring instrument workflows: load a *preset* YAML by name and
  override one or two knobs from the CLI;
* programmatic sweeps: build a ``StageFitSettings`` dataclass in Python
  and pass it as ``settings=``.

The same call producing the same fit is the goal regardless of which
surface you use. This page walks through the mental model, the three
input surfaces, persistence behaviour, and how to write your own
presets.

The mental model
----------------

A fit's parameters come from layers that are merged per field. Highest
precedence first:

1. **explicit kwargs** — ``fit_peaks(..., max_decay_factor=3.0)``
2. **preset / settings** — a YAML preset loaded by name, or a
   ``StageFitSettings`` instance you built in Python
3. **persisted** — what the previous fit on this ``.ftmw`` file used
4. **recommended** — a Stage 2b hint (e.g., "Gaussian fits this
   experiment better"); currently a placeholder for a forthcoming
   discriminator
5. **hard defaults** — the library's stock values

Each Stage 5 knob (there are about thirty) walks this chain
independently. If you explicitly set ``max_decay_factor=3.0``, your
value wins. Everything else falls through one layer at a time until it
hits a concrete value. The hard defaults are guaranteed to fill any
remaining gap so the resolved settings instance is always complete.

The same idea applies stage-wide: Stage 1's ``FTSettings`` already
works this way (``explicit > persisted > recommended > hard default``),
and the remaining stages will follow Stage 5's pattern over time.

Three ways to drive the fit
---------------------------

Stock defaults
~~~~~~~~~~~~~~

The simplest call uses every hard default — Lorentzian shape, ``max_decay_factor=5``,
``rescue.max_rounds=5``, and the rest of the documented stock values:

.. code-block:: python

   import ftmwpipeline.api as ftmw
   ftmw.fit_peaks("exp.ftmw")

Equivalent at the CLI::

   ftmwpipeline fit-peaks exp.ftmw

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

The CLI exposes a flag for each kwarg that's already wired up
(``--shape``, ``--max-decay-factor``, ``--max-residual-rescue-rounds``,
…). Knobs not yet on the CLI surface — for instance, the seeder
thresholds — are reachable through the Python ``settings=`` kwarg or a
preset YAML.

Presets
~~~~~~~

A preset is a named bundle of knob values that bring an experiment or
instrument's recipe under version control. ``ftmwpipeline`` ships
three:

* ``gaussian_default`` — clean Gaussian baseline; otherwise stock.
* ``lorentzian_legacy`` — the historical Lorentzian default, named
  explicitly for A/B comparisons.
* ``instrument_bc_2638`` — starting point for the BlackChirp 2638
  fixture: Gaussian shape, per-band τ routing on.

Use a packaged preset by bare name:

.. code-block:: python

   ftmw.fit_peaks("exp.ftmw", preset="instrument_bc_2638")

.. code-block:: shell

   ftmwpipeline fit-peaks exp.ftmw --preset instrument_bc_2638

Or load a YAML file you wrote yourself by path:

.. code-block:: shell

   ftmwpipeline fit-peaks exp.ftmw --preset ./my_lab_recipe.yaml

Presets and explicit kwargs compose: kwargs win per field, so you can
adopt a preset's recipe and tweak one knob:

.. code-block:: shell

   ftmwpipeline fit-peaks exp.ftmw \
     --preset instrument_bc_2638 \
     --max-residual-rescue-rounds 3

The Python ``settings=`` kwarg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For programmatic sweeps — comparing several knob variants from a
notebook, or building a recipe at runtime — construct a
:class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings`
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

``settings=`` and ``preset=`` are alternative ways to populate the same
layer — passing both raises ``ValueError``. (If you want both a preset
and dataclass-level overrides, use a preset name plus explicit kwargs
for the override.)

Persistence and auto-inheritance
--------------------------------

Every time you call ``fit_peaks``, the resolved settings are stamped
into the ``.ftmw`` file under ``processing_parameters/stage5_fit``.
The next call on the same file inherits those settings unless you
override them, so a sequence like:

.. code-block:: shell

   ftmwpipeline fit-peaks exp.ftmw --preset instrument_bc_2638
   ftmwpipeline visualize-fit exp.ftmw
   # ... look at the fit, decide to tighten rescue ...
   ftmwpipeline fit-peaks exp.ftmw --max-residual-rescue-rounds 3

does what you probably expect: the second fit keeps the Gaussian shape
and the per-band τ routing from the first call's preset, and just
tightens the rescue. You don't have to re-supply ``--preset`` to keep
the prior recipe.

The persisted block is structured to be inspectable on disk::

   $ h5dump -A exp.ftmw | head -40
   /processing_parameters/stage5_fit
     @creation_time = "2026-05-26T17:42:11..."
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

Every sub-block (``shape/``, ``tau/``, ``seeder/``, ``conservative/``,
``penalties/``, ``rescue/``, ``thaw/``) is its own HDF5 group so you
can grep one block in isolation. Unset fields use the ``__None__``
sentinel string (same convention Stage 1's ``FTSettings`` uses).

The ``preset_name`` attribute records the bare name (or path) you
supplied to ``--preset`` for that fit. The reproducibility recipe is
straightforward: the resolved values plus that name describe the fit
exactly.

Writing your own preset
-----------------------

A preset is a small YAML file. The Stage 5 settings sit under a
top-level ``fit:`` block so future stage-spanning presets can carry an
``ft:`` block alongside without breaking the format:

.. code-block:: yaml

   name: my_lab_recipe
   description: |
     Whatever your lab calls this recipe. Multi-line markdown ok.

   fit:
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

* ``shape`` is the line-shape selector. Use the short form
  (``shape: gaussian``); the long form
  (``shape: {kind: gaussian}``) also works and leaves room for future
  shape-specific parameter blocks (e.g., Voigt).
* ``tau``, ``seeder``, ``conservative``, ``penalties``, ``rescue``,
  ``thaw`` are the six sub-blocks; each holds a flat map of
  ``field_name: value``.
* Only set fields you care about. Anything omitted stays ``None`` so
  the resolver falls through to the next layer (probably the hard
  defaults).
* Unknown keys raise ``ValueError`` at load time, so typos surface
  immediately rather than silently doing the wrong thing.
* ``name`` and ``description`` at the top level are documentation —
  the parser preserves them but the fit doesn't use them.

To use a YAML you wrote, pass its path:

.. code-block:: shell

   ftmwpipeline fit-peaks exp.ftmw --preset ./my_lab_recipe.yaml

To ship one alongside the package, drop it into
``src/ftmwpipeline/presets/`` and refer to it by bare name.

The shape recommendation hint
-----------------------------

Stage 2b (the τ calibration) writes a ``recommended_shape`` attribute
on its output group as a contract for a forthcoming L/G discriminator
that compares the Lorentzian and Gaussian τ calibrations and suggests
whichever fits the experiment better. The attribute carries the
``__None__`` sentinel by default — there's no recommendation until the
discriminator is written.

When a concrete recommendation lands there, Stage 5 will pick it up
automatically as the *recommended* layer of the resolution chain — one
step weaker than what you've persisted on the file, two steps weaker
than an explicit kwarg or preset. The mental model is: the library has
an opinion about the line shape, but you always get to override.

If you've persisted ``shape: gaussian`` and Stage 2b later recommends
Lorentzian, the persisted value wins (you already chose). Likewise, an
explicit ``--shape lorentzian`` always wins, regardless of the
recommendation.

Migration from per-kwarg calls
------------------------------

The legacy keyword arguments still work — they collect into a
``StageFitSettings`` under the hood. Code written before this design
landed:

.. code-block:: python

   ftmw.fit_peaks(
       "exp.ftmw",
       shape="gaussian",
       max_decay_factor=3.0,
       max_residual_rescue_rounds=3,
   )

does exactly the same thing as it always did. No deprecation warnings
are emitted today; they'll arrive on the next release cycle. If you're
running parameter sweeps, the win is more about the ergonomics of one
YAML diff per variant than about new behaviour, and that's where the
preset surface earns its keep.

Where to look in the codebase
-----------------------------

* :mod:`ftmwpipeline.core.stage_fit_settings` — the
  ``StageFitSettings`` dataclass, the resolution chain
  (:func:`~ftmwpipeline.core.stage_fit_settings.resolve`), YAML I/O,
  and :func:`~ftmwpipeline.core.stage_fit_settings.load_preset`.
* ``src/ftmwpipeline/presets/`` — the packaged preset YAML files.
* :mod:`ftmwpipeline.io.stage_fit_settings_serialization` — HDF5
  persistence at ``processing_parameters/stage5_fit``.
* :mod:`ftmwpipeline.core.settings` — the Stage 1 ``FTSettings``
  precedent the Stage 5 pattern extends.
