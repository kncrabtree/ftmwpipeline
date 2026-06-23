.. index::
   single: performance
   single: parallelism
   single: jobs; worker count
   single: FTMW_MAX_WORKERS
   single: report; cost
   single: fit; cost

Performance
===========

The pipeline parallelizes the expensive stages automatically and ships with
defaults tuned to give a good fit out of the box. The controls that remain are
few: how many CPU cores the work spreads across, how to make a report cheaper
when not every figure is needed, and a handful of settings that trade fitting
thoroughness against time.

Two of the stages do enough work to be worth parallelizing:

- :doc:`Stage 5 <stage5_fitting>` fits each analysis window, and independent
  windows are fit concurrently across a pool of worker processes.
- The :doc:`Stage 6 <stage6_review>` HTML report renders one figure set per
  window, and those renders run concurrently too.

Everything else (import, the FT, noise, decay-time calibration, peak detection,
window planning) is fast enough to run in a single process.

Controlling the number of cores
-------------------------------

By default each pool sizes itself to the machine: it uses ``cpu_count() - 2``
worker processes (leaving two cores for the operating system and the parent
process), and pins the linear-algebra libraries inside each worker to a single
thread so that *N* workers each running multi-threaded math do not oversubscribe
the machine. On a dedicated workstation this is usually what you want.

An override is warranted when the default does not fit the machine — a shared
login node where cores must be left free for other users, or a batch scheduler
that allocated a fixed core count the pipeline cannot see. Two controls set the
worker count, highest precedence first:

- the ``--jobs N`` (``-j N``) flag on ``fit run`` and ``report run``;
- the ``FTMW_MAX_WORKERS`` environment variable.

If neither is set, the ``cpu_count() - 2`` default applies. The flag overrides the
environment variable, which overrides the default; the value is clamped to at
least one.

.. code-block:: console

   # Cap the fit to 8 worker processes
   $ ftmwpipeline fit run exp.ftmw --jobs 8

   # Cap every pool for a whole session (e.g. a scheduler core allocation)
   $ export FTMW_MAX_WORKERS=16
   $ ftmwpipeline fit run exp.ftmw
   $ ftmwpipeline report run exp.ftmw

   # Force a single process (sequential) -- useful for debugging or profiling
   $ ftmwpipeline fit run exp.ftmw --jobs 1

The same control is available programmatically as a ``jobs`` argument:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   ftmw.fit_peaks("exp.ftmw", jobs=8)
   ftmw.report_run("exp.ftmw", jobs=8)

The worker count does **not** change the result — the fit and the rendered
figures are identical regardless of how many workers run them — so it is purely a
speed/occupancy choice, safe to set per run. And because each worker already pins
its math libraries to one thread, the ``OMP_NUM_THREADS`` /
``OPENBLAS_NUM_THREADS`` variables need not be set by hand for the pool; the
pipeline manages that to avoid oversubscription.

Making a report cheaper
-----------------------

The cost of ``report run`` is dominated by rendering the per-window figures, so
the largest savings come from rendering fewer of them. When you do not need the
full per-window atlas, scope the output (see :doc:`Stage 6 <stage6_review>` for
what each produces):

- ``--windows attention`` — render detail sections only for the windows the
  review flagged for attention, instead of every window. The index still lists
  them all.
- ``--summary`` — emit the index and methods pages only, with no per-window detail
  sections (no per-window figures rendered at all).
- ``--level1-only`` — write just the Level-1 line table (CSV/JSON/LaTeX) and skip
  the HTML report entirely; the fastest option when you only want the numbers.
- ``--no-table`` — the converse, skipping the table; minor, since the table is
  cheap.

On a large experiment ``--windows attention`` or ``--summary`` turns a
many-minute report into a quick one while keeping the parts most runs actually
read.

Trading fitting thoroughness for time
-------------------------------------

The Stage 5 fit spends most of its time on dense windows and on the iterative
passes that recover hard structure — the conservative add-one-peak loop, the
residual-rescue rounds, and the thaw/replan renegotiation. The defaults are
deliberately patient; if a run is slower than you want and you are willing to
trade some recovery of marginal lines, a few settings bound that work. Unlike
``--jobs``, these **change the result** (they are quality/cost tradeoffs, not pure
speed), so reach for them only when the default cost is a problem.

Exposed as ``fit run`` flags:

- ``--max-residual-rescue-rounds`` — cap the residual-rescue passes (each round
  re-fits the window looking for a missed line). Lower is faster; the default
  recovers more weak lines.
- ``--max-thaw-rounds`` / ``--max-replan-rounds`` — bound the cross-window
  renegotiation (local thaw and structural merge). Lower is faster.
- ``--rescue-snr-threshold`` — raise it to nominate fewer rescue candidates.
- ``--residual-edge-threshold`` — raise it to trigger thaw/replan less often.
- ``--no-fit-tau`` — hold the decay time fixed at the :doc:`Stage 2b <stage2b_tau>`
  value instead of freeing it per window; one fewer free parameter, so the solver
  converges faster (at some cost to per-window line-width fidelity).

A few cost-relevant settings are not individual flags and are set through the
:doc:`settings or a preset <settings_and_presets>` — most notably
``conservative.max_peaks`` (a hard cap on the number of peaks fit per window;
``0`` = uncapped, width-bounded) and ``baseline.enabled`` / ``baseline.order``
(the leakage-wing baseline, an extra fit on a triggered window). The default of no
peak cap is what lets the fit resolve dense forests; set a cap only if a
pathological window is dominating the run.

Earlier stages set later cost
-----------------------------

Stage 5's cost is not decided only at Stage 5. The number of fits it runs is fixed
upstream: :doc:`Stage 3 <stage3_peaks>` promotes detected peaks above an SNR
cutoff, :doc:`Stage 4 <stage4_windows>` groups the promoted peaks into analysis
windows, and Stage 5 fits each window. So the single most effective control over
fit time is often the **Stage 3 promotion cutoff**, ``peaks run --min-snr`` —
raise it and fewer candidates are promoted, which means fewer (and smaller)
windows and fewer nonlinear fits downstream. This is a detection-completeness
tradeoff, not a free speedup: the cutoff sits near the noise floor by default
(about 3\ :math:`\sigma`), and pushing it higher drops genuine weak lines along
with the noise-grass candidates. But when a run is swamped by marginal candidates,
tightening the cutoff is higher-leverage than anything inside Stage 5.

Conversely, a *better* upstream result can make the fit both better and faster.
Running :doc:`Stage 2b decay-time calibration <stage2b_tau>` gives Stage 5 a
data-driven starting decay time (per frequency band when available) instead of a
generic guess, so the solver begins near the solution and tends to converge in
fewer iterations. Stage 2b itself costs about a second, so a good decay-time
calibration can pay for itself in the fit.

Performance is best read across the whole pipeline rather than stage by stage: a
choice made at detection or calibration time propagates into the most expensive
stage. Tune the earlier, cheaper stages first.

Where the time goes
-------------------

For a sense of scale: on a typical experiment the **report** and the **dense
per-window fits** are the two costs that matter, and both scale down with more
cores (the ``--jobs`` control above); the earlier stages are seconds at most. For
a slow run, in order: make sure the pools have the cores you intend (``--jobs`` /
``FTMW_MAX_WORKERS``); check whether the fit is doing more work than you need
because of upstream choices (the Stage 3 cutoff and Stage 2b calibration above);
reach for the Stage 5 thoroughness settings if a dense spectrum is genuinely the
bottleneck; and scope the report output.
