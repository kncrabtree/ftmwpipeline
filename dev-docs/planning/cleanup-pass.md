# Repository cleanup pass

Tracking doc for the post-documentation repository cleanup: dead-code removal,
duplication extraction, top-level-doc refresh, stray-artifact removal, and the
batched lint/format pass. Multi-session; this doc is the durable record so work
survives context boundaries.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.

## Scope decisions (user, locked)

- **Dead code:** remove the 5 pure-stub modules **and** `io/experimental_formats.py`
  (migrate its one test consumer onto the live loader path first).
- **Duplication:** extract **everything** — all of F1–F12 below, including the
  large settings-framework / serialization unifications.
- **`output/freqcal-repro/`:** delete.
- Docs refresh (README / STATUS / strategy `rdc`) and the final
  black/isort/mypy + green-test pass are in scope regardless.

## Acceptance bar

- Touched stage tests + cross-interface consistency green after each phase;
  full suite green at the end.
- Behavior-preserving refactors must not change results. For the
  serialization (F2) and fit-cleanup (F9) changes, a fresh-fixture golden of the
  persisted fields / fitted table must be byte-identical before vs after
  (`fit_peaks` is deterministic — table equality is the bar). Settings round-trip
  is already golden-tested.
- **Golden harness** (`scratch/cleanup-golden/golden.py`, gitignored, persists
  across sessions): builds the 2638 experiment end-to-end and snapshots the fit
  (512 per-peak rows + per-window tau/χ²ᵣ scalars). `python
  scratch/cleanup-golden/golden.py check` diffs the current tree against the
  stored `fit_2638.golden.txt`; `... record` refreshes the reference. Use `check`
  for every byte-identity-sensitive item instead of the stash→rebuild→diff dance;
  only `record` after a *deliberate* behavior change. Reference recorded at the
  F9-committed state (== pre-F9 output).
- New code black/isort/mypy-clean; the repo-wide format pass is the last step so
  its churn stays separate from the content changes.

## Phase 1 — stray artifacts (untracked housekeeping) — DONE

- [x] Delete root `param_persistence_enhanced_spectrum.png`,
  `pipeline_to_functional_params_enhanced_spectrum.png`,
  `test_pipeline_enhanced_spectrum.png`.
- [x] Delete leftover `src/ftmwpipeline/window_assignment/` (only a stale
  `__pycache__` survives the deleted Stage 4 stub package).
- [x] Delete `examples/blackchirp_data/363/.ipynb_checkpoints/`.
- [x] Delete `output/` (the `freqcal-repro/` run output — misplaced from
  `scratch/`).

## Phase 2 — dead code — DONE

Five pure-`NotImplementedError` "Phase 2" scaffold modules, no live callers
(real implementations live elsewhere — stats in `fitting/validation.py`, loaders
in `io/data_loaders/`):

- [x] `core/fit_metrics.py` (not re-exported anywhere — fully orphaned).
- [x] `preprocessing/data_loading.py` (+ drop `load_blackchirp_data`,
  `load_fid_data` from `preprocessing/__init__.py` import + `__all__`).
- [x] `preprocessing/data_validation.py` (+ drop `validate_fid_data`,
  `validate_frequency_data`).
- [x] `utils/statistical_tests.py` (+ drop `f_test`, `aic_comparison`,
  `chi_squared_test`).
- [x] `utils/physics_utils.py` (+ drop its 3 names from `utils/__init__.py`).
- [x] `io/experimental_formats.py` — legacy parallel Blackchirp loader. Migrated
  the five test consumers (`test_pipeline_cache.py`, `test_three_stage_workflow.py`,
  `test_file_manager.py`, `test_fid_serialization.py`) onto
  `BlackChirpLoader().load_fid(...)` (the live loader returns an `FID`; its
  metadata uses `source_path`, not the old `experiment_path`), then deleted the
  module + the `io/__init__.py` export. Whole suite collects (1943 tests, no
  import errors); the migrated files pass.

## Phase 3 — low-risk duplication extractions

(Finding numbers from the duplication audit.) **F3/F4/F6/F8/F10/F11 all done;
F5 deferred to Phase 4 — it pairs with the F1 settings work and has
per-call-site churn.** 700 touched-area tests green after F6/F8/F10.

- [x] **F4** baseband↔molecular conversion (5+ sites, two spellings). The actual
  duplication was the **sign convention**, spelled two ways: inline
  `-1.0 if "lower"` (data_structures `apply_molecular_frequency` ×2, tau_calibration
  ×3, tau viz ×1) vs `peak_model.sideband_sign` (stage3/5, spur_detection). The
  probe-relative conversion itself already had canonical helpers
  (`peak_model.molecular_frequency`/`baseband_offset`), so no new converter was
  added. Fix: added a **`Sideband.sign`** property (single source of truth, lives on
  the enum in `data_structures` so there is no `data_structures`→`fitting` cycle);
  `sideband_sign` now delegates to it; the two `apply_molecular_frequency` copies
  collapse to `probe + sideband.sign * scope`; the inline tau_calibration/viz signs
  route through `Sideband.coerce(...).sign`. Fresh 2638 e2e fit table **byte-identical**
  before vs after (512 peaks, `diff` clean); 286 core + 39 tau/viz tests green; black +
  mypy clean.
- [x] **F6** sideband string→enum coercion → `Sideband.coerce` classmethod on the
  enum (lsb/usb-aware); deleted the three local resolvers
  (`stage5_impl._resolve_sideband`, `stage6_impl._sideband_from_value`,
  `active_ft_support._resolve_sideband`) and repointed all 11 call sites.
- [x] **F5** `_required*` post-resolve coercion (stage2/3/4/5_impl,
  tau_settings_resolution) → `require_resolved(value, name, *, cast=None,
  owner=...)` in `_internal/shared_utils.py`. Each stage's `_required` /
  `_required_{int,float,bool,str}` now delegates to it, preserving the
  per-owner assertion message; golden byte-identical.
- [x] **F10** protected-offset matcher (3 identical in `residual_rescue.py`) →
  `make_protected_matcher(protected, tol)` factory in `fitting/validation.py`.
- [x] **F3** HDF5 attr helpers (`_load_json_attr`, NaN-sentinel float coercion,
  group-reset + stage-header stamp) scattered across `io/*_serialization.py` →
  `io/_hdf5_helpers.py` (`load_json_attr`, `nan_if_none`/`none_if_nan`/`opt_float`,
  `reset_group`, `stamp_stage_header`). Repointed fitting/window/peak/stage6_review/
  timebase/tau serializers; deleted the two local `_load_json_attr`, the
  `_nan_if_none`/`_none_if_nan` pair, and `stage6_review._opt_float`. `stamp_stage_header`
  writes the count attrs before `creation_time`/`stage_name` to match the prior order
  (HDF5 attrs are order-independent on disk regardless). 341 io unit tests + 33
  cross-interface green; touched files black + mypy clean.
- [x] **F8** τ₀ fallback (Stage 5 seed + Stage 6 frozen-background dependent τ) →
  `default_tau0_us(acquisition_us)` in `active_ft_support.py` (a *function*
  returning `acquisition_us / 3.0`, not a `1/3` constant — preserves the exact
  division so the fit seed stays bit-identical). The `_active_acquisition_us`
  dup (active_ft_support + stage3_impl) and the padded-active-rFFT block are
  deferred (cross-module import + χ²/noise byte-equivalence).
- [x] **F11** deleted `fit_detail._apply_bare_style` (byte-for-byte dup of public
  `report_style.apply_bare_style` — same `#dbe0e6` grid) and repointed its 6 call
  sites + the `report_html_impl` import to the public name; added
  `set_log_spectrum_ylim(ax, rms_noise, top, y_max_factor)` to `report_style.py`,
  collapsing the identical floor/yscale/ylim block in `peak_visualization` and
  `window_visualization`; consolidated the de-ramp + rolling-`S_coh` + threshold
  computation into `edge_coherence.coherence_curve(freqs, spec, rms, params)` used
  by both `window_visualization` and `tuning/plots._coherence_curve`. The latter
  **fixes the divergent defaults** — `tuning/plots` hardcoded `edge_m=64`/
  `edge_threshold=8.0`, which only happened to equal `DEFAULT_EDGE_M`/
  `DEFAULT_EDGE_THRESHOLD`; the curve is byte-identical today but the helper now
  pulls the constants so the two surfaces can never silently diverge. Local
  `deramp_to_active_start` import inside `coherence_curve` keeps the
  `edge_coherence`↔`leakage` cycle at bay. 173 viz/tuning/preprocessing tests
  green; real 2638 window-plan + peak figures render; black + mypy clean.

## Phase 4 — large duplication refactors (byte-identity sensitive)

The self-contained pair (**F9** sort/broadcast preamble, **F12** `fork_map`) are
done — see their `[x]` entries lower in this section. The settings-framework trio
(F1 + F5, then F2) is done.

- [x] **F1** settings resolve/preset framework across `core/stage_fit_settings.py`,
  `tau_calibration_settings.py`, `peak_detection_settings.py`,
  `window_planning_settings.py` → `core/settings_framework.py`. The structural
  pieces (layer walk, sub-block merge, attrs/YAML loops, preset file resolution +
  block extraction) live there once, parameterized by the settings class,
  `sub_names`, `hard_defaults`, and per-module value codecs. stage_fit keeps its
  `shape`-aware resolve/to_attrs/from_attrs + the legacy `fit:` / flat-fallback
  `load_preset`; tau keeps its tuple codecs; the other two delegate wholesale.
  golden byte-identical, 206 settings/serialization round-trip tests + 33
  cross-interface green; black/isort/mypy clean.
- [x] **F2** settings serialization save/load/present across the five
  `io/*_settings_serialization.py` modules → `io/_settings_serialization.py`
  (`save_settings`, `load_subblock_settings`, `load_flat_settings`,
  `settings_block_present`). A dict-valued `to_attrs` entry → subgroup, a scalar →
  top-level attr, so one save path covers the flat Stage 2 layout, the sub-block
  layouts, and Stage 5's `shape` subgroup-or-sentinel. tau's tuple lifting and
  Stage 5's `shape` reader ride in as `write_attr` / `read_sub` / `extra_top`
  callbacks. golden byte-identical, 46 io round-trip + 33 cross-interface green.
- [x] **F9** sort + σ-broadcast cleanup preamble (residual_rescue ×3 +
  window_fit.conservative_fit) → `window_fit.sort_window_arrays(offset_grid,
  spectrum, rms_noise, *, extras=...)`, returning `(u, z, sigma, order,
  sorted_extras)`. `extras` maps a name to an `(array_or_None, dtype)` pair
  (budget→float, background→complex128); `order` is returned so the
  `iterative_aicc_cleanup` identity-permutation `initial_refits` guard is
  preserved unchanged. 2638 fit table byte-identical (golden `check`); 74
  window_fit/residual_rescue tests green; black + mypy clean.
- [x] **F12** the two fork-pool figure-render orchestrators in
  `_internal/report_html_impl.py` (`_render_methods_figures`,
  `_render_all_window_figures`) → `fork_map(items, worker, *, jobs, override,
  progress)`. The helper owns the serial/fork decision, the capped
  `ProcessPoolExecutor("fork")`, and the in-order `ex.map` drain (with the
  optional `progress(i, n)` callback carrying the ordered per-window log); each
  caller still sets/clears its own fork-inherited module-global in a `try/finally`
  around the call (`fork_map` does not manage it, per the BLAS/fork notes). The
  serial paths now also route through the global-reading worker — output-identical.
  42 `test_report_full` + 6 render-path tests green (real end-to-end HTML render);
  black + mypy clean.

## Phase 5 — top-level docs refresh — DONE

- [x] **README.md** — status block now describes the full Stages 0–6 pipeline;
  the CLI quickstart leads with `ftmwpipeline run` (whole-experiment) and lists
  the per-stage verbs through `report run`.
- [x] **STATUS.md** — stage table carries the 2b/6/timebase rows; the impl glob
  is `stage{0,1,2,2b,3,4,5,6}_impl.py`; test counts re-measured (1943 collected);
  cross-interface 20→33; `timebase_calibration` + `stage6_review` added to the
  `.ftmw` group layout (plus the `processing_parameters/stage*` settings groups);
  CLI grammar lists `timebase`/`review`/`report`/`clocks`/`run`; the lint-debt
  note reflects the now-clean state.
- [x] **Strategy docs** — the retired `rdc` knob is dropped from the
  display/scaling list in `API_STRATEGY.md` and `SERIALIZATION_STRATEGY.md`
  (it survives only as an unconditional internal function arg).

## Phase 6 — lint/format + tests — DONE

- [x] Repo-wide `black` + `isort` over `src/`. `mypy --strict` was already clean
  (132 source files — the earlier doc effort had normalized most of the debt), so
  the residual was a handful of `black`/`isort` files; normalized in its own
  commit.
- [x] Full test suite green (`-o addopts=""`): 1943 collected, all passing after
  realigning the one drifted Stage 4 tuning metric-columns assertion (a
  pre-existing staleness, unrelated to the refactors).

## Outcome

All phases complete. The cleanup removed the dead scaffold modules and the legacy
Blackchirp loader, extracted the duplication findings F1–F12, refreshed the
top-level docs, and normalized the formatting. Byte-identity held throughout
(golden `check` clean after every byte-sensitive refactor).

- `0aeb922` — Phase 1 (stray artifacts) + Phase 2 (dead modules + the legacy
  Blackchirp loader; five test files migrated to `BlackChirpLoader`).
- `77ac107` — **F6** (`Sideband.coerce`), **F10** (`make_protected_matcher`),
  **F8** (`default_tau0_us`).
- `9d7565c` — **F3** (`io/_hdf5_helpers.py`).
- `d150bc4` — **F4** (`Sideband.sign` single source of truth).
- `5215555` — **F11** (bare-style dedup + `set_log_spectrum_ylim` +
  `edge_coherence.coherence_curve`).
- `71b7c79` — **F9** (`window_fit.sort_window_arrays`).
- `f8b9719` — **F12** (`fork_map` render orchestrator).
- `4977ab9` — **F1** + **F5** (`core/settings_framework.py` + `require_resolved`).
- `3833522` — **F2** (`io/_settings_serialization.py`).

## Provenance

- Findings: dead-code + artifacts from a direct grep/graph sweep
  (codebase-memory MCP confirmed no edgeless nodes before disconnecting);
  duplication F1–F12 from the duplication audit; docs staleness from the
  docs-freshness audit.
- Validation: the golden harness (`scratch/cleanup-golden/golden.py check`,
  gitignored) snapshots the 2638 end-to-end fit; it stayed byte-identical across
  every byte-sensitive refactor. Settings round-trip is unit-test-covered.
