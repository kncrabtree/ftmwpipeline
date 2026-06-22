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
- [ ] **F5** `_required*` post-resolve coercion (6 copies: stage2/3/4/5_impl,
  tau_settings_resolution, re-imported by stage6) → `require_resolved(value, name,
  *, cast=None, owner=...)` in `_internal/shared_utils.py`. **DEFERRED to Phase 4**
  (pairs with F1; many call sites, must preserve the per-class assertion message).
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

- [ ] **F1** settings resolve/preset framework across `core/stage_fit_settings.py`,
  `tau_calibration_settings.py`, `peak_detection_settings.py`,
  `window_planning_settings.py` → `core/settings_framework.py` parameterized by
  `(settings_cls, sub_names, hard_defaults, preset_block_key, value_codec)`.
  Capture stage_fit's sibling-key stripping + tuple/ClockSource/shape codecs as
  hooks. Round-trip is test-covered.
- [ ] **F2** settings serialization save/load/present across
  `io/noise_settings_serialization.py`, `peak_detection_settings_serialization.py`,
  `window_planning_settings_serialization.py`, `stage_fit_settings_serialization.py`,
  `tau_calibration_serialization.py` → `io/_settings_serialization.py`
  (`save_subblock_settings`, `load_subblock_settings`, `settings_block_present`),
  with read/write callbacks for stage_fit's `shape` subgroup + tau's tuples.
  (Persistence twin of F1; depends on F3's helpers.)
- [x] **F9** sort + σ-broadcast cleanup preamble (residual_rescue ×3 +
  window_fit.conservative_fit) → `window_fit.sort_window_arrays(offset_grid,
  spectrum, rms_noise, *, extras=...)`, returning `(u, z, sigma, order,
  sorted_extras)`. `extras` maps a name to an `(array_or_None, dtype)` pair
  (budget→float, background→complex128); `order` is returned so the
  `iterative_aicc_cleanup` identity-permutation `initial_refits` guard is
  preserved unchanged. 2638 fit table byte-identical (golden `check`); 74
  window_fit/residual_rescue tests green; black + mypy clean.
- [ ] **F12** the two fork-pool figure-render orchestrators in
  `_internal/report_html_impl.py` (`_render_methods_figures`,
  `_render_all_window_figures`) → `fork_map(items, worker, *, jobs, override)`.
  The fork-inherited module-global + ordered progress log are load-bearing
  (BLAS/fork notes) — preserve ordering and clear-on-finally.

## Phase 5 — top-level docs refresh

- [ ] **README.md** — replace the "Stages 3–5 not yet implemented" status block
  (Stages 0–6 all ship); extend/repoint the Stage-2-capped quickstart (mention
  `ftmwpipeline run` for the whole-experiment path).
- [ ] **STATUS.md** — add Stage 6 + timebase rows to the stage table; fix the
  `_internal/stage{...}_impl.py` glob (drop deleted `stage2b_g_impl`, add `6`);
  re-measure + reconcile the test counts; cross-interface 20→33; add
  `timebase_calibration` + `stage6_review` to the `.ftmw` group layout; add
  `timebase`/`review`/`report`/`clocks`/`run` to the CLI grammar list. (Reconcile
  against the dead-module removals too.)
- [ ] **Strategy docs** — drop the retired `rdc` knob from the display/scaling
  list in `API_STRATEGY.md` (~line 84) and `SERIALIZATION_STRATEGY.md` (~line 48).

## Phase 6 — lint/format + tests

- [ ] Repo-wide batched `black` + `isort` + `mypy` (strict) pass over `src/`
  (the long-deferred formatting debt; its own commit(s) so churn stays separate).
- [ ] Full test suite green (`-o addopts=""`), including `slow`/regen guards
  where touched.

## Start here (current state)

Working tree clean; Phase 3 complete (all of F3/F4/F6/F8/F10/F11 committed):

- `0aeb922` — Phase 1 (stray artifacts) + Phase 2 (dead modules + the legacy
  Blackchirp loader; five test files migrated to `BlackChirpLoader`).
- `77ac107` — Phase 3 **F6** (`Sideband.coerce`), **F10** (`make_protected_matcher`),
  **F8** (`default_tau0_us`). 700 fitting/core/cross-interface tests green; touched
  files black + mypy clean.
- `9d7565c` — **F3** (`io/_hdf5_helpers.py`: `load_json_attr`/`nan_if_none`/
  `none_if_nan`/`opt_float`/`reset_group`/`stamp_stage_header`).
- `d150bc4` — **F4** (`Sideband.sign` single source of truth; 2638 fit table
  byte-identical).
- **F11** (bare-style dedup + `set_log_spectrum_ylim` + `edge_coherence.coherence_curve`
  with the default-divergence fix) committed alongside this doc update.

**Next up: Phase 4 (the heavy refactors), then Phases 5–6.** Suggested order:

1. **Phase 4** (F1, F2, F5, F9, F12) — the heavy, golden/byte-identity-sensitive
   refactors. F5 (`require_resolved`) pairs with F1. F2 (settings serialization)
   builds on F3's new `io/_hdf5_helpers.py`. Build a fresh 2638 fixture and diff the
   persisted fields / fitted table before vs after for F2 and F9 (the
   stash → rebuild → `diff` harness used for F4 works here too).
2. **Phase 5** (README/STATUS/strategy refresh) then **Phase 6** (repo-wide
   black/isort/mypy + full green suite) last.

## Resume notes

- Run commands via `conda run -n ftmwpipeline-dev`; `-o addopts=""` drops
  coverage; scope tests to the touched surface mid-effort, save one full run for
  the end.
- Commit in logical groups (artifacts+deadcode; each refactor family; docs;
  lint) so diffs stay legible — the `commit` skill governs the messages.
- Findings provenance: dead-code + artifacts from a direct grep/graph sweep
  (codebase-memory MCP confirmed no edgeless nodes before disconnecting);
  duplication F1–F12 from the duplication audit; docs staleness from the
  docs-freshness audit.
