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

(Finding numbers from the duplication audit.)

- [ ] **F4** baseband↔molecular conversion (5+ sites, two spellings) → one
  `baseband_to_molecular` / `molecular_to_baseband` in `utils/signal_processing.py`
  (or a `Sideband.sign` property), accepting `Sideband` or `"lower"/"upper"`.
  Sites: `fitting/tau_calibration.py`, `visualization/tau_calibration_visualization.py`,
  `core/data_structures.py` (`PreprocessedFID`/`FID.apply_molecular_frequency`),
  `_internal/stage3_impl.py`, `_internal/stage5_impl.py`, `fitting/spur_detection.py`.
  Watch the `data_structures`→`utils` import direction for a cycle.
- [ ] **F6** sideband string→enum coercion: `_resolve_sideband` (stage5_impl) and
  `_sideband_from_value` (stage6_impl) are byte-identical; `active_ft_support._resolve_sideband`
  is a simpler variant. Keep the richer one (lsb/usb-aware), delete the others.
- [ ] **F5** `_required*` post-resolve coercion (6 copies: stage2/3/4/5_impl,
  tau_settings_resolution, re-imported by stage6) → `require_resolved(value, name,
  *, cast=None)` in `_internal/shared_utils.py`.
- [ ] **F10** protected-offset matcher (3 identical in `residual_rescue.py`) →
  `_protected_offset_matcher(protected, tol)` factory in `fitting/validation.py`.
- [ ] **F3** HDF5 attr helpers (`_load_json_attr`, NaN-sentinel float coercion,
  group-reset + stage-header stamp) scattered across `io/*_serialization.py` →
  `io/_hdf5_helpers.py` (`load_json_attr`, `nan_if_none`/`none_if_nan`/`opt_float`,
  `reset_group`, `stamp_stage_header`).
- [ ] **F8** `_active_acquisition_us` dup (active_ft_support + stage3_impl) →
  import the single one; add `DEFAULT_TAU0_ACTIVE_FRACTION = 1/3` (Stage 5/6 must
  stay in lockstep — Stage 6 replays Stage 5). Defer the padded-active-rFFT block
  (feeds χ²/noise; byte-equivalence sensitive) — extract only the conversion line.
- [ ] **F11** delete `fit_detail._apply_bare_style` (private dup of public
  `report_style.apply_bare_style`; repoint the `report_html_impl` import to the
  public name); add `set_log_spectrum_ylim` to `report_style.py`; consolidate the
  edge-coherence curve into `preprocessing/edge_coherence.py`
  **fixing the divergent `edge_m`/`edge_threshold` defaults** (latent bug:
  `window_visualization.py` uses `DEFAULT_*`, `tuning/plots._coherence_curve`
  hardcodes 64/8.0).

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
- [ ] **F9** sort + σ-broadcast cleanup preamble (residual_rescue ×3 +
  window_fit.conservative_fit) → `sort_window_arrays(offset_grid, spectrum,
  rms_noise, *, extras=...)`. Must preserve byte-identical ordering and the
  `iterative_aicc_cleanup` identity-permutation `initial_refits` guard.
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
