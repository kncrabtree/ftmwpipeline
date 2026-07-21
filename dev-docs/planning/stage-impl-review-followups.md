# Stage-implementation review follow-ups

Findings from a correctness-focused review of the pipeline stage implementations
(`preprocessing/`, `fitting/`, `core/` — not `_internal/`, which is the interface
layer). Merges the maintainer's manual pass with an independent automated review;
each independent-review item cross-references the maintainer note it confirms,
refutes, or extends (`review #N`).

Scope reminder: `_internal/` is out of primary scope, but the Active-FT
data-flow audit (below) necessarily traced into the stage impls to settle
`review #5`.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.

Any item tagged **byte-sensitive** must pass `scratch/cleanup-golden/golden.py
check` (the gitignored 2638 end-to-end fit snapshot) before and after — behavior
must not change unless the item is a deliberate behavior change.

> **Golden-harness note (2026-07-20).** `scratch/cleanup-golden/` was absent from
> the working copy and the `ftmwpipeline-dev` conda env did not exist, so the
> harness was **rebuilt from scratch** this session rather than restored. The
> current reference (`fit_2638.golden.txt`, 262 windows / 510 peaks) is a valid
> before/after gate for changes made from this point on, but it is **not** the
> same reference the earlier cleanup pass used (that one recorded "512 per-peak
> rows"), and today's tree was never verified against that historical baseline.
> The harness records the recipe: `trim=(26500, 40000)`,
> `guard_margin_us=0.67`, `FTMW_MAX_WORKERS` pinned (the default
> `cpu_count()-2` OOMs a fit worker on a 30 GB box).

## Checklist

Correctness (silent wrong numbers / real defects):

- [x] **C1 (HIGH, latent).** Thaw path discards fresh fit statistics —
  `_install_cofit_outcome` overwrites peaks but keeps stale `peak_errors`,
  `covariance`, `chi_squared`, `n_params`, `n_data`, `tau_error`.
  *(Done, commit 9c62d50; golden byte-identical.)*
- [x] **C2 (MED-HIGH).** `sigma_tau_floor` knob + `DEFAULT_SIGMA_TAU_FLOOR_US`
  are unwired no-ops — wire through or delete (subsumes `review #6`/`#7`).
- [x] **C3 (MED, review #14).** Spur detection is not eps-aware and uses a fixed
  MHz tolerance instead of bin-width units; share machinery with the timebase
  calibrator. *(Resolved. Part 1: timebase now runs right after Stage 1 (its true
  Stage 0 + Stage 1 dependency, previously under-declared as Stage 0 only), so its
  measured `eps` is available to Stage 5 — byte-neutral for the eps measurement.
  Part 2: the nearest-bin match window is now bin-width-derived
  (`max(integer_tol, 0.5·Δf)`), honors each `LatticePoint.window_mhz`, and is
  eps-aware — a scale error is corrected by **shifting** each lattice point's
  search anchor to the measured position `f·(1+eps)` (widening the tolerance
  alone cannot help: the detector inspects the single nearest bin to the
  prediction, so a tone displaced past ~half a bin lands in a different bin the
  widened window never reaches). 2638 golden byte-identical; the 7-fixture set
  shows 4 fixtures gain genuine on-lattice clock spurs with χ²ᵣ health preserved,
  no regression. Full design + implementation notes in
  [[timebase-early-eps-aware-spurs]] — `planning/timebase-early-eps-aware-spurs.md`.)*

Code-vs-doc divergences (reconcile per ROADMAP divergence discipline — decide
whether code or spec is authoritative, then fix the other):

- [x] **D1 (review #5).** Active-FT invariant HOLDS in code; the defect is stale
  docs/comments that say later stages use the zero-substituted Stage 1 FT. Fix
  every misleading string; decide whether Stage 1 should display the Active
  FID/FT. *(Docs scrubbed in 8d741c4; Stage 1 now displays the Active FID/FT in
  d0d8073. Both golden byte-identical / display-only.)*
- [x] **D2 (MED).** Rescue τ-seeding seeds from global `tau_maj` for every
  window when a calibration exists — the opposite of what the method doc says.
  *(Resolved doc-side: code is authoritative + validated — in production the
  seed is the per-band anchor, not a single global `tau_maj` (routed via
  `resolve_window_tau_anchor`, `plan_execution.py:2660-2665`). Spec corrected;
  ROADMAP D17. Doc-only, non-byte-sensitive.)*
- [x] **D3 (MED).** Frozen-contributor skirt is drawn at the dependent window's
  τ, not the contributor's own `τ_c` as the spec requires.
  *(Resolved doc-side per maintainer: code authoritative — the shared-τ skirt is
  a deliberate, documented far-wing approximation. Spec model eq + prose amended
  to `h_T(u−δ_c; τ)`; ROADMAP D18. Doc-only, non-byte-sensitive.)*
- [x] **D4 (MED).** `leakage_touched_intervals` has zero callers, yet ROADMAP D8
  and the `leakage.py` module docstring present it as the live Stage 3/4 leakage
  map. Wire it or retire it + correct D8.

Dead / vestigial code:

- [x] **DC1 (review #15).** `coherence_screen.py` — confirmed dead in `src/`
  (intentional, documented). Curation decision: delete or keep as research helper.
- [x] **DC2 (MED).** `leakage.py` module docstring is wrong (see D4); the
  `deramp_to_active_start` helper it names is NOT dead (refutes `review #16`).
- [x] **DC3.** Dead fitting exports: `result_line_views`,
  `remove_and_refit_cleanup`, `evaluate_fixed_contributor`,
  `passes_significance_test`, `calculate_rms_residuals`, unused `calculate_aicc`
  import.
- [x] **DC4 (review #11).** `peak_model.py` deramp narrative is largely accurate
  (helper is live for Stage 4 display) — trim only the confusing "de-ramped
  [0,T]" phrasing.

Constant / structure hygiene:

- [x] **H1 (review #6).** Move `DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN`
  (tau_calibration.py:262) and `DEFAULT_BAND_LABELS` (:311) into the top constant
  block (both are used).

Terminology & comments (cleanup):

- [x] **T1 (review #4).** Assess the term "canonical" — 31 doc uses + user-facing
  CLI/plot strings; likely a holdover. Reconcile "canonical spectrum" vs "active
  FT", especially in user-facing strings. *(Done in 8d741c4: scrubbed the FT/
  spectrum sense across src + docs, preferring "standard" in user-facing prose;
  generic-English uses left alone. Golden byte-identical.)*
- [x] **T2 (review #8).** `tau_maj` retirement — already tracked as an
  architectural backlog item in `cleanup-pass.md`. Verified correct today
  (per-band routing sound; `tau_maj` load-bearing → versioned rename). Update
  stale comments now; do the rename with the golden gate.
- [x] **T3 (review #1).** Remove source-evolution comments.
- [x] **T4 (review #2).** Remove backward-compatibility code (`_internal`).
  *(Done. `_internal` had almost no dead back-compat: only the Stage 4
  `processing_parameters/window_assignment` JSON write was truly dead (no
  reader) and was removed. The Stage 3 `peak_detection` block its comment called
  a "back-compat shim" is actually the report's live data source — kept, comment
  corrected. The remaining `legacy`-tagged branches (stage5/stage6/
  start_detection promotion/τ/acquisition fallbacks) are live older-file readers
  and were kept. Golden byte-identical. Retiring the write-only public-API
  `save_peak_parameters`/`save_window_parameters` writers is an api.py + spec
  change, left as a follow-up.)*
- [x] **T5 (review #3).** Remove progress/dev markers ("pre-D7", etc.;
  `_internal`).
- [x] **T6 (review #9).** Remove "reference implementation"/`bcfitting`
  provenance framing; scrutinize the ported algorithms (see the peak-detection
  padding seam under T6 detail). *Framing removal done; edge-padding fixed to a
  true edge-value hold (`np.pad(mode="edge")`) — golden byte-identical (no 2638
  peak sits within `half` bins of a band edge), so it is a latent-correctness
  improvement with no numeric change here.*
- [x] **T7 (review #10).** Remove `dev-docs/planning/` and `scratch/` pointers in
  source comments; keep only specific-claim citations.
- [x] **T8 (review #12).** ASCII scientific notation in comments (`1e6`, not
  `10⁶`).

Robustness (assess; not confirmed bugs):

- [x] **R1 (review #13).** Timebase local noise reference is fragile but
  self-protecting — optional hardening (more probes / scatter estimator).
  *(Done via more probes: `DEFAULT_N_NOISE_PROBES` 8 → 32, stabilizing the
  25th-percentile noise floor without changing its upper-outlier
  (line-contamination) robustness. Across the 7 fixtures eps moves toward the
  lower-variance estimate where the 8-probe floor was noisy (2638 +2.7%, 1512
  −6.5%, others ≤0.7%); the gated spur set, line counts, and χ²ᵣ are identical on
  all 7 (eps accuracy is second-order for spurs) and the 2638 golden is
  byte-identical. The Rayleigh-scatter estimator (option B) is left as a possible
  future refinement.)*
- [x] **R2.** Misleading diagnostics that do not change any number but can
  mislead a maintainer (mislabeled `aicc_delta`, wrong phase-penalty/`max_nfev`
  docstrings, Rayleigh false-alarm comment).
- [x] **R3.** Informational: Stage 1 amplitude normalization divides by full
  record length (uniform scalar, no statistical impact); the "not padded length"
  comment is vestigial.

---

## Detail

### C1 — Thaw path discards fresh fit statistics (HIGH, latent)

`fitting/plan_execution.py:4380-4449` (`_install_cofit_outcome`). After an
accepted thaw it overwrites `peaks`, `tau_us`, `tau_was_fit`, `fitted_spectrum`,
`residual` and clears baseline fields, but never updates `peak_errors`,
`covariance`, `chi_squared`, `cost`, `n_params`, `n_data`, `tau_error`. The
window then reports positions/amplitudes from the joint co-fit but uncertainties
and reduced-χ² from the stale pre-thaw independent fit. That stale `reduced_chi2`
feeds the per-window log, Stage 6 χ²ᵣ attention routing, and `fit check` grading.
Contradicts `stage5_fitting.rst` ("re-determined jointly … replacing the frozen
copy") and the Stage 6 doc.

Confirmed as an oversight, not design: the rescue path
(`_apply_rescue_to_outcome:4532`) reassigns the whole `fit.fit`, and the baseline
path (`_apply_baseline_to_outcome:4781-4795`) explicitly copies
`peak_errors`/`covariance`/`chi_squared`/`n_params`/`n_data` — only the thaw path
omits them. **Latent:** thaw-accept count is 0 on every validated fixture, so it
never fires today; it is a real defect on the documented 36350/36389 doublet
mechanism. Fix by mirroring `_apply_baseline_to_outcome`. Byte-sensitive only
once a thaw-exercising fixture exists.

### C2 — Unwired `sigma_tau_floor` knob (MED-HIGH; review #6/#7)

The knob `aggregation.sigma_tau_floor_us` is declared
(`core/tau_calibration_settings.py:167`, hard default 0.5 at :361, preset at
`presets/defaults.yaml:51`, registry column at
`_internal/tuning/registry.py:957`) and the matching constant
`DEFAULT_SIGMA_TAU_FLOOR_US` is defined and exported
(`fitting/tau_calibration.py:55`, `__all__` :123) — but **neither is ever
consumed.** `stage2b_impl.py` never puts `sigma_tau_floor_us` in `kernel_kwargs`;
`extract_tau_majority` takes no such parameter. The only σ_τ floor actually
applied is the hardcoded literal `0.5` inside `compute_band_majorities`
(tau_calibration.py:324, applied :395), whose caller `_finalize` (:1908) does not
pass `sigma_floor_us` and does not reference the constant.

Failure: a user or a tuning sweep over `grid=(0.0, 0.5, 1.0)` sets the knob; it is
resolved, persisted, echoed in `parameters_used`, and shown by the tuning harness
as if it mattered, but the floor stays pinned at 0.5 — the knob is a no-op and any
sweep result attributing σ_τ differences to it is spurious. Also violates the
settings modules' stated invariant that `_HARD_DEFAULTS` mirrors `DEFAULT_*`
constants still imported by the kernels. Fix: wire it through
(`kernel_kwargs → extract_tau_majority → _finalize → compute_band_majorities`,
with the default referencing the constant) **or** delete knob + constant + preset
line + registry column. This is the sole vestigial `DEFAULT_*` constant
(`review #7`); all others are live.

### C3 — Spur detection not eps-aware, fixed MHz tolerance (MED; review #14)

`fitting/spur_detection.py:98` `DEFAULT_INTEGER_TOL_MHZ = 0.04`, applied at :479
and :512, and wired at `_internal/stage5_impl.py:1404`/:1435. The nearest-bin
match uses a fixed 0.04 MHz window (~half a bin only at the reference
`T_active ≈ 13 µs`) — it is neither computed from the actual active-FT bin
spacing `1/T_active` nor widened by the clock scale error ε (grep for
`epsilon`/`eps_` in the spur/active-FT path returns nothing; neither
`build_clock_lattice`, `ClockLattice`, nor `detect_active_ft_spurs` take an ε).
`detect_active_ft_spurs` also ignores each `LatticePoint.window_mhz` in lattice
mode, using the scalar tolerance instead.

Failure: a digitizer scale error ε displaces every measured tone by ε·f_bb (on
the reference instrument ε ≈ 2.13 ppm → 32–77 kHz across the baseband range),
right at or beyond the 0.04 MHz edge; a genuine clock spur can fall outside its
match window and go undetected, leaving a CW tone in the residual to inflate a
window's χ². Rework in bin-width units, make it eps-aware, and share machinery
with `timebase_calibration.py` (which already measures ε) — the maintainer note
that spur detection was done first and more ad hoc is correct.

**Scoped + deferred (2026-07-21).** eps-awareness is blocked by stage ordering:
`timebase_calibration` (which measures ε) is registered as running only after
Stage 5, so no ε is available to the spur sweep. Its true dependency is Stage 0 +
Stage 1 (`timebase_impl.py:152-172` reads `start_us`/`end_us` from persisted
Stage 1 settings; the registered dep `file_manager.py:219` under-declares this as
Stage 0 only). The fix is to run timebase right after Stage 1 (byte-preserving
for the ε measurement — same bounds) so Stage 5 can consume ε, then do the three
spur sub-parts (bin-width tolerance, per-point `LatticePoint.window_mhz`,
ε-widening). Maintainer chose the after-Stage-1 placement (not pure Stage 0 — the
active bounds are canonically a Stage 1 product) and deferred implementation to a
clean boundary; the ε benefit needs the 7-fixture spur set to validate (2638
gates byte-identity only). Full design:
`planning/timebase-early-eps-aware-spurs.md`.

**Resolved.** Both parts landed. One design correction surfaced during
implementation: ε-awareness is a **search-anchor shift**, not a window widening.
`detect_active_ft_spurs` inspects the single active-FT bin nearest each lattice
point's *predicted* frequency, so once a scale error displaces a tone past ~half
a bin it lands in a different bin and a wider match tolerance never reaches it
(verified empirically). The fix shifts each lattice point's search frequency to
the measured position `f·(1+eps)` (`s·ε·f_bb` in the molecular frame) and widens
the window only by the ε *uncertainty* term `N·σ_ε·f_bb`. 2638 golden
byte-identical; 4 of the 7 fixtures gain genuine on-lattice clock spurs (line
counts drop only by the masked tones, χ²ᵣ medians hold/improve).

### D1 — Active-FT invariant: code correct, docs stale (review #5)

The invariant HOLDS in code. Verified data-flow (read, not trusting comments):

- Stage 2 (control): `stage2_impl.py:184` `build_trimmed_active_ft` →
  `estimate_active_ft_noise` on the active FT (:187).
- Stage 2b: does **not** touch the persisted Stage 1 spectrum. It loads the raw
  FID, slices the active region (`tau_calibration.py:2104`
  `active = fid_arr[start_idx:end_idx]`), and runs its own sliding STFT on the
  active samples. The maintainer's suspicion that 2b uses the zero-substituted FT
  is **refuted**. (The STFT frames are internally zero-padded to the active length
  to hold a common grid — intrinsic to a decay-vs-time measurement, bounded to
  the active region, and the noise floor is measured on the same active tail; not
  the Stage 1 full-record zeroing.)
- Stage 4: `stage4_impl.py:183` `build_active_grid_with_noise` → passes
  `active_ft.freq_array`/`complex_spectrum`/`active_rms` into `build_window_plan`.
- Stage 5: `stage5_impl.py:1261` `compute_active_ft`.
- Stage 3 (the one exception): intentionally uses modified/front-zeroed FTs for
  rough peak finding, then re-scores on the active FT.

So no stage outside Stage 3 measures on the zero-substituted FT. The actionable
defect is **stale documentation** — reconcile against `stage2_noise.rst`, which
states the rule correctly:

- `docs/source/stage1_ft.rst:19-21` — says the canonical settings "bind the
  spectrum that noise estimation, peak detection, window assignment, and fitting
  all reproduce," implying the zero-substituted FT.
- `fitting/active_ft.py:23-24` — "The active-FT is internal to Stage 5" is wrong;
  it is the domain for Stages 2/4/5.
- `_internal/stage4_impl.py:12-14, 70-73, 176-178` — docstrings claim the "Stage 1
  persisted canonical spectrum."
- `fitting/tau_calibration.py:1426, 1469` — "full-record FT noise floor" (it is
  the active region).
- `preprocessing/window_planning.py:538-540` — param docstring "persisted user
  spectrum."
- User-facing CLI: `cli/peak_commands.py:169` "Detection operates on the Stage 1
  persisted canonical spectrum."

Separately, the maintainer wants Stage 1 to present the **Active FID / Active FT**
to the user rather than the zero-replaced version. The infrastructure exists
(`build_trimmed_active_ft`, `compute_display_ft`); this is a deliberate display
change to decide, not a correctness fix (no statistics are computed on the Stage 1
display spectrum). See also T1 (the "canonical" terminology assessment).

### D2 — Rescue τ-seeding contradicts the doc (MED)

`fitting/residual_rescue.py:1205-1217`: when a Stage 2b calibration is present
(the normal case) rescue seeds τ from the global `tau_maj` for **every** window
"regardless of the initial fit's outcome"; the converged-τ-with-pinned-fallback
logic only runs in the no-calibration `else` branch.
`methods/stage5_fitting.rst:222-225` states rescue seeds from the *converged* τ
except when pinned. Code is self-consistent; the doc (or the code) must be picked
as authoritative. Real numeric consequence when a window's true τ diverges from
the band `tau_maj` (wrong FWHM for the detector separation/shape). Intersects the
`tau_maj` retirement (T2).

**Resolution (2026-07-21): code authoritative, spec amended.** The seed is
*not* the global `tau_maj` in production: Stage 5 routes the per-band anchor into
`ck_for_window["tau_maj_us"]` per window (`plan_execution.py:2660-2665`, via
`resolve_window_tau_anchor`), and the rescue reads that key
(`residual_rescue.py:1084-1086`), so a window's rescue basis already uses its own
band's τ, not a single global value. Seeding from the calibration rather than the
converged per-window LSQ τ is a deliberate, cross-fixture-validated choice (the
"broken-initial-fit pathology" closed by the code comment at
`residual_rescue.py:1074-1082`). The spec (`stage5_fitting.rst` §"Residual
rescue") was corrected to describe the calibration-anchored seeding; ROADMAP D17.
Doc-only change, non-byte-sensitive.

### D3 — Frozen-contributor skirt drawn at the wrong τ (MED)

`FrozenPeak` stores no τ, so `subtract_frozen_background`
(`plan_execution.py:1038/1652/1879/1887`) and `evaluate_edge_free_contributors`
(:3951) draw the contributor skirt at the dependent window's `tau0_us`.
`stage5_fitting.rst:125,134-135` specifies `h_T(u − δ_c; τ_c)` — the
contributor's own `τ_c`. Impact is usually small (the far wing `~1/(i2πΔf)` is
τ-independent), but code and spec literally disagree for cross-band contributor
pairs; reconcile per divergence discipline.

**Resolution (2026-07-21): code authoritative, spec amended (maintainer call).**
The shared-τ skirt is a deliberate, documented modeling choice (FrozenPeak +
`subtract_frozen_background` docstrings: "`h_T` carries one `tau` per window"),
not an oversight, and the numeric difference is negligible where frozen
contributors actually contribute (their center is out-of-window, so only the
τ-independent far wing enters). The correct spec citation is
`docs/source/stage5_fitting.rst` (the stage page, model eq ~:126 and prose
~:133-135), not the methods note; those were changed from `τ_c` to the shared `τ`
with the far-wing justification added. ROADMAP D18. Doc-only, non-byte-sensitive.
(The current spec no longer front-zeros; the review's `:125,134-135` line numbers
predate the D15 rewrite.)

### D4 / DC2 — `leakage_touched_intervals` is dead but documented as live (MED)

`preprocessing/leakage.py:92-135` `leakage_touched_intervals` has zero callers in
`src/` (tests only), yet the module docstring (`leakage.py:1-17`) asserts it is
"the authoritative leakage-extent map shared by the Stage 3 gap-pass mask and
Stage 4 window assignment," and ROADMAP **D8**'s resolution says both stages
consume the de-ramped leakage-touched map. In reality Stage 3
(`stage3_impl.py:162`) and Stage 4 (`window_planning.py:659`) both call
`active_edge_coherence`, which does no de-ramp. A maintainer editing
`leakage_touched_intervals` would expect Stage 3/4 behavior to move and it would
not — a maintenance trap. Either wire the intended map in or delete the function
and correct D8 + the module docstring.

Note this refutes `review #16`: the sibling `deramp_to_active_start` is **live**
via the display path (`edge_coherence.coherence_curve` →
`window_visualization`/`tuning.plots`), which reconstructs the plotted `S_coh`
from the full-record spectrum and genuinely needs the de-ramp. Keep it.

### DC1 — `coherence_screen.py` dead but intentional (review #15)

Entire module (`project_candidates`, `ProjectionResult`) has no `src/` callers
(tests only). Confirmed dead, but deliberate: the docstring (:21-25) says it ships
as a helper pending a wiring decision, and
`methods/matched_filter_detection.rst:129-132` records the coherence-projection
screen was "considered and rejected." Curation call — delete, or keep and drop
the `scratch/` pointer in its docstring (T7).

### DC3 — Dead fitting exports

- `result_conversion.py:191` `result_line_views` — zero references anywhere
  (not even tests); the intended post-fit user-edit surface was never wired.
- `residual_rescue.py:561-679` `remove_and_refit_cleanup` — test-only, still in
  `__all__`; would mix masked `chi_squared` with unmasked `n_data` (630-632,
  650-656) if ever wired.
- `plan_execution.py:609` `evaluate_fixed_contributor` — test-only, exported; the
  `_levelize` docstring (:3108) still points to it though production uses
  `evaluate_ancestor_leakage`.
- `validation.py` `passes_significance_test`, `calculate_rms_residuals` —
  test-only, exported.
- `window_fit.py:78` unused `calculate_aicc` import.

### H1 — Mid-file DEFAULT constants (review #6)

`fitting/tau_calibration.py:262` `DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN` and
:311 `DEFAULT_BAND_LABELS` are defined next to their consumers rather than in the
top block (46-96). Both are used (not vestigial — that is only
`DEFAULT_SIGMA_TAU_FLOOR_US`, C2). Cosmetic move.

### T1 — "canonical" terminology assessment (review #4)

"canonical" appears 31 times across 12 doc files plus user-facing strings
(`cli/peak_commands.py:169` "Stage 1 persisted canonical spectrum",
`fit_visualization.py:198` plot label "canonical sigma",
`cli/ft_commands.py` "store settings as canonical"). It reads as a holdover from
when user FT settings could dictate the working spectrum; now the Active FT is the
working domain and "canonical" mostly means "the persisted Stage 1 settings /
agreed analysis band." Decide a consistent vocabulary and scrub especially the
user-facing strings, coordinating with D1 (do not call the working spectrum
"canonical" where it is the Active FT).

### T2 — `tau_maj` retirement (review #8)

Already tracked as the "retire the band-wide `tau_maj` mechanism" backlog item in
`cleanup-pass.md`. Verified: per-band routing is correct today
(`resolve_window_tau_anchor` in `stage5_impl.py:225-251`, mirrored in stage6);
the global `tau_maj` is only the degenerate single-band fallback, so no wrong
result arises with >1 band. It is **load-bearing** — a persisted HDF5 attr
(`io/tau_calibration_serialization.py:90/221/334/350`), dataclass fields
(`TauCalibrationResult`/`BandTau.tau_maj_us`), and a fit-parameter key
(`plan_execution`/`window_fit`, read by `stage5_validation_impl.py:265`) — so
retirement is a versioned rename, not a mechanical find/replace, and is
byte-sensitive. Until then every tau anchor must go through
`resolve_window_tau_anchor` (never read persisted band-wide `tau_maj` directly).
Now: revise comments that still narrate `tau_maj` as a standalone concept
(`stage2b_impl.py:11/26`, `stage3_impl.py:626`, etc.).

### T6 — "reference implementation" framing + peak-detection padding seam (review #9)

Remove the `bcfitting` provenance framing (`peak_detection.py:7`,
`window_fit.py:4-16/103-104/653-654`, `validation.py:8/18/1503`) — `bcfitting` is
not in the codebase, and this is source-history narration. While there, note one
real limitation in the ported locator: `preprocessing/peak_detection.py:161-166`
comments "replicate the first/last `half` samples" but `y_pre = y[0:half]` /
`y_post = y[-half:]` **block-copy** those samples in forward order rather than
holding the edge value, seaming a discontinuity into the Savitzky-Golay 2nd
derivative within ~`half` bins of the band edge. A peak within `half` bins of the
band edge can be located a bin or two off, or a spurious concave-down feature can
appear at the very edge (interior detection unaffected; downstream apex-snap and
the per-window fit further mask it). LOW, but consider true edge-value replication.

### R1 — Timebase local noise reference (review #13)

`timebase_calibration.py:357-365`: noise reference = 25th percentile of the ML
scan-peak amplitude at off-lattice probe frequencies (`n_noise_probes=8`). The
maintainer's specific worry (a probe demodulating near a real line) is **mitigated
by design** — a probe on a line reads high and is discarded as an upper outlier by
the 25th percentile. The residual fragility is the opposite: with only 8 probes
the 25th percentile can *under*-estimate noise (inflate SNR), but the downstream
4σ shared-ε consistency rejection (:424-437) cleans admitted contaminants and ε
is a weighted fit, so the impact on reported ε is second-order. Stage 2 σ is not
directly substitutable (per-bin spectral RMS vs block-averaged coherent-demod
amplitude — different units); a cleaner hardening is more probes or a scatter
estimator. Optional, not required.

### R2 — Misleading diagnostics (no number changes)

- `window_fit.py:1556-1587/1607-1649` — stored `aicc_delta` is a penalized-score
  difference under σ_eff, not an AICc delta; accept/reject sign is correct.
- `window_fit.py:1157-1160` — phase-penalty docstring says `sqrt(λ)·|sin(Δφ/2)|`
  but the code (authoritative at :958-965, impl :1023) uses
  `sqrt(λ)·weight·cos(Δφ)`. Risk: a maintainer "fixing" code to the wrong doc.
- `window_fit.py:1140` — `max_nfev` docstring says 400 vs actual 2000.
- `residual_screening` — Rayleigh false-alarm comment says ~1% for 2.5σ (actually
  ~4.4%); detector height uses median σ while candidate SNR uses per-bin σ.
- `methods/stage5_fitting.rst:235-252` — attributes the VIF-collapse rule to
  rescue, but it lives in `_internal/stage5_impl.py` (`amplitude_vif`); self-flagged
  as issue #13 calibration debt.

### R3 — Stage 1 amplitude normalization (informational)

`core/data_structures.py:214` divides the rfft by `original_length` (full record)
while only `N_active` samples carry signal, so absolute reported amplitudes read
low by `N_active/N_full`. This is a uniform scalar across all bins — zero impact
on SNR, χ², detection, or relative amplitudes, and the robust fit recovers true
amplitudes. The adjacent comment "divide by original FID length, not padded
length" (:213) is vestigial (zero-padding was removed; `original_length ==
len(data)` on the canonical path). No action beyond the comment cleanup.

## Provenance

Maintainer manual pass (`dev-docs/review.txt`, items 1-16) merged with an
independent automated review of `preprocessing/`, `fitting/`, `core/` against the
`docs/source/` stage and methods pages. The Active-FT data-flow audit traced into
the `_internal` stage impls to settle `review #5`. No files were edited during the
review.

## Corrections found during implementation (2026-07-20)

Recorded so later readers do not re-trust the original framing:

- **DC1's "zero callers" was scoped to `src/` and `tests/` only.** Three scripts
  under `scripts/development/stage3-coherence-study/` imported
  `project_candidates`; they were removed along with the module. Treat every
  other "zero callers" claim in this doc as `src/`+`tests/`-scoped until
  re-verified against `scripts/` too.
- **T1 is far larger than "31 doc uses".** "canonical" occurs ~232 times across
  ~60 `src/` files plus ~42 in `docs/`. It is in no HDF5 key and no CLI flag
  name, so the rename is non-breaking — but many occurrences are generic English
  ("the canonical `--trim` flag") unrelated to FT settings. Scoped to the
  FT/spectrum sense, preferring "standard" in user-facing prose.
- **`"settings_source": "stage1_canonical"`** (`stage3_impl.py`) has zero readers
  anywhere in `src/`, `tests/`, or `docs/` — vestigial, removed rather than
  renamed.
- **The `_internal` "out of primary scope" line does not hold in practice.** C2,
  T2, T5 and R2 all necessarily edit `_internal/` because the dual-interface
  architecture puts the real logic there.
- **T5 was nearly empty.** Exactly one true progress marker ("pre-D7") existed;
  the `O5-*`/`D-*` tags are resolved-decision citations, not in-flight markers.
- **C1 is not the mechanical fix described.** `_install_cofit_outcome` never
  *receives* the co-fit statistics (only `new_peaks` and `tau_us`), so mirroring
  `_apply_baseline_to_outcome` requires a signature change plus a decision on
  attributing a *joint* `chi_squared`/`n_data`/`n_params` across two windows.
