# Plan: Processing-settings persistence and propagation

Status: **Complete** (D7 resolved). Cross-cutting architectural fix.

Implemented:

- `core/settings.py`: `FTSettings` dataclass — single source of truth for the
  FT settings across API/CLI/persistence; `resolve()` implements
  `explicit > persisted (ft_processing) > recommended` with hard defaults;
  `to_attrs`/`from_attrs` persist the canonical record (trim **inside**
  `ft_processing` as `trim_min_mhz`/`trim_max_mhz`). `cli/_argspec.py`
  generates argparse options from `cli_field` metadata. (Since the apodization
  removal, `FTSettings` carries only `start_us` / `end_us` / `trim` plus the
  `units_power` / `rdc` display knobs — see the banner below.)
- `stage1_impl`: `compute_ft_impl` resolves settings and, on a user-driven
  invocation, persists the resolved record incl. `trim` as canonical; a
  no-argument recompute (Stages 2/3) reproduces exactly the user-chosen
  spectrum. Changing the canonical settings via an explicit override
  invalidates every stage built on the FT (data group deleted, dropped from
  completed set, loud warning); an identical re-persist is idempotent.
- Stage 2 noise is estimated on the persisted spectrum (inherited via the
  no-arg recompute).
- `stage3_impl`: the interim `_resolve_trim`/`_resolve_zpf`/saved-Stage-3
  trim+zpf and the `--trim`/`--zpf` options are gone. Detection runs on its own
  internal grid then snaps every peak onto the persisted user grid by physical
  frequency, re-measuring amplitude and SNR against the canonical Stage 2 noise;
  internal-grid values are kept under `Peak.properties`. `peaks show` overlays
  on the user spectrum. (The Stage 3 internal detection grid is documented in
  [`stage3-peak-detection.md`](stage3-peak-detection.md).)
- `pipeline.py` / `api.py` / CLI thread `FTSettings`; `detect_peaks` no longer
  exposes trim/zpf. `SERIALIZATION_STRATEGY.md` / `API_STRATEGY.md` amended;
  ROADMAP D7 marked resolved.

The original plan (kept for provenance) follows. It predates the removal of
explicit FT apodization: its references to persisting `zpf`, `expf_us`, and
`window_function` describe the settings surface as it was then. The canonical FT
is now unconditionally unapodized and native-length — those knobs no longer
exist — and `FTSettings` carries only `start_us` / `end_us` / `trim` plus the
`units_power` / `rdc` display knobs; see
[`ft-apodization-removal.md`](ft-apodization-removal.md). The D7 persistence
contract (persist the user's chosen FT settings as canonical, later stages
inherit them) is unchanged; only the set of persisted FT knobs shrank.

Normative requirements remain in the `*_STRATEGY.md` specs; this document is
normative only for the work it tracks. Registered in
[`../ROADMAP.md`](../ROADMAP.md) (divergence **D7**).

## Problem (root cause)

The FT processing settings a user chooses are **not persisted as the
experiment's canonical settings**, and later stages silently fall back to the
import-time *recommended* defaults instead of what the user asked for.

Concretely, on experiment 2638:

- `ftmw.compute_ft(f, zpf=2, expf_us=5.0, trim=(26500, 40000))` returns a
  correctly trimmed, zpf=2 `ComplexFT` **for that call**, but persists only a
  partial `processing_parameters/ft_processing` record. `trim` is not
  persisted at all; `zpf` is written from the *recommended* block (0), not the
  user's value.
- `compute_ft_impl(file_path)` (no args), used by Stage 2 and Stage 3 to
  recompute the FT on demand, therefore rebuilds the spectrum from
  `stage0_fid_data/recommended_processing` (zpf=0, no trim) — **a different
  spectrum than the user configured**.
- Consequence: Stage 2 noise was estimated on the full untrimmed spectrum;
  Stage 3 detection, before mitigation, ran on zpf=0 / untrimmed data (DC
  edges → ~zero noise → nonsense SNR). The Stage 3 `trim`/`zpf` parameters and
  `_resolve_trim`/`_resolve_zpf` helpers added in the peak-detection work are
  **interim band-aids** for this; they must be removed/refactored by this task,
  not extended.

The recommended-processing block is a *suggestion captured at import*. Once the
user has run Stage 1 with chosen settings, those settings — not the
recommendations — are the source of truth for the rest of the pipeline.

## Required behaviour

1. **Persist what the user chose.** Stage 1 (`compute_ft` / `compute-ft`) must
   persist the *actually used* FT processing settings — `start_us`, `end_us`,
   `zpf`, `expf_us`, `window_function`, `units_power`, **and the frequency
   `trim` range** — into the `.ftmw` file as the experiment's canonical
   processing settings.
2. **Later stages respect them by default.** With no explicit override, Stages
   2–5 recompute/operate on the spectrum defined by the *persisted* settings,
   so every downstream result is expressed in terms of the settings the user
   chose. Resolution order everywhere: **explicit override > persisted user
   settings > import-time recommended**.
3. **Algorithmic deviations are internal only.** A stage MAY run an internal
   step with different settings for algorithmic reasons (e.g. peak detection
   internally at **zpf=1** for apex localization — likely the right default —
   plus an unapodized pass for gap recovery). When it does, the stage's
   persisted/returned result MUST be expressed on the user's chosen grid:
   report physical frequency and re-measure amplitude/SNR on the user-settings
   spectrum, or snap indices back onto the user grid. Internal grids must not
   leak into stored results or downstream stages.
4. **Overrides recompute, and persist intent.** If a stage exposes an explicit
   parameter override, that invocation may recompute the FT as needed; the
   override (and whether it updated the canonical settings) is recorded
   deliberately. The *normal* action — no override — uses persisted settings.

## Scope / touch points

- `stage1_impl`: persist the resolved settings (incl. `trim`) into
  `processing_parameters/ft_processing`; `compute_ft_impl` resolution becomes
  explicit > persisted > recommended; an explicit arg both recomputes and
  updates the canonical record.
- `stage2_impl`: noise estimated on the persisted-settings spectrum.
- `stage3_impl`: **remove** the interim `trim`/`zpf` ownership
  (`_resolve_trim`, `_resolve_zpf`, saved-Stage-3 trim/zpf, the `--trim` /
  `--zpf` user-facing options). Detection consumes persisted settings; it may
  *internally* use zpf=1 + unapodized for the algorithm, then snap results onto
  the persisted user grid (frequency-based; re-measure amplitude/SNR there).
  The unapodized-scoring, apex-snap, dedupe and log-overlay work from the
  peak-detection task stays — only the settings-ownership band-aid is undone.
- `SERIALIZATION_STRATEGY.md` / `API_STRATEGY.md`: amend to state that
  user-chosen FT settings are canonical persisted state and binding on later
  stages (resolves D7).
- Interaction with safe re-import (`SourceMetadata` hashing) and the
  "lightweight `.ftmw`, recompute-on-demand" model must be preserved: persist
  *settings*, not the recomputed `ComplexFT`.

## Resolved questions

These were open at planning time and are settled by the implementation above:

- Canonical set: `trim` lives **inside** `ft_processing` (as
  `trim_min_mhz`/`trim_max_mhz`), not as a separate analysis-region record.
- Peak snap-back: physical frequency + amplitude re-measured on the user grid,
  unit-tested on both sidebands.
- Stage 2 noise on canonical-settings change: invalidated and recomputed via
  the dependency tracker.
- Later-stage re-runs on canonical-settings change: an explicit override
  invalidates every downstream stage (data dropped, removed from the completed
  set, loud warning); an identical re-persist is idempotent.

## Test plan

- After `compute_ft(custom trim/zpf/expf)`, a fresh `estimate_noise` /
  `detect_peaks` (no args) operates on exactly those settings (assert grid
  size, freq range, trim).
- Override path recomputes for that call and updates canonical settings;
  subsequent no-arg stage calls see the new settings.
- Persisted settings survive close/reopen and are identical across CLI /
  Pipeline / functional API (cross-interface, per `TESTING_STRATEGY.md`).
- Stage 3 results are reported on the persisted user grid even though
  detection runs internally at zpf=1 (assert frequencies/amplitudes match the
  user-grid spectrum, not the internal one).
- Regression: 2638 end-to-end with user settings `zpf=2, expf=5,
  trim=26500–40000` yields noise and peaks consistent with that spectrum.

## Relationship to the peak-detection task

`stage3-peak-detection.md` is otherwise complete and correct (two-pass
detection, O1 leakage reach, unapodized scoring + apex-snap + dedupe,
serialization, log overlay, cross-interface/real-data tests). Only its
`trim`/`zpf` *settings ownership* is interim and is superseded here. Do not
revert the algorithmic Stage 3 work; replace the band-aid with proper
persistence + snap-back.
