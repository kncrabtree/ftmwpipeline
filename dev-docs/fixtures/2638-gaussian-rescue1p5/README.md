# Fixture: 2638 gaussian / rescue_prominence=1.5 — manual window classifications

`windows.toml` holds **hand-entered per-window classifications** of a Stage 5
fit on the 2638 BlackChirp experiment. These are an irreplaceable human-labelled
oracle (good fit vs overfit / shape_error / missed_peak / spur / …), keyed by
`freq_range_mhz` so they survive window-id renumbering across re-fits. They are
preserved here because the fit they describe lives only in an untracked
`scratch/` artifact.

## Provenance

- **Source data:** `examples/blackchirp_data/2638/` (the checked-in experiment).
- **Fit configuration:** Stage 5, `shape='gaussian'`, `rescue_prominence_threshold=1.5`,
  `rescue_snr_threshold=2.5`, `max_residual_rescue_rounds=5`,
  `tau_calibration_source='persisted'` (`tau_maj ≈ 6.41 µs`). Active region
  `T_active = 12.65 µs` (`1/T_active ≈ 79.05 kHz`), `n_active = 632499`.
- **Labels finalized:** 2026-05-28 (last edit 22:55 local).
- **Code state when labelled — IMPORTANT:** the fit **predates** two Stage 5
  features that shipped the next day:
  - clock/LO **spur masking** (commit `c75bd48`, 2026-05-29), and
  - the evidence-triggered **leakage-wing baseline** (commit `96128fc`, 2026-05-29).

  The labelled fixture itself has `spur_masking_enabled` and `baseline_enabled`
  unset (features absent at fit time).

## How to use these labels with current code

The labels remain a valid oracle for **which windows are overfit vs real**, but
the fit they were made on is stale. To validate against current `main`:

1. Re-run Stage 5 on 2638 with the same config (`shape='gaussian'`,
   `rescue_prominence_threshold=1.5`) using current code (spur masking +
   leakage-wing baseline now default-on).
2. Re-map labels by `freq_range_mhz` (NOT `window_id`, which can shift).
3. Expect divergence on the classes the new features address:
   - `spur`-classified windows (e.g. 28460/29440/30720/35840/39040 MHz) should
     now be gated by spur masking.
   - `baseline_offset`-classified windows (e.g. w222/224/225/226/227, the
     33820–33900 MHz chain) should now be improved by the leakage-wing baseline.
   The `overfit`, `shape_error`, `missed_peak`, and good-fit classifications on
   isolated/clean lines are not touched by spur/baseline work and should
   transfer.

## Sub-resolution overfit study (GitHub issue #13)

This fixture is the substrate for the sub-resolution-overfit discriminant in
issue #13. The discriminant is closest-pair separation in active-FT resolution
elements, `Δf / (1/T_active)`:

- **Overfits to collapse** (separation 0.80–1.07 elements, amp ratio 4–10:1):
  w012, w022, w040, w069, w071, w091, w143, w281.
- **Good tight blends to preserve** (separation 1.08–1.73 elements, amp ratio
  1–4:1): w020, w075, w159, w189, w218, w303.

(Window ids above are from this labelling pass; re-map by `freq_range_mhz`.)

## Classification legend

The legend and conventions are in the header comment of `windows.toml`. Empty
`classification` = "good fit / no concerns".
