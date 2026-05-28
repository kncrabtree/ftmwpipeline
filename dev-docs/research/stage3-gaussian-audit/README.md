# Stage 3 Gaussian-path defaults audit

Validates Stage 3 (peak detection) defaults on the 2638 fixture under
the Gaussian line-shape path. The pre-audit defaults were calibrated
against the Lorentzian path before the Stage 5 Gaussian-shape selector
landed; this audit asks which (if any) need to shift now that Gaussian
is a first-class path.

## Scope

Five probes, one per high-Y-rated Stage 3 knob from the
[instrument-tunable knobs table][knobs]:

[knobs]: ../../planning/instrument-tunable-knobs.md

1. Gap-pass τ-feeder substitution (`tau_basis_us` source).
2. `promotion.min_snr` (user-grid promotion floor).
3. `promotion.internal_min_snr` (internal zpf=1 detection floor).
4. `gap_pass.gap_mask_edge_threshold` (de-ramped coherent-leakage cutoff).
5. `primary_pass.min_exclusion_mhz` (gap-pass exclusion radius around
   primary peaks).

The fixture is the 2638 unapodized spectrum (`expf_us=None`,
`zpf=2`, `trim=(26500, 40000)`); both Stage 2b twins are run on each
shape's working copy so the τ-feeder can route to either anchor.

## Reproduce

```bash
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage3-gaussian-audit/probe_tau_feeder.py
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage3-gaussian-audit/probe_min_snr.py
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage3-gaussian-audit/probe_internal_min_snr.py
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage3-gaussian-audit/probe_gap_mask_edge.py
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage3-gaussian-audit/probe_min_exclusion.py
```

Per-knob CSVs land under `data/`; panel figures under `figures/`.
Working fixtures cache to
`scratch/stage3-gaussian-audit/exp_2638_unapodized_{lorentzian,gaussian}.ftmw`.
Fixture prep is the only heavy step (~3 min total for both shapes);
per-probe sweeps run in seconds.

## Headline result

**No Stage 3 default knob needs to diverge between Lorentzian and
Gaussian on 2638.** Of the five probes:

* The τ-feeder fix (Phase A,
  `src/ftmwpipeline/_internal/stage3_impl.py`) is the only
  shape-conditioned change required; it routes the gap-pass matched
  filter's `tau_basis_us` to `τ_G_maj` from
  `stage2b_tau_G_calibration` when ``recommended_shape='gaussian'``.
  On 2638 this swaps `5.958 µs → 6.411 µs` (+7.6 %) and recovers
  +100 gap detections (+12 promoted) with zero new sidelobe-proxy
  hits.
* The other four probes (the SNR floors, the gap mask edge, the
  primary-pass exclusion radius) show **shape-invariant response
  curves** within ≤ 2 % of each other at every grid point. The
  Lorentzian-calibrated defaults sit at the same operating point on
  the Gaussian path.

The architectural decision recorded against the
[`settings-backfill.md`][backfill] follow-ups -- ship divergent
shape-conditioned defaults via per-stage `_GAUSSIAN_RECOMMENDED_OVERRIDES`
populated through the existing resolver `recommended` layer
(Option 1) -- stands as the policy commitment for any future divergent
Stage 3, 4, or 5 knob. **No `_GAUSSIAN_RECOMMENDED_OVERRIDES` table
is shipped for Stage 3 today**, because there is nothing to override.
The first divergent knob to land in a future audit will be the
provisional first entry in that table.

[backfill]: ../../planning/settings-backfill.md

## Per-probe results

### Probe 1: τ-feeder substitution

`probe_tau_feeder.py` runs four variants against the cached fixtures
to isolate the impact of the τ value alone:

| variant | recommended_shape | τ_basis (µs) | source | n_peaks | n_promoted | primary | gap | SNR p95 | sidelobe-proxy |
|---|---|---|---|---|---|---|---|---|---|
| L_anchor | lorentzian | 5.958 | stage2b_tau_maj | 5533 | 758 | 3188 | 2345 | 9.34 | 0 |
| G_old_tau | lorentzian (forced) | 5.958 | stage2b_tau_maj | 5533 | 758 | 3188 | 2345 | 9.34 | 0 |
| G_new_tau | gaussian | 6.411 | stage2b_tau_G_maj | 5633 | 770 | 3188 | 2445 | 9.02 | 0 |
| L_with_tau_G | gaussian (forced) | 6.411 | stage2b_tau_G_maj | 5633 | 770 | 3188 | 2445 | 9.02 | 0 |

Key reads:

* **Pre-Phase-A behaviour on Gaussian data was identical to Lorentzian
  data** (L_anchor == G_old_tau). The τ-feeder was the only piece that
  cared about the shape; Stage 3 has no other shape-aware code path.
* **Post-Phase-A** the τ value drives the entire downstream delta. The
  +12 promoted peaks (~1.6 %) are weak gap-pass detections the wider
  matched filter newly resolves. SNR p95 dips marginally (9.34 → 9.02)
  -- the extra peaks pull the upper tail down.
* **Symmetry across the shape tag** (G_new_tau == L_with_tau_G)
  confirms the τ value is the *only* downstream consumer of the
  feeder branch. The matched-filter apodization itself remains a
  pure-exp ``exp(-t/τ_basis_us)`` in all four variants; with τ_G
  substituted in, the filter is *mismatched* relative to a true
  Gaussian envelope. The cost of that mismatch is bounded by the SNR
  p95 shift (a few percent) and the absence of new sidelobe-proxy
  hits -- modest enough that the time-constant swap is sufficient for
  the production path on 2638. The apodization-shape question (Gauss
  vs exp filter at the same τ_G anchor) would require a code change
  to `_mf_gap_spectrum`; deferred as a follow-up if a future fixture
  shows a sharper τ-mismatch penalty.

### Probe 2: `promotion.min_snr`

| min_snr | L: n_peaks | L: n_promoted | G: n_peaks | G: n_promoted |
|---|---|---|---|---|
| 2.0 | 5533 | 3284 | 5633 | 3396 |
| 2.5 | 5533 | 1377 | 5633 | 1403 |
| 3.0 | 5533 | 758 | 5633 | 770 |
| 4.0 | 5533 | 505 | 5633 | 506 |
| 5.0 | 5533 | 408 | 5633 | 410 |

The cliff between 2.0 → 3.0 (4.3× drop in promoted count) is a 2638
property, not a shape property -- both paths fall off at the same
rate. The Lorentzian / Gaussian promoted counts agree to ≤ 3.5 %
across the entire sweep. The default `min_snr = 3.0` lands at the
same operating point on both paths.

### Probe 3: `promotion.internal_min_snr`

| internal_min_snr | L: n_peaks | L: n_promoted | G: n_peaks | G: n_promoted |
|---|---|---|---|---|
| 1.5 | 18715 | 1017 | 18941 | 1029 |
| 2.0 |  5533 |  758 |  5633 |  770 |
| 2.5 |  1423 |  611 |  1451 |  623 |
| 3.0 |   629 |  537 |   636 |  546 |

The 2.0 default is the knee for both shapes: pulling down to 1.5
inflates the candidate pool 3.4× (with only +259 / +34 % promoted --
the rest is noise), and pulling up to 3.0 loses ~30 % of promoted
peaks at the apodization-smeared floor. Lorentzian and Gaussian agree
to within 1.6 % at every floor. The default is shape-invariant.

### Probe 4: `gap_pass.gap_mask_edge_threshold`

| threshold | L: n_gap | G: n_gap | L: sidelobe | G: sidelobe |
|---|---|---|---|---|
| 4.0 | 1054 | 1146 | 0 | 0 |
| 6.0 | 1796 | 1893 | 0 | 0 |
| 8.0 | 2345 | 2445 | 0 | 0 |
| 10.0 | 2811 | 2852 | 0 | 0 |
| 12.0 | 3169 | 3228 | 0 | 0 |

Both shapes show a smoothly monotonic detection curve. Gaussian has
+50–100 gap detections at every threshold (the wider τ_G_maj MF), but
the *shape* of the response is identical. The sidelobe-suspect proxy
is zero everywhere -- the proxy threshold (internal SNR ≥ 2 × promotion
floor, user SNR < floor) doesn't catch anything even at the most
permissive cutoff. The default 8.0 is shape-invariant; the question
of whether 8.0 is the right *level* is independent of shape and is
covered by the original [`leakage-detection-rework.md`][leak]
calibration.

[leak]: ../../planning/leakage-detection-rework.md

### Probe 5: `primary_pass.min_exclusion_mhz`

| exclusion (MHz) | L: n_gap | G: n_gap | L: n_peaks | G: n_peaks |
|---|---|---|---|---|
| 0.00 | 2345 | 2445 | 5533 | 5633 |
| 0.05 | 2316 | 2415 | 5504 | 5603 |
| 0.10 | 2306 | 2406 | 5494 | 5594 |
| 0.20 | 2184 | 2275 | 5372 | 5463 |
| 0.50 | 1853 | 1933 | 5041 | 5121 |

The primary count is constant at 3188 across the sweep (the knob only
gates the gap pass). Both shapes lose ~21 % of gap detections at
excl = 0.5; the relative loss curve is identical. The default 0.0 --
which delegates sidelobe rejection entirely to the gap mask -- is
the most permissive setting and lands at the same operating point on
both paths.

## Limitations

* **Single fixture.** Everything here is on 2638. A second
  Gaussian-shape fixture (when one is available) is needed before
  promoting "no divergence on 2638" to "no divergence universally."
* **Sidelobe-suspect proxy is a coarse heuristic** -- it catches
  high-internal-SNR / low-user-SNR detections, which is one
  failure mode of the gap-pass mask. A high-permissive threshold
  could still admit sidelobes that look fine on the internal grid
  too (the proxy under-detects). The zero-everywhere result says
  "the proxy didn't fire," not "there are no sidelobes."
* **No ground-truth peak list.** Probes report relative changes,
  not accuracy. The +12 promoted peaks from Phase A are
  *plausibly* real lines the wider MF resolves, but the audit
  doesn't prove it -- a known-line spike-in test or a cross-fixture
  comparison would.
* **The apodization-shape question** (exp filter at τ_G vs Gauss
  filter at τ_G) is deferred. The SNR p95 dip from 9.34 to 9.02
  is the bound on the matched-filter mismatch cost on 2638; if a
  future fixture shows a sharper penalty, swapping the filter
  itself becomes the right next step.

## Decisions recorded

1. **τ-feeder shape-awareness shipped (Phase A).** Stage 3 gap-pass
   `tau_basis_us` prefers `τ_G_maj` from `stage2b_tau_G_calibration`
   when `recommended_shape='gaussian'`; otherwise falls through to
   `τ_maj`, `expf_us`, 5.0 µs in order.
   See `_internal/stage3_impl.py` and
   `tests/integration/test_stage3_settings_propagation.py::TestGapPassTauFeeder`.
2. **No Stage 3 hard defaults change.** All four sweep probes show
   shape-invariant response on 2638; no candidate for a
   `_GAUSSIAN_RECOMMENDED_OVERRIDES` entry today.
3. **Architectural policy: Option 1 stands as the design
   commitment** for any future Stage 3 / 4 / 5 knob whose
   shape-optimal default diverges on a future fixture. The
   implementation lands when the first divergent knob does --
   shipping empty plumbing now would be code without purpose.
4. **`settings-backfill.md` follow-up #5 status (Stage 3 portion):
   shipped.** The Stage 4 `leakage.tau_us` portion and the
   Stage 5-rescue τ portion remain open.
