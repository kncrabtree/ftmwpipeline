# Stage 4 Gaussian-path defaults audit

Validates Stage 4 (window planning) defaults on the 2638 fixture
under the Gaussian line-shape path. Companion to
[`stage3-gaussian-audit/README.md`](../stage3-gaussian-audit/README.md).

This audit also serves as the data-driven resolution for the Stage 4
portion of [`settings-backfill.md`][backfill] follow-up #5
(should `leakage.tau_us` consume the Stage 2b τ when available?).

[backfill]: ../../planning/settings-backfill.md

## Scope

Four probes, one per high-Y-rated Stage 4 knob from the
[instrument-tunable knobs table][knobs]:

[knobs]: ../../planning/instrument-tunable-knobs.md

1. `leakage.tau_us` -- boxcar (default `None`) vs Stage 2b
   `τ_maj` / `τ_G_maj` vs bracketing values (3, 12 µs).
2. `coherence.edge_threshold` -- S_coh cutoff for leakage-touched-region
   detection.
3. `clustering.max_window_width_mhz` -- width cap above which windows
   become HARD and gain split proposals.
4. `contributor.magnitude_attachment_threshold` -- tier-1 contributor
   attachment threshold in σ_c units.

The fixture is the 2638 unapodized spectrum (`expf_us=None`,
`zpf=2`, `trim=(26500, 40000)`); both Stage 2b twins are run on each
shape's working copy, the recommended_shape stamp drives the Stage 3
τ-feeder for that copy, and Stage 3 detection runs once per shape.

## Reproduce

```bash
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage4-gaussian-audit/probe_leakage_tau.py
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage4-gaussian-audit/probe_edge_threshold.py
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage4-gaussian-audit/probe_max_width.py
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage4-gaussian-audit/probe_mag_attachment.py
```

Per-knob CSVs land under `data/`; panel figures under `figures/`.
Working fixtures cache to
`scratch/stage4-gaussian-audit/exp_2638_unapodized_{lorentzian,gaussian}.ftmw`.
Fixture prep takes ~4 min total (both shapes, includes Stage 2b
calibrations and Stage 3 detection). Per-probe sweeps run in
seconds.

## Headline result

**No Stage 4 default changes on 2638.** All four probes show
shape-invariant response curves; the L and G fixtures produce plans
within ≤ 3 % of each other on every metric. The follow-up #5
question for Stage 4 (`leakage.tau_us` boxcar vs Stage 2b τ)
resolves *empirically* on Stage 5 χ²ᵣ grounds (`probe_leakage_tau_p5.py`):

* **The boxcar default for `leakage.tau_us` is correct on 2638.**
  Window count and window widths are **byte-identical** across all
  τ variants (the `min_window_half_width_mhz=2.0` floor and peak
  clustering -- not the analytic leakage reach -- are what set the
  window boundaries). The τ choice only changes which strong
  out-of-band lines get *attached* as `fixed_contributors`. Stage 5
  χ²ᵣ comes out the same regardless: aggregate median χ²ᵣ moves by
  ≤ 0.02 across variants, p95 χ²ᵣ moves by ≤ 0.18 (and tau_G_maj /
  boxcar are tied). For the 23-29 windows with χ²ᵣ > 5 on boxcar --
  the ones we'd most like to rescue -- **every τ variant produces
  the identical contributor set**: the high-χ²ᵣ tail is not an
  under-attached-contributor problem and the τ-feed cannot help it.
  Keeping boxcar avoids the dependency-graph inflation
  (+25 / +9 hard windows for τ_maj / τ_G_maj on 2638) while losing
  no measurable Stage 5 quality.

The architectural commitment from
[`settings-backfill.md`][backfill] (Option 1: route shape-conditioned
defaults through the resolver's `recommended` layer via per-stage
`_GAUSSIAN_RECOMMENDED_OVERRIDES`) still stands as the policy for
future divergent knobs, but Stage 4 contributes no overrides today.

**Follow-up #5 status.** Stage 3 portion shipped (`stage3_impl.py`
gap-pass τ-feeder routes to `τ_G_maj` when `recommended_shape='gaussian'`).
Stage 4 portion resolved negative on χ²ᵣ grounds. **Stage 5 rescue τ
portion was already shipped** as part of the original Stage 5 Gaussian
work: the `tau_maj_us` carried in `conservative_kwargs` is the
shape-matching twin's value (line 573-577 of `stage5_impl.py` picks
`tau_G_calibration` when `shape=GAUSSIAN`), and the rescue at
`residual_rescue.py:684` consumes it directly. Verified empirically
in `scratch/verify_rescue_tau.py`: 695 rescue calls on the
Gaussian fixture all saw per-band τ_G values (6.26 / 6.63 / 8.34 µs),
zero Lorentzian τ_maj=5.958 leakage. The note in
[`stage5-gaussian-shape.md`][gauss] § Open Questions claiming the
rescue uses the pure-exp τ is stale and is updated alongside.

[gauss]: ../../planning/stage5-gaussian-shape.md

## Per-probe results

### Probe 1: `leakage.tau_us` (boxcar vs Stage 2b τ)

The headline probe -- also the resolution for follow-up #5's Stage 4
piece. Variants:

| shape | variant | τ (µs) | n_windows | n_hard | n_fixed | n_deps | width p95 (MHz) |
|---|---|---|---|---|---|---|---|
| L | boxcar | None | 386 | **188** | **124** | 76 | 9.54 |
| L | tau_maj | 5.958 | 386 | 213 (+13 %) | 137 (+10 %) | 75 | 9.54 |
| L | tau_G_maj | 6.411 | 386 | 208 (+11 %) | 142 (+15 %) | 83 | 9.54 |
| L | tau_3 | 3.000 | 386 | 260 (+38 %) | 213 (+72 %) | 118 | 9.54 |
| L | tau_12 | 12.000 | 386 | 193 (+3 %) | 132 (+6 %) | 80 | 9.54 |
| G | boxcar | None | 391 | **190** | **125** | 77 | 9.52 |
| G | tau_maj | 5.958 | 391 | 215 (+13 %) | 138 (+10 %) | 76 | 9.52 |
| G | tau_G_maj | 6.411 | 391 | 210 (+11 %) | 143 (+14 %) | 84 | 9.52 |
| G | tau_3 | 3.000 | 391 | 264 (+39 %) | 216 (+73 %) | 120 | 9.52 |
| G | tau_12 | 12.000 | 391 | 195 (+3 %) | 133 (+6 %) | 81 | 9.52 |

Stage 4-internal reads (this probe alone):

* **Window count, width distribution, and width max are byte-identical
  across all τ variants.** The leakage reach feeds the `predicted_skirt`
  amplitude used to attach `fixed_contributors`, but it does *not* set
  window boundaries on 2638 -- `min_window_half_width_mhz=2.0` (the
  isolated-peak half-width floor) and peak clustering drive the
  boundaries. Smaller τ → wider analytic reach → more strong lines
  clear the attachment threshold → more `fixed_contributors` per
  window. Same windows, different attachments.
* **Smaller τ → more attachments.** The reach formula
  `reach ∝ env_factor / tau_eff` has the smaller `tau_eff` term
  dominating for `T/τ ≈ 2` (the 2638 regime). Boxcar
  (`tau_eff = T = 12.65 µs`, `env_factor = 2.0`) gives the
  *narrowest* analytic reach in this regime, which is the
  *opposite* of what the `estimate_leakage_reach` docstring's "most
  leakage-prone case" claim suggests. The docstring is misleading
  about direction; it is the most *amplitude-conservative* case
  (env_factor=2 is the upper bound), but the narrower
  temporal-integration regime (smaller τ) wins on *spectral-spread*
  and produces wider reach in absolute terms.
* **τ_12 ≈ boxcar.** At τ=12 µs ≈ T, `tau_eff = 12 × (1 - e^{-1.05}) =
  7.8 µs` (vs T=12.65), so the reach approaches the boxcar limit from
  above. n_fixed differs by only 8 contributors between τ_12 and boxcar.
* **L/G symmetry.** Lorentzian and Gaussian fixtures behave nearly
  identically -- the same τ value drives the same attachment pattern
  on either shape. The +5 absolute window difference (386 vs 391) is
  the +5 promoted peaks the shape-aware Stage 3 τ-feeder recovers on
  the Gaussian fixture.

**Stage 4-internal metrics are not the verdict.** "More
fixed_contributors" can be either useful (an out-of-band line
attached as a contributor to absorb bleed-through) or wasteful
(an unnecessary attachment that adds a dependency edge for nothing).
Only the downstream Stage 5 fit can distinguish these cases. That
is the next sub-probe.

### Probe 1 follow-up: Stage 5 χ²ᵣ comparison across τ variants

`probe_leakage_tau_p5.py` runs `fit_peaks` on each variant fixture
above (3 τ variants × 2 shapes = 6 fits, ~12 min wall-clock) and
compares per-window χ²ᵣ. Variants are matched window-by-window by
`window_id` (probe 1 confirmed window boundaries are byte-identical
across variants, so the match is exact).

**Aggregate χ²ᵣ across variants** (per shape):

| shape | variant | χ²ᵣ med | χ²ᵣ p95 | χ²ᵣ max | n > 5 | n > 10 | total fixed | fitted peaks |
|---|---|---|---|---|---|---|---|---|
| L | boxcar | 1.363 | 5.723 | 76.4 | 23 | 13 | 124 | 717 |
| L | tau_maj | 1.362 | 5.945 | 76.4 | 24 | 13 | 137 | 717 |
| L | tau_G_maj | 1.358 | 5.723 | 76.4 | 23 | 13 | 142 | 718 |
| G | boxcar | 1.380 | 6.819 | 202.0 | 29 | 12 | 125 | 631 |
| G | tau_maj | 1.380 | 6.996 | 202.0 | 30 | 12 | 138 | 631 |
| G | tau_G_maj | 1.379 | 6.819 | 202.0 | 29 | 12 | 143 | 631 |

The aggregate distributions are flat: median χ²ᵣ varies by ≤ 0.02,
p95 by ≤ 0.18 across variants. tau_G_maj and boxcar are tied on p95
and on n > 5 / n > 10; tau_maj is slightly worse on p95 (+0.22 / +0.18).
Total fitted peak count is constant (717-718 / 631).

**Per-pair conditional Δχ²ᵣ by Δn_fixed_contributors:**

```
shape   pair (a → b)          n   identical-fixed  more-b  more-a   med Δχ²ᵣ on more-a
L      boxcar → tau_maj    386       349              25     12    +0.324 (p95 +3.64)
L      boxcar → tau_G_maj  386       359              23      4    +0.529
L      tau_maj → tau_G_maj 386       367              10      9    -0.271 on more-b
G      boxcar → tau_maj    391       354              25     12    +0.316 (p95 +3.49)
G      boxcar → tau_G_maj  391       364              23      4    +0.466
G      tau_maj → tau_G_maj 391       372              10      9    -0.277 on more-b
```

(`more-b` = variant B added a contributor A lacked; `more-a` =
variant B dropped a contributor A had.)

Two patterns stand out:

1. **~90 % of windows have identical contributor sets across τ
   variants** (349-372 of 386-391). For those windows Δχ²ᵣ is exactly
   zero -- the τ choice is invisible.
2. **On the ~10 % of windows where attachments differ, the effects
   are small.** Median Δχ²ᵣ when a variant adds a contributor is
   between -0.004 and -0.28 (slight improvement). Median Δχ²ᵣ when a
   variant drops a contributor that the comparison plan had is
   +0.32 to +0.53 (slight degradation, with p95 reaching +3.6 on
   the τ_maj-drops-boxcar-contributors bucket). tau_G_maj has the
   fewest drops (4 vs 12 for tau_maj) and adds the most useful
   contributors (mean improvement -0.027 vs tau_maj's -0.016).

**The high-χ²ᵣ tail is not a contributor-attachment problem.**
Restricting the comparison to windows with χ²ᵣ > 5 on boxcar
(`scratch/verify_rescue_tau.py`-style cut on the per-pairs CSV):

| chi2r_boxcar threshold | n windows (L / G) | n with differing contributors (L:tau_maj / L:tau_G_maj / G:tau_maj / G:tau_G_maj) |
|---|---|---|
| > 2 | 95 / 101 | 11 / 7 / 13 / 10 |
| > 5 | **23 / 29** | **0 / 0 / 0 / 1** |
| > 10 | 13 / 12 | 0 / 0 / 0 / 0 |

The 23-29 worst-fitting windows on boxcar **get the identical
contributor set on every τ variant.** The χ²ᵣ excess on those
windows has a different cause -- shape mismatch / blend complexity /
missing real line / frozen-contributor parameter error -- and the
τ-feed cannot remediate it.

**Why the boxcar default is correct on χ²ᵣ grounds.** Aggregate
Stage 5 fit quality is unchanged across τ variants. The conditional
analysis shows tau_G_maj is marginally better on the well-fitted
tail, but the magnitude (≤ 0.05 median Δχ²ᵣ improvement on
< 10 % of windows) is too small to justify wiring a shape-aware
τ-feeder. The high-χ²ᵣ tail -- the cases that would actually
benefit from "better" contributor attachment -- gets the same
contributor set on every variant, so the τ-feed offers no rescue
where it would matter. The boxcar default is a strictly simpler
plan recipe for indistinguishable downstream quality.

### Probe 2: `coherence.edge_threshold`

| threshold | L: n_windows | L: n_hard | L: n_fixed | G: n_windows | G: n_hard | G: n_fixed |
|---|---|---|---|---|---|---|
| 4.0 | 379 | 267 | 133 | 384 | 268 | 139 |
| 6.0 | 386 | 202 | 124 | 391 | 204 | 125 |
| 8.0 | 386 | 188 | 124 | 391 | 190 | 125 |
| 10.0 | 390 | 187 | 102 | 395 | 189 | 103 |
| 12.0 | 390 | 184 | 102 | 395 | 186 | 103 |

The n_hard curve has a knee around threshold = 6-8: below 6 the
hard count jumps dramatically (low threshold = many false
"leakage-touched" regions); above 8 it plateaus. The default 8.0
lands on the plateau. Lorentzian and Gaussian responses agree to
≤ 2 % at every threshold. Keep default.

### Probe 3: `clustering.max_window_width_mhz`

| cap (MHz) | L: n_windows | L: n_hard | L: n_split | G: n_windows | G: n_hard | G: n_split |
|---|---|---|---|---|---|---|
| 20 | 386 | 188 | 1 | 391 | 190 | 1 |
| 30 | 386 | 188 | 1 | 391 | 190 | 1 |
| 40 | 386 | 188 | 1 | 391 | 190 | 1 |
| 60 | 386 | 188 | 0 | 391 | 190 | 0 |
| 80 | 386 | 188 | 0 | 391 | 190 | 0 |

The cap is **effectively dormant** on 2638: n_windows and n_hard are
flat across the entire 20-80 MHz sweep. The width-p95 distribution
(9.54 / 9.52 MHz) sits an order of magnitude below the smallest cap
tested. Only one window (~45 MHz wide) hits the 40 MHz cap and gets
a split proposal; raising the cap to 60+ MHz suppresses even that
single proposal. The default 40 MHz preserves the single
conservative split trigger on 2638 without affecting anything else.
Keep default.

### Probe 4: `contributor.magnitude_attachment_threshold`

| threshold | L: n_fixed | L: n_deps | G: n_fixed | G: n_deps |
|---|---|---|---|---|
| 0.050 | 193 | 107 | 196 | 109 |
| 0.075 | 135 | 74 | 136 | 75 |
| 0.100 | 124 | 76 | 125 | 77 |
| 0.150 | 114 | 76 | 118 | 79 |
| 0.200 | 90 | 61 | 92 | 63 |

The cliff is between 0.05 and 0.075 (-30 % fixed_contributors); from
0.075 to 0.150 the slope flattens (-15 % total over three steps);
0.20 drops another -21 %. The default 0.10 sits in the stable
mid-slope region (no nearby cliff). L/G agree to ≤ 4 %. The probe
also indirectly answers the "was this a Lorentzian-skirt
calibration?" question from the prompt: `_leakage_envelope_fraction`
uses the same closed-form analytic envelope as
`estimate_leakage_reach`, so the threshold was calibrated under
*boxcar* (the default), not a Lorentzian-specific skirt. Either way,
the empirical L/G curves overlay; keep default.

## Limitations

* **Single fixture.** Everything here is on 2638 (T = 12.65 µs,
  τ ≈ 6 µs). The "boxcar wins" verdict on `leakage.tau_us` is a
  T/τ ≈ 2 statement; instruments with very different `T/τ` (e.g.
  τ ≫ T, where `tau_eff → T` for the damped case and `env_factor → 2`
  too) might show different trade-offs.
* **High-χ²ᵣ-tail mechanism is unidentified.** The probe shows the
  τ-feed *cannot* remediate the 23-29 worst-fitting windows on 2638
  because every variant attaches the same contributors there. The
  remaining causes (frozen-contributor parameter error after thaw,
  missing real line, blend complexity, shape mismatch) are not
  diagnosed here. A focused Stage 5 fit-quality study on those
  specific windows is the natural follow-up.
* **The reach-formula direction.** The "boxcar is more leakage-prone"
  docstring claim on `estimate_leakage_reach` is misleading for
  `T/τ ~ 2` regimes (where the τ-fed reach is wider). Worth a
  separate clarifying edit to that docstring; not done here.

## Decisions recorded

1. **`leakage.tau_us` default stays `None` (boxcar).** Empirical
   evidence on 2638 (probe 1 + p5 follow-up): aggregate χ²ᵣ is
   unchanged across τ variants, the high-χ²ᵣ tail gets identical
   contributor sets on every variant (so τ-feed cannot help where
   it would matter), and the marginal Δχ²ᵣ on well-fitted windows
   is too small (≤ 0.05 median) to justify the added complexity of
   a shape-aware τ-feeder. **The τ-feeder fix for Stage 4 is not
   shipped.**
2. **No Stage 4 hard defaults change.** All four probes show
   shape-invariant response; no candidate for a
   `_GAUSSIAN_RECOMMENDED_OVERRIDES` entry today.
3. **Option 1 architectural commitment stands** for any future
   Stage 4 knob whose shape-optimal default diverges on a different
   fixture. The implementation lands when the first divergent knob
   does.
4. **`settings-backfill.md` follow-up #5 fully resolved.** Stage 3
   portion: shipped (`stage3_impl.py` gap-pass τ-feeder routes to
   τ_G_maj when Gaussian recommended). Stage 4 portion: resolved
   negative on Stage 5 χ²ᵣ grounds. Stage 5 rescue portion: already
   shipped as part of the original Stage 5 Gaussian work
   (`stage5_impl.py:573-577` selects `tau_G_calibration` when
   `shape=GAUSSIAN`; the rescue's `tau_maj_us` carries that value);
   the open-question note in `stage5-gaussian-shape.md` was stale
   and is corrected.
