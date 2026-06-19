# Frequency calibration and the σ_f uncertainty budget — research-report plan

Status: **planned (not written).** This document is the outline and the
findings-to-date for a self-contained research report (in the style of
[`../noise-snr-scaling/report.md`](../noise-snr-scaling/report.md)) covering the
complete frequency-calibration and frequency-uncertainty source analysis. The
findings below are already established; what remains is to consolidate them into
one narrative `report.md` backed by **one self-contained reproducer script** that
regenerates every number and figure from the `.ftmw` fixtures + catalogs.

The settled engineering decision this analysis produced is shipped and recorded
separately ([`../../planning/stage6-reports.md`](../../planning/stage6-reports.md)
§B): report **precision only** — `σ_f = sqrt(σ_stat² + (σ_ε·f_baseband)² +
σ_floor²)` with `σ_floor` a user-settable accuracy floor, default 0. This report
is the *evidence record* behind that decision, not a new decision.

## Why a report

The σ_f investigation spans six intertwined sub-analyses (calibration,
catalog comparison, the SNR relationship, pedestal influence, and both
within- and between-acquisition reproducibility), each currently scattered
across `scratch/` scripts, the
`stage6-reports.md` §B prose, and the
[`frequency-uncertainty-digitizer-clock`] memory. The conclusions are strong and
non-obvious (the optimistic σ_f is real precision; the dominant accuracy term is
not self-calibratable) and will be cited by reports, user docs, and any future
absolute-calibration work. They deserve one durable, reproducible write-up.

## Sections (the eventual report)

1. **The symptom.** Reported σ_f is ~60–77× too optimistic vs the vinyl-cyanide
   catalog (1512: |Δf| median 16 kHz vs reported σ_f 0.2 kHz).
2. **Ruled-out causes (the important negatives).** Not a covariance bug (the full
   joint noise-weighted `JᵀJ` is inverted; per-line σ already marginalizes
   off-diagonals → σ_f is honest Cramér–Rao precision); not χ²-scaling (√χ²ᵣ
   closes only a fraction; the catalog residual is **SNR-independent**, ~9–15 kHz
   flat from SNR 90 → 5900 while σ_f falls 0.33 → 0.06 kHz); not blend-fitting
   (isolated lines carry the bias *more* than blended); not catalog uncertainty
   alone (see §6).
3. **The calibration mechanism.** An unlocked digitizer sample clock with
   fractional scale error ε ≈ 2 ppm scales the baseband: `Δf = ε·(probe − f_mol)
   = ε·f_baseband` — simultaneously a ~20 kHz offset and a ~2–3 kHz/GHz tilt, one
   parameter. Catalog regression gives ε = 2.28 ppm (1512) / 2.20 ppm (655); the
   clock spurs recover the **same** ε prior-free (`calibrate_timebase`:
   +2.20/+1.98 ppm, σ 0.08–0.12). Applying it drops the catalog residual
   **21 → 6 kHz (1512) / 20 → 4 kHz (655)** with no catalog. Cross-references the
   timebase work in
   [`../../planning/instrument-clock-declaration.md`](../../planning/instrument-clock-declaration.md).
4. **The σ_f budget.** Three terms; σ_stat is already the CRB (Spearman ρ = +0.96
   vs `linewidth/√SNR` on isolated lines, and — §7 — empirical sub-acquisition
   scatter equals ~1.25× the reported CRB across 1.5 decades of shots); σ_ε·f_baseband
   is the timebase parameter uncertainty propagated per line; σ_floor is the user's
   accuracy declaration (default 0), whose *random* component is now directly
   measured sub-kHz (§7) and whose *systematic* component (`δ_down`) is documented
   but not self-calibratable (§9). The precision-only decision and its justification.
5. **The SNR relationship.** σ_stat tracks the CRB; the *accuracy* residual does
   not shrink with SNR (the SNR-independence of §2), which is what distinguishes a
   systematic from a statistical error.
6. **Catalog comparison — the ruler can't be trusted at the kHz level.** Catalog
   uncertainties (~0.35 kHz) are *model* precision (a global-Hamiltonian-fit
   parameter covariance projected per transition), not single-measurement
   accuracy (cavity-FTMW inputs are eyeballed ~10 kHz centres with ~4 kHz
   assigned). Quadrature on the run-to-run results (§8) implies catalog accuracy
   ≈ 3 kHz — so most of the apparent "~5 kHz catalog floor" is the catalog itself.
7. **Within-acquisition reproducibility — the random floor, measured directly.**
   A single acquisition is split into independent sub-measurements that are refit
   against the *frozen* co-average model (targeted pruned-window Stage 5; match by
   frequency, not window id) and their per-line frequency scatter is the
   within-acquisition reproducibility. Two sub-acquisition providers: kept-separate
   scope frames (direct-sampling instrument), and Blackchirp backup differencing —
   the stored cumulative ADC sums difference *exactly* into disjoint shot chunks,
   and caching them gives free re-binning (trade chunk SNR vs sample count). Two
   regimes observed: a **drift-dominated** case (coherent timebase ramp resolved as
   a per-frame common offset, plus a per-line floor ~0.6–1 kHz) and a
   **statistics-limited** case where the chunk scatter tracks the CRB across a
   re-bin sweep (45k→900k shots) — the excess *shrinks* with shots (no
   shot-independent floor) and the empirical scatter is a steady **~1.25× the
   reported CRB**, directly validating σ_stat as the Cramér–Rao precision (§4).
   Co-average random floor lands **sub-kHz** in both. This is the direct
   measurement of σ_floor's *random* component; it is structurally blind to the
   constant per-acquisition offset (§8), which only the between-acquisition test
   sees.
8. **Between-acquisition reproducibility (the catalog-free ruler).** Two
   vinyl-cyanide acquisitions (1512, 655) and two MTBE (360, 363), each
   independently ε-corrected, differenced pairwise (no catalog). Random
   reproducibility shrinks with SNR to ~2.6–2.7 kHz (below the catalog floor),
   agreeing across both pairs; **plus a per-acquisition flat absolute offset
   `δ_down`** (+4.2 kHz VC, −12.2 kHz MTBE). Flat in frequency ⇒ not Doppler
   (∝f_mol) and not residual ε (∝f_baseband); MTBE is collision-free
   post-expansion ⇒ the offset is instrumental. The comparator is N-acquisition
   ready: each added VyCN acquisition contributes new pairwise differences that
   test stable-per-instrument vs random-per-acquisition `δ_down`.
9. **`δ_down` is irreducible from self-calibration.** ε (multiplicative baseband
   scale) self-cals from the spurs; the additive absolute offset needs an
   RF-native spur through the downconversion mixer, and the clock lattice cannot
   pin it: the candidate spur `6400 = probe − 3×upconvLO` carries `δ_down − 3·δ_up`
   (upconv contamination a molecule never sees) **and** sits on the δ-free
   downconv-derived 1280-comb (`6400 = 5×1280`, `40960 = 32×1280`) so its
   deviation is diluted toward zero (2-LO reconstruction under-predicts the clean
   instrumental pair, +0.5 vs +4.2 kHz — the dilution signature).
10. **Pedestal influence (tested).** Hypothesis: an asymmetric
   leakage pedestal biases a line's fitted centre, possibly contributing to
   `δ_down`. Result on the 7–10 isolated VC lines common to 1512/655 (SNR>50
   both): **three independent pedestal suppressors converge** on the
   inter-acquisition offset — production per-window fit (leakage-wing baseline)
   +4.4…+6.3 kHz, explicit polynomial-baseline refit +4.7…+6.9 kHz, Hann-apodized
   centroid +6.0…+6.7 kHz (std ≤ 3.6) — while pedestal-*exposed* estimators
   (no-baseline fit, boxcar centroid) are wildly unstable (per-line 30–75 kHz
   jumps; inter-acq std 10–51 kHz). Conclusions: (a) `δ_down` does **not** collapse
   under pedestal suppression → it is a genuine per-acquisition absolute-reference
   offset, not a pedestal artifact; (b) the pedestal bias *itself* is larger in
   the dense 655 than the noisy 1512 → driven by neighbour-line leakage density,
   **not absolute noise level** (refutes the original noise-level-dependent
   framing, supports the line-strength/temperature framing); (c) the production
   per-window fit already agrees with the apodized centre → production `δ_down`
   is the post-mitigation residual.
11. **Conclusions and the shipped decision.** Separate precision (σ_stat, σ_ε,
    and the directly-measured within-acquisition random floor) from accuracy
    (`δ_down`); the dominant accuracy limit is not self-calibratable; ship
    precision with a user-owned accuracy floor; the pull is a user-calibration
    tool, not a gate.

## The self-contained reproducer script

One script (`reproduce.py` beside the report, like
`../noise-snr-scaling/legacy_adaptive.py`) that, given the fixture `.ftmw` files
and the catalogs, regenerates: the ε catalog-regression + the prior-free spur ε;
the residual-reduction table (§3); the SNR-independence figure (§2/§5); the
within-acquisition floor + re-bin sweep + CRB-validation figures (§7); the
catalog-vs-run-to-run quadrature (§6); the between-acquisition offset table (§8);
the lattice-can't-pin-it spur reconstruction (§9); and the pedestal cross-method
convergence table (§10). It should print the report's numbers and write the
figures to an output dir (gitignored; never the working tree).

**Existing seeds** (gitignored `scratch/`, regenerate the numbers today): the
within-acquisition sub-frame/backup floor machinery (`scratch/floor-uxr/`,
`scratch/floor-bc/` — frozen pruned-window targeted refit + backup differencing +
re-bin sweep); the catalog-free between-acquisition comparator
(`scratch/vycn-crossacq/{build,extract,compare}.py`, N-acquisition ready); the
σ_ε / baseband analyses (`scratch/unc_*.py`, `scratch/sigma_eps.py`,
`scratch/floor_vs_baseband.py`); the lattice-intercept / spur reconstructions
(`scratch/lattice_intercept*.py`, `scratch/spur_2lo.py`, `scratch/spur6400.py`);
and the pedestal test (`scratch/pedestal-test/{recon,pedestal_test,apod_arm}.py`).
The report consolidates these into the single reproducer.

## Pending input — a third vinyl-cyanide acquisition

A further VyCN acquisition strengthens §8 materially: the VC run-to-run is
currently a single pair (1512↔655, n = 2 acquisitions → one pairwise difference).
A third VC acquisition gives three pairwise differences and tests whether `δ_down`
is a **stable** per-instrument offset or a **random** per-acquisition draw — the
decisive distinction for whether any fixed correction is even possible. The
`scratch/vycn-crossacq/compare.py` comparator already accepts an arbitrary list of
acquisitions, so an added fixture drops straight in. An in-progress grid (vinyl
cyanide; single- vs multi-frame, free-running vs Rb-locked digitizer) is the
intended source — and it doubles as the §7 within-acquisition coherence test
(free-running scope clock → frame phase-incoherence → co-average signal loss
that scales with baseband frequency). Write the report with the two-acquisition
caveat and append the grid results when they land.

## Deferred / non-blocking instrument-side tests (from §B)

Cross-instrument comparison (same molecule, second spectrometer — the shared
catalog cancels) and the bench counter test on the downconversion LO
(Valon 5009 → 5120 ×8 = 40960 vs Rb) would *characterize* `δ_down` further but
are instrument-side and not required to write this report.
