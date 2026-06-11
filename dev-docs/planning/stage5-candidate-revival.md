# Stage 5 — Candidate revival and user-directed window re-fit

**Status: proposed** (planning only; no implementation). Parent plan:
[`stage5-fitting.md`](stage5-fitting.md). Companion context: the accept-gate
conservatism record in [`stage5-context-invariant-gate.md`](stage5-context-invariant-gate.md)
and the blend-escape arc in the Stage 5 session handoffs.

## Problem

The Stage 5 accept machinery is deliberately conservative: the gate bars sit
near the false-positive edge, and the remaining cross-fixture misses are
predominantly *weak lines in near-blends* that were **considered and
rejected as marginal**, not lines the pipeline never saw. Verified on the
v8 fits (every missed line appears in the persisted audit record):

| window | where the miss lives in the audit record |
|---|---|
| 1231 w50 | rescue candidate at −1.737 MHz (SNR 11) nominated in *every* rescue round; all rounds rejected — "joint refit + merge collapsed 4 pair(s) (consolidated χ² did not improve)" |
| 1231 w425 | two `tentative` add-loop candidates "held pending a jointly-significant batch"; the batch never formed |
| 1231 w426 | a blend-split trial that failed the gate + two `peak-separation constraint` rejects; an SNR-5 rescue candidate re-nominated every round |
| 363 w76 | two final `peak-separation constraint` rejects after 10 accepts |

Pushing the automatic bars down to capture these would buy the misses back
at the cost of dust everywhere else (measured repeatedly in the gate arc;
see "Do NOT chase by lowering bars" in the session handoff). The principled
resolution for genuinely marginal evidence is **human arbitration**: the
spectroscopist knows the catalog, the experiment, and the local context.

The same need appears in the opposite direction: very-high-SNR fixtures
(e.g. 1019) may *overfit* — imperfect lineshape absorbed by extra peaks
that the user does not believe (no splitting expected) — and there is
currently no way to say "re-fit this window without that peak".

## Proposal

Two coupled features, shipped together because the second is what makes the
first useful:

### 1. Candidate ledger (expose what was rejected)

At fit time, derive from the existing audit machinery a per-window list of
**revivable candidates**: peaks that were nominated but not installed.
Sources already persisted today (`FittingResult.audit_trail`,
`rescue_events`): add-loop `reject`/`tentative` steps with their offsets and
evidence, rescue-round `candidates` lists with per-candidate SNR, blend-split
trial failures, separation rejects.

What the ledger adds over the raw audit trail:

- **normalization** — one entry per distinct candidate (molecular MHz),
  deduplicated across rescue rounds and decision sites (w50's candidate
  appears five times in the raw record), carrying: frequency, best evidence
  seen (Δχ², f-statistic/p-value, or residual SNR — whichever site produced
  it), rejection reason(s), decision site(s);
- **a surfacing bar** — only candidates above a quality floor enter the
  ledger (strawman: residual/rescue SNR ≥ 3, or gate evidence within an
  order of magnitude of the bar). The dust the gates correctly kill must
  not be thrown at the user; the bar is a *display* threshold, tunable,
  and orthogonal to the accept gates;
- **persistence** — serialized with the fit (sibling of
  `diagnostics["gated_spurs"]`), so `fit show` can render it without
  re-fitting.

UX surfaces:

- `fit show --window N` detail view marks ledger candidates (hollow/grey
  ticks at candidate frequencies, distinct from fitted-peak markers), with
  a table in the text output: frequency, evidence, reason.
- A machine-readable accessor on all three interfaces (functional API
  returns the ledger; CLI `fit show --candidates` prints it).

### 2. User-directed window re-fit (revive / remove)

A `refit` verb scoped to one window, accepting explicit edits:

```
ftmwpipeline fit refit exp.ftmw --window 50 --add 28051.88 --remove 28054.73
```

- Pipeline: `Pipeline.refit_window(window_id, add=[...], remove=[...])`;
  functional API mirror. All three interfaces delegate to one `_internal`
  impl per the dual-interface rule.
- `--add F` snaps to the nearest ledger candidate within a tolerance
  (default ~1 bin) when one exists — reviving its recorded seed — and
  otherwise seeds a fresh peak at F. `--remove F` snaps to the nearest
  fitted peak.
- The re-fit re-runs the production single-window machinery (same
  contributors, baseline, spur masks, τ anchoring) with the edits applied:
  removed peaks blacklisted from every nomination site; added peaks seeded
  unconditionally.
- **Gate semantics for user adds (key design decision):** a user add
  bypasses the *accept* gate (the user is the gate) but still faces the
  NLS itself — if the optimizer drives the peak to zero amplitude or it
  collapses onto a neighbor, the result reports that honestly rather than
  silently keeping a phantom. Knockout/χ² diagnostics are computed and
  reported for the user peak like any other, flagged `user`. Subsequent
  automatic passes (merge cleanup, AICc cleanup, rescue) must NOT silently
  collapse or prune a user-added peak, and must not re-add a user-removed
  one; both would make the verb feel broken.

### Provenance (non-negotiable)

The persisted fit must record the intervention: per-window
`user_edits` diagnostics (adds, removes, timestamp-free), audit-trail
steps `decision="user-add"` / `"user-remove"`, and a per-peak `origin`
flag (`user` vs `auto`) surviving serialization round-trip. Validation
tooling (pass metrics, catalog matching) must be able to separate curated
from automatic results — a curated fixture must never silently masquerade
as an automatic benchmark.

## Scope notes and open questions

- **Neighbor consistency.** A re-fit window changes its own peaks only. If
  the window is a *contributor primary* for neighbors' frozen skirts, the
  neighbors' fits become stale. Phase 1: document the staleness and report
  which neighbors reference the window; do not cascade automatically.
  Cascading re-fit is a possible phase 2.
- **Replan interaction.** The mid-fit structural replan can change window
  geometry between fits. The ledger and `refit` address the *persisted*
  windows of the current fit; a later full re-fit owns its own decisions
  (user edits are not replayed automatically — replay is out of scope).
- **Ledger bar calibration.** Strawman bar above; calibrate on the v8
  fixtures so that the four verified miss windows surface their candidates
  while a quiet window surfaces none. Measure ledger sizes across all
  seven fixtures before freezing the default.
- **Serialization.** New per-window dataset(s); old files without a ledger
  open fine (empty ledger); a refit updates the stage5 payload in place
  with the provenance above.
- **Tests.** Unit: ledger normalization/dedupe, bar, snap semantics,
  user-peak immunity in merge/cleanup/rescue, provenance round-trip.
  Integration: cross-interface consistency for `refit` + ledger accessors;
  an end-to-end revive on a synthetic w50-topology window.

## Why this over more automatic levers

The blend arc closed every miss class that carried decisive evidence
(blend-split trial, relative-evidence lane, residual re-seed). What remains
is evidence the data genuinely cannot adjudicate at the configured
false-positive budget. A curated lane converts that boundary from a silent
loss into a visible, reversible decision — and the same machinery gives the
overfit direction (user removes) for free.
