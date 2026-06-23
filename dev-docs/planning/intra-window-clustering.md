# Plan: Intra-window parameter clustering for Stage 5

Status: **retired (will-not-build).** Registered to capture a design idea that
replaced the *proposed* Stage 5 `split` structural renegotiation primitive
([`stage5-fitting.md`](stage5-fitting.md) § "Renegotiation handshake with
Stage 4"). That primitive was never built — Stage 5 ships merge-only, and
`FitWindow.split_proposal` remains a carried-but-unconsumed field.

Retired because the gain it promised does not exist by construction. Stage 5
already fits every window **jointly** — all its peaks, one shared `τ`, one NLS —
so the maximum-likelihood parameter estimates and their full covariance are
already what the joint fit returns. A block-diagonal covariance only means the
joint fit *decomposed* into independent sub-fits with no information loss; acting
on it (the "Reduce" step below) is purely cosmetic relabeling of the
per-cluster parameter count and buys no precision. The speculative
non-contiguous joint-fit direction has no driver either: the merge handshake
plus the leakage-wing baseline covered the practical need on 2638, and no fixture
has surfaced a window where intra-window decomposition would be a measurable win.
The design is preserved below as a record; reopen only if such a fixture appears.

## Motivation

Stage 4 builds **tight** windows: each is sized to its in-band peaks' cores
plus a `min_window_half_width_mhz` (~2 MHz on 2638), with strong out-of-band
lines carried as fixed contributors rather than by widening. As a result a
typical window has only a handful of free peaks, and a "too wide" window with
many independent clusters is rare in practice.

When it *does* happen — a hard window whose interior holds two or more
groups of peaks that are decoupled in their parameters — the conventional
remedy is to split the window structurally (the `SplitRequest` primitive the
plan originally listed). That works, but it is the wrong abstraction: the
underlying fact is that **the joint-fit covariance matrix already tells us
which peaks are coupled.** A block-diagonal covariance means the joint fit
decomposes exactly into a sum of independent sub-fits with no information
loss. Splitting the window forces the decomposition through plan-revision
machinery; clustering does it for free at the fitter.

## Approach

1. **Fit jointly.** Run the conservative add-one-peak loop as today — one
   window, all its peaks, one shared `τ`.
2. **Read the covariance.** `WindowFitResult.covariance` is the inverse
   `(JᵀJ)⁻¹` at the solution. Group its columns by peak (3 parameters per
   peak: amplitude, offset, phase).
3. **Score off-diagonal blocks.** For each pair of peaks `(i, j)`, compute a
   normalised coupling
   `c_ij = max |cov_ij| / sqrt(diag_i · diag_j)` over the 9 entries of the
   3×3 off-diagonal block. Peaks with `c_ij < ε` for some small ε
   (~0.1; calibrate) are *decoupled*.
4. **Partition.** Find the connected components of the coupling graph
   (peaks as nodes, edges where `c_ij ≥ ε`). Each component is a *cluster*.
5. **Reduce.** When the partition is non-trivial (>1 cluster), the joint
   fit's parameters within each cluster are already its independent
   best-fit values — no re-fit needed. The reduction is cosmetic: report
   each cluster as its own fit-result block in the window outcome, and let
   the AIC / F-test accounting use the per-cluster parameter count rather
   than the global one.

## The non-contiguous fit question

A related, more speculative direction: if two clusters of peaks live in two
*different* windows and their joint covariance shows strong coupling
(e.g. via a shared `τ`), is there value in fitting them on a
**non-contiguous offset grid** — concatenated active-FT slices with a gap
between, model evaluated at both regions, one set of shared parameters?

The infrastructure half-exists already: `local_thaw_cofit` already
concatenates two windows under a shared baseband coordinate to handle the
doublet thaw case. Generalising it would not be hard.

The empirical question is whether the gap data carries enough information
about the shared parameters (mainly `τ`) to justify the orchestration cost.
For a damped cosine the gap *does* see both clusters' leakage skirts, so
the answer is non-zero — but for typical FTMW separations the prior is
that the contribution is small.

Both questions (intra-window clustering, non-contiguous joint fits) are
expected to be informed by the same experiment: take a 2638 window with
multiple clusters of free peaks, fit it three ways (separate windows,
single joint window, joint window + post-fit clustering), and compare
parameter errors and AIC.

## Open questions

- **ε threshold.** Calibrated against 2638's hard windows.
- **Shared τ inside a decomposition.** Does each cluster get its own τ
  after decomposition, or do they all keep the joint fit's τ? Probably the
  joint τ — that is the maximum-likelihood estimate constrained by the
  full window's data — but worth confirming on synthetic.
- **Interaction with the conservative add-one-peak loop.** Clustering
  applies *after* the loop converges; the loop itself should not be
  cluster-aware. (Adding a candidate to a cluster it does not statistically
  belong to fails the F-test naturally.)
- **Non-contiguous fits.** Is the precision gain large enough to justify
  the implementation? Settled empirically.

## Task breakdown

Deferred. Will be filled in when this plan is activated.
