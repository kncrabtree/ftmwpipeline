# Specification: Serialization and the `.ftmw` File

Status of this document: **normative specification**. It defines storage
invariants and requirements, not the literal on-disk layout. Exact HDF5
group/attribute names are an implementation detail; the code is their source of
truth.

## Scope

How an experiment's analysis is persisted. One experiment is one self-contained
`.ftmw` file (HDF5 container). This document specifies the *storage* invariants
that realize the scientific requirements in
[`SCIENCE_STRATEGY.md`](SCIENCE_STRATEGY.md) — lossless raw data, honest
uncertainties, and a reproducible self-contained record — and references that
document rather than restating the science.

## Principles

1. **Lightweight files.** Persist only what is needed to reconstruct results.
   Large derived arrays that are cheap to recompute are not stored; what is
   stored uses a compact lossless representation (for example indices or
   coefficients rather than dense masks).
2. **Bit-perfect raw data.** The imported FID is stored losslessly and
   reconstructs exactly.
3. **Parameter flexibility.** A single stored FID must serve unlimited FT
   parameter combinations; exploring parameters never requires re-importing.
4. **Self-contained and portable.** A `.ftmw` file carries everything needed to
   continue or reproduce the analysis on another machine, with no external
   dependencies. This is the storage realization of the reproducible,
   self-contained result required by
   [`SCIENCE_STRATEGY.md`](SCIENCE_STRATEGY.md): it follows that **no external
   artifact may silently override a setting the file persists** — see *Settings
   resolution and reproducibility* below.
5. **Provenance.** Source identity and import parameters are recorded (see
   [`API_STRATEGY.md`](API_STRATEGY.md)).
6. **Sufficient for honest uncertainties.** Where a result carries
   uncertainties, the file persists enough to reconstruct them *with their
   correlations*, not as independent error bars — the honest-uncertainties
   requirement of [`SCIENCE_STRATEGY.md`](SCIENCE_STRATEGY.md).

## Settings resolution and reproducibility

A stage's effective settings are resolved from layers in this **normative
precedence** (highest first):

```
explicit override  >  persisted (.ftmw)  >  preset (.yml)  >  recommended  >  hard default
```

- **explicit override** — a value passed by the caller for this invocation. A
  deliberate, per-run act: it recomputes the stage, persists the new intent, and
  invalidates the results of any stage downstream of it, which must be re-run.
- **persisted (.ftmw)** — the value stamped into the file when the stage was
  last run. **Authoritative over any external artifact.**
- **preset (.yml)** — an instrument preset the runner opted into for this
  invocation. It supplies values the file has *not* persisted; it must **never**
  override a value the file already persists.
- **recommended** — an upstream advisory value (import-time recommended
  parameters; a Stage 2b shape recommendation). Reserved/`None` where unused.
- **hard default** — the package constant.

The invariant — **persisted outranks preset** — is what makes Principle 4 hold:
a `.yml` a recipient happens to have (possibly tuned for a different instrument)
cannot change the output of a shared, fully-processed `.ftmw`. The preset layer
exists to *seed* fields the file has not yet fixed, not to second-guess fields
it has. It is consistent with the canonical (data-selection) settings, whose
resolution omits the preset layer (`explicit > persisted > recommended`); the
preset layer slots directly below persisted for every stage that has one.

Because the layers are distinct, an explicit override and a preset may be
supplied in the same invocation — they are not mutually exclusive. The explicit
values win per field, the preset seeds the fields the explicit layer and the
file leave unset, and persisted still outranks the preset. This is what lets a
runner adopt a preset recipe and override a field or two in one call without
losing the shared-file reproducibility guarantee.

## Stage tracking

The file records which stages are complete and the dependency graph between
stages. The recorded completion key for each stage must be the single canonical
key used by the dependency tracker and by all interfaces. Validation of a
completed stage checks for that stage's actual persisted artifact, which is not
necessarily a group named after the stage (Stage 1 has no result group by
design).

## Reconstruction guarantees

- FID reconstructs bit-perfectly.
- ComplexFT recomputed from a stored FID with given parameters is identical
  across all interfaces (this is asserted by cross-interface tests).
- Any persisted stage result reconstructs to scientific equivalence with the
  value originally computed.

## Multi-format input

Import is via an extensible loader registry with format auto-detection. Adding
a format must not require changes to the serialized file structure: all formats
normalize to the same stored FID representation. Implementing a custom format
must be minimal — declare recognized inputs and provide a single function
returning an FID (plus optional recommended parameters); loaders must be able
to signal problems by raising, not by implementing validation hooks.

## Non-goals / constraints

- Storing ComplexFT, or any large array recomputable in interactive time, is
  prohibited — it defeats the lightweight-file invariant.
- Storage-size figures are requirements only when accompanied by a benchmark
  that measures them; otherwise they are non-normative.
