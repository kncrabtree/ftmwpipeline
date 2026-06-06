# Specification: Serialization and the `.ftmw` File

Status of this document: **normative specification**. It defines storage
invariants and requirements, not the literal on-disk layout. Exact HDF5
group/attribute names are an implementation detail; the code is their source of
truth and the current layout is recorded in [`../STATUS.md`](../STATUS.md).

## Scope

How an experiment's analysis is persisted. One experiment is one self-contained
`.ftmw` file (HDF5 container).

## Principles

1. **Lightweight files.** Persist only what is needed to reconstruct results.
   Large derived arrays that are cheap to recompute are not stored.
2. **Bit-perfect raw data.** The imported FID is stored losslessly and
   reconstructs exactly.
3. **Parameter flexibility.** A single stored FID must serve unlimited FT
   parameter combinations; exploring parameters never requires re-importing.
4. **Self-contained and portable.** A `.ftmw` file carries everything needed to
   continue or reproduce the analysis on another machine, with no external
   dependencies. Sharing the file is sufficient to reproduce the result: a
   recipient running a compatible package version obtains identical output from
   the file alone, with no instrument preset or other side artifact required.
   It follows that **no external artifact may silently override a setting the
   file persists** — see *Settings resolution and reproducibility* below.
5. **Provenance.** Source identity and import parameters are recorded (see
   [`API_STRATEGY.md`](API_STRATEGY.md)).

## Storage model by stage

- **Stage 0 — FID.** The raw time-series and its acquisition metadata
  (sample spacing, probe frequency, sideband, shot count, point count,
  duration) are stored losslessly, together with any source-format
  *recommended* processing parameters (advisory defaults the user may
  override).
- **Stage 1 — ComplexFT.** Not persisted. The frequency-domain result is
  recomputed on demand from the stored FID plus the active processing
  parameters. Only the processing parameters are persisted (for reproducibility
  and parameter persistence). Consequently a completed Stage 1 is proven by the
  presence of its persisted parameters, not by a stored result array.

  The canonical FT is unconditionally unapodized, un-windowed, and
  native-length — there are no `expf_us` / `window_function` / `zpf` settings.
  The user-chosen Stage 1 FT processing settings are data selection (`start_us`,
  `end_us`, **and the frequency `trim` range**) plus display/scaling
  (`units_power`, `rdc`); they are persisted in
  `processing_parameters/ft_processing` as the experiment's *canonical*
  settings.  Legacy `.ftmw` files carrying the retired apodization keys open
  with a warning and are recomputed unapodized.  All later stages operate on the
  spectrum they define.  Setting resolution order is **explicit override >
  persisted canonical > import-time recommended**.  Changing canonical settings
  via an explicit override invalidates downstream stage results (Stages 2–5
  must be re-run).
- **Stage 2 — NoiseResult.** Persisted, using a compact representation
  sufficient for exact reconstruction (store indices/coefficients rather than
  full dense masks where that is lossless).
- **Stages 3–5.** Expensive derived results (peaks, window definitions, fitted
  parameters) are persisted; anything cheaply reconstructible from them and the
  on-demand ComplexFT is not.

## Settings resolution and reproducibility

A stage's effective settings are resolved from layers in this **normative
precedence** (highest first):

```
explicit override  >  persisted (.ftmw)  >  preset (.yml)  >  recommended  >  hard default
```

- **explicit override** — a value passed by the caller for this invocation. A
  deliberate, per-run act; it recomputes and persists intent (see
  [`planning/processing-settings-persistence.md`](planning/processing-settings-persistence.md)).
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
it has. This matches the Stage 1 canonical-settings order already specified above
(`explicit > persisted > recommended`); the preset layer slots directly below
persisted for every stage.

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
