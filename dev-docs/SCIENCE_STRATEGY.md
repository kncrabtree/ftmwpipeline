# Specification: Scientific Requirements

Status of this document: **normative specification**. It states the scientific
invariants the pipeline must uphold, independent of how an analysis is exposed
(Python API, CLI) or stored on disk. The interface specifications
([`API_STRATEGY.md`](API_STRATEGY.md),
[`CLI_STRATEGY.md`](CLI_STRATEGY.md),
[`SERIALIZATION_STRATEGY.md`](SERIALIZATION_STRATEGY.md),
[`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)) reference this document for the
*why* behind their requirements and must not restate or weaken these
invariants. Where the code diverges, see the divergence log in
[`ROADMAP.md`](ROADMAP.md).

## Scope

What must be true of the analysis *as science* — the integrity of the data, the
spectrum, the statistics, and the result — regardless of interface or storage
layout. These are requirements on outcomes, not on algorithms: how a stage
achieves them is the code's to decide and [`../STATUS.md`](../STATUS.md)'s to
record.

## Requirements

1. **Faithful raw data.** The imported signal is the ground truth from which all
   science derives. It is preserved without loss and reconstructs exactly; no
   stage mutates it. Exploring processing parameters never alters or re-imports
   the raw record.

2. **Unbiased canonical spectrum.** The canonical frequency-domain spectrum that
   every later stage measures is computed without apodization, time-domain
   windowing, or zero-padding. These operations are prohibited on it because
   they corrupt the science the later stages depend on: apodization and windows
   trade frequency resolution and bias the line shape, and zero-padding
   interpolates bins so that the per-bin noise and χ² statistics are no longer
   valid. The intended way to trade variance for robustness is the per-window
   fit, never a transform-level window. A stage may use a throwaway finer grid
   internally (for example for sub-bin position finding) provided that grid
   never becomes the spectrum any later stage measures or fits.

3. **Statistical integrity.** The spectrum later stages score must carry valid,
   independent per-bin noise statistics. No processing step may correlate bins,
   fabricate resolution, or otherwise invalidate the noise model on which
   detection thresholds, reduced χ², and fitted uncertainties rest. Noise is
   characterized on the same grid the later stages plan, score, and fit on — not
   on a different spectrum.

4. **Honest uncertainties.** Fitted quantities carry uncertainties, and the
   analysis retains enough information to propagate them *with their
   correlations*, not as independent error bars. A reported precision must
   reflect what the data constrains; an uncertainty that cannot be determined
   from the data is surfaced as such, never silently assumed away or fabricated.

5. **Reproducible, self-contained result.** The analysis record is sufficient on
   its own to reproduce the scientific result: given the record and a compatible
   package version, the same inputs yield the same output on any machine, with
   no external artifact required. It follows that **no external artifact may
   silently change the science of a completed record** — anything that would
   alter a result requires explicit, recorded intent. The record's provenance
   must be sufficient to detect whether a re-import refers to the same source and
   to reproduce the analysis. (How settings layers and storage realize this is
   specified in [`SERIALIZATION_STRATEGY.md`](SERIALIZATION_STRATEGY.md).)

6. **Identical results across interfaces.** The same inputs and parameters
   produce the same scientific result whichever interface drives the analysis;
   the interface is a means of invocation and carries no analysis logic of its
   own. Divergence between interfaces is a defect, not a variant.

7. **Real-data correctness.** Scientific correctness is judged against real
   experiment data, which is the reference for any claim of correctness.
   Synthetic cases support but do not replace it, and a numerical claim is
   normative only if a test measures it on representative data.
