# Specification: Scientific Requirements

Status of this document: **normative specification**. It states the scientific
invariants the pipeline must uphold, independent of how an analysis is exposed
(Python API, CLI) or stored on disk. The interface specifications
([`API_STRATEGY.md`](API_STRATEGY.md),
[`CLI_STRATEGY.md`](CLI_STRATEGY.md),
[`SERIALIZATION_STRATEGY.md`](SERIALIZATION_STRATEGY.md),
[`TESTING_STRATEGY.md`](TESTING_STRATEGY.md)) reference this document for the
*why* behind their requirements and must not restate or weaken these
invariants.

## Scope

What must be true of the analysis *as science* — the integrity of the data, the
spectrum, the statistics, and the result — regardless of interface or storage
layout. These are requirements on outcomes, not on algorithms: how a stage
achieves them is the code's to decide.

## Requirements

1. **Faithful raw data.** The imported signal is the ground truth from which all
   science derives. It is preserved without loss and reconstructs exactly; no
   stage mutates it. Exploring processing parameters never alters or re-imports
   the raw record.

2. **Unbiased active FT.** The **active FT** — the transform of the FID between
   the persisted active-region bounds `[start_us, end_us]`, and the single grid
   every measuring stage consumes — is computed without apodization, time-domain
   windowing, or zero-padding. These operations are prohibited on it because
   they corrupt the science the later stages depend on: apodization and windows
   trade frequency resolution and bias the line shape, and zero-padding
   interpolates bins so that the per-bin noise and χ² statistics are no longer
   valid. The intended way to trade variance for robustness is the per-window
   fit, never a transform-level window. A stage may use a throwaway finer grid
   internally (for example for sub-bin position finding) provided that grid
   never becomes the spectrum any later stage measures or fits.

   **There is no "canonical" spectrum. The term is retired**: it invited a model
   in which some full-length transform, rather than the active FT, is what later
   stages measure, and that model is false. The vocabulary is exactly two
   spectra — the **active FT** above, and the **magnitude-display FT**, which is
   zero-padded by exactly a factor of two to maximize phase information for
   visualization and is never measured on. Where the untrimmed original must be
   named, call it the **raw**, **full-length**, or **imported** FT/FID; Stage 0
   start detection uses it because it necessarily precedes knowledge of the
   active region.

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

8. **Frequency tolerances scale with resolution.** Any tolerance, threshold,
   window width, or increment that expresses a *spectral distance* is defined as
   a multiple of the active-FT bin spacing, never as an absolute frequency. A
   constant frozen in MHz silently encodes one laboratory's acquisition length:
   it is correct only there, and it fails in both directions elsewhere — too
   coarse at long acquisitions, where it merges genuinely resolved lines, and
   too fine at short ones, where it stops doing its job. Acquisition length is
   an experimental choice, not a property of the analysis, so no constant may
   assume one.

   The exceptions are quantities that are genuinely absolute and owe nothing to
   the transform: float-comparison tolerances on user-declared values (a clock
   frequency the user typed), and a user-declared accuracy floor. These must say
   so at their definition; anything else expressed in frequency units is a
   defect.

   **Terminology is normative here.** The **active FT** is the transform of the
   FID between the persisted active-region bounds `[start_us, end_us]` only —
   truncated, never zero-padded, never apodized, never windowed. It is *the
   single grid every measuring stage consumes* (`stage2_impl.py` states this at
   the Stage 2 measurement itself), so its spacing,
   `df = 1 / (end_us - start_us)`, is the spacing every spectral tolerance is
   defined against. There is no second measurement grid to choose from, and a
   tolerance defined against anything else — a record length, a padded length,
   an `n_padded` field — is wrong by that ratio, silently.

   Only two transforms are not the active FT, and neither is ever measured on:
   Stage 0 start detection, which necessarily precedes knowledge of the active
   region, and the display magnitude spectrum, which is zero-padded by exactly a
   factor of two to maximize phase information for visualization. Padding for
   display is legitimate; padding, or any full-record transform, as an input to
   a measurement is not.

   The multiple is chosen on its own merits — a quarter bin, a half bin — for
   the scientific reason that justifies it. It is **not** back-fitted to
   reproduce whatever absolute frequency a previous constant happened to have
   at one laboratory's acquisition length. Reproducing the old number exactly
   would preserve the very accident the requirement exists to remove.
