# Planning documents

Per-feature implementation plans and tracking. One document per significant
piece of work (typically one per pipeline stage; subdirectories allowed for
large efforts).

## Lifecycle

1. **Plan.** Before implementation starts, create a document here describing
   the approach, the algorithms/data structures, the interface surface across
   CLI/Pipeline/functional API, the serialization, and the test plan. Register
   it in [`../ROADMAP.md`](../ROADMAP.md).
2. **Track.** While work is in progress, the document carries the task
   breakdown and open questions. Keep status detail here, not in the roadmap.
3. **Summarize.** On completion, rewrite the document as an implementation
   overview: what was built, how it works, how to use it. Remove the
   task-tracking scaffolding.
4. **Seed docs.** At the end of development these overviews are the raw
   material for end-user documentation.

## Conventions

- A planning document is normative only for the work it tracks; the
  `*_STRATEGY.md` specs remain the authority on requirements.
- No emojis. No dated progress logs — git history and `../STATUS.md` cover
  state.
- Naming: `stage<N>-<topic>.md` (e.g. `stage3-peak-detection.md`), or a
  subdirectory for multi-document efforts.
