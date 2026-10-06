# Specification: Testing

Status of this document: **normative specification**. It states testing
requirements, not the current state of the suite.

## Scope

Test requirements for the three user interfaces (CLI, Pipeline class,
functional API) that share one implementation core.

## Core principle

All interfaces must produce identical scientific results for identical inputs
and parameters (the cross-interface invariant in
[`SCIENCE_STRATEGY.md`](SCIENCE_STRATEGY.md)). This is the central invariant the
suite exists to protect; the remaining scientific invariants that document
states are likewise what the suite verifies.

## Required test categories

1. **Unit tests.** Algorithms, data structures, parameter validation, and
   serialization round-trips, in isolation. Serialization tests must assert
   exact reconstruction for losslessly-stored data (FID) and scientific
   equivalence for derived results.
2. **Per-interface workflow tests.** Each interface independently exercised
   through a real multi-stage workflow on real experiment data.
3. **Cross-interface consistency tests.** The same operation performed via CLI
   (as a subprocess), the Pipeline class, and the functional API must yield
   numerically identical results and consistent pipeline state. This category
   is mandatory and gates any change to a stage implementation.
4. **Parameter-persistence tests.** Parameters saved through one interface
   reproduce identical results when reloaded through any interface.
5. **File-management tests.** Creation vs. opening semantics, safe-reimport
   detection, dependency enforcement, and error/corruption handling.
6. **Performance tests.** Benchmarks for interactive responsiveness and for any
   storage-efficiency figure that is stated as a requirement. A storage or
   timing claim is normative only if a test measures it.

## Requirements

- Real experiment data is used for workflow, consistency, and persistence
  tests; it is the reference for scientific correctness.
- A stage is not "complete" until it has unit tests *and* a cross-interface
  consistency test.
- Stage 6 refusals are cross-interface tests that also assert the refused file
  is byte-identical afterwards and that no window was refit: a bare
  `review edit`, and every write of a file the replay engine cannot curate (a
  fit without `peak_uid`, or curation without the engine's stamps), on the CLI,
  the Pipeline class, the functional API and a `ReviewSession`.
- Every Stage 6 write test asserts the write invariant: after each write, the
  persisted curated state equals, bit for bit, the reference replay of the
  persisted decision log under the persisted review parameters
  (`tests/_replay_support.py`, the `every_write_is_reference` fixture, which a
  Stage 6 test module opts into). A test that counts work (fit-context builds,
  refits) or deliberately persists a state that is not the reference turns the
  check off, with its reason stated. The check is never part of a production
  write path.
- Tests must not write artifacts into the working tree or repository; outputs
  go to a temporary location.
- Tests run in the project dev environment
  (`conda run -n ftmwpipeline-dev ...`). The coverage plugin is configured in
  project settings; commands that disable it must also neutralize the
  configured coverage arguments.
- Markers distinguish slow, integration, unit, and performance tests so subsets
  can be selected in CI.

## Coverage targets

- Unit coverage of core algorithms and data structures: high (>90%).
- Every user-facing workflow covered by an integration test.
- Every major operation covered across all three interfaces.

These are targets the suite must trend toward; they are not satisfied by
asserting them in documentation.
