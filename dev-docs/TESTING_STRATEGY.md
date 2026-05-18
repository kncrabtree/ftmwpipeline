# Specification: Testing

Status of this document: **normative specification**. It states testing
requirements, not the current state of the suite. For the current suite and
counts see [`../STATUS.md`](../STATUS.md).

## Scope

Test requirements for the three user interfaces (CLI, Pipeline class,
functional API) that share one implementation core.

## Core principle

All interfaces must produce identical scientific results for identical inputs
and parameters. This is the central invariant the suite exists to protect.

## Required test categories

1. **Unit tests.** Algorithms, data structures, parameter validation, and
   serialization round-trips, in isolation. Serialization tests must assert
   exact reconstruction for losslessly-stored data (FID) and scientific
   equivalence for derived results.
2. **Per-interface workflow tests.** Each interface independently exercised
   through a real Stage 0 → 1 → 2 workflow on real experiment data.
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
