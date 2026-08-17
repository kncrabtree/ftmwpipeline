# AGENTS.md — tests/

Invariants for working in the test suite. See the repo-root `AGENTS.md` for
commands and the cross-interface rule, and `dev-docs/TESTING_STRATEGY.md` for
the normative test requirements.

## Rebuild fixtures fresh after a code change

Never reuse or patch an existing `.ftmw` fixture after changing code — rebuild
it from the example data. A persisted setting in a `.ftmw` outranks the code
default (the settings-resolution precedence: persisted beats recommended/
default), so an old file silently keeps the knob values it was built with, and
the test then exercises stale behavior while looking correct. Build fixtures
in place from `examples/`; let versioning, not a hand-patched file, carry a
legitimate algorithm change. Pass no explicit line shape when building — let the
Stage 2b vote stamp the recommended shape.

Freshness is about **not inheriting persisted state**, so apply it once, at the
point the state could be inherited — not reflexively at every layer. A `.ftmw`
carries a 750k-point FID and costs ~7 MB, so a copy is not free: a full suite
writes gigabytes of them. If a fixture already hands you a private per-test copy,
copying *that* again buys no isolation and doubles the cost. The Stage 6
fixtures make the distinction explicit — `stage5_small_file` is a fresh writable
copy for a test that edits the file it was handed, and `stage5_small_source` is
the shared read-only build for a test that makes its own copy first. Copy from
the source, not from another copy.

Note that a module-local fixture can shadow a conftest one of the same name
(`test_report_full.py` and `test_report_table.py` both define their own
`stage5_small_file`), so the fixture a test receives is not decidable from the
name alone. Check for an override before repointing anything.

## Write only to tmp_path

Tests must write exclusively to pytest's `tmp_path` / `tmp_path_factory`. Never
write into the working tree or the repository; never leave an artifact behind.
Trees for passing tests are deleted (`tmp_path_retention_policy = "failed"`), so
a passing test cannot leave anything for a later one to find.

## Scope test runs to what you touched

Run the affected stage's targeted suites (plus type-checking on the files you
changed) on the inner loop; reserve a single full-suite run for the end. The
full suite is minutes long, so running it after every edit is the slow path.
