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

## Write only to tmp_path

Tests must write exclusively to pytest's `tmp_path` / `tmp_path_factory`. Never
write into the working tree or the repository; never leave an artifact behind.

## Scope test runs to what you touched

Run the affected stage's targeted suites (plus type-checking on the files you
changed) on the inner loop; reserve a single full-suite run for the end. The
full suite is minutes long, so running it after every edit is the slow path.
