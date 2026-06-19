# Stage 6 — interactive CLI review shell

Status: **dropped (superseded); will-not-build.** This terminal REPL was a
stopgap for the planned C++/Qt graphical shell, but the workflow it targeted is
now covered without it: the **CSV review language** (`review apply` / `review
log` / `review undo`, [`stage6-report-curation.md`](stage6-report-curation.md))
provides batch, replayable, hand-editable curation, and the **in-report
curation cart** turns the Level-3 HTML report into an authoring surface that
emits the same curation file. A terminal REPL would add a third interaction
surface with no remaining unique value, so it is not being built. The design
sketch below is retained for its rationale — file-based rendering (no live
matplotlib / `ipywidgets`), the display-letter↔peak map, and the
byte-identical-to-verb test lever — should a REPL ever be reconsidered.

---

Original proposal (design sketch, not implemented). A terminal-interactive
front-end over the existing Stage 6 `review` verbs. It adds **no** fitting logic
and **no** new persisted state — it is a navigation and dispatch loop that calls
the same `_internal` impl functions the non-interactive verbs already call.

## Purpose

The non-interactive verbs (`review show/edit/merge/split/accept`,
[`stage6-finalization.md`](stage6-finalization.md)) require the user to retype
the file path, the `--window N`, and the target frequencies on every action,
and to re-run `review show --window N --output …` and reopen the image after
each edit. For a real review pass over dozens of flagged windows that is
tedious. An interactive shell keeps the *current window* as session state,
renders it on demand, and offers a single-keystroke menu whose actions map
one-to-one onto the existing verbs — so a full attention sweep becomes
"look, act, next" without leaving the loop.

## Scope and non-goals

- **In scope:** a CLI REPL that selects a window, renders it to an image
  file, presents a text menu (add / remove-by-letter / merge / split /
  accept), applies the chosen action through the existing impl, and advances
  to the next window. Three entry selections: the attention queue, the full
  window list (browse), or "start at the window nearest a frequency."
- **Out of scope — deliberately:**
  - **No `ipywidgets`** controls, and no Jupyter-bound UI of any kind.
  - **No live/interactive matplotlib** (no `plt.show()` event loop, no
    blocking canvas, no animation). Matplotlib interactive mode is too slow
    for this workload; rendering is **to a file**, decoupled from the menu.
  - **No new fit semantics, gates, or persisted fields.** Every mutating
    action is one of the already-specified Stage 6 edits and persists exactly
    as that verb does (decision log, `origin=user`, provenance flip,
    candidate-flag clearing). Quitting mid-session always leaves a consistent
    file.

## Architecture fit

This is a **CLI-only** convenience and does not exist on the `Pipeline` class
or functional API — an interactive REPL has no meaningful stateless/file-bound
analogue, and the dual-interface rule governs *stage behaviour*, not terminal
ergonomics. The rule is honoured the important way: the shell introduces no
behaviour of its own. Each menu action is a thin call to the same function the
corresponding `review` subcommand calls —
`refit_window_impl` (`add`/`remove`), `merge_peaks_impl`, `split_peak_impl`,
`review_accept_impl` — so an interactive `r B` is byte-identical to
`review edit --window N --remove <freq-of-B>`. Read-side state comes from
`get_review_status_impl` (attention ranking, provenance, decision log) and
`get_candidate_ledger_impl` (revivable lines); rendering reuses
`render_fit_detail_impl`. The shell is pure orchestration.

## Proposed surface

A new `review` verb. Working name **`review interactive`** (synonym
`review browse`); final name is an open question. Selection flags mirror the
existing `review show` conventions:

```
ftmwpipeline review interactive FILE [--attention] [--all] [--near FREQ_MHZ]
                                     [--bar BAR] [--open] [--render-dir DIR]
```

- `--attention` (default): queue = windows flagged for attention, worst
  severity first (the `review show --attention` ranking). Resolving a
  window's flag advances the queue.
- `--all`: queue = every window, ascending `window_id` (or ascending centre
  frequency — open question), for a fast browse regardless of flags.
- `--near FREQ_MHZ`: queue = all windows, but the cursor *starts* at the
  window whose `freq_range` contains or is closest to `FREQ_MHZ`. Compatible
  with `--all`/`--attention` as the ordering for subsequent next/prev.
- `--open`: after each render, shell out once to the OS image opener
  (`xdg-open` / `open`) so an auto-refreshing viewer updates in place. Off by
  default (prints the written path instead).
- `--render-dir`: where the per-window PNG is written (default an untracked
  scratch path; **never** the working tree — same artifact discipline as the
  rest of the CLI).

### The loop

On entering a window, the shell prints a compact text block — window id,
freq range, χ²ᵣ, provenance label, attention reasons, and the **lettered
fitted-peak table** (A, B, C … with frequency / amplitude / SNR / origin) —
and writes/refreshes the consolidated detail figure
(`render_fit_detail_impl`) to the render path. It does **not** re-render on
every keystroke; rendering happens on window entry and on the explicit `v`
command, so the slow step is paid only when the view actually changed.

Menu (single-letter commands; full words accepted):

| Key | Action | Delegates to |
|---|---|---|
| `a F [F …]` | add peak(s) at molecular MHz F | `refit_window_impl(add=…)` |
| `r L [L …]` | remove fitted peak(s) **by display letter** | `refit_window_impl(remove=…)` |
| `m L L [L …]` | merge ≥2 lettered peaks into one | `merge_peaks_impl` |
| `s L [K]` | split lettered peak into K (default 2) | `split_peak_impl` |
| `c F` | revive candidate at F (accept-by-candidate) | `review_accept_impl(candidate_freq=F)` |
| `k` | accept window as-is (reviewed, no change) | `review_accept_impl` |
| `v` | (re)render + open the current window | `render_fit_detail_impl` |
| `n` / `p` | next / previous window in the queue | navigation |
| `w N` | jump to window id N | navigation |
| `g F` | jump to the window nearest frequency F | navigation |
| `l` | list the queue with provenance + flags | `get_review_status_impl` |
| `d` | show this window's decision log | `get_review_status_impl` |
| `?` | help | — |
| `q` | quit (file already consistent) | — |

### Letter resolution (the one shared invariant)

`r B` must remove exactly the peak the figure and the table label **B**. The
display letters are assigned by `_peak_labels(len(fitted_peaks))` over
`window_fit.fitted_peaks` order (`visualization/fit_detail.py`). The shell
must derive the letter→peak map from that **same** call so the figure, the
printed table, and the menu agree, and must **re-derive it after every edit**
(an add/remove/merge/split changes the peak set, hence the lettering). The
letter is resolved to the peak's molecular frequency before delegating, since
the underlying verbs address peaks by frequency. This shared-ordering
requirement is the only new coupling the feature introduces; factoring the
A.. labelling into one helper consumed by both the renderer and the shell
keeps it single-source.

## Relationship to the planned Qt frontend

The long-term plan is a **C++/Qt application** as the rich graphical shell for
`.ftmw` files (live, fast plotting; direct manipulation). That app and this
CLI shell are **two clients of the same Stage 6 contract**: the persisted
decision log, per-peak `origin`, per-window provenance, and the candidate
ledger. Nothing in this feature should encode interaction state the Qt app
cannot reconstruct from the file, and the menu actions deliberately stay a
1:1 image of the documented verbs so the two clients can never diverge in
*what an edit means*. This CLI shell is the low-effort interim review tool;
the Qt app is where interactive graphics live. Keeping rendering file-based
(not a matplotlib event loop) is consistent with that division — the CLI
never tries to be the graphics shell.

## Test plan

A REPL is awkward to test directly, so the testable logic is factored into
pure functions and exercised without a live terminal:

- **Command parser:** `parse_menu_input(line) -> Action | error`. Table-driven
  tests over every command form, bad letters, missing args, out-of-range K.
- **Letter resolver:** given a `FittingResult`, `letters_to_frequencies(["B"])`
  returns the right molecular frequencies and rejects letters not present;
  pinned against `_peak_labels` so the figure/table/menu mapping cannot drift.
- **Queue/navigation:** attention ordering, `--all` ordering, `--near`
  start-index selection, next/prev/jump bounds, and "advance on flag
  resolved."
- **Action equivalence (the consistency guarantee):** drive a scripted input
  stream (a fake stdin / injected line iterator) through the loop on a fixture
  and assert the resulting file is **byte-identical** to running the
  equivalent non-interactive `review edit/merge/split/accept` sequence — the
  Stage 6 analogue of the cross-interface consistency tests. This is what
  certifies the shell adds no behaviour.
- No image-content assertions: rendering only needs to be exercised for
  smoke (a PNG is written), since `render_fit_detail_impl` is already covered.

## Open questions

1. Verb name: `interactive` vs `browse` vs `shell`.
2. `--all` ordering: ascending `window_id` or ascending centre frequency?
3. Auto-open policy: default to printing the path (current proposal) or
   default `--open` with a one-time viewer launch? Portable opener detection
   (`xdg-open`/`open`/`$BROWSER`).
4. Optional terminal-only preview (an ASCII magnitude sparkline of the window)
   for triage without leaving the terminal — nice-to-have, or leave all
   visuals to the file/Qt path?
5. In-session `undo`: edits persist immediately and the decision log records
   them; is a convenience `u` (re-fit to the prior peak set) worth it, or is
   re-editing sufficient? Lean defer.
