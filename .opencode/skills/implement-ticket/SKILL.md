---
name: Implement Ticket
description: Implement a ticket from the llm-from-scratch backlog (docs/tickets.md, mirrored as GitHub issues in calculon102/llm-from-scratch): check dependencies, implement to the "Done when" criterion, run QA, commit, and close the GitHub issue. Use this whenever the user asks to implement, work on, or complete a ticket (T-xxx), a GitHub issue in this repo, or "the next ticket" / "the first two tickets" / "next ready ticket" — even if they don't use the word "ticket".
---

# Implement a ticket

Work in this repo is driven by tickets in `docs/tickets.md`, mirrored as GitHub issues in `calculon102/llm-from-scratch` (issues #2–#43, in order; issue #1 is the roadmap board). Each ticket's **"Done when"** line is its acceptance criterion. A ticket is not done until that criterion is verified with evidence and the GitHub issue is closed.

## 1. Identify the ticket

- **T-id given** (e.g. T-101): read the ticket in `docs/tickets.md`.
- **Issue number given**: issue titles start with the T-id, so confirm with `gh issue view <N> --repo calculon102/llm-from-scratch --json number,title,state` and read that ticket.
- **"Next / first / ready ticket(s)"**: scan `docs/tickets.md` in order and pick tickets whose "Depends on" items are all closed on GitHub (see step 2). If several are ready, pick in ticket order; if anything is ambiguous (open dependency, unclear scope), confirm the pick with the user before starting.

## 2. Check dependencies

For every "Depends on" item, verify the matching GitHub issue is closed:

```sh
gh issue view <N> --repo calculon102/llm-from-scratch --json state,title
```

If a dependency is still open, stop and tell the user — do not implement out of order.

## 3. Implement

- Keep the "Done when" line as the target while coding.
- Follow the architecture rules in `AGENTS.md`: reuse the shared Trainer loop for all training phases, YAML config via `--config`, runs through the artifact manager, scale profile ladder `tiny_debug → tiny_train → small → medium → max_7900xtx`, offline-only (no external API calls), German-first corpus, ROCm with CPU fallback.
- Add tests where the ticket implies them ("tests pass", "round-trip passes", "shape tests").
- Keep changes minimal and consistent with existing code and structure.

## 4. QA — verify the acceptance criterion

QA means demonstrating the "Done when" line with concrete evidence, not just writing code that looks right:

- **Environment**: use the project venv `.venv` (create it with `python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"` if missing). Reuse it across tickets — the torch download is large.
- **Code tickets**: run `pytest` from `.venv`; run the actual entrypoint/command the ticket is about (use a tiny config where one exists) and observe the output.
- **Artifact-producing tickets**: verify the produced files (plots, logs, checkpoints, persisted vocab/merges) actually exist and look sane.
- **Qualitative criteria** ("learns XOR reliably", "robust ingestion on noisy data"): run a small experiment and record the result (final loss/accuracy, round-trip success, OOV count).
- **Docs tickets**: check the note exists at the expected path and covers what the ticket lists (e.g. equations + intuition + failure modes).
- **UI tickets**: start the app and verify the routes/behaviour (a browser check via playwright is fine when available).

If QA fails, fix and re-run. Do not proceed to closing until the criterion is met. If it cannot be met in this environment (e.g. no GPU), verify the CPU fallback path and state the limitation in the issue comment.

## 5. Commit

Commit with the ticket id in the subject, e.g.:

```
Add YAML config loader and schema validation (T-003)
```

with a short body listing what was added/changed.

## 6. Close the GitHub issue

Only after QA passes and the commit exists:

```sh
gh issue comment <N> --repo calculon102/llm-from-scratch --body "<QA summary>"
gh issue close <N> --repo calculon102/llm-from-scratch
```

The comment must include: what was implemented, how the "Done when" criterion was verified (test output, commands run, artifacts produced), and the commit hash.

## Multiple tickets

When several tickets are requested, do them one at a time in dependency order — implement → QA → commit → close — so each issue gets its own verifiable comment and the board stays accurate.
