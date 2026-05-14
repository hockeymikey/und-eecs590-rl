# AI Usage Disclosure

This project used **Claude** and **Codex** CLIs as collaborators throughout the semester. This document records what they were used for and what they were not used for, so a reviewer can calibrate expectations against the V3 spec's framing of AI assistance.

## What it was used for

- **Code review and debugging.** Reading through diffs before commit, spotting bugs in PPO/BC/env code, and walking through stack traces. This helped turn my rough implementation style into cleaner submission-ready code.
- **Architecture and design discussion.** Sounding-board work on the env decomposition (gym env vs. Isaac env, shared `IsaacSingleEnvWrapper` interface), observation/reward design (the brush-coverage vs. centre-trace reward bug, observation extension), and training-workflow tooling (`uv` workspace layout, `just` recipe shape, per-run reproducibility snapshot).
- **Diagnosis on training runs.** Talking through completed runs, pointing out details I had missed, and helping me keep the debugging narrative organized.
- **Documentation drafting and editing.** Markdown in this repo went through an iterative editing pass for grammar, structure, and narrative clarity.
- **Isaac Lab / Arch Linux debugging.** Pair-debugging the multi-day platform setup, including stack traces, candidate fixes, and failure-mode triage.

## What it was not used for

- **No autonomous code generation.** Every committed change was read, accepted, and edited by me before landing. Everything was done through PyCharm like any other project of mine.
- **No autonomous decision making on experimental design.** Algorithm choice (PPO over SAC/TD3/etc.), reward iterations, observation iterations, the BC warmstart approach, and other design choices — those decisions are mine. LLMs were used as a sounding board.
- **No reporting on outcomes I didn't see.** Training-run results, plots, and conclusions in `docs/decisions.md` and `docs/technical-challenges.md` reflect runs I actually ran on my own hardware. Where a result is one seed I say so; where a number is shaky I say so.
- **No hidden authorship.** Every place AI-drafted prose lives in the repo is also a place I edited and signed off on. The voice across `docs/` is intentionally consistent because I do the final editing pass.

## Honest caveat on documentation drafting

Where LLMs were most heavily involved is in *drafting prose* — taking my notes, decisions, and summaries and turning them into the longer-form documentation under `docs/`. The structure, framing, and content are mine. If a reviewer reads `docs/decisions.md` and `docs/technical-challenges.md` side by side and notices a consistent house style, that's the editing pass; the underlying technical content reflects what I built and what I saw during training.

## Tools

- **Claude Code CLI** in-editor pair-coding sessions.
- **Codex CLI** in-editor pair-coding sessions.

No other AI tools (Copilot, MinMax, DeepSeek, Cursor's own models, etc.) were used on this project.

## Course policy

This disclosure matches the spirit of the EECS 590 V3 §Final Content rubric: the decisions are mine, the implementation is mine, the conclusions are mine; AI was a code-review and design-discussion aid.

See also: `CITATIONS.md` for the broader citation list and `docs/decisions.md` for the implementation decisions themselves.
