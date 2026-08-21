# Test Log Convention

Every local test outcome — a benchmark match, a correctness check, a before/after comparison — gets an
entry in [`LOG.md`](LOG.md), newest first. This is a project rule (see `../../CLAUDE.md`), not just a
suggestion: it's what lets us tell "we measured this" apart from "we assume this," days or weeks later
when the numbers matter for a real decision (which config to submit, whether a refactor actually changed
behavior).

Episodes aren't seeded by default, so a single run is a data point, not a verdict — note when a result
comes from one run vs. several trials.

## Run artifacts

Single-match runs (`src/run_match.py`) save a `result.json` + `replay.html` + a `best_config.json` snapshot
into a timestamped folder under `output/games/` by default (gitignored — local visibility only, not
committed; this is for one-off test/verification games only, not RL training runs, which log to
`src/rl_checkpoints/`). Open the `replay.html` in a browser to watch the actual match instead of just
reading the summary here. Prefer `run_match.py` over one-off inline Python for anything worth inspecting
later; if a check genuinely needs custom ad hoc code (e.g. a multi-trial loop, an ablation), still write
its result to a file under `output/` rather than letting it live only in a terminal that scrolls away.

## Entry format

```markdown
## YYYY-MM-DD — short title
- Command: `...` (or a one-line description if it was ad hoc Python)
- Result: agent A `<reward>` vs agent B `<reward>` — status/status (note win count if multiple trials)
- Notes: what this confirms or rules out, one or two sentences
```

Keep entries short. The point is a scannable record, not a full transcript — link to the relevant file
(`src/robust_agent.py`, etc.) rather than pasting code.
