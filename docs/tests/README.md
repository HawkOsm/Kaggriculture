# Test Log Convention

Every local test outcome — a benchmark match, a correctness check, a before/after comparison — gets an
entry in [`LOG.md`](LOG.md), newest first. This is a project rule (see `../../CLAUDE.md`), not just a
suggestion: it's what lets us tell "we measured this" apart from "we assume this," days or weeks later
when the numbers matter for a real decision (which config to submit, whether a refactor actually changed
behavior).

**Before trying a strategic idea (a spending posture, a dispatch approach, a config direction),
check [`IDEAS_TRIED.md`](IDEAS_TRIED.md) first** — it's a concept-indexed "have we already tried
this?" ledger distilled from `LOG.md`, so a plausible-sounding idea that's already been tested and
regressed doesn't get re-tried under a different name. Add a row there (not just a `LOG.md` entry)
whenever an idea gets a real verdict.

Episodes aren't seeded by default, so a single run is a data point, not a verdict — note when a result
comes from one run vs. several trials.

## Run artifacts

Single-match runs (`src/run_match.py`) save a `result.json` + `replay.html` + a `best_config.json`
snapshot + `actions.csv` into a timestamped folder under `output/games/` by default (gitignored —
local visibility only, not committed; this is for one-off test/verification games only, not RL
training runs, which log to `src/rl_checkpoints/`). Open the `replay.html` in a browser to watch
the actual match instead of just reading the summary here.

`actions.csv` is a continuous per-turn, per-unit log — one row per unit (farmer/hand) per step per
player, both players, with position and the exact action taken — reconstructed from `env.steps`
after the run (works for any opponent, including black-box ones we don't control the source of).
This replaces the pattern of writing a fresh ad hoc instrumentation script every time a "what is
each unit actually doing" question comes up (this project did that repeatedly during development —
tier/movement audits, production audits, distance probes); now it's just there for every run. Filter
it in a spreadsheet or with `pandas`/`csv` for whatever the question is (idle-tile fraction, task
category mix, movement-vs-action ratio, a specific unit's path) rather than re-deriving from scratch.

Prefer `run_match.py` over one-off inline Python for anything worth inspecting later; if a check
genuinely needs custom ad hoc code beyond what `actions.csv` covers (e.g. a multi-trial loop, an
ablation), still write its result to a file under `output/` rather than letting it live only in a
terminal that scrolls away.

## Entry format

```markdown
## YYYY-MM-DD — short title
- Command: `...` (or a one-line description if it was ad hoc Python)
- Result: agent A `<reward>` vs agent B `<reward>` — status/status (note win count if multiple trials)
- Notes: what this confirms or rules out, one or two sentences
```

Keep entries short. The point is a scannable record, not a full transcript — link to the relevant file
(`src/robust_agent.py`, etc.) rather than pasting code.
