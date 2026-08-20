# Test Log Convention

Every local test outcome — a benchmark match, a correctness check, a before/after comparison — gets an
entry in [`LOG.md`](LOG.md), newest first. This is a project rule (see `../../CLAUDE.md`), not just a
suggestion: it's what lets us tell "we measured this" apart from "we assume this," days or weeks later
when the numbers matter for a real decision (which config to submit, whether a refactor actually changed
behavior).

Episodes aren't seeded by default, so a single run is a data point, not a verdict — note when a result
comes from one run vs. several trials.

## Entry format

```markdown
## YYYY-MM-DD — short title
- Command: `...` (or a one-line description if it was ad hoc Python)
- Result: agent A `<reward>` vs agent B `<reward>` — status/status (note win count if multiple trials)
- Notes: what this confirms or rules out, one or two sentences
```

Keep entries short. The point is a scannable record, not a full transcript — link to the relevant file
(`src/robust_agent.py`, etc.) rather than pasting code.
