# Project rules

- **Every test outcome gets logged to `docs/tests/`.** Any time an agent is run locally — a benchmark
  match, a correctness check, a before/after comparison after a fix or refactor — add an entry to
  `docs/tests/LOG.md` before moving on. See `docs/tests/README.md` for the entry format. This applies
  whether the run was requested explicitly or done as part of verifying other work (e.g. confirming a
  refactor didn't change behavior).
- **Before trying a strategic idea (spending posture, dispatch approach, config direction), check
  `docs/tests/IDEAS_TRIED.md` first.** It's a concept-indexed ledger of what's already been tried and
  its verdict, distilled from `LOG.md` — most "spend/expand more aggressively" variants have already
  been tried and regressed; don't re-run one under a new name without checking there first. Add a row
  when a new idea gets a real verdict.

## Token-conscious doc reading

- **`docs/GAME_GUIDE.md` is the canonical game-rules reference** — read it first for any rules question.
  `docs/kaggriculture/` (official starter-kit bundle) and `docs/RULES.md`/`docs/starter_kit/` overlap it
  heavily; only open those for a specific cross-check (rules wording, CLI/submit workflow) rather than as
  a first read.
- **Never `Read` a full `replay.json` into context.** A 720-turn episode replay can be enormous. Write or
  reuse a small Python script to summarize it (final scores, money-over-time, specific-turn traces)
  instead of dumping the raw JSON.
