# AGENT.md

Instructions for any AI agent working in this repo (Claude Code, `agy`/Antigravity, or otherwise).
Read this first. It replaces the old `CLAUDE.md` — same rules, expanded with the working
conventions this project has actually needed in practice.

## What this repo is

An autonomous agent for Kaggle's Kaggriculture competition (see [README.md](README.md) for the
competition/deadline details and repo layout).

**The live submission architecture is `src/adaptive_agent/`** (`constants.py`/`config.py`/
`state.py`/`market_forecast.py`/`dispatch.py`/`farm_plan.py`/`market.py`/`agent.py`), promoted
2026-08-27 to replace `src/robust_agent/` as what `submission/main.py` is built from. It's a
from-scratch rebuild modeled on `src/opponents/pilkwang_agent.py`'s real architecture (public
Kaggle notebook on this competition's own Code tab; competition rule 6.b deems publicly-shared
Code-tab code OSI-licensed, see `docs/RULES.md` — a genuine derivative work, not a republish) —
a staged phase state machine (`LIQUIDATE`/`CRISIS`/`BOOTSTRAP`/`COMPOUND`/`REALIZE`), scheduled
herd/land targets instead of a pure reactive gate, real price-curve forecasting
(`_price_at`/`_shape`, reconstructing the actual `MARKET_PARAMS` formula), and a global greedy
bipartite worker↔mission dispatch instead of a tiered priority queue. Verified via a real
150-trial Optuna search + `verify_candidate`/`verify_holdout` gate against the former
`robust_agent` champion: 20W-0L-0T, +79k avg margin, independently re-verified multiple times
(see `docs/tests/LOG.md`'s 2026-08-27 pilkwang-rebuild entries) — not a marginal tune, a decisive,
robust win across the whole searched parameter space. `src/adaptive_best_config.json` holds its
tuned config (a completely different schema from `robust_agent`'s — `HERD_EXPANSION_DAY`,
`PRIORITY_BONUS`, `RESERVE_FRACTION`, etc., not the old 46-knob set). Regenerate
`submission/main.py` with `python src/build_adaptive_submission.py` any time `src/adaptive_agent/`
or `src/adaptive_best_config.json` changes; never hand-edit it directly.

**`src/robust_agent/` has been deleted** (2026-08-27, same day as the promotion above), along with
its now-unusable support code: `src/build_submission.py` (its flattener), `src/optimize.py` (its
CLI shim), and all of `src/tuning/` (its Optuna search/verification harness — `sample_config`'s
46-knob schema, `verify_candidate`/`verify_holdout`, the risk-adjusted scoring). None of it applies
to `adaptive_agent`'s different config shape (`HERD_EXPANSION_DAY`/`PRIORITY_BONUS`/
`RESERVE_FRACTION`/etc., not the old knob set), and keeping it around as dead imports was worse
than removing it — the real logic patterns (dual-agent verification, holdout rotation, risk-
adjusted Optuna scoring) are preserved in `.claude/scratch/pilkwang_rebuild/search.py` and
documented in `docs/tests/LOG.md` if a rework for `adaptive_agent`'s schema is ever wanted.
`src/agent.py` is a compatibility shim, repointed to re-export `adaptive_agent.agent.make_agent`
(so `import agent` / `run_match.py`'s `agent:<name>` convention keep working). **There is
currently no Optuna search/promotion pipeline for `adaptive_agent`** — that's the real, open
follow-up if further tuning is wanted; `submission/main.py` reflects the config the original
150-trial `pilkwang_rebuild` search found, not anything searched since. Lint/complexity:
`ruff check` (config in `pyproject.toml`, `max-complexity = 15`; `src/agent.py` has a
`per-file-ignores` exemption for F401 since its import is a deliberate re-export, not dead code —
`ruff --fix` will strip it otherwise, as it did once already, see `docs/tests/LOG.md`'s 2026-08-27
restructuring entry) and `radon cc` for complexity metrics.

## Standing rules

1. **Every test outcome gets logged to `docs/tests/LOG.md`.** Any local agent run — a benchmark
   match, a correctness check, a before/after comparison after a fix or refactor — gets an entry
   before moving on, whether the run was requested explicitly or done to verify other work. See
   [`docs/tests/README.md`](docs/tests/README.md) for the entry format. Newest entries go right
   after the `---` at the top of the file.
2. **Before trying a strategic idea (spending posture, dispatch approach, config direction),
   check [`docs/tests/IDEAS_TRIED.md`](docs/tests/IDEAS_TRIED.md) first.** It's a concept-indexed
   ledger distilled from `LOG.md` — most "spend/expand more aggressively" variants have already
   been tried and regressed. Don't re-run one under a new name without checking there. Add a row
   when a new idea gets a real verdict.
3. **The only valid promotion signal is `src/optimize.py`'s own `verify_candidate` (direct
   candidate-vs-champion self-play) followed by `verify_holdout` (regression check against the
   held-out opponent rotation).** A config that looks better against a fixed third-party opponent
   pool can still lose head-to-head to the actual champion — this is a real, repeatedly observed
   failure mode (non-transitivity), not a hypothetical. Never promote or trust an "improvement" on
   a third-party-benchmark comparison alone.
4. **`docs/GAME_GUIDE.md` is the canonical game-rules reference** — read it first for any rules
   question. `docs/kaggriculture/` (official starter-kit bundle) and `docs/RULES.md`/
   `docs/starter_kit/` overlap it heavily; only open those for a specific cross-check (rules
   wording, CLI/submit workflow), not as a first read.
5. **Never `Read` a full `replay.json` into context.** A 720-turn episode replay can be enormous.
   Write or reuse a small Python script to summarize it (final scores, money-over-time,
   specific-turn traces) instead of dumping the raw JSON.

## Working conventions (learned the hard way — follow these)

- **Scratch/temp files go in `.claude/scratch/<task-name>/`, never the repo root and never
  `/tmp`.** One-off debug scripts, intermediate `results.json`, trace harnesses — all of it.
  Verify the path with `pwd`/`ls` before writing, especially from a delegated agent process that
  may start in an unexpected working directory.
- **Git and Kaggle actions stay manual.** Never run `git commit`/`git push`/`git checkout`/
  `kaggle competitions submit`, even when asked to "update" something — hand the exact command to
  the user instead. This applies to any agent working in this repo, delegated or not.
- **Never trust a delegated agent's "improvement" or "passed" claim at face value.** Re-derive the
  actual numbers from the raw `results.json`/`run.log` it produced, not its prose summary. Before
  accepting a result, check it against known reference ranges for that matchup — e.g. as of the
  2026-08-27 opponent-pool refresh, `adaptive_agent`'s champion loses to the entire 2500+-tier pool
  (kawa/boatlee_v16/rayk_c95/saiteja/kaito/tran_hh/pilkwang/romanrozen/prvsiyan_frontier/
  sakhawathossen/stevenleehans, all 0W) and is roughly even with premaananda108 (2W-2L) -- see
  `CREDITS.md`'s "Current opponent pool at a glance" table for current numbers. A result far outside
  an established range is a hard stop signal, not a footnote. (The former near/mid-tier pool --
  chaitanyajamble, rajan1673, etc. -- was deleted the same day once the champion started beating it
  4W-0L across the board; don't cite old reference ranges for those matchups going forward.)
- **No monkeypatching `optimize.py`'s internals.** A comparison script must load the real champion
  (`src/agent.py`) and any candidate/variant as genuinely separate Python modules (e.g. via
  `importlib.util`), never by reassigning `optimize.make_agent = variant.make_agent` or similar —
  that silently makes "candidate vs champion" into "candidate vs itself." This exact bug produced a
  false "PASSED" result once (see `docs/tests/LOG.md`, 2026-08-26 decoupled-land-crisis entries) and
  cost real time to catch. `unittest.mock.patch` with object-identity dispatch is lower-risk but
  still needs independent, patch-free re-verification before a high-stakes result is trusted.
- **Always load `src/adaptive_best_config.json` fresh (`json.load(open(...))`) for the champion side
  of every comparison.** Never reuse or mutate one config dict across "baseline" and "variant" runs —
  a stray shared value silently contaminating both sides has produced at least one invalidated
  result. (`src/best_config.json`, without the `adaptive_` prefix, is a leftover from the deleted
  `robust_agent` era — nothing loads it anymore, don't confuse the two.)
- **A result with numbers outside every established reference range needs investigation before
  being trusted**, not just a note in passing — it has twice been the symptom of a real bug
  (wrong-path file writes, config contamination) rather than a genuine finding.
- **Small-N "improvement" claims are unreliable below N≈30** for any near-coin-flip matchup (e.g.
  `premaananda108`, currently 2W-2L) — high variance means a result at N=15 can evaporate into noise
  at N=30-40 (this happened twice with the old chaitanyajamble matchup before it was retired
  2026-08-27). Re-verify close matchups at higher N with fresh seeds before trusting a win-rate
  swing of a few points.

## Known architectural facts — `robust_agent` (deleted, historical — why it was replaced)

`robust_agent` no longer exists in this repo (deleted 2026-08-27, see above), but the reasoning
below is why `adaptive_agent` replaced it, not idle history — don't assume a from-scratch
reactive-dispatcher redesign would fare any better than these already did.

- **`in_crisis`** (`src/agent.py`, `crisis_backlog > unit_count_now * crisis_backlog_ratio`) gates
  both `BUY_ANIMAL` and `BUY_LAND`. It is load-bearing, not a bug: five structurally different
  removal/redesign attempts have all regressed (see `docs/tests/IDEAS_TRIED.md`). Land expansion is
  permanently dead in the current champion because one fully-planted quadrant alone keeps backlog
  above threshold from ~day 5 onward — confirmed from multiple independent angles, including a
  decoupled land-only gate that unlocks land mechanically but never stays competitive.
- **The current 1-2 animal count is an emergent "Goldilocks" interaction**, not a targeted
  mechanism: `max_structures=2` acts as a hard ceiling, `lambda_labor` and `crisis_backlog_ratio`
  jointly keep early backlog low enough to slip 1-2 purchases through. Moving `max_structures` by
  even 1 in either direction is catastrophic, not gradual — verified by direct ablation.
  Raising `max_hires_per_day` or `max_structures` alone to unlock more animals has been tried
  multiple times, across multiple code generations, and always regresses.
- **A from-scratch staged-target dispatcher** (pilkwang-style pre-reserved animal zones + a
  day-indexed herd/land target schedule, replacing the reactive priority-queue dispatch) was built
  and tested for the first time 2026-08-27 — loses decisively (0W-10L independently reverified).
  Cause: a hardcoded day-indexed purchase schedule, decoupled from live cash/ROI feedback, can
  commit to spending the farm can't afford. A hybrid (staged targets still gated by real-time
  ROI/cash) is a different, untested design.
- **The RL training pipeline (`train_rl.py`/`rl_policy.py`) was removed earlier** (before
  `robust_agent` itself was). `output/training/best.pt` is a stale artifact from when it existed.

## Quick commands

```bash
# Regenerate the LIVE Kaggle submission after any src/adaptive_agent/ or adaptive_best_config.json change
python src/build_adaptive_submission.py

# Run the live agent against a benchmark opponent locally
python src/run_match.py agent:adaptive_agent kawa_route_agent:kawa_route_agent
```

There is no Optuna search/tuning command for `adaptive_agent` yet — see the note above.
