# Roadmap

Written 2026-08-22, after a session that ended in a strategic step-back. Competition deadline:
entry **2026-09-23**, final submission **2026-09-30** (~5 weeks out). Live rating at time of
writing: **~500**, plateaued.

This document exists because effort was being spent on the wrong questions. It is a plan for
where effort goes, and — just as importantly — where it stops going.

---

## The honest position

**Where we stand.** We beat the near-tier field (`rajan1673`, `nagatakengo` are clean sweeps),
we're roughly even with `chaitanyajamble`/`ektarr`, and we lose to the 2500+ tier by -97k to
-142k coins. That last gap has not narrowed all session despite many attempts.

**Why the top gap isn't a tuning problem.** `kawa` and `prvsiyan` don't play the game. Both were
read directly: they replay giant hardcoded per-step action tables solved offline with full
foresight of the whole 30-day episode, with small live patches. A live heuristic reacting to
per-turn thresholds cannot reproduce a schedule computed with perfect knowledge, no matter how
well its thresholds are tuned. Nine-plus documented "spend more aggressively" failures
(`IDEAS_TRIED.md`) are all the same wall hit from different angles.

**Why the approach itself is still sound.** Lux AI Season 2 was won outright by a hand-tuned
heuristic agent with ~30 parameters — fewer knobs than ours (`input/lux_ai_2022_ryandy/`, 1st
place, Apache 2.0). Rule-based agents demonstrably win modern Kaggle simulation competitions.
That winner didn't succeed by tuning 30 parameters well; they succeeded by having the *right* 30
parameters — the right phase boundaries, roles, and horizons. That came from understanding the
game, not from search.

**The diagnosis.** Optuna tunes a decision structure you already believe in; it cannot invent
one. Most of our compute has gone into searching threshold values on a structure that hasn't been
seriously questioned, and much of it was aimed at opponents we cannot beat. The plateau is a
structure problem wearing a search problem's clothes.

---

## Phase 0 — Stop the bleeding ✅ done 2026-08-22

Correctness debt that was silently corrupting every result. All landed and verified; see
`tests/LOG.md`.

| Fix | Why it mattered |
|---|---|
| Win-based objective (`_squash`) | Trials scored raw coin margin while the rating system is win/loss/tie only. Shaving $20k off an unwinnable kawa loss scored **8x better** than flipping a matchup to a real win. |
| Objective/gate alignment | `verify_candidate` was always win-based while the objective was margin-based — **the search was optimizing a different quantity than the gate it had to pass.** Explains years of "best trial fails verification". |
| Out-of-sample holdout gate | Promotion only tested candidate-vs-champion, so a config could win head-to-head while regressing against the wider field. Exactly what happened to v11. Validated by replaying v11 through it: correctly blocked. |
| Champion drift fix | New mechanisms shipped with non-zero `DEFAULT_CONFIG` defaults; `best_config.json` doesn't pin those keys, so the champion silently stopped being the config verified 9W-1L. Cost ~5.5k margin vs `chaitanyajamble`. |

**Standing rule adopted:** a new mechanism defaults to a **no-op**, and the search turns it on.
Anything else silently mutates the champion.

---

## Phase 1 — Establish the honest ceiling (next, ~1 session)

**Goal:** find out what the heuristic approach is actually worth when pointed at the right
target. This is a measurement, not an improvement.

1. Run a full v12 search (`robust_agent_config_v12_winrate_objective`) — the first search ever
   aimed at what the promotion gate measures. Both gates now apply.
2. Test the already-built-but-unverified `opening_*` mechanism (one ablation). Grounded in five
   traced opponent openings and phase separation confirmed in three independent winners.
   Honest prior: it's still a "spend aggressively" variant, and those have failed 9+ times.

**Decision point.** Whatever rating this converges to is the heuristic-plus-search ceiling. If it
lands well short of the top tier — the likely outcome — that is the evidence to stop tuning and
commit to Phase 3, not a reason to run more searches.

---

## Phase 2 — Decision quality, not knob values (~1-2 sessions)

The work that can actually move a ceiling. Each item changes *what the agent reasons about*, not
what number a threshold holds.

1. **Continuous scores instead of binary gates.** Two independent reference winners name this:
   Halite IV's 4th place — *"continuous scoring models rather than binary discrete threshold
   conditions... a much smoother fitness landscape"* — and Kore 2022's 5th as the same point in
   negative — *"embedded magic numbers and hardcoded step-functions create brittle thresholds
   that cannot be easily tuned via automated tools like Optuna."* Our agent is built almost
   entirely from discrete gates (`remaining >= cost + cushion`, `backlog > units * ratio`,
   `day >= startup_days`), producing a landscape of flat plateaus and cliffs where a knob moves
   a long way with no effect and then falls off an edge.
   **Probe one gate first** (the hire backlog check is cleanest) and measure whether the search
   responds differently to that knob before committing to a wider refactor.
2. **Finish the ROI / shadow-pricing direction.** Pricing land and labor as real opportunity
   costs was the most promising work of the session and got parked in favour of more searching.
   `lambda_land` converged high and meaningful whenever it was actually searched.
3. **Fewer, better parameters.** We have ~45 knobs; the Lux S2 winner won with ~30. Prefer a
   small number of high-level knobs driving derived quantities over many independent flat ones.

---

## Phase 3 — The fork (decide after Phase 1's number, ~2-3 sessions)

Closing the top-tier gap needs a different class of weapon. Two candidates; pick one, don't
half-do both.

**Option A — Offline solving (match their weapon).**
`kawa`/`prvsiyan` win by precomputing routes. We have verified that **days 0-2 are byte-identical
across episodes** regardless of seed or opponent, with first divergence at day 3 (shop-unlock
RNG). If the reachable early state space is narrow enough, precomputing is available to us too.
*Open question to answer first:* how much of the game past day 3 is effectively deterministic
given our own action sequence? That measurement decides whether this is viable or a dead end.

**Option B — Reinforcement learning.**
Reward directly on win/loss, which sidesteps both the discrete-gate landscape problem and the
hand-tuned-threshold problem entirely. `src/train_rl.py` already exists as a hook. Precedent is
strong: Lux S1 1st, Hungry Geese 1st, and the RPS 1st place were all learned policies with no
interpretable knobs (`input/index.json`).
*Cost:* it is the biggest time investment here, and ~5 weeks is not generous for it.

---

## Explicit non-goals

Things to stop spending time on. Each is here because it already consumed effort and produced a
documented negative result.

- **Chasing kawa/prvsiyan's spending curve.** Wrong target for a live heuristic; concluded in
  `IDEAS_TRIED.md` and re-confirmed repeatedly. Post-v12 the objective deliberately writes these
  opponents off — correctly, since losing by less earns zero rating.
- **More "spend/expand more aggressively" variants** under new names. Nine-plus documented
  failures. Check `IDEAS_TRIED.md` before proposing one.
- **Trusting small samples.** Two episodes per opponent has produced at least three false
  positives, including a promotion that had to be reverted. A result you revert costs more than
  the run you saved.
- **Copying opponent action tables.** Their openings are absolute per-step movement
  choreography for a fixed actor count; altering any quantity desyncs the paths. Generalize the
  *goals*, never the table.

---

## Where things live

| What | Where |
|---|---|
| Game rules (canonical) | `GAME_GUIDE.md` |
| Every test outcome, newest first | `tests/LOG.md` |
| What's been tried, by concept, with verdicts | `tests/IDEAS_TRIED.md` |
| Competitor + prior-art research | `COMPETITOR_STRATEGY_NOTES.md` |
| Winning solutions' source (gitignored, local reference) | `../input/` + `../input/index.json` |
| This plan | `ROADMAP.md` |
