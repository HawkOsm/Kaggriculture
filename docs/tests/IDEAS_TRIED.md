# Ideas Tried

A concept-indexed ledger of strategic ideas tried against the agent, so we stop re-testing the
same thing under a different name. [`LOG.md`](LOG.md) has the full chronological detail (command,
exact numbers, notes) for every run — this file is the "have we already tried this?" lookup, one
row/entry per **idea**, not per run. Update this whenever an idea gets a real verdict, not every
time a number changes.

Legend: ✅ worked (kept) · ❌ regressed (reverted or never landed) · ➖ null result (kept, unproven) · 🐛 bug fix

---

## Spending posture / aggressive early investment

**Consistent pattern: every attempt to spend/expand more aggressively regressed. Only waste-reduction
fixes (below) have actually helped.** Before trying another variant of "spend more," read this section.

| Idea | Verdict | Why | LOG.md |
|---|---|---|---|
| Raise `max_hires_per_day` alone (5→12) | ❌ | More hands the dispatcher can't service just means more units walking, not working — ~half of land went idle, lost to a 1-hand baseline | 2026-08-20 "scaling-strategy redesign"; retested post-dispatch-fix 2026-08-21 "rewrote _plan_units dispatch" |
| Raise `max_structures`/`pasture_target_ratio` alone (4→16) | ❌ | Animal capital cost (~$400-500 each) + shed-pickup logistics for that many pastures outweighs the water/labor savings within a 29-day season, competes with hire/land for the same early cash | 2026-08-21 "instrumented _plan_units tier usage" |
| Reserve a fraction of hands for land-expansion tasks before maintenance | ❌ | `target_structures` was already capped near current pasture count (a *different* bug, since fixed), so reserved units just planted more crop tiles — increased water demand instead of relieving it | 2026-08-21 "instrumented _plan_units tier usage" |
| Flat "aggressive" config (low reserves, high `max_structures`, earlier land, all at once) | ❌ | Regressed on 7/9 opponents; isolating `land_utilization_threshold` back out didn't recover it — not a single-knob problem | 2026-08-21 "tested a hand-built aggressive early spending config" |
| Day-phased "go broke first 40%, revert to normal after" (`broke_phase_days_frac`/`broke_phase_reserve_scale`, real mechanic, not a strawman) | ❌ | Still regressed even with a genuine time-boxed implementation — confirms the blocker is execution capacity, not that the earlier flat version's aggression was unbounded | 2026-08-21 "implemented a genuine day-phased go broke early mechanic" |
| Fix `target_structures`'s labor-count gating (build toward *intended* labor, not today's headcount) + raise `max_structures` | ❌ (mechanism ✅, outcome ❌) | The build-ramp fix itself works (pasture count now grows correctly, confirmed via trace) — but money still nets worse; animals lag pasture badly (12 pasture, 5 animals) | 2026-08-21 "fixed a real bug (pasture target gated on today's headcount)" |
| Also loosen `animal_reserve_multiple` (2.18→1.1) on top of the above | ❌ (dramatically) | Buying animals faster just drains cash with no compensating income mechanism — final money crashed to 2,461 | same entry as above |

**Update 2026-08-21 — production/sell-side hypothesis refuted.** Audited our own watering/harvest/
feed/sell compliance directly (`.claude/scratch/production_audit.py`): unwatered tiles near-zero
all game, harvest-ready backlog **zero for the entire 29-day game** (both crops and animal
products), shed inventory doesn't pile up (except `FERTILIZER`, a small ~$1000 leak — collected
from animals but never sold down). **Production/sell execution is clean at whatever scale we're
operating at.** The real constraint is capacity *scale* itself (small `max_structures`, hands,
land) — and raising that scale keeps backfiring not because we can't use the extra capacity once
built, but because of cash-flow/capital competition while *acquiring* it. Current best guess: a
live heuristic agent reacting to reserve/backlog thresholds each turn can't reproduce the tight,
non-starving spending *schedule* that lets kawa's pre-solved replay (see "Dispatch / routing
efficiency" below — most strong opponents don't decide live at all) scale up without ever running
dry. A genuine forward-looking spending plan (a few days of projected income ahead, not reactive
per-turn gates) is the untested next idea here — not attempted yet.

---

## Dispatch / routing efficiency

The one area with actual verified wins, because these reduce waste rather than increase spending.

| Idea | Verdict | Why | LOG.md |
|---|---|---|---|
| Cross-turn target "stickiness" (don't re-litigate a target every turn) | ✅ | Cut movement waste 74.2% → 69.3% | 2026-08-21 "added cross-turn target stickiness" |
| Global nearest-pair batch matching (replace fixed-unit-order sequential greedy) | ✅ modest | ~15% money improvement in head-to-head trace, no regression — but doesn't close the larger gap alone | 2026-08-21 "rewrote _plan_units dispatch to global nearest-pair batch matching" |
| Spatial zoning (partition board into per-unit home regions) | **ruled out before building** | Measured water-tier assignment distance under the *existing* matcher: mean 1.62 tiles, 78% ≤2 — already near-optimal. Zoning can't reduce distance that's already minimal; the cost is trip *count*, not trip *distance* | 2026-08-21 "traced submission vs kawa_route_agent... found the real bottleneck is dispatch scaling" |
| Clustering-biased empty-tile placement (grow footprint as a contiguous block instead of pure nearest-available) | ➖ null | Implemented correctly (real `cost_fn` mechanism), but no measurable change in water-tier distance or match margin at `CLUSTER_BONUS=2.0` — likely too weak relative to typical distances, or swamped by run-to-run variance at this sample size. Kept the code, unproven | 2026-08-21 "clustering-biased empty-tile placement" |
| Move `CARE` from priority tier 12 to right after `FEED` (found via `actions.csv`: CARE was 0.9% of our unit-turns vs kawa's 4.6%, a 5.1x gap, the largest of any action category) | ✅ small | CARE's share rose to 1.5% (now tracking FEED's rate as intended), margins slightly better on both test opponents, no regression | 2026-08-21 "used the new actions.csv log to find and fix a real dispatch-priority gap" |
| `_market_orders` checked every spend gate against a stale `money` snapshot that never accounted for what earlier orders the same turn had committed — `BUY_ANIMAL` (last in the function) lost the race for cash most often | ✅ small | Real running balance + moved animal purchases ahead of hire/land — ~15% better vs prvsiyan in one check, flat vs kawa, no regression | 2026-08-21 "fixed a real stale-money bug" |
| Force-sell shed backlog past a quantity threshold regardless of price (to fix `MILK` piling up unsold once price craters) | ❌ | Shed inventory doesn't decay like crops do — forcing a sale at a crashed price permanently realizes a loss instead of waiting for the market's slow-but-real recovery. Reverted; `sell_backlog_multiple` config key stays inert for a smarter version later (e.g. only force-sell near an actual hard cap) | 2026-08-21 "reverted the sell_backlog_multiple force-sell fix" |

**Known root cause, not yet solved**: watering alone is 33-48% of all unit-turns (measured via
tier instrumentation), and the opponents that beat us decisively (kawa, romanrozen, and most of
the pool) turn out to **not run a live dispatcher at all** — they replay a pre-recorded action
sequence indexed by turn number (`_ACTIONS[step]`), with narrow patches for exceptions. This is a
fundamentally different, offline-solved-once architecture. Explicitly considered and **rejected**
in favor of staying with a live reactive dispatcher (fragility tradeoff vs. a fixed script) — see
"Approach pivot" discussion, 2026-08-21.

---

## Pasture / animal capacity bugs (both real, both fixed)

| Bug | Fix | LOG.md |
|---|---|---|
| `target_structures` scaled only with labor, never land — froze pasture at its day-0 value forever once land expanded past quadrant 1 | Multiply by `n_quadrants` | 2026-08-21 "replay review found two real bugs" |
| `target_structures` gated on *today's* headcount (`len(units)`), capping near 1 on day 0 regardless of `max_structures` — contradicted the code's own "building is free" comment | Replace `len(units)` with `1 + max_hires_per_day` (intended labor scale) | 2026-08-21 "fixed a real bug (pasture target gated on today's headcount)" |
| `_market_orders` checked every spend gate (`BUY_LAND`/`HIRE`/`BUY_ANIMAL`) against the same stale `money` snapshot read once at the top — never accounted for what earlier orders queued the *same turn* already committed. `BUY_ANIMAL` was last in the function, so it lost the race for cash most often | Real running balance (`remaining`, decremented per queued order) + moved animal purchases ahead of hire/land | 2026-08-21 "fixed a real stale-money bug" |

Both `target_structures` fixes are real and kept. Neither closed the money gap on its own — see
"Spending posture" above. The stale-money fix + animal-priority reorder gave a small real
improvement (~15% vs prvsiyan, flat vs kawa, 6-episode verification) — first genuine win in the
"spending posture" family after seven regressions, likely because it's a real bug fix (correctness)
rather than a spending-aggressiveness change.

**`MILK` accumulation leak — root cause confirmed, but the obvious fix ❌ regressed.** `MILK` (and
likely other premium products — `above_target > 1` in the price table) can pile up unsold once
production scales up, because once our own selling craters its price below our
`sell_fraction`-based threshold, the gate stops firing entirely rather than selling at *some* lower
price. **Tried**: force a sale once shed backlog crosses a multiple of `max_sell_chunk`, regardless
of price (`sell_backlog_multiple`, added to `DEFAULT_CONFIG`/`KNOB_SPECS`/`optimize.py`'s search
space — code path itself reverted, the config key stays inert). **Regressed on both test
opponents** (kawa -122k→-147k, prvsiyan -103k→-131k, 6-episode check) — the reasoning was wrong:
unlike crops, shed inventory doesn't decay while waiting, so forcing a sale at a crashed price
permanently realizes a loss instead of waiting for the market's real (if slow) price recovery.
"Some revenue beats zero" only holds when the alternative is actually zero — here it's "more
revenue, later." A smarter version (only force-sell near an actual hard cap — shed capacity or
`max_held` on the source tile, where the alternative is genuinely wasted production) is untested.

---

## Opponent-adaptive behavior

- **`opponent_race_discount`'s optimal value is opponent-dependent, not universal.** Ablation
  (4 episodes each): vs kawa, the current static tuning wins; vs prvsiyan, a much more aggressive
  discount wins by an even bigger margin, and even fully disabling opponent-awareness beats the
  static baseline. Swings of 15-25k in opposite directions — a real trade-off, not noise. No single
  static value can be right for both. See `LOG.md` for the numbers.
- **✅ Built genuine opponent-adaptive behavior from this**, not another static-value guess:
  `_opponent_profile` (renamed from `_opponent_incoming_supply`) now also computes the opponent's
  crop/product *concentration* and overall *scale* from their public tiles/land/labor (never
  anything private), and the race-discount scales by `concentration × scale` instead of being one
  fixed number. Verified it lands *between* the two static extremes for both kawa and prvsiyan
  (avoiding each one's worst case, not matching either one's best case) — exactly the signature of
  a working adaptive mechanism. New `opponent_concentration_sensitivity` knob (default 0.3) is in
  the Optuna search space to find the right default. Whether the trade is net-positive across the
  *full* 9-opponent pool (only 5 spot-checked so far) is still open.
- This is the first opponent-*adaptive* mechanism in the agent — everything else (hire/land/animal
  spending, dispatch priority) is still static per-game, set once from config, not observed and
  reacted to per-opponent. If this pans out, the same public-tile-scan approach (concentration,
  scale) could extend to other knobs later.

## Config search (Optuna)

- **Four full 150-trial searches this session against the 9-opponent (2500+) pool, none promoted.**
  Verification results: 0W-10L, 0W-10L, 0W-10L, 2W-8L (best so far). All landed on similar
  conservative configs (`max_structures` 4-7, low `pasture_target_ratio`). See the [Search Ledger
  artifact](../../.claude/scratch/artifact_build/optuna_dashboard.html) — every completed trial
  landed in the first ~150 of 684 total; the last 3 searches spent ~530 trials each re-exploring
  the same space and found nothing new.
- **Silent bug (fixed 2026-08-21)**: `evaluate_config` had lost `_resolve_agent` at some point —
  every non-champion opponent silently resolved to a meaningless default-config agent. Polluted
  every exploration trial (not the promotion gate, which always played candidate-vs-champion
  directly) for an unknown span of prior searches.
- **`opponent_awareness_enabled` was left as a searchable boolean** — a noisy 2-episode trial
  toggled it off with no real justification (it has no game-theoretic downside). Hard-coded to
  `True` in `optimize.py`'s `_suggest_config`; ✅ small real win when fixed (~4% vs kawa).
- **Lesson from the "1st Place" Kaggle discussion thread**: only win/loss counts for rating, not
  margin. Once we have any non-zero win rate at all, switch primary comparison metric from
  avg-margin to win-rate — margin-based comparisons risk optimizing something the actual ranking
  doesn't reward.

---

## RL training

| Finding | LOG.md |
|---|---|
| Reward was pure own-money delta, never opponent-relative, despite the game only scoring win/loss | 2026-08-21 "RL training review before relaunch" |
| `opp_cash_frac` feature saturated at 1.0 almost immediately (reused `cash_scale`, tuned for a much smaller number) — policy couldn't tell "opponent modestly ahead" from "astronomically ahead" | same entry |
| PPO replay-buffer staleness: 12,000-episode buffer meant most updates were importance-sampled against a policy dozens of generations stale | 2026-08-21 "train_rl.py: found and fixed a real PPO replay-buffer staleness regression" |
| `best.pt` trained without kawa in the opponent pool lost 0W-6L to kawa specifically — RL can't fix what its own opponent pool never included, and can't touch the mechanical/dispatch layer at all (only tunes per-day knobs on top of it) | 2026-08-21 "RL-trained best.pt vs kawa_route_agent" |
| Elite-retention safeguard (revert to `best.pt` after 3 consecutive regressed evals) got stuck reverting to the same checkpoint (gen 235) repeatedly during a 2h run, never escaping | referenced across several 2026-08-21 entries |
| Discussion-forum finding (not our own test, but relevant): a PPO plateau training against a diverse opponent pool may reflect a genuine non-transitive game (real rock-paper-scissors cycles among strategies) rather than an optimization failure — a mixed strategy, not a bug | this session's discussion-forum research, not logged as a LOG.md test entry |

**Not yet retrained** with all of today's mechanical fixes (dispatch rewrite, opponent-awareness,
pasture-scaling) combined — the plan is to do this only once the mechanical layer stops being the
bottleneck, since retraining against a broken execution layer just re-learns the same ceiling.

---

## Benchmarking / evaluation methodology

- **The entire 9-opponent pool is 2500+ Elo — far above our live rating (~467).** Local 0W-9L
  against it was never representative of near-term competitive standing. Pulled two additional
  opponents genuinely near our own rating (`rajan1673_agent` #4478/434.7, `chaitanyajamble_agent`
  #4058/507.2) — result: **4W-0L vs rajan1673** (our first win all session), 0W-4L vs
  chaitanyajamble but by an order of magnitude smaller margin than anything against the 2500+ pool.
  See 2026-08-21 "pulled two near-tier opponents."
- **Important correction (user pushback, 2026-08-21)**: beating near-tier opponents proves the
  agent isn't fundamentally broken and gives a non-degenerate win/loss signal for iteration — it
  does **not** prove progress toward the prize goal (top 10 needs ~2900+ Elo). Don't over-read a
  near-tier win as closing the real gap.
- **Rating noise**: per the discussion forum, don't judge a submission before ~60 games (~5
  hours); byte-identical agents have been reported diverging by 1000+ points purely from early
  matchmaking luck. Treat <50-point live rating swings as noise.

---

## Opponent pool sourcing

- Standing methodology: `kaggle kernels pull`, decode if packed, AST def-name-set diff against the
  existing pool (jaccard > ~0.15-0.3 on function names, or identical docstring/codename, is a real
  duplicate-strategy flag), verify vs `random` and vs the pool before installing. Full detail and
  attribution policy in [`CREDITS.md`](../../CREDITS.md).
- **Excluded as duplicates**: `salemali7` (republish of kawa's reconstruction), `bruceqdu`
  (duplicate of saiteja's reconstruction), `denizeryilmaz`/`deniz_v111_agent` (100% function-name
  overlap with `boatlee_v16_agent`, attribution stripped).
- **`nagatakengo/kaggriculture-movements-top-xx`**: looked like a near-perfect near-tier score
  match (474.5 vs our 467) but turned out to be a replay-analysis notebook, not a submittable
  agent — check for a real `agent(obs)` function before investing further in a candidate.
- Most strong opponents in the pool (`kawa`, `romanrozen`, likely others) are **replay-based**
  (`_ACTIONS[step]`), not live dispatchers — see "Dispatch / routing efficiency" above.

---

## Recurring bug class: `build_submission.py`'s hand-maintained function list

Happened **twice** — a new helper function added to `agent.py` (`_dynamic_sell_fraction`, then
later `_assign_nearest`) without updating `build_submission.py`'s `ROBUST_AGENT_FUNCS` list. Both
times, the built `submission/main.py` silently `NameError`'d into `SAFE_FALLBACK` (PASS) every
single turn — a real, playing submission that actually does nothing, with no visible error unless
you check stderr. Fixed structurally the second time: `_check_completeness()` in
`build_submission.py` now walks every extracted function's AST for references to other
module-level names and errors before writing the file if anything's missing, instead of relying on
remembering to update a list by hand a third time.

**Always run `python src/run_match.py submission/main.py <opponent>` (real file-path loading) after
any `agent.py` change that adds a new top-level function or constant** — `_check_completeness`
catches the missing-from-list case, but still verify the actual built file runs clean.

---

## Game-rules / market mechanics (confirmed against `docs/GAME_GUIDE.md` and the rules)

- Ranking is Elo-like, win/loss only — coin margin never matters, confirmed in `docs/GAME_GUIDE.md`
  and independently by the "1st Place" discussion thread.
- Animals are structurally favored over crops: "Ongoing" (indefinite production, no replanting),
  structure is free to build (only the animal costs money), and critically — animals need
  FEED+CARE, not WATER, sidestepping the single biggest measured labor sink (33-48% of unit-turns).
- Market price is one shared inventory pool (`I0=10,000`) read by both players — dumping a product
  to hurt an opponent's price equally hurts our own future sales of it. Real lever, not a one-sided
  exploit; only clearly worth it if we're exiting that product anyway or the opponent is far more
  concentrated in it than we are. Discussed, never implemented.
- Two mid-competition balance patches landed (Town Center demand halved and de-escalated; small
  carrot/tomato/egg scarcity-pricing tweak) — confirmed our local `kaggle-environments==1.32.7`
  already reflects both.
- No rule against studying public opponent notebooks for strategic insight (`docs/RULES.md` §3.5:
  public code sharing is explicitly allowed and openly licensed) — the restriction is only on
  *private* sharing outside a team. We're already more conservative than required: never derive
  code directly into `submission/main.py`, only used as local sparring opponents.
