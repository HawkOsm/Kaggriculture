# Ideas Tried

A concept-indexed ledger of strategic ideas tried against the agent, so we stop re-testing the
same thing under a different name. [`LOG.md`](LOG.md) has the full chronological detail (command,
exact numbers, notes) for every run — this file is the "have we already tried this?" lookup, one
row/entry per **idea**, not per run. Update this whenever an idea gets a real verdict, not every
time a number changes.

Legend: ✅ worked (kept) · ❌ regressed (reverted or never landed) · ➖ null result (kept, unproven) · 🐛 bug fix

---

## Spending posture / aggressive early investment

> **⚠️ REVISED 2026-08-22 — read this before trusting the failures below.** The v14 champion was
> promoted with `money_reserve: 140` (down from 420), `seed_money_floor: 7` (from 25) and
> `hire_money_floor: 10` (from 60) — i.e. **the most aggressive spending posture ever promoted
> here**, and it verified 10W-0L seeded, improved against all three held-out opponents, and won
> 6W-0L on unseen holdout seeds. That does not make the rows below wrong, but it does change what
> they mean, for two reasons:
>
> 1. **Every failure below was measured with unseeded evaluation**, whose noise floor was later
>    measured directly: an identical config beats a copy of *itself* 3W-1L with a +1,285 margin.
>    Several of those "regressions" may have been noise. They were not re-tested after seeding
>    landed.
> 2. **The v14 champion is not "spend more and hope."** It couples the low reserves with much
>    stronger opponent reactivity (`opponent_incoming_threshold` 9 → 2, `opponent_race_discount`
>    0.966 → 0.434) and a tighter risk band (`risk_scale_min/max` 0.89–2.35 → 1.45–1.80). The
>    failed attempts loosened spending *in isolation*. This is a different idea that happens to
>    share one parameter direction.
>
> Practical guidance: do not treat "aggressive spending" as settled-negative any more. Do still
> check whether a proposed variant is isolated loosening (the failed pattern) or coupled with
> reactivity (the promoted pattern) — and measure it seeded.

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
| "Highly favor animal buying": `max_structures` 4→14 + `animal_reserve_multiple` 2.6→0.3, retested *after* the reserve-architecture rework (proportional-to-cost cushions, not a dominant flat buffer) | ❌ | Regressed against both kawa and prvsiyan at 6 episodes each, every single episode worse than baseline — the architecture fix didn't change the verdict, so it isn't that the old reserve design was artificially suppressing beneficial aggression | 2026-08-22 "tested highly favor animal buying" |
| Lower `land_reserve_multiple` alone (3.284→1.0), inspired by pilkwang Kim's published closed-form hire-reserve formula | ❌ | That formula only covers HIRE (which we already match in shape); land/animal are different capital-risk profiles. Regressed 7/8→6/8 wins, avg margin +15,940→+12,934, made our worst matchup (chaitanyajamble) worse — same isolated-loosening failure pattern as every row above, just on a knob not previously isolated-tested | 2026-08-22 "tested lowering land_reserve_multiple alone" |
| Replace the flat reserve multiplier with an actual ROI/payback projection (`roi_gate_enabled`, only buy if projected $/day × days-left clears `roi_margin`× cost) — a more "principled" mechanism, not just a smaller number | ❌ (worse than the flat cut above) | 6/8 wins, avg margin +10,503 (vs the flat cut's +12,934, vs champion's +15,940) — daisy023/kiykhoi both took double-digit-thousand hits. A purchase that pencils out in isolation still competes with everything else for the same thin labor pool and the same simple dispatcher, which the ROI formula can't see — confirms the bottleneck is execution/dispatch capacity, not reserve-formula sophistication (see the "production/sell-side hypothesis refuted" update below) | 2026-08-22 "implemented + tested an ROI-payback purchase gate" |
| Additive pilkwang-style phase-window gates: late-game land cutoff (`land_min_days_left=12`), animal-purchase cutoff (`animal_purchase_last_day=18`), backlog-triggered "crisis" pause on new capital spending — his real numbers, layered on top of (not replacing) the existing gates | ❌ (third in a row, same fingerprint) | 6/8 wins, avg margin +12,960 — chaitanyajamble worse again, rajan1673 the only gainer again. Three structurally different mechanisms (flat cut, ROI check, phase windows) all regress the same way on this pool — the common factor is "restrict land/animal spending further," not the specific formula. Pilkwang's own numbers were tuned against his surrounding system (staged herd targets, per-quadrant crop mix); lifting just the day-window constants without that system is the same isolated-transplant failure as the reserve-formula copy | 2026-08-22 "implemented + tested pilkwang-style phase-window gates" |

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

**Update 2026-08-21 — CONFIRMED: kawa and prvsiyan are pre-solved fixed-route scripts, not live
strategies.** Read both agents' source directly (`src/opponents/kawa_route_agent.py`,
`_kawa_actions`, line 139; `src/opponents/prvsiyan_frontier_agent.py`, docstring literally says
"Soil modal route", reuses `_kawa_actions` under a renamed key). Both select and replay one of a
handful of giant hardcoded per-step action tables computed offline with full foresight of the
whole 30-day game, with only small live patches (weed detours, feed timing, market-crash-aware
sell ordering). This **retroactively explains every row in the table above** — a live reactive
heuristic cannot replicate a schedule solved with perfect foresight by tuning reserve thresholds
harder, no matter how many combinations get tried. Also reconfirmed the isolated `max_structures`
4→16 test still regresses even with today's dispatch/CARE/stale-money fixes in place (final money
roughly halved vs both), so this isn't an artifact of pre-fix dispatch either. **Reframe, not a
dead end**: stop chasing kawa/prvsiyan's exact animal/pasture curve specifically — it's the wrong
target for a live heuristic. Weight near-term comparisons toward the genuinely-live near-tier
opponents (`rajan1673_agent`, `chaitanyajamble_agent`) instead, and treat closing the kawa/prvsiyan
gap as a stretch goal, not a bug hunt. Separately: `best_config.json`'s `max_structures: 4` is
still a real, independent bug (below the day-0 starting pasture count of 7, so it silently blocks
*all* new pasture building for the entire game, confirmed via `actions.csv`) — worth fixing on its
own merits, just not expected to close the money gap by itself. See `LOG.md`.

---

## Dispatch / routing efficiency

The one area with actual verified wins, because these reduce waste rather than increase spending.

| Idea | Verdict | Why | LOG.md |
|---|---|---|---|
| **Batch multiple `HIRE` orders in one turn** (was: only ever queued one per turn, so reaching a 7-unit workforce took ~7 turns even though nothing enforced that) — matches pilkwang's own `while` loop, discovered by having agy read pilkwang's source + real match data | ✅ **biggest win of the whole animal-buying investigation** | chaitanyajamble avg margin -1,320 -> +717 (1/6 -> 4/6 wins, better in 6/6 episodes); daisy023 +12,631 -> +14,484 (still 6/6, better in 6/6); pilkwang -101,042 -> -95,068 (still unwinnable, net positive). Does NOT fix `in_crisis`/animal count (identical in every episode) — pure labor-throughput win, repeats every day since hands expire daily | 2026-08-23 "batch hiring" |
| Cross-turn target "stickiness" (don't re-litigate a target every turn) | ✅ | Cut movement waste 74.2% → 69.3% | 2026-08-21 "added cross-turn target stickiness" |
| Global nearest-pair batch matching (replace fixed-unit-order sequential greedy) | ✅ modest | ~15% money improvement in head-to-head trace, no regression — but doesn't close the larger gap alone | 2026-08-21 "rewrote _plan_units dispatch to global nearest-pair batch matching" |
| Spatial zoning (partition board into per-unit home regions) | **ruled out before building** | Measured water-tier assignment distance under the *existing* matcher: mean 1.62 tiles, 78% ≤2 — already near-optimal. Zoning can't reduce distance that's already minimal; the cost is trip *count*, not trip *distance* | 2026-08-21 "traced submission vs kawa_route_agent... found the real bottleneck is dispatch scaling" |
| Clustering-biased empty-tile placement (grow footprint as a contiguous block instead of pure nearest-available) | ➖ null | Implemented correctly (real `cost_fn` mechanism), but no measurable change in water-tier distance or match margin at `CLUSTER_BONUS=2.0` — likely too weak relative to typical distances, or swamped by run-to-run variance at this sample size. Kept the code, unproven | 2026-08-21 "clustering-biased empty-tile placement" |
| Move `CARE` from priority tier 12 to right after `FEED` (found via `actions.csv`: CARE was 0.9% of our unit-turns vs kawa's 4.6%, a 5.1x gap, the largest of any action category) | ✅ small | CARE's share rose to 1.5% (now tracking FEED's rate as intended), margins slightly better on both test opponents, no regression | 2026-08-21 "used the new actions.csv log to find and fix a real dispatch-priority gap" |
| `_market_orders` checked every spend gate against a stale `money` snapshot that never accounted for what earlier orders the same turn had committed — `BUY_ANIMAL` (last in the function) lost the race for cash most often | ✅ small | Real running balance + moved animal purchases ahead of hire/land — ~15% better vs prvsiyan in one check, flat vs kawa, no regression | 2026-08-21 "fixed a real stale-money bug" |
| Force-sell shed backlog past a quantity threshold regardless of price (to fix `MILK` piling up unsold once price craters) | ❌ | Shed inventory doesn't decay like crops do — forcing a sale at a crashed price permanently realizes a loss instead of waiting for the market's slow-but-real recovery. Reverted; `sell_backlog_multiple` config key stays inert for a smarter version later (e.g. only force-sell near an actual hard cap) | 2026-08-21 "reverted the sell_backlog_multiple force-sell fix" |

**Global per-job dispatch rewrite (`S_ij = priority_weight + value - travel_cost*distance`, directly matching pilkwang's own real dispatch formula) — implemented, tuned twice (once flawed, once correctly), REGRESSES both times. ❌ Closed.** Replaces the tier-exhaustion winner-take-all loop with a single unified per-turn scoring pool across all 9 non-critical task types (feed/care stay a separate fixed-priority pre-pass, matching pilkwang's own critical/non-critical split). First test (untuned, hand-guessed `travel_cost=3.0` + champion's tier-exhaustion-tuned `priority_weight_*` values): modest regression. First search attempt (40 trials): objective was a flat margin average across opponents, which pilkwang's -85k-to-118k swings dominated, making the search nearly blind to the two contestable matchups — inconclusive, not a real verdict. **Re-run with both flaws fixed** (`.claude/scratch/global_dispatch_rewrite/search_travel_cost_v2.py`): reused `optimize.py`'s own `_risk_adjusted_score`/tanh-squash objective (the exact fix for pilkwang-magnitude domination) and based the search on the newly `optimize.py`-promoted champion, not the stale pre-optimization one. **Still a clean, unambiguous regression** — chaitanyajamble collapses to 0W-15L (down from the champion's 30-53% range), pilkwang gets worse, daisy023's margin roughly halves. **This closes the idea for real, on both previously-unresolved counts.** Together with pocket-carry, real-options-gating, target-structures-reactive, and roi-delay-penalty (all also modeled directly on pieces of pilkwang's or other top competitors' real behavior), the consistent lesson: pilkwang's individual mechanisms do not transplant in isolation onto this reactive architecture — they were tuned as parts of one coherent, fully-scripted system. "Decide like pilkwang" would require building and tuning a comparable whole system, not porting pieces of it. 2026-08-26 LOG.md (two entries: the re-tested search results, and the earlier flawed-objective attempt).

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
| 2026-08-23 boolean-toggle removal refactor merged the ROI check and `animal_reserve_multiple` cushion into an always-both conjunction (previously either/or via `roi_gate_enabled`) — real bug, but measured to have **zero effect** on animal count (16-episode ablation, bit-identical results on one opponent) | Restored exclusivity via `roi_margin > 0` as the numeric switch | 2026-08-23 "fixed the roi/reserve double-gate bug, measured ZERO effect" |
| `target_structures` ignores `animal_purchase_last_day` entirely — keeps commanding new pasture/coop builds (day 23, day 25 observed in a real replay) 12+ days after the animal-purchase window closed, i.e. structures that can never hold an animal. **✅ Real fix, applied, verified positive** | Also zero `target_structures` once `day > animal_purchase_last_day` | 2026-08-23 "fixed wasted late-pasture builds" — better in 18/18 episodes, no exceptions |

**Prioritize build/animal infrastructure over planting (dispatcher-order flip) — DOES hit 4-6 animals, CRATERS score. ❌** Swapping `priority_weight_empty_build` and `priority_weight_empty_plant` (so structures/animals claim labor before crops each turn) unlocks 1 -> 6 animals/game, every episode — mechanically this IS "buy animals like top opponents." Measured across 18 episodes (chaitanyajamble/daisy023/pilkwang): every single one got dramatically worse (chaitanyajamble 18x worse margin, daisy023 flips from 6/6 wins to 0/6, pilkwang worse). **Follow-up, more surgical version also tested and also fails**: a `build_labor_reserve` knob (guarantees N idle units/turn to construction *before* the plant-race runs, rather than a full priority swap) was added and tested at 1-2 units — even the smallest dose (1 unit) is nearly as bad as the full swap (daisy023 flips 6/6 wins -> 0/6 at reserve=1 already). **Root cause is NOT dispatch-priority or construction speed — pasture builds barely changed (7->7) between reserve=0 and reserve=1.** It's that `feed`/`care` are the two highest-priority tiers in the whole system (weight 100/95, above every crop-maintenance tier), so every animal bought creates a permanent daily labor draw for the rest of the season, not a one-time cost — 5-6 animals' worth of feed/care taxes the same ~7-unit labor pool crops already saturate, for the whole remaining game. Optuna already searched the plant/build weight tradeoff and converged on plant-dominant for a reason. **Do not touch `priority_weight_empty_build`/`empty_plant` relative to each other, and do not add any mechanism that increases animal count without also solving the ongoing upkeep-labor cost** — this is now doubly-confirmed dead ground (2026-08-23 LOG.md, both entries).

**Raise the workforce ceiling itself (`max_hires_per_day`/`opening_hires_day0`, 6/2 -> 9/9), re-tested AFTER the batch-hiring fix — REGRESSES WORSE than the original pre-fix refutation. ❌** This closes off the natural "give the fixed-priority system enough labor that nothing has to starve" idea definitively. Result: chaitanyajamble +717 (4/6 wins) -> -35,775 (0/6); daisy023 +14,484 (6/6) -> -16,442 (0/6, flips to a clean loss); pilkwang -95,068 -> -110,736. Mechanism: more labor doesn't sit idle as spare capacity — it plants proportionally more tiles (so `in_crisis` still trips on day 0 regardless, exactly as before), while front-loading 3-6 *extra* animal purchases into the same brief pre-crisis window, each adding permanent FEED/CARE upkeep for the rest of the season. Bigger workforce = bigger commitment on every front, not slack to redirect. (2026-08-23 LOG.md, "re-tested raise the workforce ceiling post-batch-hire-fix")

**Throttle planting by labor capacity, and/or redesign `in_crisis` to match pilkwang's (dynamic, consecutive-neglect-only, animal-decoupled) — both REGRESS, RE-TESTED TWICE, both times. ❌** Hypothesis confirmed with real numbers first: our agent plants 24 tiles by step 20 (7 units) vs pilkwang's 10 (10 units), and routine morning watering backlog alone (24) exceeds the `in_crisis` threshold (17.5), tripping every morning. First test (2026-08-23, pre-batch-hire, 18 episodes): both a planting throttle and a pilkwang-style crisis redesign unlock 6-7 animals/game and regress (chaitanyajamble +716 -> as low as -22,877). **Re-tested 2026-08-26, AFTER batch hiring was applied, at N=20/opponent (not 4-6) given the same day's finding that small samples overstated chaitanyajamble**: still regresses, worse than before — chaitanyajamble 55% win rate/+141 -> **0% win rate/-19,327**; daisy023 100%/+13,877 -> **45%/-5,698** (flips a clean sweep into a losing record); pilkwang -91,954 -> -103,969. Animal count rose only modestly (1.0 -> 1.45-1.9 avg) and still devastated both working matchups. **This closes the `in_crisis` line of investigation for real — 5 structurally different approaches now tried (remove gate, throttle planting, raise workforce, redesign trigger, decouple by category), all regress.** The gate is load-bearing for the two matchups that currently make this agent competitive, not an arbitrary bug. Do not re-attempt loosening it without a fundamentally different mechanism than anything tried so far (2026-08-23 and 2026-08-26 LOG.md entries).

**Animal purchases stuck at 1/game — root cause found, but the fix REGRESSES. ❌** `in_crisis` (pauses new land/animal spend when backlog > `unit_count * crisis_backlog_ratio`) trips true partway through day 0 once hiring caps out, and never recovers for the rest of the game — the actual reason we buy ~1 animal vs 4-5 for top opponents. **Tried**: drop the `in_crisis` requirement from the animal-purchase gate only (land/hire untouched). **Confirmed**: this does unlock a 2nd animal, in 18/18 episodes — but the 2nd animal made the score *worse* in every one of 3 opponents tested (chaitanyajamble, daisy023, pilkwang), no exceptions. Verdict: our current architecture cannot make >1 animal pay for itself yet; top opponents' higher animal counts almost certainly come from a structurally different system (staged herd targets, dedicated dispatch capacity), not a looser gate that would work here too. **Do not chase animal count as a target** — see 2026-08-23 LOG.md entries for both the root-cause find and this ablation.

**ROI-delay-penalty purchase (let `animal_roi_ok`'s own math decide the batch size, penalized by an estimated placement delay, instead of hardcoding a count) — theoretically well-motivated (converged on by 4/5 independent investment-theory research passes: NPV/DCF, portfolio/profitability-index, Kelly sizing, course-toolkit gap analysis), still REGRESSES, and the reason is diagnosable. ❌** Implementation: `animal_roi_ok(..., placement_delay)` shrinks usable-days by an escalating delay (1x/2x/3x a config estimate) for each animal beyond current empty-pasture capacity; `market_orders()`'s hard "only buy if a pasture is already empty" gate replaced with a loop that keeps buying while delay-adjusted ROI clears the bar. Self-limiting mechanism works exactly as designed (a clean, consistent 3.00 avg animals bought, not 1 and not a hardcoded 4) — **but still regresses on every opponent** (N=15): chaitanyajamble 33.3%/-473 -> 13.3%/-1,790; daisy023 stays 15W-0L but margin drops 13,848 -> 12,088; pilkwang stays 0W-15L but margin worsens -81,224 -> -84,216. **Root cause traced, not just re-confirmed**: `target_structures` (agent.py, `plan_units()`) is a function of labor headcount and quadrant count ONLY — zero feedback from how many animals are actually sitting in the shed waiting for a pasture. A real chaitanyajamble trace showed only 1 of 3 purchased animals ever got placed through day 11; the other two were dead capital priced against a delay estimate with no mechanism guaranteeing it would ever resolve. **The ROI-math framing isn't wrong, but pricing a delay only helps if the delay is bounded by something real** — this variant assumed a finite wait without making `target_structures` reactive to the actual backlog it was pricing. If revisited: either make `target_structures` reactive to shed-pending animal count (make the delay estimate true), or don't loosen the gate past "pasture is a near-certainty" in the first place. 2026-08-26 LOG.md, "ROI-delay-penalty animal purchase".

**"Pocket carry" — batch-buy 4 animals in the crisis-free day-0 window (before any pasture exists), carry in unit inventory, place once pastures finish — deliberately avoids touching `in_crisis` at all. Mechanism works exactly as designed; still a severe REGRESSION. ❌** Directly modeled on kaito/boatlee/rayk's real decompiled action tapes, which batch-buy their full animal count in one turn-0 order before building any pasture, then carry and place on completion. Traced execution confirms the variant genuinely achieves this: ~4.1 animals bought and ~4.3 placed by day 11 (vs. champion's 1.0/1.0) — no implementation bug. **Result (N=15/opponent): every matchup got worse, including the previously-clean daisy023 sweep, which flips to a total loss** — daisy023 15W-0L(+13,803) -> 0W-15L(-9,385); chaitanyajamble 5W-10L(-981) -> 0W-15L(-27,965); pilkwang 0W-15L(-87,096) -> 0W-15L(-118,815, worse margin). This is the 7th structurally distinct approach to the animal-count problem to regress, and the first that deliberately sidesteps `in_crisis` entirely (buys during the pre-crisis window, only relies on `PLACE` — confirmed unblocked by the gate — happening later) — confirming the ceiling isn't specifically about the crisis gate's trigger logic, it's that the current dispatcher's early economy (day-0 cash, unit-turns) can't absorb a bigger animal commitment however it's timed. Top competitors' openings are one piece of a fully scripted, whole-system macro-schedule; transplanting just the purchase-timing piece onto a reactive priority dispatcher starves everything else. **Do not attempt another purchase-timing/sequencing variant of this idea** — a fix here would need to change dispatcher capacity/throughput itself (see "Dispatch / routing efficiency" below), not when/how the animal purchase is issued. 2026-08-26 LOG.md, "pocket carry opening".

**`animal_purchase_last_day`'s day-cutoff role is redundant with the ROI gate (`roi_margin`) for the animal-buy decision specifically — confirmed empirically, not just by math.** At the champion's tuned `roi_margin=2.27`, `animal_roi_ok()`'s own payback math already stops COW purchases by ~day 7-8 and SHEEP by ~day 4 — both earlier than the hard `animal_purchase_last_day=11` cutoff, so ROI is the binding constraint before the day-cutoff ever matters. A direct ablation (`animal_purchase_last_day=999`, N=15/opponent) confirmed this: `avg_animals_bought` was bit-identical (1.00 in both configs, every opponent/seed). **Not free to delete, though** — the same key also zeroes `target_structures` once `day > animal_purchase_last_day` (prevents building pastures too late to ever hold an animal, the fix from the row above this section). Setting it to 999 reopens that window too and showed a small chaitanyajamble dip consistent with wasted late-game pasture builds returning. If revisited, decouple the two uses first. 2026-08-26 LOG.md.

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

## Relative-wealth risk scaling (CPPI / tournament theory) — ➖ built, verified as a no-op so far

Since only win/loss matters, not final margin (`docs/GAME_GUIDE.md`), the theoretically right
target isn't "maximize expected money," it's "stay ahead of the opponent's total wealth." Two real
finance frameworks map onto this directly: **CPPI** (`cushion = value - floor`, `risky_allocation =
multiplier * cushion`, with the floor replaced by the opponent's total value) and **tournament
theory / "gambling for resurrection"** (mutual fund managers trailing their peer benchmark increase
risk to catch up; managers leading it reduce risk to lock in the win).

- **🐛 Naive version undercounts opponent wealth.** First pass valued the opponent's "unrealized"
  side using `_opponent_profile`'s existing `supply` (only already-ripe or near-ripe-within-
  `lookahead_days` yield) — meaning a cash-poor, asset-rich opponent mid-build-out (exactly kawa/
  prvsiyan's early game) looked artificially poor. Fixed with `_standing_asset_value`: values every
  standing crop at its full expected yield and every live animal at its purchase cost, regardless
  of growth stage. Confirmed via direct trace that opponent value is now visible from day 1, not
  just near-harvest.
- **❌ Symmetric version (loosens reserves when behind) regressed**, 8 episodes vs kawa/prvsiyan —
  same failure mode as every other "spend more when behind" attempt in the Spending-posture section
  above, for the same underlying reason (dispatch/execution capacity, not the trigger for spending).
- **➖ Asymmetric version (only tightens up when ahead, `risk_scale_min=1.0`) — looked like a win at
  4 episodes/opponent, evaporated at 10.** rajan1673 was already a clean 10W-0L at baseline; the
  mechanism made it 9W-1L. chaitanyajamble was an unchanged 2W-8L either way. **This is a
  methodology lesson as much as a mechanism result: a 4-episode/opponent sample is not enough to
  trust — it flipped a losing comparison into a "clean win" that fully reversed at 10 episodes.**
  Every other verified change this session used 6+ episodes; this one initially didn't.
- **Current state**: kept in the code (correctly implemented, verified to compute as designed) but
  `risk_sensitivity` defaults to `0.0` — a structural no-op — rather than a hand-picked value from
  the debunked small sample. Left in `optimize.py`'s search space for a real search with the
  dedicated multi-episode verification gate to find actual support, if any exists. See `LOG.md`.

## Shadow pricing (labor/land opportunity cost) — 🐛✅ land kept, ❌ "will make animals win" hypothesis

Prompted by an external second-opinion review (pasted into this session) recommending `lambda_labor`/
`lambda_land` shadow prices in `_crop_score`/`_animal_score`, on the theory that correctly pricing
labor scarcity would make animals (no watering, but real daily FEED+CARE) naturally outscore crops
without needing `animal_enabled` hardcoded.

- **✅ Land shadow pricing has real value.** A controlled Optuna search (`v10_shadow_pricing`,
  identical setup to the `v9` solo-optimization search below except shadow pricing is now live) had
  `lambda_land` converge to 24.9 out of an allowed [0, 80] range — a real, non-trivial value, not
  near a boundary by accident.
- **❌ The specific "animals will naturally win" hypothesis was not confirmed.** `lambda_labor`
  converged to 0.45 out of [0, 15] — the search, completely free to set this coefficient to whatever
  value would help, converged near-zero across every high-scoring trial shown (0.43-0.53 band). The
  best trial still has `animal_enabled: false`, same as the pre-shadow-pricing baseline. Verification
  did improve (v9's 2W-8L, avg -1,666.5 → v10's 4W-6L, avg +288.4) but not enough to promote.
- **Best current explanation**: shadow pricing only affects the *decision* of what to plant — it
  doesn't touch the deeper, already-established gap (see "Dispatch / routing efficiency" above) that
  the animal pipeline (build → buy → place → feed → care → collect, six separate dispatch-tier steps
  each competing independently for the same scarce labor) is mechanically more fragile to execute
  reliably than crops' simpler plant → water → harvest chain, regardless of how correctly the
  planting decision itself is priced. See `docs/tests/LOG.md` for both search results.

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
- ✅🐛 **2026-08-22 — the bullet directly above was finally acted on, having sat unactioned for a
  long time while every search in between optimized the wrong quantity.** `evaluate_config`
  scored risk-adjusted **raw coin margin**, but the rating system is win/loss/tie only. Since
  kawa (~-142k) and prvsiyan (~-97k) dominate the pool average by an order of magnitude over the
  contestable near-tier matchups (±6k..25k), "maximize average margin" mostly meant "lose
  slightly less lopsidedly to opponents we cannot beat", which earns zero rating. Worked example
  on a representative margin vector: improving kawa by 20k (still a loss) scored **8x better**
  than flipping a matchup to a genuine win. Fixed by squashing margins through
  `tanh(margin/win_scale)` (`--win-scale`, default 15000; `0` restores legacy behavior) so
  blowouts saturate and near-boundary matchups keep gradient. tanh rather than a hard win/loss
  indicator because at 2 episodes/opponent a 3-valued signal is too noisy for TPE and gives zero
  gradient on a matchup where every episode loses. **Deliberate trade-off:** the search now
  writes off kawa/prvsiyan entirely — correct for rating, but it means v12+ will not try to close
  that gap. Study renamed `v12_winrate_objective`; **prior studies' trial values are in different
  units and are not comparable.**
- ✅🐛 **Objective/gate mismatch (same fix, worth its own row).** `verify_candidate`'s promotion
  gate has always been win-based (`wins > losses`) while the objective was margin-based — **the
  search was optimizing a different quantity from the gate it had to pass.** This retroactively
  explains the recurring "search finds a best trial, then fails verification" pattern that runs
  through most of this file's Optuna rows.

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
- ✅🐛 **2026-08-22 — promotion gate had no out-of-sample check, and it cost us a real promotion.**
  `verify_candidate` only ever played candidate-vs-current-champion, so a config could win that
  head-to-head while genuinely regressing against the wider field. v11 did exactly that: 6W-4L
  (+1,942.7) against the champion, promoted, then found to regress and reverted. Fixed with
  `verify_holdout()` + `HOLDOUT_OPPONENTS = [chaitanyajamble, ektarr, rajan1673, nagatakengo]`,
  held out of **every** search pool (argparse now refuses a run that puts one in `--opponents`,
  since that silently destroys the out-of-sample property). Promotion now needs both gates.
  Validated by replaying v11's exact params through it: correctly blocked on `ektarr` (-6,875.8
  vs champion -2,825.5). The check asks *"is the candidate worse than the CHAMPION here"*, not
  *"does the candidate win here"* — several holdout opponents are losing matchups for the
  champion too, so demanding wins would reject everything. Same structural idea as the Halite IV
  4th-place solution, which evolves against a standing pool of baseline bots plus retained older
  genomes rather than against the current best alone (`../../input/HaliteIV_0Zeta/`).
- ⚠️ **Beware comparing a new result against *historical* numbers in LOG.md instead of a freshly
  measured baseline.** The v11 robustness check did this and concluded v11 had "doubled the
  deficit" vs `chaitanyajamble`; measured side-by-side against a concurrent champion, the two
  were within noise (-15,413 vs -14,666) — the gap was champion drift (below), not v11. The
  revert still stood on a different, real regression, but the stated reason was wrong. Always
  re-measure the champion in the same run as the candidate.

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

## Recurring bug class: a new mechanism's non-zero default silently mutates the champion

🐛 **Found 2026-08-22, and it had already corrupted a promotion decision.** `best_config.json`
only pins the keys a search actually wrote; every other key falls through to `DEFAULT_CONFIG`.
So adding a new mechanism with a *non-zero* default retroactively changes the champion — it stops
being the config that was actually verified. Shadow pricing and cash-flow lookahead shipped with
`lambda_labor=3.0`, `lambda_land=15.0`, `income_lookahead_days=2`, `income_discount=0.7` on the
reasoning "default to something nonzero so the mechanism does something to test against". That
reasoning is wrong. Measured cost to the champion (4 episodes each): vs `chaitanyajamble`
**-7,650.8 → -13,173.0**; vs `ektarr` **+173.0 → -2,317.2**. It also produced a false diagnosis
of v11's regression (see the evaluation-methodology section). All four defaults set to 0.0/0;
champion re-measured back in band (-6,755.0 / +683.0).

**Standing rule: a new mechanism defaults to a NO-OP, and the search turns it on.** Already
followed correctly by `opening_enabled` (False) and `broke_phase_*` (frac=0, scale=1). The
mechanism stays fully searchable in `KNOB_SPECS`/`sample_config` — only its default changes.
Any config key that gates behavior and isn't pinned in `best_config.json` is a live instance of
this bug class waiting to happen.

---

## Scripted opening window (`opening_*`) — ➖ built, NOT yet tested

Days 0-2 (steps 0-71) verified **100% deterministic** across episodes regardless of seed or
opponent — byte-identical observations, first divergence exactly at day 3 (shop-unlock RNG *and*
market prices together). Tracing five independently-written strong opponents in that window found
tight convergence: wheat+melon only (never carrot/tomato/strawberry), pasture+sheep/cow only
(never coop/geese), 4-5 hands hired on day 0, and $2,804-2,991 of the starting $3,000 spent by
end of day 2. Full per-opponent numbers in `../../input/index.json`.

Implemented as `opening_enabled` (default **False**, no-op) / `opening_days` /
`opening_reserve_scale` / `opening_hires_day0`: relaxes the animal-buy `started_up` gate and the
reserve cushions during the window, plus a day-0 hire target that bypasses the backlog gate
(backlog *cannot* justify a day-0 hire — nothing is planted yet). Deliberately **not** a copy of
any opponent's table: kawa's and prvsiyan's sources were read directly and both are absolute
per-step movement choreography for a fixed actor count, which desyncs if any quantity changes.

**Status (robust_agent-era `opening_*`): unverified, never tested** — `robust_agent` was deleted
before this got run. Honest prior at the time — this is still a "spend aggressively early"
variant, and those have failed 9+ times (see the first section). Phase separation itself is
independently corroborated by two reference winners (Halite IV's full second `EARLY_PARAMETERS`
set; Lux S2's `END_PHASE`/`ICE_MINE_RUSH` step thresholds), which is why it was built rather than
dismissed.

**Status (adaptive_agent-era, narrower re-test): ✅ real improvement, promoted — 2026-08-30.**
First re-measured our own agent's actual day-0-2 behavior instead of assuming a gap: hiring (8
hands day 0) and spend pace ($2,848 of $3,000 by end of day 2) already matched the strong-opponent
range, and animal placement was already COW/SHEEP-only — so this was NOT the "spend more
aggressively" failure pattern the 9+ prior attempts were. The one real divergence was crop
selection: we planted CARROT and STRAWBERRY during the opening alongside WHEAT/MELON, where the
documented strong-opponent convergence is wheat+melon only. Fix was narrower than the old
`opening_*` design — one condition on `_field_jobs`' PLANT job (`src/adaptive_agent/dispatch.py`):
while `day <= OPENING_RESTRICT_DAYS` (config key, set to 2), only WHEAT/MELON planting jobs are
generated; other crops defer to day 3+. No gate relaxation, no hire-target change, no reserve-scale
change. Verified via `compare_agents.py` — own money improved in 5/6 opponents, reactive-opponent
(pilkwang) margin flipped from -12,032 to +4,831, tool flagged the expected fixed-tape confound
(reduced overproduction stops crashing the shared market, so price-blind fixed sellers land a
better price without us doing worse). Full entry: `docs/tests/LOG.md`, 2026-08-30.

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
