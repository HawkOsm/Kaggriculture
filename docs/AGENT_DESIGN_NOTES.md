# Agent design notes

Design rationale, ablation history, and "why is this gate shaped like this"
context for `src/agent.py`, moved out of inline comments so the code itself
stays readable. Cross-reference: `docs/tests/LOG.md` (ablation results),
`docs/tests/IDEAS_TRIED.md` (closed lines of investigation),
`docs/COMPETITOR_STRATEGY_NOTES.md` (opponent research this config draws on).

Function names below match `src/agent.py` (no leading underscore — see that
file's own note on the 2026-08-26 rename).

## Module docstring / overall design

Builds on `opponents/melon_maxxer.py` / `opponents/multi_crop.py` with three
things they don't do: (1) never raises — every decision path is wrapped so a
bug degrades to a harmless `PASS` instead of an engine "Error" status
(presumably scored as a loss), (2) actually spends money — hires, land,
coops/pastures, animals, fertilizer, all previously-unused mechanics, and
(3) controls every unit (farmer + all hired hands) via a shared,
priority-ranked task queue instead of one farmer following one crop.

All knobs live in `DEFAULT_CONFIG` so `optimize.py`'s search harness can tune
them against the two things measurable offline: reward vs. a benchmark
agent, and reward vs. self (self-play).

The config below encodes a scaling-first strategy reverse-engineered from the
#1 leaderboard player's public replays (`docs/tests/LOG.md`): hire and expand
land aggressively from turn one, run a large pasture (COW/SHEEP) operation,
and tolerate near-zero cash early on the bet that the resulting labor/land
capacity compounds well before the 30-day season ends. The previous
Optuna-searched config converged on the opposite (`max_hires=1`, animals
off) because it was only ever tested against passive opponents that never
punish staying small — it lost a real ladder match to an opponent that
scaled hands 1→9 and land 1→3 quadrants while we sat on a single unexpanded
quadrant the whole game.

## DEFAULT_CONFIG rationale, by key

**`sell_fraction_base/day_weight/cash_weight`, `sell_fraction_min/max`,
`cash_scale`** — Sell threshold isn't one fixed number for the whole game;
it's a small linear function of two state features (days remaining, cash on
hand), computed fresh every turn by `dynamic_sell_fraction()`. Optuna
searches the 3 weights instead of one constant — this is the first piece of
a "hybrid RL" layer: everything mechanical (movement, task priority) stays
rule-based, but a few genuinely strategic scalars become state-dependent
policies whose coefficients get tuned the same way as any other config
value — derivative-free policy search, the same family (evolution
strategies / CMA-ES) real RL pipelines use when a full gradient-based policy
network isn't worth the engineering cost. Sign intuition: positive
`day_weight` = pickier early (more season left to wait for a better price),
positive `cash_weight` = pickier when already cash-flush, both push toward
`sell_fraction_min` when broke/late so cash-flow wins over holding out.
`sell_fraction_min/max` are a hard safety clamp, not searched. `cash_scale`
normalizes the cash-headroom feature (`money / cash_scale`, clamped to
`[0,1]`) against the game's actual price/cost scale rather than hardcoding
to the $3000 starting balance.

**`max_sell_chunk`** — caps per-crop sell size per turn so one order doesn't
crash its own price mid-sale (selling is one-unit-at-a-time and price moves
as it goes).

**`sell_backlog_multiple`** — meant to force a sale once shed backlog for one
item crosses this many sell chunks, regardless of price threshold, so a
premium item whose price has crashed doesn't accumulate in the shed forever.
**Currently unused** — the force-sell logic that read it was tried and
reverted (a 6-episode ablation found it strictly worse: unlike crops, shed
inventory doesn't decay while waiting, so a forced sale at a crashed price
permanently realizes a loss instead of waiting for the slow-but-real price
recovery). Kept in `DEFAULT_CONFIG`/`KNOB_SPECS` in case a smarter version of
the idea gets revisited, but deliberately **not** in `KNOB_SPECS`'s search
space right now — searching an inert knob just burns a dimension.

**`money_reserve`** — a small, FIXED survival floor for land/animal
purchases, never multiplied by `risk_scale`. Used to be the *dominant* term
in those gates (`remaining - money_reserve*scale >= cost*multiple*scale`,
champion-tuned to 420 — already more cash than the #1 player's entire bank
during the early-game window this was meant to emulate, $25-760), which
meant `risk_scale` could only ever nudge spending caution by ~11% either way
regardless of how lopsided the actual game state was — the flat floor
swallowed the signal. Reserve requirements are now purely proportional to
what's being bought (`animal_reserve_multiple` / `land_reserve_multiple`,
both scaled directly by `risk_scale`), so this key's only remaining job is
what `hire_money_floor`/`seed_money_floor` already do: stop the agent from
spending to literal $0 and getting stuck, not express a spending *policy*.
Lowered to match that narrower job; not yet re-verified at this magnitude.

**`broke_phase_days_frac`, `broke_phase_reserve_scale`** — optional "broke
phase": scale every spending reserve/gate down for the first
`broke_phase_days_frac` of the season, then back to normal. Deliberately
spend into the red early since only the final day's coin count is scored
(`docs/GAME_GUIDE.md`: margin doesn't matter, only win/loss), matching
traced top opponents (kawa, prvsiyan) who are visibly near-broke through day
6-8 before exploding from day 10+. Disabled by default (`frac=0`, `scale=1`
— no behavior change) since a flat "always aggressive" version of this
already regressed across the pool; this version is narrower — only the early
phase gets cheaper, not the whole game.

**`seed_money_floor`** — seed costs are small; allow buying seeds even close
to the reserve.

**`hire_money_floor`, `hire_reserve_multiple`** — hire cost is a fibonacci
curve that resets to $1 every day — even the 12th hire of a day only costs
$144. Gating it behind the same reserve used for $400-4000 land/animal
purchases created a poverty trap: once income dipped below `money_reserve`
for any reason, hiring (the thing that would fix low income) got blocked
too, for potentially days at a stretch. Hiring gets its own much smaller
floor instead, matching how the #1 player keeps hiring 9-12 hands/day while
running as low as $73.

**`max_hires_per_day`** — the #1 player sustains 8-12 hired hands/day almost
the whole game, but that assumes routing sophisticated enough to keep that
many hands productively busy across a wide board. Our simple
greedy-nearest-tile dispatcher isn't that: local testing at
`max_hires_per_day=12` left ~half of all unlocked land permanently empty
(units spend most turns walking, not working) and lost head-to-head to a
modest 1-hand baseline. 5 is a middle ground that empirically holds its own
— like most knobs here, this is exactly what `optimize.py` exists to tune
rather than hand-guess further.

**`coop_ratio`, `max_structures`, `pasture_target_ratio`** — coops/GOOSE are
deliberately near-unused: the #1 player's replays show a large pasture
(COW/SHEEP) operation and literally zero coops across two independent games.
`coop_ratio` (fraction of structures built as coops rather than pastures,
0.0 = never, 1.0 = parity) is continuous so the search can express "a few
coops" instead of only all-or-nothing. `max_structures` caps total
pasture+coop tiles (#1 player runs 14-18 pastures). `pasture_target_ratio`
sets target pasture count as a multiple of current unit count (farmer +
hands), capped at `max_structures` — builds pasture capacity roughly in step
with labor instead of all at once or only as leftover land; `BUILD_PASTURE`
itself costs nothing (only the animal does), so there's no cash reason to
delay it.

**`animal_reserve_multiple`** — post-purchase cash cushion required before
`BUY_ANIMAL`, as a multiple of *that animal's own cost* ($400-500), scaled
directly by `risk_scale` — gate is `remaining >= cost + max(money_reserve,
animal_reserve_multiple * risk_scale * cost)`. No longer stacks with a
separate flat buffer (see `money_reserve`) — a bigger animal purchase now
naturally demands a bigger absolute cushion without an independent flat
number tuned separately.

**`startup_days`, `land_startup_days`** — land is delayed a few days
(`land_startup_days`) so hiring/seed spending on day 0 doesn't compete with
it for the same thin starting cash pile. The #1 player's own land timing (NE
~day 5, SW ~day 10) is consistent with "buy it once affordable after a few
days of income," not day 0.

**`land_utilization_threshold`** — moderate (0.75), not the old 0.87, which
created a trap when labor was hard-capped at 1 hand (occupancy could never
rise to meet it). At realistic hand counts, 0.5+ is enough signal that the
current footprint is actually being worked before unlocking more board that
costs real travel time to reach.

**`land_reserve_multiple`** — post-purchase cash cushion before `BUY_LAND`,
as a multiple of that specific quadrant's own cost ($1000/$2000/$4000),
scaled by `risk_scale` — same shape as `animal_reserve_multiple`. Land
previously had no cost-scaled reserve of its own, just the flat
`money_reserve` buffer — a $4000 SE quadrant now demands a proportionally
bigger cushion than a $1000 NE one.

**`roi_margin`** — ROI gate: replaces `animal_reserve_multiple`/
`land_reserve_multiple`'s flat "hold N times the cost in cash" cushion with
an actual projected-payback check — only buy if the animal/quadrant is
expected to earn back at least `roi_margin` times its own cost before
wind-down, using the same $/day scoring (`animal_score`, `crop_score`)
already used to rank crops/animals elsewhere. This is the ROI-based idea
`money_reserve`'s own history already argued for (the flat multiple was
never a value judgment, just a survival-margin guess) — when the ROI check
is active (`roi_margin > 0`), the multiplier cushions above are skipped
entirely and only the flat `reserve_floor` is used (still need SOME
liquidity margin so a purchase can't spend to literal $0), so a purchase
that clears the ROI bar isn't also blocked by an unrelated cash-hoarding
requirement — see the note on the `roi_margin`/`animal_purchase_last_day`
interaction below.

**`land_min_days_left`, `animal_purchase_last_day`, `crisis_backlog_ratio`**
— phase-window gates, modeled directly on
`src/opponents/pilkwang_agent.py` (the highest-scoring genuinely-reactive
agent surveyed, see `docs/PARAMETER_MODEL_FINDINGS.md`) — additive
restrictions on top of the existing land/animal gates, not a replacement
(unlike the ROI gate / reserve multiples, which loosen spending and
regressed twice; these only ever say "don't buy here", the direction that's
had verified wins — see `docs/tests/IDEAS_TRIED.md`'s "Dispatch / routing
efficiency" section vs. its "Spending posture" section). Three gaps this
closes, each confirmed present in pilkwang's real source (not guessed): (1)
land has no late-game cutoff of its own — only the blanket `wind_down_days`
(3 days) applies, far too late for a $1000-4000 purchase to pay back;
pilkwang requires `days_left >= 12` (→ `land_min_days_left`). (2) animals
have no day-based cutoff independent of wind-down either; pilkwang
hard-stops new animal purchases after day 18 regardless of cash/pasture
availability (→ `animal_purchase_last_day`). (3) neither gate reacts to
whether the CURRENT footprint is already backlogged — pilkwang's CRISIS
phase pauses new land/animal capital spending whenever at-risk items exceed
available labor, so it never buys more than the dispatcher can service (the
exact failure mode behind every "spend more" regression in this project's
history) (→ `crisis_backlog_ratio`, feeds `in_crisis` in `market_orders()`).

  *2026-08-26 finding:* `animal_purchase_last_day`'s day-cutoff role turns
  out to be redundant with `roi_margin` under the champion's tuned values.
  Computing `animal_roi_ok()`'s natural cutoff at `roi_margin=2.27`: COW's
  ROI-only cutoff lands around day 7-8, SHEEP's around day 4 — both *earlier*
  than the hard `animal_purchase_last_day=11` cutoff, meaning ROI is already
  the binding constraint by the time the day-cutoff would matter. A direct
  15-seed×3-opponent ablation setting `animal_purchase_last_day=999`
  confirmed this: `avg_animals_bought` was bit-identical (1.00 in both
  configs, every opponent) — see `docs/tests/LOG.md`, 2026-08-26 entry.
  **But** the same key is also read in `plan_units()` to zero out
  `target_structures` once `day > animal_purchase_last_day` (prevents
  building pastures too late to ever hold an animal — the "wasted pasture"
  bug fixed earlier the same day). Setting it to 999 reopens *that* window
  too, and the ablation showed a small chaitanyajamble dip consistent with
  wasted late-game pasture builds returning. **Conclusion: the parameter
  is real and load-bearing for `target_structures`, but redundant for the
  `animal_phase_ok` animal-buy gate specifically. Don't delete it outright —
  decouple the two uses first if removing it from the animal-buy gate.**

**`fertilizer_stock_target`** — proactively buying fertilizer is usually not
worth it; collecting it free from animals is enough once any are running.
Buy fertilizer until the shed holds this many units; 0 = never buy.
Animal-collected fertilizer is free, so buying is a marginal spend the
search can dial rather than toggle.

**`hire_backlog_ratio`** — hire only if pending work (harvest/water/
feed/weeds/fertilize/empty tiles) exceeds this many tasks per current unit.
This is what actually produces the observed hiring ramp (5→9→11→12 hands
over the first ~8-10 days) without hardcoding it: on day 0 there's a full
unopened quadrant (25 tiles) of backlog to justify hiring toward ~25/ratio
hands, and backlog rises again each time `BUY_LAND` unlocks a fresh
quadrant. Raised from the old config's 1.2 (tuned against passive opponents,
produced a 1-hand agent) but still well below a value that would recreate
that trap.

**`diversification_weight`** — score penalty per tile already growing a
given crop, so units planting in the same turn spread across crops instead
of piling into whichever one currently scores highest (self-inflicted price
crash risk).

**`lambda_labor`, `lambda_land`** — shadow prices: how much a scarce unit of
labor/land is worth right now, priced directly into `crop_score`/
`animal_score` ($/day units, same as those functions' own output) rather
than left implicit. Motivated by an external second-opinion review plus a
real competitor's published formula for the land term specifically
(`docs/COMPETITOR_STRATEGY_NOTES.md`, pilkwang's `lambda_L * g_c`). Idea: a
crop that's cheap on paper but needs daily watering should score worse when
workers are already backlogged, and an animal (no watering, but real daily
FEED+CARE) should compete on the same footing instead of needing a hardcoded
toggle. `lambda_labor` multiplies a per-turn labor-scarcity signal (pending
tasks per unit, same backlog formula the hire gate uses); `lambda_land`
multiplies current land utilization (occupied/unlocked).

  Both default to 0.0 (no-op) — corrected 2026-08-22 after these shipped
  with nonzero defaults (3.0/15.0) "so the mechanism does something to test
  against." That reasoning was wrong and caused a real, measured regression:
  `best_config.json` doesn't pin these keys, so the CHAMPION silently
  inherited the nonzero defaults and stopped being the config that was
  actually verified 9W-1L. Measured cost to the champion (4 episodes each):
  chaitanyajamble -7,650.8 → -13,173.0, ektarr +173.0 → -2,317.2. It also
  corrupted a promotion decision: v11's apparent "regression vs
  chaitanyajamble" was mostly this drift, not v11. Rule going forward, same
  as `opening_days`/`broke_phase_*`: a new mechanism defaults to a no-op and
  the search turns it on; anything else silently mutates the champion. Note:
  the `season_ok` admissibility check in `crop_score` (a tile with no time
  left to mature is never plantable at all) is a hard constraint, unchanged
  by shadow pricing — it only affects crops that could be planted, not
  whether they're allowed to be.

**`income_lookahead_days`, `income_discount`** — cash-flow lookahead (see
`market_orders()`'s `guaranteed_income` note): how far ahead to scan our own
tiles for near-ripe/ripening yield that can offset the reserve cushion, and
how much to discount its priced value before trusting it (a scripted
opponent's projected income is certain; ours isn't — price risk and
execution risk both apply). Motivated by an external second-opinion review's
"MPC-lite" proposal; unverified, same status as `lambda_labor`/`lambda_land`.

**`opening_days`, `opening_reserve_scale`, `opening_hires_day0`** — scripted
opening window: days 0-2 (steps 0-71) are 100% deterministic regardless of
opponent or seed — confirmed empirically by diffing independent episodes'
full observation stream, byte-identical through day 2, first divergence
(both market prices AND town shop unlocks) exactly at day 3. Tracing 5
independently-written strong opponents (kawa, prvsiyan, boatlee_v16,
rayk_c95, saiteja — all separately confirmed fixed-route scripts) in this
window found a tight, convergent pattern: spend nearly the entire starting
$3000 by day 2, hire 4-5 hands immediately on day 0. Two concrete, isolated
gaps this closes without touching crop selection or reserve architecture
elsewhere: (1) `started_up`'s day-0 animal-buy block is a poor fit for a
window verified safe to spend aggressively in — see `opening_days`; (2) the
hire gate is backlog-triggered, which structurally can't fire hard on day 0
since no backlog exists before anything's been bought/planted yet — see
`opening_hires_day0`. Deliberately NOT a literal replay of any opponent's
action table: reading kawa's and prvsiyan's source directly shows both are
hardcoded per-step absolute movement choreography for a fixed actor count,
which does not survive generalization to different quantities (confirmed by
reading, not assumed) — these are reserve/gate relaxations layered on top of
the existing reactive dispatch/shadow-pricing engine, which still decides
tile-by-tile where units walk and what they plant. `opening_reserve_scale`
multiplies (not replaces) the existing risk-scale-adjusted cushion for
animal/hire/land purchases while `day < opening_days`; small rather than 0
so a purchase still leaves a trivial buffer, not because a tighter number is
known-unsafe. `opening_hires_day0` is a day-0-only hire target overriding
(not replacing) the normal backlog-triggered gate for the reason above,
still gated on affordability.

**`season_days`, `wind_down_days`** — total in-game days (720 turns / 24
turns-per-day); used to stop planting crops whose harvest cycle wouldn't
finish before season end. In the final `wind_down_days` days, stop
hiring/land/animal spending and sell shed inventory regardless of price —
reward is money at game end, so anything still sitting in the shed or any
hand hired too late to earn back its cost is pure waste. The #1 player
visibly winds crop mix back down to fast-cycle WHEAT only in the final ~5
days.

**`wealth_margin_scale`, `risk_sensitivity`, `risk_scale_min/max`** —
relative-wealth risk scale. RESTORED 2026-08-22 after a prior session
deleted it: `wealth_margin_scale` was the single highest-importance
parameter in the v14 search (importance 0.204) and a real promoted effect in
that champion, so deleting the mechanism discarded the strongest measured
signal available. Restored behind a toggle that defaults to a no-op
(`risk_sensitivity=0.0`) per the standing rule that a restored/new mechanism
starts inert until the search turns it on.

**`opponent_incoming_threshold`, `opponent_race_discount`,
`opponent_lookahead_days`, `opponent_concentration_sensitivity`** — if the
opponent's public tiles show a crop about to ripen in bulk, lower our own
sell-price bar so we sell into the current, still-healthy price instead of
after their dump craters it.
`opponent_concentration_sensitivity` makes the race discount opponent-
adaptive (their concentration in this item × their overall scale, both
0-1 — see `public_supply_forecast()`) instead of one fixed discount for
every opponent; an ablation found the single best static discount for one
opponent was a clear loss against a different one.

**`crash_sell_margin`, `crash_sell_multiplier`, `crash_sell_chunk`** —
item-holding race/crash mechanic, replacing the earlier total-wealth CPPI
risk-scale (removed 2026-08-22): that mechanism scaled every reserve gate by
*total* money+asset margin vs. the opponent, judged the wrong signal —
"opponent money based" rather than reacting to per-item exposure. Only
win/loss at game end is scored, not coin margin (`docs/GAME_GUIDE.md`), so
if the opponent is sitting on significantly more near-ripe/ripening supply
of an item than we are, deliberately dumping our own (smaller) holding early
can crash the price before their bigger dump lands — their loss on the
crashed price for their larger exposure plausibly exceeds our own loss on
our smaller one, a net gain in relative standing even though our own
absolute revenue on that item drops. Distinct from `opponent_race_discount`
(which only lowers our own sell-price bar so we sell before their crash —
purely defensive); this is the offensive version: cause a bigger crash than
we'd otherwise choose to. `crash_sell_margin` is the minimum raw unit gap
required to trigger a crash (avoids reacting to a trivial 1-2 unit
difference); `crash_sell_multiplier` requires the opponent's exposure to
also be at least this many times ours (combined with the margin so neither a
small ratio on a big pile nor a big ratio on a trivial pile alone triggers
it); `crash_sell_chunk` is the dump size once triggered, bypassing
`max_sell_chunk` and the normal price threshold — the point is to move the
price, not sell profitably this turn. OFF by default, unverified.

**`priority_weight_*`** — `plan_units()`'s dispatch tiers execute in
descending-weight order, sorted fresh each turn — a real number per tier
(flat keys, like every other tunable knob, so `KNOB_SPECS`/`optimize.py`'s
search space can pick each up without special-casing) instead of a
hardcoded sequence, so a search (Optuna or the RL policy) can find a better
ordering instead of it only changing when someone reads `actions.csv` and
hand-edits the function (which is exactly how CARE's starvation got fixed
the one time before this — CARE was tier 12 of ~16, chronically shut out).
Defaults reproduce the hand-tuned order this function used before this
change exactly, so nothing changes until the weights do: feed > care >
harvest > fertilize > collect_fertilizer > water > empty_coop_place >
empty_pasture_place > weeds > empty_build > empty_plant. Shed-pickup tiers
and the final drop/pass fallback aren't part of this — they're per-unit
prerequisites for a future turn, not tasks competing for this turn's labor
the same way these 11 are.

**`build_labor_reserve`** — guaranteed slice of each turn's leftover labor
pool (after the tier loop) given to `empty_build` BEFORE the
`empty_build`/`empty_plant` weight race runs on the rest — the race itself
is winner-take-all (whichever tier outranks the other gets every unit in the
pool that turn, not just more of them), so when planting demand alone can
absorb the whole pool, `empty_build` gets zero units, not fewer. Default 0
(no-op): reserved units, if there's no build work available for them, flow
back into the normal race unassigned. A full priority swap (build ranked
above plant entirely, not just a reserve) was tried instead of this small
reserve and craters the score — it starves crop revenue. See
`docs/tests/IDEAS_TRIED.md`.

**`cluster_shed_weight`** — penalty weight (per tile of distance to the
nearest shed tile) added to the tile-claim ranking for
`empty_build`/`empty_plant`. 0.0 default (unchanged behavior) means claims
are ranked purely by distance-to-unit and local-neighbor clustering; a real
trace analysis found occupied-tile distance from the shed drifts +3.57 tiles
on average from game start to end, so a positive value here is expected to
help, but that's not yet verified by an actual search/promotion.

## Scoring functions (`crop_score`, `animal_score`, `animal_roi_ok`,
`land_roi_ok`)

Rough $/tile/day estimates used to rank crops and animals against each other
using the *current* market price. Not exact — doesn't account for
fertilizer, watering misses, or market impact — just a ranking signal.

`animal_roi_ok`/`land_roi_ok` implement the ROI payback check described
under `roi_margin` above: only worth buying if the steady-state $/day rate
can earn back `roi_margin` × its own cost in the days actually left before
wind-down, accounting for the ramp-up to `first_yield_day` (an animal placed
today earns nothing until then). `land_roi_ok` prices a new quadrant at the
best $/day rate currently achievable across crops/animals (same scoring
functions), times `land_utilization_threshold` (the fraction of a new
quadrant already assumed to get worked, reused here rather than assuming
100% occupancy from day one).

## `plan_units()` — unit dispatch

Decides one action per unit (farmer + hands), consuming tiles from `info` so
two units never chase the same tile in the same turn. Two layers on top of
the naive "each unit runs its own priority ladder, grabbing whichever tile
is nearest to itself" design:

1. **Sticky continuation** — `unit_targets` (a `{unit_idx: (task_key,
   target)}` dict mutated in place across calls) keeps a unit already
   walking toward a target heading there instead of re-running the full
   priority ladder from scratch every turn. Without this, a unit several
   tiles into a walk toward (say) a water target gets pulled onto a
   newly-appeared, higher-priority-tier task the instant one shows up
   anywhere on the board, abandoning the walk and starting a new one.
2. **Batched global assignment within a tier** — within each priority tier
   that draws from a shared, contested tile list (feed/harvest/fertilize/
   collect_fertilizer/water/care/weeds/empty_coop/empty_pasture/empty),
   assignment is a greedy *global* nearest-pair match (`assign_nearest`)
   across all still-unassigned eligible units at once, not a fixed
   unit-processing-order sequential grab. Instrumentation showed most
   unit-turns in the water tier were pure travel (~90% not-yet-arrived) —
   confirming forced long walks from fixed processing order (farmer always
   resolved first regardless of position) were a real cost. Tiers without
   shared contention (shed pickups, drop, pass) stay per-unit.

`assign_nearest` repeatedly assigns the single lowest-cost (unit, tile) pair
across all of `pending` × `candidates`, instead of processing units in a
fixed array order and letting each grab whatever's nearest to *itself*
regardless of whether some other still-unassigned unit is actually closer to
that same tile. `continue_fn`, if given, is re-checked before every pair
pick and stops the tier early once it goes false (used by
`empty_build`/`empty_plant`, whose eligibility changes as the tier itself
assigns). `cost_fn(pos, target)`, if given, replaces plain Manhattan
distance (used by `empty_build`/`empty_plant` to prefer clustering new tiles
against the existing footprint over pure proximity).

**Target structure count** (`target_structures`) scales with current labor
AND unlocked land so pasture capacity builds out in step with land instead
of all at once, or (the bug this quadrant factor fixes) frozen forever once
the starting pasture count already meets a units-only target — with
`unit_count` roughly constant for most of the game, a units-only formula
never grows again after quadrant 1, so a 2nd/3rd quadrant's worth of new
land never gets any pasture, confirmed live via replay review (pasture flat
at its day-0 value the entire game). Uses *intended* labor scale
(`1 + max_hires_per_day`), not actual headcount — the original units-only
version capped the day-0 target near 1 regardless of `max_structures`, since
there's only ever a lone farmer before any hire has landed, even though
building is free (only the animal costs money) so there's no cash reason to
delay it.

**Found 2026-08-23 (audit):** a unit sticky-walking toward a plant target
committed to on an earlier turn (when some crop was still viable) can arrive
to find every crop's shadow-priced score has since gone negative
(labor/land scarcity rose during the walk). Without releasing the unit back
to `pending` for a fresh assignment that same turn, the action silently
stayed `None` and got committed as a wasted turn — the engine no-ops a
`None` action, so the failure was invisible except as elevated idle turns.
Fixed by releasing the unit back to `pending` instead.

**New-tile claiming** uses a clustering-biased cost (`cluster_cost`) instead
of pure distance-to-unit: prefer tiles touching the existing worked
footprint (or another tile claimed earlier this same turn) over an equally
close tile that would start a new, disconnected patch. Distance-to-unit
alone already turned out to be near-optimal (measured mean 1.62 tiles per
water-tier assignment) — the actual cost isn't long walks, it's that *every*
tile still needs its own multi-turn trip regardless of how short. A tight,
contiguous footprint turns a day's worth of water/harvest/feed visits into
one short sweep instead of many separate short hops scattered across the
board.

## `market_orders()` — market spending

**Ordering matters.** Every gate used to check the SAME `money` snapshot
read at the top of this function, never accounting for what earlier orders
THIS SAME TURN had already committed to spend — so on any turn where land +
hire + an animal were each individually affordable but not all three
combined, all three got queued anyway, and whichever the engine processes
last silently lost the race for cash (orders are "processed in order, per
player" — `docs/GAME_GUIDE.md`). Since `BUY_ANIMAL` used to be the last
block in this function, it was the most likely to get squeezed out — a real,
previously-undiagnosed contributor to animals growing much more slowly than
the strong opponents'. `remaining` now tracks an honest running balance
across every order queued in the function; sell proceeds aren't added
speculatively (same-turn availability isn't guaranteed), only spends are
subtracted, so this is a conservative fix, not an optimistic one. Animal
purchases are placed near the top of the order (ahead of hire/land) so they
get first claim on `remaining`, matching replay review showing strong
opponents hyper-optimize the *current* footprint with animals rather than
treating land expansion as the priority.

**Emergency wheat buy** (only when wheat inventory is truly zero everywhere
— shed and carried — so it can't re-trigger just because a unit is
mid-transit with a full load) is deliberately queued FIRST, ahead of
seeds/animals/hire/land: market order position is itself a contested
resource (an opponent's own order can drain shared WHEAT price/stock before
a feed purchase queued behind other spending lands) — matches a documented
competitor bug pattern (Rayk Kretzschmar, `docs/COMPETITOR_STRATEGY_NOTES.md`).
Ablation (6 seeds × 3 opponents): no measurable difference for us
specifically (we only ever run 1 animal, so feed demand is tiny), but zero
regression and correct in principle — kept, same precedent as the ROI/
reserve double-gate fix below.

**Double-gate bug (fixed 2026-08-23):** when `roi_margin > 0` (the ROI check
is actually active, not a no-op), only the cheap `reserve_floor` cushion is
needed on top of it — stacking the full `animal_reserve_multiple`/
`land_reserve_multiple` cushion on an already-active ROI check double-gates
every purchase. This was an unintended side effect of the `*_enabled`
boolean removal, which had merged what used to be two exclusive branches.
Every animal/land cushion call site branches on `roi_margin > 0` for this
reason.

**Hire batching:** the engine allows up to `maxMarketOrdersPerTurn` (10)
market orders per turn, and pilkwang's own source batches multiple `HIRE`
orders in the same turn (a `while` loop) to reach a ~10-unit workforce by
step 2 of day 0. Matched here 2026-08-23 — the previous single-hire-per-turn
code took ~7 turns to reach 7 units; this was the single biggest measured
win of that day's work. Backlog can't justify a hire on day 0 (nothing's
been bought or planted yet, so there IS no backlog regardless of how many
hands would actually be useful once the day-0 spending spree lands) — this
is the one gate the opening window overrides by target count
(`opening_hires_day0`) rather than by relaxing a reserve multiplier, still
subject to the same affordability check as any other hire.

**Land timing:** buy the next quadrant as soon as affordable, rather than
waiting for high occupancy of the current footprint — occupancy is a
lagging, circular signal; a thin labor force can't reach high utilization in
the first place, so an occupancy-gated trigger can permanently starve itself
of the very land expansion that would fix the labor shortage. The #1 player
buys NE by day 5 and SW by day 10 on a near-fixed schedule that falls
straight out of "buy it the moment you can afford it" given their income
curve, not out of watching occupancy. But land is only worth buying once the
crew can actually walk to and work it — a wider board means more travel time
per action, so unlocking a 4th quadrant while half the first three still sit
empty just creates more unreachable dead land, not more income (confirmed
locally: an earlier version that bought on affordability alone ended up with
~half of all unlocked tiles permanently empty). Hence still requiring decent
utilization of the current footprint (`land_utilization_threshold`),
matching land growth to demonstrated crew capacity instead of pure cash
availability. `land_cushion` uses `opening_scale` (not plain `scale`) — found
2026-08-22: every other cushion (animal, hire) already got the
opening-window discount, land was the one missed.

**Guaranteed income / cash-flow lookahead:** a reactive agent needs a cash
cushion because it can't otherwise distinguish "$0 in the bank, empty soil"
from "$0 in the bank, $600 of tomatoes ripening tomorrow" — both look
equally risky by liquid cash alone. `our_supply` is the same near-ripe/
ripening-within-lookahead tile scan `public_supply_forecast()` does for the
opponent-awareness mechanism, reused on our own farm (it's a generic
function over "a farm dict", not opponent-specific). Discounted for two real
risks a scripted opponent with perfect foresight doesn't have to price in:
the sale might land at a worse price than today's quote, and (unlike a
precomputed route) there's no guarantee the projected labor actually arrives
to harvest/sell it on schedule. Guaranteed income can only offset the safety
cushion, never let a purchase proceed without covering its own cost out of
liquid cash — that half of the gate (`remaining >= cost`) is untouched,
only the `+ cushion` term shrinks.

**Wind-down:** reward is money at game end, full stop — unsold shed
inventory and freshly-hired hands with no time left to earn back their cost
are pure waste in the closing days. The #1 player visibly winds crop mix
back down in the final ~5 days rather than planting things that won't
mature; applied here to market orders too: stop paying for anything that
can't pay itself back, dump inventory for whatever it fetches instead of
holding out for a better price that will never be realized.

**Crisis gate:** pauses new land/animal capital spending (not hiring — more
hands is the actual fix for this) whenever pending work already exceeds
current labor, same backlog formula the HIRE gate uses, so a turn already
drowning in unfed/unwatered/unharvested tiles doesn't also buy more capacity
the dispatcher has no chance of servicing this season. **Status as of
2026-08-26: this gate is real and correctly diagnosing genuine capacity
saturation, not a bug** — a 90-episode traced check found our maxed 7-unit
crew's daily backlog bottoms out at ~21-23 every day for the first 11 days,
above even the tuned champion's real threshold (7 × 2.177 ≈ 15.2). Six
different redesigns/removals of this gate were tried across two sessions and
all six regressed; this is closed — see `docs/tests/IDEAS_TRIED.md`. The
live lever going forward is dispatcher throughput (see `plan_units()` notes
above and the pocket-carry/walking-efficiency investigations in
`docs/tests/LOG.md`), not this gate's math.

**Selling:** sell everything sellable above threshold in bounded chunks. If
the opponent's about to dump a lot of an item on the market (visible from
their public tiles), lower our own bar so we sell into the current,
still-healthy price instead of after their sale craters it. Offensive crash
sell (see `crash_sell_*` above) bypasses the normal threshold/chunk size
entirely when the opponent's exposure clearly exceeds ours, both in absolute
gap and ratio, to move the price rather than sell well. In wind-down, ignore
the threshold entirely.

## `public_supply_forecast()`

Everything legally visible about a farm's public tiles/land/labor — never a
shed or seed counts, which stay private — reduced to three things worth
reacting to. Called on the OPPONENT's farm (with `opponent_lookahead_days`)
to forecast their incoming market dumps, and on OUR OWN farm (with
`income_lookahead_days`) to forecast our own near-ripe income for the
cash-flow cushion in `market_orders()` — same tile-walk math either way,
just pointed at a different farm dict with a different lookahead knob.
`opponent_farm` is the parameter name because that was this function's
original, narrower use case (renamed from `opponent_profile` 2026-08-26);
pass whichever farm dict you want a forecast for.

- `supply`: how much of each product they're likely to dump on the market
  soon (already-ripe, or a one-time crop maturing within `lookahead_days`).
- `concentration`: `{item: fraction of occupied tiles producing it}`. An
  opponent running 80% wheat has a much bigger relative stake in wheat's
  price than one running an even 5-crop split.
- `scale`: 0-1, how big their whole operation currently is (land + labor),
  independent of what they're growing — normalized against the range this
  project's traced strong opponents actually reach at their peak (3-4
  quadrants, 12-14 hands), not an arbitrary guess.

## `standing_asset_value()`

Estimates the market value of everything currently growing/held on a farm,
priced at current rates — the "unrealized gains" half of total wealth for
the relative-wealth risk scale. Deliberately broader than
`public_supply_forecast()`'s `supply` (which only counts already-ripe or
near-ripe-within-lookahead output, tuned for "what's about to hit the
market"): every standing crop counts at its full expected yield value
regardless of growth stage, and every live animal counts at its purchase
cost (the sunk capital already committed) plus anything it's already
produced. A freshly-planted crop or a just-bought animal is real future
money even though it contributes nothing to `supply` — valuing it at $0 (the
original version) meant an opponent mid-way through a heavy build-out phase
(near-broke on cash, but sitting on a large just-planted/just-stocked
position) looked artificially poor and the mechanism reacted to their threat
a turn too late.

## RL policy hook (`extract_features`, `decode_knobs`, `KNOB_SPECS`,
`FEATURE_NAMES`)

A trained network can re-decide strategic knobs once per day (not per turn —
30 decisions/game is tractable for RL, 720 isn't, and none of these need
finer granularity than a day). Everything mechanical (movement, task
priority, harvesting) stays the proven rule-based code completely untouched
— the network only ever outputs values merged into `cfg` and read by the
same functions Optuna already tunes. See `train_rl.py` for the training
side; `agent.py` stays torch-free on purpose so the submission build never
needs torch.

`KNOB_SPECS` entries are `(key, low, high, is_int)`, ranges matching
`optimize.py`'s Optuna search space so a trained policy and a searched
static config are directly comparable. `priority_weight_*` entries are
generated from `PRIORITY_TIER_NAMES` so this list can't silently drift out
of sync with what `plan_units()` actually reads.

`FEATURE_NAMES` is kept small and hand-picked (not the raw board) since the
mechanical layer already turns the board into a handful of meaningful
numbers every turn (`scan_farm`) — reusing that instead of learning grid
perception from scratch is what makes this tractable to train on a single
GPU in hours, not days. `RL_MONEY_SCALE` (50000.0) is deliberately separate
from `config["cash_scale"]` (which normalizes OUR OWN sell-aggressiveness in
`dynamic_sell_fraction`, tuned around a few thousand) — reusing it here
saturated `cash_frac`/`opp_cash_frac` at 1.0 well before day 10 against real
strong opponents (kawa reaches ~$15k by day 12, $150k+ by day 29), making the
RL policy unable to tell "modestly ahead/behind" from "astronomically
ahead/behind" for most of any competitive game.

## 2026-08-26 renames

Every previously-underscore-prefixed module function (`_crop_cycle_days`,
`_crop_score`, `_diversified_crop_score`, `_animal_score`, `_animal_roi_ok`,
`_land_roi_ok`, `_base_price`, `_dynamic_sell_fraction`,
`_plant_harvest_ready`, `_scan_farm`, `_assign_nearest`, `_plan_units`,
`_carried_total`, `_public_supply_forecast`, `_standing_asset_value`,
`_market_orders`) had its leading underscore dropped. They were never
actually private in the Python sense (`build_submission.py` extracts them by
name into `submission/main.py`, `train_rl.py` imports several directly), so
the underscore was cosmetic-only; dropping it removes noise without changing
any behavior. `src/build_submission.py` and `src/train_rl.py` were updated
to match. The local variable that used to be named `market_orders` (holding
the *result* of calling `_market_orders(...)` in `make_agent`'s per-turn
closure) was renamed to `market_order_list` to avoid colliding with the
now-unprefixed function name of the same call.
