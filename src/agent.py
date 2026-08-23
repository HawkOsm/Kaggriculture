"""Kaggriculture robust, multi-unit, parameterized economic agent.

Builds on opponents/melon_maxxer.py / opponents/multi_crop.py with three things they don't do:

  1. Never raises. Every decision path is wrapped so a bug degrades to a
     harmless PASS instead of an engine "Error" status (which is presumably
     scored as a loss on the ladder).
  2. Actually spends money: hires farm hands, buys land, builds coops/
     pastures and buys+places animals, and fertilizes -- all previously
     unused mechanics.
  3. Controls every unit (farmer + all hired hands) via a shared,
     priority-ranked task queue instead of one farmer following one crop.

All the knobs live in DEFAULT_CONFIG so a local search harness (see
optimize.py) can tune them against the two things we can actually measure
offline: reward vs. a benchmark agent, and reward vs. self (self-play).
"""

import math
import sys

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    LAND_PRICES,
    MARKET_PARAMS,
)

from farm_utils import act_or_move, closest, shed_tiles

# Config below encodes a scaling-first strategy reverse-engineered from the
# #1 leaderboard player's public replays (see docs/tests/LOG.md): hire and
# expand land aggressively from turn one, run a large pasture (COW/SHEEP)
# operation, and tolerate near-zero cash early on the bet that the resulting
# labor/land capacity compounds well before the 30-day season ends. Our
# previous Optuna-searched config converged on the opposite (max_hires=1,
# animals off) because it was only ever tested against passive opponents
# that never punish staying small -- it lost a real ladder match to an
# opponent that scaled hands 1->9 and land 1->3 quadrants while we sat on a
# single unexpanded quadrant the whole game.
DEFAULT_CONFIG = {
    # Sell threshold is no longer one fixed number for the whole game -- it's
    # a small linear function of two state features (days remaining, cash on
    # hand), computed fresh every turn by _dynamic_sell_fraction(). Optuna
    # searches the 3 weights below instead of one constant, which is the
    # first piece of the "hybrid RL" layer: everything mechanical (movement,
    # task priority) stays rule-based, but a few genuinely strategic scalars
    # become state-dependent policies whose coefficients get tuned the same
    # way as any other config value -- derivative-free policy search, the
    # same family (evolution strategies / CMA-ES) real RL pipelines use when
    # a full gradient-based policy network isn't worth the engineering cost.
    # Intuition for the signs Optuna will actually search: positive
    # day_weight = pickier early (more season left to wait for a better
    # price), positive cash_weight = pickier when already cash-flush (can
    # afford to hold out), and both push toward sell_fraction_min when
    # broke/late so cash-flow wins over holding out.
    "sell_fraction_base": 0.5,
    "sell_fraction_day_weight": 0.1,
    "sell_fraction_cash_weight": 0.1,
    # Hard floor/ceiling regardless of what the linear policy computes --
    # not searched, just a safety clamp so a bad coefficient combo can't
    # produce a nonsensical (negative or >1) threshold.
    "sell_fraction_min": 0.15,
    "sell_fraction_max": 0.9,
    # Reference cash level the cash-headroom feature is normalized against
    # (money / cash_scale, clamped to [0, 1]) -- how much money "counts as
    # flush" depends on the game's actual price/cost scale, so this is
    # tunable rather than hardcoded to the $3000 starting balance.
    "cash_scale": 1500,
    # Cap per-crop sell size per turn so one order doesn't crash its own
    # price mid-sale (selling is one-unit-at-a-time and price moves as it
    # goes).
    "max_sell_chunk": 10,
    # Force a sale once shed backlog for one item crosses this many sell
    # chunks, regardless of the price threshold below -- otherwise a
    # premium item whose price has crashed (from our own selling or the
    # shared market) never clears a price-only gate again and just
    # accumulates in the shed for the rest of the game (see
    # docs/tests/LOG.md). Some revenue at a depressed price beats zero.
    "sell_backlog_multiple": 2.5,
    # A small, FIXED survival floor for land/animal purchases -- never
    # multiplied by risk_scale. This used to be the *dominant* term in
    # those gates (`remaining - money_reserve*scale >= cost*multiple*scale`,
    # with a champion-tuned value of 420 -- already more cash than the
    # reigning #1 player's entire bank during the early-game window this
    # was meant to emulate, $25-760), which meant risk_scale could only
    # ever nudge spending caution by its own narrow bound (~11% either way)
    # regardless of how lopsided the actual game state was -- the flat
    # floor swallowed the signal. Reserve requirements are now purely
    # proportional to what's being bought (see animal_reserve_multiple /
    # land_reserve_multiple below, both scaled directly by risk_scale), so
    # this key's only remaining job is what hire_money_floor/
    # seed_money_floor already do: stop the agent from spending down to
    # literal $0 and getting stuck, not express a spending *policy*. Value
    # lowered to match that narrower job -- not yet re-verified at this
    # magnitude, expect this to need a fresh search (see docs/tests/LOG.md).
    "money_reserve": 40,
    # See _market_orders' phase_scale: disabled by default (frac=0 means
    # "day < 0" is never true, scale is never applied).
    "broke_phase_days_frac": 0.0,
    "broke_phase_reserve_scale": 1.0,
    # Seed costs are small; allow buying seeds even close to the reserve.
    "seed_money_floor": 20,
    # Hire cost is a fibonacci curve that resets to $1 every day -- even the
    # 12th hire of a day only costs $144. Gating it behind the same reserve
    # used for $400-4000 land/animal purchases created a poverty trap: once
    # income dipped below money_reserve for any reason, hiring (the thing
    # that would fix low income) got blocked too, for potentially days at a
    # stretch. Hiring gets its own much smaller floor instead, matching how
    # the #1 player keeps hiring 9-12 hands/day while running as low as $73.
    "hire_money_floor": 10,
    "hire_reserve_multiple": 2.0,
    # #1 player sustains 8-12 hired hands/day almost the whole game, but
    # that assumes routing sophisticated enough to keep that many hands
    # productively busy across a wide board. Our simple greedy-nearest-tile
    # dispatcher isn't that: local testing at max_hires_per_day=12 left
    # ~half of all unlocked land permanently empty (units spend most turns
    # walking, not working) and lost head-to-head to a modest 1-hand
    # baseline. 5 is a middle ground that empirically holds its own; this
    # (like most of the knobs below) is exactly what optimize.py exists to
    # tune properly instead of hand-guessing further -- see docs/tests/LOG.md.
    "max_hires_per_day": 5,
    "animal_enabled": True,
    # Coops/GOOSE are deliberately unused -- the #1 player's replays show a
    # large pasture (COW/SHEEP) operation and literally zero coops across
    # two independent games. Kept as a switch in case future data disagrees.
    "enable_coop": False,
    # Ceiling on total pasture+coop tiles; #1 player runs 14-18 pastures.
    "max_structures": 16,
    # Target pasture count as a multiple of current unit count (farmer +
    # hands), capped at max_structures -- builds out pasture capacity
    # roughly in step with labor instead of all at once or only as leftover
    # land. BUILD_PASTURE itself costs nothing (only the animal does), so
    # there's no cash reason to delay it.
    "pasture_target_ratio": 0.3,
    # Post-purchase cash cushion required before BUY_ANIMAL, as a multiple
    # of *this specific animal's own cost* ($400-500), scaled directly by
    # risk_scale -- gate is `remaining >= cost + max(money_reserve,
    # animal_reserve_multiple * risk_scale * cost)`. No longer stacks with
    # a separate flat buffer (see money_reserve's comment) -- a bigger
    # animal purchase now naturally demands a bigger absolute cushion
    # without needing an independent flat number tuned separately.
    "animal_reserve_multiple": 2.5,
    "startup_days": 2,
    # Land is different: delay it a few days so hiring/seed spending on day
    # 0 doesn't compete with it for the same thin starting cash pile. The
    # #1 player's own land timing (NE ~day 5, SW ~day 10) is consistent
    # with "buy it once affordable after a few days of income," not day 0.
    "land_startup_days": 4,
    # Moderate, not the old 0.87 -- that value created a trap when labor
    # was hard-capped at 1 hand (occupancy could never rise to meet it).
    # At realistic hand counts, 0.5 is enough signal that the current
    # footprint is actually being worked before unlocking more of a board
    # that costs real travel time to reach.
    "land_utilization_threshold": 0.75,
    # Post-purchase cash cushion required before BUY_LAND, as a multiple of
    # the *specific quadrant's own cost* ($1000/$2000/$4000), scaled by
    # risk_scale -- same shape as animal_reserve_multiple, and new: land
    # previously had no cost-scaled reserve of its own at all, just the
    # flat money_reserve buffer directly (see that key's comment). A $4000
    # SE quadrant now demands a proportionally bigger cushion than a $1000
    # NE one, instead of the same flat number regardless of purchase size.
    "land_reserve_multiple": 1.5,
    # ROI gate: replaces animal_reserve_multiple/land_reserve_multiple's flat
    # "hold N times the cost in cash" cushion with an actual projected-payback
    # check -- only buy if the animal/quadrant is expected to earn back at
    # least roi_margin times its own cost before wind-down, using the same
    # $/day scoring (_animal_score, _crop_score) already used to rank
    # crops/animals elsewhere. This is the thing money_reserve's own comment
    # already argued for (the flat multiple was never a value judgment, just
    # a survival-margin guess) -- when enabled, the multiplier cushions above
    # are skipped entirely and only the flat reserve_floor is used (still
    # need SOME liquidity margin so a purchase can't spend to literal $0),
    # so a purchase that clears the ROI bar isn't also blocked by an
    # unrelated cash-hoarding requirement. OFF by default (no-op) per this
    # project's rule that a new mechanism defaults inert until a real search
    # verifies it -- NOT yet verified.
    "roi_gate_enabled": False,
    "roi_margin": 1.5,
    # Phase-window gates, modeled directly on src/opponents/pilkwang_agent.py
    # (the highest-scoring genuinely-reactive agent surveyed, see
    # docs/PARAMETER_MODEL_FINDINGS.md) -- ADDITIVE restrictions on top of
    # the existing land/animal gates, not a replacement (unlike
    # roi_gate_enabled/the reserve multiples, which loosen spending and
    # regressed twice; these only ever say "don't buy here", the one
    # direction that's actually had verified wins in this project's history
    # -- see docs/tests/IDEAS_TRIED.md's "Dispatch / routing efficiency"
    # section vs. its "Spending posture" section). Three gaps this closes,
    # each confirmed present in pilkwang's real source (not guessed):
    # (1) land has no late-game cutoff of its own -- only the blanket
    # wind_down_days (3 days) applies, far too late for a $1000-4000
    # purchase to ever pay back; pilkwang requires `days_left >= 12`.
    # (2) animals have no day-based cutoff independent of wind-down either;
    # pilkwang hard-stops new animal purchases after day 18
    # (ANIMAL_PURCHASE_LAST_DAY) regardless of cash/pasture availability.
    # (3) neither gate reacts to whether the CURRENT footprint is already
    # backlogged -- pilkwang's CRISIS phase pauses new land/animal capital
    # spending whenever at-risk items exceed available labor, so it never
    # buys more than the dispatcher can service (the exact failure mode
    # behind every "spend more" regression this session, and this session's
    # own roi_gate_enabled/land_reserve_multiple experiments). OFF by
    # default (no-op) per this project's rule -- NOT yet verified.
    "phase_gate_enabled": False,
    "land_min_days_left": 12,
    "animal_purchase_last_day": 18,
    "crisis_backlog_ratio": 2.5,
    # Proactively buying fertilizer is usually not worth it -- collecting it
    # free from animals is enough once any are running. Off by default.
    "buy_fertilizer": False,
    # Which crops we're willing to grow at all; scoring picks among these.
    "crops": list(CROPS.keys()),
    # Hire only if pending work (harvest/water/feed/weeds/fertilize/empty
    # tiles) exceeds this many tasks per current unit. This is what actually
    # produces the observed hiring ramp (5 -> 9 -> 11 -> 12 hands over the
    # first ~8-10 days) without hardcoding it: on day 0 there's a full
    # unopened quadrant (25 tiles) of backlog to justify hiring toward
    # ~25/ratio hands, and backlog rises again each time BUY_LAND unlocks a
    # fresh quadrant, justifying hiring further. Raised from the old
    # config's 1.2 (which was tuned against passive opponents and produced
    # a 1-hand agent) but still well below a value that would recreate that
    # trap.
    "hire_backlog_ratio": 1.5,
    # Score penalty per tile already growing a given crop, so units planting
    # in the same turn spread across crops instead of piling into whichever
    # one currently scores highest (self-inflicted price crash risk).
    "diversification_weight": 0.15,
    # Shadow prices: how much a scarce unit of labor / land is worth right
    # now, priced directly into _crop_score/_animal_score ($/day units,
    # same as those functions' own output) rather than left implicit.
    # Motivated by an external second-opinion review (see docs/tests/LOG.md)
    # plus a real competitor's published formula for the land term
    # specifically (docs/COMPETITOR_STRATEGY_NOTES.md, pilkwang's
    # `lambda_L * g_c`). The idea: a crop that's cheap on paper but needs
    # daily watering should score worse when workers are already backlogged,
    # and an animal (no watering, but real daily FEED+CARE) should compete
    # on the same footing instead of needing `animal_enabled` hardcoded on.
    # lambda_labor multiplies a per-turn labor-scarcity signal (pending
    # tasks per unit, same backlog formula _market_orders already uses for
    # the hire gate); lambda_land multiplies current land utilization
    # (occupied/unlocked).
    #
    # BOTH DEFAULT TO 0.0 (no-op), corrected 2026-08-22 after these shipped
    # with nonzero defaults (3.0/15.0) "so the mechanism does something to
    # test against". That reasoning was wrong and caused a real, measured
    # regression: best_config.json doesn't pin these keys, so the CHAMPION
    # silently inherited them and stopped being the config that was actually
    # verified 9W-1L. Measured cost to the champion (4 episodes each):
    # chaitanyajamble -7,650.8 -> -13,173.0, ektarr +173.0 -> -2,317.2. It
    # also corrupted a promotion decision: v11's apparent "regression vs
    # chaitanyajamble" was mostly this drift, not v11 (see docs/tests/LOG.md).
    # Rule going forward, same as opening_enabled and broke_phase_*: a new
    # mechanism defaults to a no-op, and the search turns it on. Anything
    # else silently mutates the champion.
    # crop_score's existing
    # season_ok admissibility check (a tile with no time left to mature is
    # never plantable at all) is a hard constraint, unchanged by this --
    # shadow pricing only affects crops that could be planted, not whether
    # they're allowed to be.
    "lambda_labor": 0.0,
    "lambda_land": 0.0,
    # Cash-flow lookahead (see _market_orders' guaranteed_income comment):
    # how far ahead to scan our own tiles for near-ripe/ripening yield that
    # can offset the reserve cushion, and how much to discount its priced
    # value before trusting it (a scripted opponent's projected income is
    # certain; ours isn't -- price risk and execution risk both apply).
    # Motivated by an external second-opinion review's "MPC-lite" proposal
    # -- unverified, same status as lambda_labor/lambda_land above.
    "income_lookahead_days": 0,
    "income_discount": 0.0,
    # Scripted opening window: days 0-2 (steps 0-71) are 100% deterministic
    # regardless of opponent or seed -- confirmed empirically by diffing
    # independent episodes' full observation stream, byte-identical through
    # day 2, first divergence (both market prices AND town shop unlocks)
    # exactly at day 3. Tracing 5 independently-written strong opponents
    # (kawa, prvsiyan, boatlee_v16, rayk_c95, saiteja -- all separately
    # confirmed fixed-route scripts) in this exact window found a tight,
    # convergent pattern: spend nearly the entire starting $3000 by day 2,
    # hire 4-5 hands immediately on day 0. Two concrete, isolated gaps this
    # closes without touching crop selection or reserve architecture
    # elsewhere: (1) `started_up`'s day-0 animal-buy block is a poor fit for
    # a window verified safe to spend aggressively in -- see `opening_days`
    # below; (2) the hire gate is backlog-triggered, which structurally
    # can't fire hard on day 0 since no backlog exists before anything's
    # been bought/planted yet -- see `opening_hires_day0`. Deliberately NOT
    # a literal replay of any opponent's action table: read kawa's and
    # prvsiyan's source directly, both are hardcoded per-step absolute
    # movement choreography for a fixed actor count, which does not survive
    # generalization to different quantities (confirmed via that reading,
    # not assumed) -- these are reserve/gate relaxations layered on top of
    # the EXISTING reactive dispatch/shadow-pricing engine, which already
    # decides tile-by-tile where units walk and what they plant. `_enabled`
    # defaults False (a no-op merged into best_config.json, unlike
    # `broke_phase_*`'s always-on-but-inert-by-fraction pattern) so this
    # can't silently change the current champion's behavior outside the
    # normal ablate-then-promote pipeline -- unverified until tested.
    "opening_enabled": False,
    "opening_days": 3,
    # Multiplies (not replaces) the existing risk_scale-adjusted cushion for
    # animal/hire/land purchases specifically while `day < opening_days`, on
    # top of `opening_enabled`'s gate on `started_up` for animals below --
    # small rather than 0 so a purchase still leaves a trivial buffer, not
    # because there's evidence a tighter number is unsafe (there isn't yet).
    "opening_reserve_scale": 0.15,
    # Day-0-only hire target that overrides (does not replace) the normal
    # backlog-triggered gate -- see this block's own comment above for why
    # backlog can't do this job on day 0. Still gated on affordability
    # (remaining >= cost + the reduced opening cushion above), never forced
    # past what's actually affordable.
    "opening_hires_day0": 5,
    # Total in-game days (720 turns / 24 turns-per-day). Used to stop
    # planting crops whose harvest cycle wouldn't finish before season end --
    # the #1 player visibly winds crop mix back down to fast-cycle WHEAT
    # only in the final ~5 days rather than planting things that won't mature.
    "season_days": 30,
    # In the final `wind_down_days` days, stop hiring/land/animal spending
    # and sell shed inventory regardless of price -- reward is money at
    # game end, so anything still sitting in the shed or any hand hired too
    # late to earn back its cost is pure waste.
    "wind_down_days": 3,
    # If the opponent's public tiles show a crop about to ripen in bulk,
    # sell ours first -- their harvest dump will crash the price shortly
    # after, so getting ahead of it captures the higher price.
    # Relative-wealth risk scale. RESTORED 2026-08-22 after a prior session
    # deleted it: `wealth_margin_scale` was the single highest-importance
    # parameter in the v14 search (0.204) and a real promoted effect in that
    # champion, so deleting the mechanism discarded the strongest measured
    # signal we had. Restored behind a toggle that defaults False -- per the
    # standing rule, a restored/new mechanism is a NO-OP by default and the
    # search decides. risk_sensitivity 0.0 is independently a no-op too, so
    # this is off twice over until a search turns it on.
    "relative_wealth_enabled": False,
    "wealth_margin_scale": 3000.0,
    "risk_sensitivity": 0.0,
    "risk_scale_min": 1.0,
    "risk_scale_max": 2.0,
    "opponent_awareness_enabled": True,
    "opponent_incoming_threshold": 3,
    "opponent_race_discount": 0.7,
    "opponent_lookahead_days": 2,
    # How much extra discount to apply per unit of "threat" (their
    # concentration in this item x their overall scale, both 0-1 -- see
    # _opponent_profile) on top of opponent_race_discount -- makes the
    # race-discount opponent-adaptive instead of one fixed number for every
    # opponent. An ablation found the single best static discount for one
    # opponent was a clear loss against a different one (docs/tests/LOG.md).
    "opponent_concentration_sensitivity": 0.3,
    # Item-holding race/crash mechanic, replacing the earlier total-wealth
    # CPPI risk-scale (removed 2026-08-22 -- see docs/tests/LOG.md): that
    # mechanism scaled every reserve gate by *total* money+asset margin vs.
    # the opponent, which the project owner judged the wrong signal --
    # "opponent money based" rather than reacting to what actually matters,
    # per-item exposure. Only win/loss at game end is scored, not coin
    # margin (docs/GAME_GUIDE.md), so if the opponent is sitting on
    # significantly more near-ripe/ripening supply of an item than we are,
    # deliberately dumping our own (smaller) holding early can crash the
    # price before their bigger dump lands -- their loss on the crashed
    # price for their larger exposure plausibly exceeds our own loss on our
    # smaller one, a net gain in relative standing even though our own
    # absolute revenue on that item drops. Distinct from the existing
    # opponent_race_discount below (which only lowers our own sell-price bar
    # so we sell before their crash -- purely defensive); this is the
    # offensive version: cause a bigger crash than we'd otherwise choose to.
    # OFF by default, per this project's rule that a new mechanism defaults
    # to a no-op until a real search finds support for it (see
    # opening_enabled/lambda_labor's comments) -- NOT yet verified.
    "crash_sell_enabled": False,
    # Minimum raw unit gap (opponent_exposure - our_exposure) required to
    # trigger a crash -- avoids reacting to a trivial 1-2 unit difference.
    "crash_sell_margin": 5,
    # Opponent's exposure must also be at least this many times ours --
    # combined with crash_sell_margin so neither a small ratio on a big pile
    # nor a big ratio on a trivial pile alone can trigger it.
    "crash_sell_multiplier": 1.5,
    # Dump size once triggered, bypassing max_sell_chunk and the normal
    # price threshold entirely -- the point is to move the price, not to
    # sell profitably this turn.
    "crash_sell_chunk": 50,
    # _plan_units' dispatch tiers execute in descending-weight order, sorted
    # fresh each turn -- a real number per tier (flat keys, like every other
    # tunable knob here, so KNOB_SPECS/optimize.py's search space can pick
    # each up without any special-casing) instead of a hardcoded sequence,
    # so a search (Optuna or the RL policy) can find a better ordering
    # instead of it only changing when someone reads actions.csv and
    # hand-edits the function (which is exactly how CARE's starvation got
    # fixed the one time before this -- see docs/tests/LOG.md). These
    # defaults reproduce that same fixed order exactly, so nothing changes
    # until the weights do: feed > care > harvest > fertilize >
    # collect_fertilizer > water > empty_coop_place > empty_pasture_place >
    # weeds > empty_build > empty_plant. Shed-pickup tiers and the final
    # drop/pass fallback aren't part of this -- they're per-unit
    # prerequisites for a future turn, not tasks competing for this turn's
    # labor the same way these 11 are.
    "priority_weight_feed": 100.0,
    "priority_weight_care": 95.0,
    "priority_weight_harvest": 90.0,
    "priority_weight_fertilize": 85.0,
    "priority_weight_collect_fertilizer": 80.0,
    "priority_weight_water": 75.0,
    "priority_weight_empty_coop_place": 50.0,
    "priority_weight_empty_pasture_place": 45.0,
    "priority_weight_weeds": 20.0,
    "priority_weight_empty_build": 15.0,
    "priority_weight_empty_plant": 10.0,
    # Penalty weight (per tile of distance to the nearest shed tile) added to
    # _cluster_cost's tile-claim ranking for empty_build/empty_plant -- 0.0
    # (default, unchanged behavior) means claims are ranked purely by
    # distance-to-unit and local-neighbor clustering, same as before this
    # existed; a real trace analysis (docs/tests/LOG.md, 2026-08-22) found
    # occupied-tile distance from the shed drifts +3.57 tiles on average
    # from game start to end, so a positive value here is expected to help,
    # but that's not yet verified by an actual search/promotion.
    "cluster_shed_weight": 0.0,
}

PRIORITY_TIER_NAMES = (
    "feed", "care", "harvest", "fertilize", "collect_fertilizer", "water",
    "empty_coop_place", "empty_pasture_place", "weeds", "empty_build", "empty_plant",
)

SAFE_FALLBACK = {"farmer": ["PASS"], "hands": [], "market": []}


# --------------------------------------------------------------------------
# Scoring: rough $/tile/day estimates used to rank crops and animals against
# each other using the *current* market price. Not exact -- doesn't account
# for fertilizer, watering misses, or market impact -- just a ranking signal.
# --------------------------------------------------------------------------

def _crop_cycle_days(crop):
    c = CROPS[crop]
    if c["ongoing"]:
        return max(1, c["first_yield_day"] + c["interval"] * (c["max_yield"] - 1))
    return max(1, c["max_yield_day"])


def _crop_score(crop, price, labor_penalty=0.0, land_penalty=0.0):
    # labor_penalty/land_penalty are shadow prices ($/day each, both default
    # 0.0 -- no behavior change unless a caller actually computes and passes
    # them): the opportunity cost of the daily watering labor and the tile
    # itself, given how scarce those two things currently are. Subtracted
    # after the existing $/day amortization (both already the same units),
    # not folded into revenue -- a crop can score negative once labor/land
    # are tight enough, which is the point: see _plan_units' use of this to
    # skip planting instead of always planting whatever's "least bad."
    c = CROPS[crop]
    revenue = price * c["max_yield"]
    return (revenue - c["seed"]) / _crop_cycle_days(crop) - labor_penalty - land_penalty


def _diversified_crop_score(crop, price, crop_counts, weight, labor_penalty=0.0, land_penalty=0.0):
    # Discount by how much of this crop is already growing, so several units
    # deciding what to plant in the same turn don't all pile into whichever
    # single crop currently scores highest.
    return _crop_score(crop, price, labor_penalty, land_penalty) / (1 + crop_counts.get(crop, 0) * weight)


def _animal_score(animal, price, labor_penalty=0.0):
    a = ANIMALS[animal]
    # Steady-state $/day once producing; ignores the wheat feed cost and the
    # ramp-up to first_yield_day, both small relative to season length.
    # labor_penalty: same shadow-price idea as _crop_score's, in the same
    # $/day units -- an animal needs daily FEED+CARE, not zero labor, so it
    # isn't automatically exempt from labor scarcity just because it skips
    # watering.
    return price / max(1, a["interval"]) - labor_penalty


def _animal_roi_ok(animal, price, day, config):
    """See DEFAULT_CONFIG's roi_gate_enabled comment: only worth buying if
    its steady-state $/day rate can earn back roi_margin x its own cost in
    the days actually left before wind-down, accounting for the ramp-up to
    first_yield_day (an animal placed today earns nothing until then)."""
    a = ANIMALS[animal]
    usable_days = config["season_days"] - day - a["first_yield_day"] - config["wind_down_days"]
    if usable_days <= 0:
        return False
    projected_revenue = _animal_score(animal, price) * usable_days
    return projected_revenue >= a["cost"] * config["roi_margin"]


def _land_roi_ok(cost, day, n_quadrants, unlocked_tiles, config, prices):
    """See DEFAULT_CONFIG's roi_gate_enabled comment. Land itself produces
    nothing -- its value is the extra tiles it unlocks, priced at the best
    $/day rate currently achievable (same _crop_score/_animal_score used to
    rank what to plant) times land_utilization_threshold (the fraction of a
    new quadrant this config already assumes will actually get worked,
    reused here rather than assuming 100% occupancy from day one)."""
    tiles_per_quadrant = unlocked_tiles / max(1, n_quadrants)
    best_rate = 0.0
    for crop in config["crops"]:
        best_rate = max(best_rate, _crop_score(crop, prices.get(crop, _base_price(crop))))
    if config["animal_enabled"]:
        for animal in ("COW", "SHEEP"):
            product = ANIMALS[animal]["product"]
            best_rate = max(best_rate, _animal_score(animal, prices.get(product, _base_price(product))))
    usable_days = config["season_days"] - day - config["wind_down_days"]
    if usable_days <= 0 or best_rate <= 0:
        return False
    projected_value = tiles_per_quadrant * config["land_utilization_threshold"] * best_rate * usable_days
    return projected_value >= cost * config["roi_margin"]


def _base_price(item):
    return MARKET_PARAMS.get(item, {}).get("base", 0)


def _dynamic_sell_fraction(config, day, money):
    """How picky to be about sale price this turn, as a linear function of
    two state features instead of one constant for the whole game. See
    DEFAULT_CONFIG's sell_fraction_* comment for why -- this is the "hybrid
    RL" piece: the coefficients are just more numbers Optuna searches, but
    the resulting behavior can actually adapt within a single game."""
    season_days = max(1, config["season_days"])
    days_left_frac = max(0.0, config["season_days"] - day) / season_days
    cash_frac = min(1.0, max(0.0, money / max(1.0, config["cash_scale"])))
    raw = (
        config["sell_fraction_base"]
        + config["sell_fraction_day_weight"] * days_left_frac
        + config["sell_fraction_cash_weight"] * cash_frac
    )
    return min(config["sell_fraction_max"], max(config["sell_fraction_min"], raw))


# --------------------------------------------------------------------------
# Farm scanning: one pass over the grid, bucketed by task type.
# --------------------------------------------------------------------------

def _plant_harvest_ready(tile, day):
    crop = tile["crop"]
    c = CROPS[crop]
    age = day - tile["planted_day"]
    if age < c["first_yield_day"]:
        return False
    if not c["ongoing"]:
        # Wait for full maturity so a single harvest captures the whole
        # watering-bonus window instead of an early partial yield.
        return age >= c["max_yield_day"]
    return True


def _scan_farm(farm, board_size, day):
    info = {
        "harvest": [],
        "water": [],
        "fertilize": [],
        "weeds": [],
        "empty": [],
        "feed": [],
        "care": [],
        "collect_fertilizer": [],
        "empty_coop": [],
        "empty_pasture": [],
        "occupied": 0,
        "unlocked": 0,
        "crop_counts": {},
        "coop_count": 0,
        "pasture_count": 0,
    }
    tiles = farm["tiles"]
    for y in range(board_size):
        row = tiles[y]
        for x in range(board_size):
            tile = row[x]
            if tile == "LOCKED":
                continue
            info["unlocked"] += 1
            if tile is None:
                info["empty"].append((x, y))
                continue
            info["occupied"] += 1
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "WEED":
                info["weeds"].append((x, y))
            elif kind == "PLANT":
                info["crop_counts"][tile["crop"]] = info["crop_counts"].get(tile["crop"], 0) + 1
                if tile["yield_units"] > 0 and _plant_harvest_ready(tile, day):
                    info["harvest"].append((x, y))
                if not tile["watered_today"]:
                    info["water"].append((x, y))
                if tile.get("fertilized_until_day", -1) < day:
                    info["fertilize"].append((x, y))
            elif kind in ("COOP", "PASTURE"):
                info["coop_count" if kind == "COOP" else "pasture_count"] += 1
                if "animal" not in tile:
                    info["empty_coop" if kind == "COOP" else "empty_pasture"].append((x, y))
                else:
                    if tile["yield_units"] > 0:
                        info["harvest"].append((x, y))
                    if not tile["fed_today"]:
                        info["feed"].append((x, y))
                    elif not tile["cared_today"]:
                        info["care"].append((x, y))
                    if tile["fertilizer_available"]:
                        info["collect_fertilizer"].append((x, y))
    return info


# --------------------------------------------------------------------------
# Task assignment (uses step_toward / closest / act_or_move / shed_tiles
# from farm_utils.py for the actual movement mechanics).
# --------------------------------------------------------------------------

STATIC_TASK_ACTIONS = {
    "feed": ["FEED"], "harvest": ["HARVEST"], "fertilize": ["FERTILIZE"],
    "collect_fertilizer": ["COLLECT_FERTILIZER"], "water": ["WATER"],
    "care": ["CARE"], "weeds": ["DIG"], "empty_coop_place": ["PLACE", "GOOSE"],
}
TASK_INFO_KEY = {
    "feed": "feed", "harvest": "harvest", "fertilize": "fertilize",
    "collect_fertilizer": "collect_fertilizer", "water": "water", "care": "care",
    "weeds": "weeds", "empty_coop_place": "empty_coop", "empty_pasture_place": "empty_pasture",
    "empty_build": "empty", "empty_plant": "empty",
}


def _assign_nearest(pending, candidates, act_fn, actions, continue_fn=None, cost_fn=None):
    """Repeatedly assign the single lowest-cost (unit, tile) pair across all
    of `pending` x `candidates`, instead of processing units in a fixed
    array order and letting each grab whatever's nearest to *itself*
    regardless of whether some other still-unassigned unit is actually
    closer to that same tile. Mutates `pending` (list of [idx, pos, inv])
    and `candidates` in place, removing what gets assigned, and writes
    `actions[idx]`. `continue_fn`, if given, is re-checked before every
    pair pick and stops the tier early once it goes false (used by
    empty_build/empty_plant, whose eligibility -- structures/seed
    availability -- changes as the tier itself assigns). `cost_fn(pos,
    target)`, if given, replaces plain Manhattan distance as the ranking
    metric (used by empty_build/empty_plant to prefer clustering new
    tiles against the existing footprint over pure proximity -- see
    docs/tests/LOG.md).
    """
    cost_fn = cost_fn or (lambda pos, t: abs(pos[0] - t[0]) + abs(pos[1] - t[1]))
    while pending and candidates:
        if continue_fn is not None and not continue_fn():
            break
        best = None
        best_cost = None
        for i, (idx, pos, inv) in enumerate(pending):
            for j, t in enumerate(candidates):
                c = cost_fn(pos, t)
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best = (i, j)
        i, j = best
        idx, pos, inv = pending[i]
        target = candidates[j]
        actions[idx] = act_fn(idx, pos, inv, target)
        del pending[i]
        del candidates[j]


def _plan_units(farm, private, board_size, day, info, config, prices, unit_targets=None):
    """Decide one action per unit (farmer + hands), consuming tiles from
    `info` so two units never chase the same tile in the same turn.

    Two layers on top of the naive "each unit runs its own priority ladder,
    grabbing whichever tile is nearest to itself" design:

    1. `unit_targets`, if given, is a {unit_idx: (task_key, target)} dict
       mutated in place across calls (one call per turn) so a unit already
       walking toward a target keeps heading there instead of re-running
       the full priority ladder from scratch every turn. Without this, a
       unit several tiles into a walk toward (say) a water target gets
       pulled onto a newly-appeared, higher-priority-tier task the instant
       one shows up anywhere on the board, abandoning the walk and
       starting a new one (see docs/tests/LOG.md).
    2. Within each priority tier that draws from a shared, contested tile
       list (feed/harvest/fertilize/collect_fertilizer/water/care/weeds/
       empty_coop/empty_pasture/empty), assignment is a greedy *global*
       nearest-pair match (`_assign_nearest`) across all still-unassigned
       eligible units at once, not a fixed unit-processing-order sequential
       grab. Instrumentation showed most unit-turns in the water tier were
       pure travel (~90% not-yet-arrived) -- confirming forced long walks
       from fixed processing order (farmer always resolved first regardless
       of position) were a real cost, not just a theoretical one. Tiers
       without shared contention (shed pickups, drop, pass) stay per-unit.
    """
    if unit_targets is None:
        unit_targets = {}

    units = [(0, tuple(farm["farmer"]))]
    for i, pos in enumerate(farm.get("hands", []) or []):
        units.append((i + 1, tuple(pos)))

    inventories = private.get("inventories", []) or []
    shed = private.get("shed", {}) or {}

    shed_spots = shed_tiles(board_size)
    seeds = private.get("seeds", {}) or {}

    crop_pool = [c for c in config["crops"] if seeds.get(c, 0) > 0]
    crop_counts = dict(info.get("crop_counts", {}))
    # Don't plant anything whose harvest cycle wouldn't finish before season
    # end -- a tile with a crop that can't mature in time is wasted for the
    # rest of the game.
    season_ok = {c: day + _crop_cycle_days(c) <= config["season_days"] for c in config["crops"]}

    # Shadow prices for this turn (see DEFAULT_CONFIG's lambda_labor/
    # lambda_land comment): labor_scarcity is the same pending-tasks-per-unit
    # signal _market_orders' hire gate already uses; land_scarcity is
    # current occupancy of the unlocked footprint. Both 0 (no penalty) when
    # nothing's actually backed up yet, growing as the board fills up.
    unit_count_for_labor = 1 + len(farm.get("hands", []) or [])
    labor_backlog = (
        len(info["harvest"]) + len(info["water"]) + len(info["feed"])
        + len(info["weeds"]) + len(info["fertilize"])
    )
    labor_scarcity = labor_backlog / max(1, unit_count_for_labor)
    crop_labor_penalty = config["lambda_labor"] * labor_scarcity
    land_scarcity = info["occupied"] / info["unlocked"] if info["unlocked"] > 0 else 0.0
    crop_land_penalty = config["lambda_land"] * land_scarcity

    def _best_crop(viable_crops):
        # Shared by both crop-selection call sites (sticky continuation and
        # fresh assignment) so shadow pricing behaves identically in both.
        # Unlike the un-penalized version, a crop can lose here even with no
        # competitor: if every viable crop's labor/land-adjusted score is
        # <=0, this returns None exactly like "no viable crops" -- the tile
        # goes unplanted this turn instead of taking whatever's least bad,
        # freeing that labor for higher-priority tiers (feed/care/harvest)
        # or a future turn once labor/land scarcity eases.
        scored = [
            (c, _diversified_crop_score(
                c, prices.get(c, _base_price(c)), crop_counts, config["diversification_weight"],
                crop_labor_penalty, crop_land_penalty,
            ))
            for c in viable_crops
        ]
        positive = [(c, s) for c, s in scored if s > 0]
        return max(positive, key=lambda cs: cs[1])[0] if positive else None

    def inv_of(idx):
        return inventories[idx] if idx < len(inventories) else {}

    # Target structure count scales with current labor AND unlocked land so
    # pasture capacity builds out in step with land instead of all at once,
    # only once crop land runs out, or (the bug this quadrant factor fixes)
    # frozen forever once the starting pasture count already meets a
    # units-only target: with unit_count roughly constant for most of the
    # game, a units-only formula never grows again after quadrant 1, so a
    # 2nd/3rd quadrant's worth of new land never gets any pasture -- newly
    # unlocked land unconditionally became crop tiles instead, confirmed
    # live (pasture flat at its day-0 value the entire 29-day game across
    # multiple matches) via replay review, see docs/tests/LOG.md.
    #
    # Uses *intended* labor scale (1 + max_hires_per_day), not len(units)
    # (today's actual headcount) -- the original units-only version capped
    # day-0 target near 1 regardless of max_structures, since there's only
    # ever a lone farmer before any hire has landed. That directly
    # contradicted the very next sentence in this comment: building is
    # free (only the animal costs money), so there's no cash reason to
    # delay it -- yet the formula delayed it anyway, gated on a headcount
    # that hasn't caught up yet. Replay review of the strong opponents in
    # our benchmark pool confirmed they build out full pasture capacity in
    # the first few turns, well ahead of their own hiring curve, rather
    # than growing it in step with current hands (see docs/tests/LOG.md).
    # This still isn't "build everything on turn 0 no matter what" --
    # max_structures and land availability (info["empty"] itself) remain
    # the real caps -- it just stops using *today's* labor as an artificial
    # third one.
    n_quadrants = len(farm.get("unlocked_quadrants", ["NW"]))
    structures_committed = info["pasture_count"] + info["coop_count"]
    if config["season_days"] - day <= config["wind_down_days"]:
        target_structures = 0  # no time left for a new pasture to pay back
    else:
        intended_units = 1 + config["max_hires_per_day"]
        target_structures = min(
            config["max_structures"],
            math.ceil(intended_units * config["pasture_target_ratio"] * n_quadrants),
        )

    actions = {}

    def commit(idx, pos, target, task_key, action):
        # Called after every pick below: remember it if the unit hasn't
        # arrived yet (still walking, needs to survive to next turn),
        # forget it once acted on (nothing left to remember).
        if pos == target:
            unit_targets.pop(idx, None)
        else:
            unit_targets[idx] = (task_key, target)
        return action

    # --- Phase 1: sticky continuation, per unit (unchanged semantics) ---
    pending = []
    for idx, pos in units:
        inv = inv_of(idx)
        remembered = unit_targets.get(idx)
        if remembered is not None:
            task_key, remembered_target = remembered
            candidates = info.get(TASK_INFO_KEY.get(task_key, ""))
            if candidates is not None and remembered_target in candidates:
                target = remembered_target
                candidates.remove(target)
                action = None
                if task_key in STATIC_TASK_ACTIONS:
                    action = act_or_move(pos, target, STATIC_TASK_ACTIONS[task_key])
                elif task_key == "empty_pasture_place":
                    animal = "COW" if inv.get("COW", 0) > 0 else "SHEEP"
                    action = act_or_move(pos, target, ["PLACE", animal])
                elif task_key == "empty_build":
                    build = "BUILD_COOP" if config["enable_coop"] and info["coop_count"] <= info["pasture_count"] else "BUILD_PASTURE"
                    action = act_or_move(pos, target, [build])
                    if pos == target:
                        if build == "BUILD_COOP":
                            info["coop_count"] += 1
                        else:
                            info["pasture_count"] += 1
                elif task_key == "empty_plant":
                    viable_crops = [c for c in crop_pool if season_ok.get(c, True)]
                    best_crop = _best_crop(viable_crops)
                    if best_crop is not None:
                        action = act_or_move(pos, target, ["PLANT", best_crop])
                        if pos == target:
                            seeds[best_crop] = seeds.get(best_crop, 0) - 1
                            if seeds[best_crop] <= 0:
                                crop_pool = [c for c in crop_pool if c != best_crop]
                            crop_counts[best_crop] = crop_counts.get(best_crop, 0) + 1
                actions[idx] = action
                if pos == target:
                    unit_targets.pop(idx, None)
                else:
                    unit_targets[idx] = (task_key, target)
                continue
            else:
                unit_targets.pop(idx, None)
        pending.append([idx, pos, inv])

    # --- Phase 2: batch tiers over shared, contested tile lists ---
    #
    # Tier EXECUTION order used to be a fixed sequence of calls -- discovering
    # a starved tier (CARE was tier 12 of ~16, chronically shut out, see
    # docs/tests/LOG.md) meant hand-editing this function and re-verifying.
    # Instead, each tier is registered once as a runner keyed by name, and
    # `config["priority_weight_<name>"]` (one real number per tier, all
    # independently tunable -- KNOB_SPECS/optimize.py's search space)
    # decides execution order by sorting descending each turn.
    # A float-per-tier weight rather than a searched permutation directly:
    # Optuna's default samplers handle independent continuous ranges far
    # better than a combinatorial ordering, and "sort by score" is standard
    # for exactly this kind of tunable scheduling problem. Defaults below
    # reproduce the hand-tuned order this function used before this change
    # (feed > care > harvest > fertilize > collect_fertilizer > water >
    # empty_coop_place > empty_pasture_place > weeds > empty_build >
    # empty_plant), so nothing changes unless the weights are.
    def elig(p, pred):
        return [u for u in p if pred(u[2])]

    def _tier_feed(p):
        _assign_nearest(
            elig(p, lambda inv: inv.get("WHEAT", 0) > 0),
            info["feed"],
            lambda idx, pos, inv, t: commit(idx, pos, t, "feed", act_or_move(pos, t, ["FEED"])),
            actions,
        )

    def _tier_care(p):
        _assign_nearest(
            list(p),
            info["care"],
            lambda idx, pos, inv, t: commit(idx, pos, t, "care", act_or_move(pos, t, ["CARE"])),
            actions,
        )

    def _tier_harvest(p):
        _assign_nearest(
            list(p),
            info["harvest"],
            lambda idx, pos, inv, t: commit(idx, pos, t, "harvest", act_or_move(pos, t, ["HARVEST"])),
            actions,
        )

    def _tier_fertilize(p):
        _assign_nearest(
            elig(p, lambda inv: inv.get("FERTILIZER", 0) > 0),
            info["fertilize"],
            lambda idx, pos, inv, t: commit(idx, pos, t, "fertilize", act_or_move(pos, t, ["FERTILIZE"])),
            actions,
        )

    def _tier_collect_fertilizer(p):
        _assign_nearest(
            list(p),
            info["collect_fertilizer"],
            lambda idx, pos, inv, t: commit(idx, pos, t, "collect_fertilizer", act_or_move(pos, t, ["COLLECT_FERTILIZER"])),
            actions,
        )

    def _tier_water(p):
        _assign_nearest(
            list(p),
            info["water"],
            lambda idx, pos, inv, t: commit(idx, pos, t, "water", act_or_move(pos, t, ["WATER"])),
            actions,
        )

    def _tier_empty_coop_place(p):
        _assign_nearest(
            elig(p, lambda inv: config["enable_coop"] and inv.get("GOOSE", 0) > 0),
            info["empty_coop"],
            lambda idx, pos, inv, t: commit(idx, pos, t, "empty_coop_place", act_or_move(pos, t, ["PLACE", "GOOSE"])),
            actions,
        )

    def _tier_empty_pasture_place(p):
        _assign_nearest(
            elig(p, lambda inv: config["animal_enabled"] and (inv.get("COW", 0) > 0 or inv.get("SHEEP", 0) > 0)),
            info["empty_pasture"],
            lambda idx, pos, inv, t: commit(
                idx, pos, t, "empty_pasture_place",
                act_or_move(pos, t, ["PLACE", "COW" if inv.get("COW", 0) > 0 else "SHEEP"]),
            ),
            actions,
        )

    def _tier_weeds(p):
        _assign_nearest(
            list(p),
            info["weeds"],
            lambda idx, pos, inv, t: commit(idx, pos, t, "weeds", act_or_move(pos, t, ["DIG"])),
            actions,
        )

    tier_order = sorted(
        ("feed", "care", "harvest", "fertilize", "collect_fertilizer", "water",
         "empty_coop_place", "empty_pasture_place", "weeds"),
        key=lambda name: -config.get(f"priority_weight_{name}", 0.0),
    )
    tier_runners = {
        "feed": _tier_feed, "care": _tier_care, "harvest": _tier_harvest,
        "fertilize": _tier_fertilize, "collect_fertilizer": _tier_collect_fertilizer,
        "water": _tier_water, "empty_coop_place": _tier_empty_coop_place,
        "empty_pasture_place": _tier_empty_pasture_place, "weeds": _tier_weeds,
    }
    for name in tier_order:
        tier_runners[name](pending)
        pending = [u for u in pending if u[0] not in actions]

    # --- Shed pickups: no shared-resource contention (shed_spots aren't
    # consumed), so per-unit is fine -- no benefit from batch matching, and
    # not part of the dynamic-priority reordering above (each is a
    # prerequisite step for a future turn's feed/fertilize/place, not a
    # competing task in the same sense). ---
    still_pending = []
    for idx, pos, inv in pending:
        if inv.get("WHEAT", 0) == 0 and info["feed"] and shed.get("WHEAT", 0) > 0:
            target = closest(pos, shed_spots)
            actions[idx] = act_or_move(pos, target, ["PICKUP", "WHEAT", min(shed["WHEAT"], 10)])
        elif inv.get("FERTILIZER", 0) == 0 and info["fertilize"] and shed.get("FERTILIZER", 0) > 0:
            target = closest(pos, shed_spots)
            actions[idx] = act_or_move(pos, target, ["PICKUP", "FERTILIZER", min(shed["FERTILIZER"], 10)])
        elif config["enable_coop"] and info["empty_coop"] and shed.get("GOOSE", 0) > 0:
            target = closest(pos, shed_spots)
            actions[idx] = act_or_move(pos, target, ["PICKUP", "GOOSE", 1])
        elif config["animal_enabled"] and info["empty_pasture"] and (shed.get("COW", 0) > 0 or shed.get("SHEEP", 0) > 0):
            animal = "COW" if shed.get("COW", 0) > 0 else "SHEEP"
            target = closest(pos, shed_spots)
            actions[idx] = act_or_move(pos, target, ["PICKUP", animal, 1])
        else:
            still_pending.append([idx, pos, inv])
    pending = still_pending

    # New land gets claimed against a clustering-biased cost instead of pure
    # distance-to-unit: prefer tiles touching the existing worked footprint
    # (or another tile claimed earlier this same turn) over an equally-close
    # tile that would start a new, disconnected patch. Distance-to-unit
    # already turned out to be near-optimal on its own (measured mean 1.62
    # tiles per water-tier assignment) -- the actual cost isn't long walks,
    # it's that *every* tile still needs its own multi-turn trip regardless
    # of how short. A tight, contiguous footprint turns a day's worth of
    # water/harvest/feed visits into one short sweep instead of many
    # separate short hops scattered across the board (see docs/tests/LOG.md).
    claimed_this_turn = set()
    tiles_grid = farm["tiles"]

    def _occupied_neighbors(x, y):
        count = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (nx, ny) in claimed_this_turn:
                count += 1
            elif 0 <= nx < board_size and 0 <= ny < board_size:
                t = tiles_grid[ny][nx]
                if t is not None and t != "LOCKED":
                    count += 1
        return count

    CLUSTER_BONUS = 2.0

    def _cluster_cost(pos, target):
        d = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
        shed_dist = min(abs(target[0] - sx) + abs(target[1] - sy) for sx, sy in shed_spots)
        return (
            d - CLUSTER_BONUS * _occupied_neighbors(target[0], target[1])
            + config["cluster_shed_weight"] * shed_dist
        )

    # Build pasture (or coop, if enabled) capacity in step with current
    # labor. Ranked above planting by default: BUILD_PASTURE is free, so
    # there's no cash trade-off, only a tile-allocation one -- but both are
    # part of the same dynamic-priority system as the tiers above (they
    # share info["empty"] as their candidate pool, so their order relative
    # to *each other*, not just to the other 9 tiers, is meaningful too).
    def _build_act(idx, pos, inv, target):
        nonlocal structures_committed
        build = "BUILD_COOP" if config["enable_coop"] and info["coop_count"] <= info["pasture_count"] else "BUILD_PASTURE"
        action = commit(idx, pos, target, "empty_build", act_or_move(pos, target, [build]))
        claimed_this_turn.add(target)
        # Count this tile toward the target immediately (not only once
        # actually built) -- otherwise every unit still mid-walk this turn
        # would see the same unmet target and all pile onto BUILD_PASTURE,
        # overshooting it in a single turn.
        structures_committed += 1
        if pos == target:
            if build == "BUILD_COOP":
                info["coop_count"] += 1
            else:
                info["pasture_count"] += 1
        return action

    def _tier_empty_build(p):
        if not config["animal_enabled"]:
            return
        _assign_nearest(
            list(p), info["empty"], _build_act, actions,
            continue_fn=lambda: structures_committed < target_structures,
            cost_fn=_cluster_cost,
        )

    def _plant_act(idx, pos, inv, target):
        nonlocal crop_pool
        viable_crops = [c for c in crop_pool if season_ok.get(c, True)]
        best_crop = _best_crop(viable_crops)
        if best_crop is None:
            # Rare: the tier-level check below already filters out turns
            # where nothing scores positive, but per-unit diversification
            # penalties can still push a borderline crop negative partway
            # through this tier's batch (crop_counts grows as earlier units
            # in the same wave commit). Don't plant something the shadow
            # price says isn't worth it -- PASS rather than waste a seed.
            return commit(idx, pos, target, "empty_plant", act_or_move(pos, target, ["PASS"]))
        action = commit(idx, pos, target, "empty_plant", act_or_move(pos, target, ["PLANT", best_crop]))
        claimed_this_turn.add(target)
        if pos == target:
            seeds[best_crop] = seeds.get(best_crop, 0) - 1
            if seeds[best_crop] <= 0:
                crop_pool = [c for c in crop_pool if c != best_crop]
            crop_counts[best_crop] = crop_counts.get(best_crop, 0) + 1
        return action

    def _tier_empty_plant(p):
        # Checks profitability (_best_crop), not just admissibility
        # (season_ok) -- avoids walking a unit all the way to a tile only to
        # find shadow pricing says nothing's worth planting there right now.
        _assign_nearest(
            list(p), info["empty"], _plant_act, actions,
            continue_fn=lambda: _best_crop([c for c in crop_pool if season_ok.get(c, True)]) is not None,
            cost_fn=_cluster_cost,
        )

    land_tier_order = sorted(
        ("empty_build", "empty_plant"),
        key=lambda name: -config.get(f"priority_weight_{name}", 0.0),
    )
    land_tier_runners = {"empty_build": _tier_empty_build, "empty_plant": _tier_empty_plant}
    for name in land_tier_order:
        land_tier_runners[name](pending)
        pending = [u for u in pending if u[0] not in actions]

    # --- Fallback: drop off whatever's left, or PASS. ---
    for idx, pos, inv in pending:
        if inv:
            target = closest(pos, shed_spots)
            actions[idx] = act_or_move(pos, target, ["DROP"])
        else:
            actions[idx] = ["PASS"]

    farmer_action = actions.get(0, ["PASS"])
    hands_actions = [actions[idx] for idx, _ in units[1:]]
    return farmer_action, hands_actions


# --------------------------------------------------------------------------
# Market orders
# --------------------------------------------------------------------------

def _carried_total(private, item):
    return sum(inv.get(item, 0) for inv in (private.get("inventories", []) or []))


def _opponent_profile(opponent_farm, board_size, day, lookahead_days):
    """Everything we can legally see about the opponent (their public
    tiles/land/labor -- never their private shed or seed counts) reduced
    to three things worth reacting to:

    - `supply`: how much of each product they're likely to dump on the
      market soon (already-ripe, or a one-time crop maturing within
      `lookahead_days`) -- same estimate as before, just folded into this
      one scan instead of a separate function, since both need the same
      tile walk.
    - `concentration`: {item: fraction of their occupied tiles producing
      it}. An opponent running 80% wheat has a much bigger relative stake
      in wheat's price than one running an even 5-crop split -- their
      dump (and their own selling pressure generally) matters more for
      that specific item.
    - `scale`: 0-1, how big their whole operation currently is (land +
      labor), independent of what they're growing -- a small opponent's
      dump barely moves the shared price no matter how concentrated it
      is; a large one's does even if diversified.
    """
    supply = {}
    item_tile_counts = {}
    total_occupied = 0
    if opponent_farm is None:
        return supply, {}, 0.0
    tiles = opponent_farm["tiles"]
    for y in range(board_size):
        row = tiles[y]
        for x in range(board_size):
            tile = row[x]
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crop = tile["crop"]
                c = CROPS[crop]
                total_occupied += 1
                item_tile_counts[crop] = item_tile_counts.get(crop, 0) + 1
                if tile["yield_units"] > 0 and _plant_harvest_ready(tile, day):
                    supply[crop] = supply.get(crop, 0) + tile["yield_units"]
                elif not c["ongoing"]:
                    days_to_ready = c["max_yield_day"] - (day - tile["planted_day"])
                    if 0 <= days_to_ready <= lookahead_days:
                        supply[crop] = supply.get(crop, 0) + c["max_yield"]
            elif kind in ("COOP", "PASTURE") and "animal" in tile:
                total_occupied += 1
                product = ANIMALS[tile["animal"]]["product"]
                item_tile_counts[product] = item_tile_counts.get(product, 0) + 1
                if tile["yield_units"] > 0:
                    supply[product] = supply.get(product, 0) + tile["yield_units"]

    concentration = (
        {item: cnt / total_occupied for item, cnt in item_tile_counts.items()}
        if total_occupied > 0 else {}
    )
    n_quadrants = len(opponent_farm.get("unlocked_quadrants", ["NW"]))
    unit_count = 1 + len(opponent_farm.get("hands", []) or [])
    # Normalized against the range this session's traced strong opponents
    # actually reach at their peak (3-4 quadrants, 12-14 hands) -- see
    # docs/tests/LOG.md -- not an arbitrary guess.
    scale = min(1.0, (n_quadrants / 4.0) * 0.5 + (unit_count / 14.0) * 0.5)
    return supply, concentration, scale


def _standing_asset_value(farm, prices):
    """Estimate the market value of everything currently growing/held on a
    farm, priced at current rates -- the "unrealized gains" half of total
    wealth for the relative-wealth risk scale below. Deliberately broader
    than `_opponent_profile`'s `supply` (which only counts already-ripe or
    near-ripe-within-lookahead output, tuned for a different question --
    "what's about to hit the market"): every standing crop counts at its
    full expected yield value regardless of growth stage, and every live
    animal counts at its purchase cost (the sunk capital already committed)
    plus anything it's already produced. A freshly-planted crop or a
    just-bought animal is real future money even though it contributes
    nothing to `supply` -- valuing it at $0 (the original version of this
    mechanism) meant an opponent mid-way through a heavy build-out phase
    (near-broke on cash, but sitting on a large just-planted/just-stocked
    position) looked artificially poor and the mechanism reacted to their
    threat a turn too late. Both farms' tiles are legally visible either
    way (docs/GAME_GUIDE.md: farm dicts are public)."""
    if farm is None:
        return 0.0
    value = 0.0
    for row in farm.get("tiles") or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crop = tile.get("crop")
                c = CROPS.get(crop)
                if c is not None:
                    value += c["max_yield"] * prices.get(crop, _base_price(crop))
            elif kind in ("COOP", "PASTURE") and tile.get("animal"):
                a = ANIMALS.get(tile["animal"])
                if a is not None:
                    value += a["cost"]
                    if tile.get("yield_units", 0) > 0:
                        value += tile["yield_units"] * prices.get(a["product"], _base_price(a["product"]))
    return value


def _market_orders(farm, private, info, config, prices, day, opponent_supply=None,
                    opponent_concentration=None, opponent_scale=0.0, opponent_farm=None,
                    our_supply=None):
    orders = []
    money = farm["money"]
    # Optional "broke phase": scale every spending reserve/gate down for the
    # first `broke_phase_days_frac` of the season, then back to normal --
    # deliberately spend into the red early (hiring/land/animals) since only
    # the final day's coin count is scored (docs/GAME_GUIDE.md: margin
    # doesn't matter, only win/loss), matching the traced top opponents
    # (kawa, prvsiyan), who are visibly near-broke through day 6-8 before
    # exploding from day 10+. Disabled by default (frac=0, scale=1 -- no
    # behavior change) since a flat "always aggressive" version of this
    # already regressed across the pool (see docs/tests/LOG.md); this is
    # narrower -- only the early phase gets cheaper, not the whole game.
    phase_scale = (
        config["broke_phase_reserve_scale"]
        if day < config["season_days"] * config["broke_phase_days_frac"]
        else 1.0
    )
    shed = private.get("shed", {}) or {}
    seeds = private.get("seeds", {}) or {}
    # See DEFAULT_CONFIG's opening_enabled comment: days 0-2 are verified
    # deterministic and 5 independent strong opponents all spend aggressively
    # in this exact window, so it gets its own gate override + tighter
    # cushion multiplier rather than just lowering startup_days globally.
    in_opening = config["opening_enabled"] and day < config["opening_days"]
    started_up = day >= config["startup_days"] or in_opening
    opponent_supply = opponent_supply or {}
    opponent_concentration = opponent_concentration or {}

    # Relative-wealth risk scale. RESTORED 2026-08-22 after a prior session
    # deleted the mechanism: it was the single highest-importance parameter in
    # the v14 search (wealth_margin_scale, importance 0.204) and a real
    # promoted effect in that champion, so removing it discarded the strongest
    # measured signal we had. Now behind an explicit toggle so the search can
    # decide rather than either of us assuming. margin > 0 (we're ahead)
    # pushes risk_scale above 1, tightening every reserve gate -- a bigger
    # lead scores no higher, only losing it hurts.
    if config["relative_wealth_enabled"] and opponent_farm is not None:
        our_value = money + _standing_asset_value(farm, prices)
        opp_value = opponent_farm.get("money", 0) + _standing_asset_value(opponent_farm, prices)
        margin_frac = (our_value - opp_value) / config["wealth_margin_scale"]
        risk_scale = min(
            config["risk_scale_max"],
            max(config["risk_scale_min"], 1.0 + config["risk_sensitivity"] * margin_frac),
        )
    else:
        risk_scale = 1.0

    scale = phase_scale * risk_scale
    # Applied only to the animal/hire/land cushions below (never the floor,
    # never the `remaining >= cost` half of any gate) -- see opening_enabled.
    opening_scale = scale * config["opening_reserve_scale"] if in_opening else scale
    # Fixed survival floor, deliberately NOT multiplied by scale -- see this
    # key's DEFAULT_CONFIG comment. Land/animal reserve requirements below
    # are `max(reserve_floor, multiple * scale * cost)`, so this only binds
    # for a cheap purchase where the proportional term would round to
    # near-nothing; it never dominates a real-sized one.
    reserve_floor = config["money_reserve"]

    # Cash-flow lookahead (see income_discount/income_lookahead_days'
    # DEFAULT_CONFIG comment): a reactive agent needs a cash cushion because
    # it can't otherwise distinguish "$0 in the bank, empty soil" from "$0
    # in the bank, $600 of tomatoes ripening tomorrow" -- both look equally
    # risky by liquid cash alone. `our_supply` is the same near-ripe/
    # ripening-within-lookahead tile scan `_opponent_profile` already does
    # for the opponent-awareness mechanism, reused here on our own farm
    # (it's a generic function over "a farm dict", not opponent-specific)
    # instead of a second, separate scan. Discounted for two real risks a
    # scripted opponent with perfect foresight doesn't have to price in: the
    # sale might land at a worse price than today's quote, and (unlike a
    # precomputed route) there's no guarantee the projected labor actually
    # arrives to harvest/sell it on schedule.
    guaranteed_income = sum(
        units * prices.get(item, _base_price(item)) for item, units in (our_supply or {}).items()
    ) * config["income_discount"]

    def _cushion(base_cushion):
        # Guaranteed income can only offset the safety cushion, never let a
        # purchase proceed without covering its own cost out of liquid cash
        # -- that half of the gate (`remaining >= cost`) is untouched at
        # every call site below, only the `+ cushion` term shrinks here.
        return max(0.0, base_cushion - guaranteed_income)
    # Reward is money at game end, full stop -- unsold shed inventory and
    # freshly-hired hands with no time left to earn back their cost are pure
    # waste in the closing days. The #1 player visibly winds crop mix back
    # down in the final ~5 days rather than planting things that won't
    # mature; this is the same idea applied to market orders: stop paying
    # for anything that can't pay itself back, and dump inventory for
    # whatever it fetches instead of holding out for a better price that
    # will never be realized.
    winding_down = config["season_days"] - day <= config["wind_down_days"]

    # See DEFAULT_CONFIG's phase_gate_enabled comment: pauses new land/animal
    # capital spending (not hiring -- more hands is the actual fix for this)
    # whenever pending work already exceeds current labor, same backlog
    # formula the HIRE gate below uses, so a turn already drowning in
    # unfed/unwatered/unharvested tiles doesn't also buy more capacity the
    # dispatcher has no chance of servicing this season.
    in_crisis = False
    if config["phase_gate_enabled"]:
        unit_count_now = 1 + len(farm.get("hands", []) or [])
        crisis_backlog = (
            len(info["harvest"]) + len(info["water"]) + len(info["feed"])
            + len(info["weeds"]) + len(info["fertilize"])
        )
        in_crisis = crisis_backlog > unit_count_now * config["crisis_backlog_ratio"]

    # Sell everything sellable that's above threshold, in bounded chunks.
    # If the opponent's about to dump a lot of this item on the market
    # (visible from their public tiles), lower our own bar so we sell into
    # the current, still-healthy price instead of after their sale craters it.
    # In the wind-down window, ignore the threshold entirely -- any price
    # beats letting it sit unsold in the shed when the season ends.
    sellable = list(config["crops"]) + [a["product"] for a in ANIMALS.values()] + ["FERTILIZER"]
    sell_fraction = _dynamic_sell_fraction(config, day, money)
    for item in sellable:
        qty = shed.get(item, 0)
        if qty <= 0:
            continue
        if winding_down:
            orders.append(["SELL", item, min(qty, config["max_sell_chunk"])])
            continue
        # Tried forcing a sale once backlog crossed a threshold (same idea
        # as wind-down's unconditional sell, but quantity-triggered) --
        # reverted: unlike crops, shed inventory doesn't decay while
        # waiting, so a forced sale at a crashed price permanently
        # realizes a loss instead of waiting for the (slow but real) price
        # recovery -- measured strictly worse in a 6-episode check (see
        # docs/tests/LOG.md). `sell_backlog_multiple` stays in
        # DEFAULT_CONFIG/KNOB_SPECS/optimize.py's search space (harmless,
        # unused here) in case a smarter version of this idea gets
        # revisited later.
        price = prices.get(item, _base_price(item))
        threshold = sell_fraction * _base_price(item)
        # Offensive crash: if the opponent's exposure to this item clearly
        # exceeds ours (both in absolute gap and ratio -- see
        # crash_sell_margin/crash_sell_multiplier's DEFAULT_CONFIG comment),
        # dump our smaller holding to depress the price ahead of their
        # bigger one landing, ignoring the normal price threshold/chunk
        # size entirely -- the point is to move the price, not sell well.
        if config["crash_sell_enabled"]:
            opp_exposure = opponent_supply.get(item, 0)
            our_exposure = (our_supply or {}).get(item, 0) + qty
            if (
                opp_exposure - our_exposure >= config["crash_sell_margin"]
                and opp_exposure >= our_exposure * config["crash_sell_multiplier"]
            ):
                orders.append(["SELL", item, min(qty, config["crash_sell_chunk"])])
                continue
        if config["opponent_awareness_enabled"] and opponent_supply.get(item, 0) >= config["opponent_incoming_threshold"]:
            # How hard to race the opponent's incoming dump isn't one fixed
            # number -- it scales with how much of a threat THIS dump
            # actually is: an opponent running 80% of their land in one
            # crop has a much bigger stake in it (and a much bigger dump
            # coming) than one growing it as a side crop, and a small-scale
            # opponent's dump barely moves the shared price no matter how
            # concentrated. Ablation (docs/tests/LOG.md) found the single
            # best static discount for one opponent was a clear loss
            # against another -- the right amount of race genuinely depends
            # on who's across the board, so make it a function of what we
            # can actually observe about them instead of one guess.
            concentration = opponent_concentration.get(item, 0.0)
            threat = concentration * opponent_scale
            dynamic_discount = config["opponent_race_discount"] - config["opponent_concentration_sensitivity"] * threat
            threshold *= max(0.1, min(1.0, dynamic_discount))
        if price >= threshold:
            orders.append(["SELL", item, min(qty, config["max_sell_chunk"])])

    # Every gate below used to check the SAME `money` snapshot read at the
    # top of this function, never accounting for what earlier orders THIS
    # SAME TURN had already committed to spend -- so on any turn where
    # land + hire + an animal were each individually affordable but not all
    # three combined, we'd queue all of them anyway, and whichever the
    # engine processes last would silently lose the race for cash (orders
    # are "processed in order, per player" -- docs/GAME_GUIDE.md). Since
    # BUY_ANIMAL was the last block in this function, it was the most
    # likely to get squeezed out -- a real, previously-undiagnosed
    # contributor to animals growing much more slowly than the strong
    # opponents' (see docs/tests/LOG.md, replay review). `remaining` tracks
    # an honest running balance across every order queued below; sell
    # proceeds aren't added speculatively (same-turn availability isn't
    # guaranteed), only spends are subtracted, so this is a conservative
    # fix, not an optimistic one.
    remaining = money

    # Keep at least one seed on hand per grown crop so a unit can always
    # plant whatever the scoring function currently prefers.
    for crop in config["crops"]:
        if seeds.get(crop, 0) == 0 and remaining >= max(CROPS[crop]["seed"], config["seed_money_floor"]):
            orders.append(["BUY_SEED", crop, 1])
            remaining -= CROPS[crop]["seed"]

    # Animals first: replay review of the strong opponents in our benchmark
    # pool shows them hyper-optimizing the *current* footprint with animals
    # (packing the field they already have, not racing to unlock more land)
    # rather than treating land expansion as the priority -- land moved
    # below hire/animal in this ordering to match, and to make sure animal
    # purchases get first claim on `remaining` each turn instead of last.
    animal_phase_ok = not in_crisis and (
        not config["phase_gate_enabled"] or day <= config["animal_purchase_last_day"]
    )
    if config["animal_enabled"] and started_up and not winding_down and animal_phase_ok:
        # Only buy up to the number of slots that are actually empty, minus
        # whatever's already bought-but-not-placed (shed + carried) -- a
        # slot being "empty" doesn't mean nothing is already en route to it.
        if config["enable_coop"]:
            goose_pending = shed.get("GOOSE", 0) + _carried_total(private, "GOOSE")
            goose_cost = ANIMALS["GOOSE"]["cost"]
            if config["roi_gate_enabled"]:
                goose_price = prices.get(ANIMALS["GOOSE"]["product"], _base_price(ANIMALS["GOOSE"]["product"]))
                goose_ok = _animal_roi_ok("GOOSE", goose_price, day, config)
                goose_cushion = _cushion(reserve_floor)
            else:
                goose_ok = True
                goose_cushion = _cushion(max(reserve_floor, config["animal_reserve_multiple"] * opening_scale * goose_cost))
            if goose_ok and len(info["empty_coop"]) > goose_pending and remaining >= goose_cost + goose_cushion:
                orders.append(["BUY_ANIMAL", "GOOSE", 1])
                remaining -= goose_cost
        if info["empty_pasture"]:
            best = max(
                ("COW", "SHEEP"),
                key=lambda a: _animal_score(a, prices.get(ANIMALS[a]["product"], _base_price(ANIMALS[a]["product"]))),
            )
            pasture_pending = shed.get("COW", 0) + shed.get("SHEEP", 0) + _carried_total(private, "COW") + _carried_total(private, "SHEEP")
            # Pasture structures are free to build (see _plan_units), so
            # they can outpace how many animals we can actually afford --
            # target_structures alone doesn't throttle spending. Animals
            # cost $400-500 each, far more than a hire, so they get their
            # own (higher) reserve multiple rather than sharing hiring's,
            # otherwise buying one for every empty pasture the moment it's
            # built drains the day-0 cash pile that hiring/land also need.
            best_cost = ANIMALS[best]["cost"]
            if config["roi_gate_enabled"]:
                best_price = prices.get(ANIMALS[best]["product"], _base_price(ANIMALS[best]["product"]))
                animal_ok = _animal_roi_ok(best, best_price, day, config)
                animal_cushion = _cushion(reserve_floor)
            else:
                animal_ok = True
                animal_cushion = _cushion(max(reserve_floor, config["animal_reserve_multiple"] * opening_scale * best_cost))
            if (
                animal_ok
                and len(info["empty_pasture"]) > pasture_pending
                and remaining >= best_cost + animal_cushion
            ):
                orders.append(["BUY_ANIMAL", best, 1])
                remaining -= best_cost

    # Hire: only if there's enough pending work to keep another hand busy
    # (backlog per current unit exceeds the ratio) AND we can comfortably
    # absorb the (fibonacci-growing) cost. Without the backlog check, extra
    # hands with nothing to do just wander to the shed and DROP/PASS. Not
    # during wind-down -- a hand hired with 1-2 days left can't earn back
    # even its own trivial cost, and existing hands are enough to keep
    # harvesting/selling out whatever's left.
    hires_today = farm.get("hires_today", 0)
    if not winding_down and hires_today < config["max_hires_per_day"]:
        unit_count = 1 + len(farm.get("hands", []) or [])
        backlog = (
            len(info["harvest"]) + len(info["water"]) + len(info["feed"])
            + len(info["weeds"]) + len(info["fertilize"]) + len(info["empty"])
        )
        # fib(n) with fib(0)=1,fib(1)=1,fib(2)=2,... matches the engine's cost curve.
        a, b = 1, 1
        for _ in range(hires_today):
            a, b = b, a + b
        hire_cost = a
        # hire_money_floor is its own fixed (unscaled) floor, same reason as
        # money_reserve -- a hire is cheap enough that scaling its floor by
        # risk_scale would barely matter either way, and keeping it fixed
        # keeps the "own dedicated smaller floor than land/animals" property
        # this key was originally introduced for.
        hire_cushion = _cushion(max(config["hire_money_floor"], config["hire_reserve_multiple"] * opening_scale * hire_cost))
        # Backlog can't justify a hire on day 0 -- nothing's been bought or
        # planted yet, so there IS no backlog regardless of how many hands
        # would actually be useful once the day-0 spending spree lands. This
        # is the one gate the opening window overrides by target count
        # rather than by relaxing a reserve multiplier, still subject to the
        # same affordability check as any other hire.
        opening_hire_forced = (
            config["opening_enabled"] and day == 0
            and hires_today < config["opening_hires_day0"]
        )
        if (
            (opening_hire_forced or backlog > unit_count * config["hire_backlog_ratio"])
            and remaining >= hire_cost + hire_cushion
        ):
            orders.append(["HIRE"])
            remaining -= hire_cost

    # Land: buy the next quadrant as soon as affordable, rather than waiting
    # for high occupancy of the current footprint. Occupancy is a lagging,
    # circular signal -- a thin labor force can't reach high utilization in
    # the first place, so an occupancy-gated trigger can permanently starve
    # itself of the very land expansion that would fix the labor shortage.
    # The #1 player buys NE by day 5 and SW by day 10 on a near-fixed
    # schedule that falls straight out of "buy it the moment you can afford
    # it" given their income curve, not out of watching occupancy.
    # ...but land is only worth buying once the crew can actually walk to
    # and work it -- a wider board means more travel time per action, so
    # unlocking a 4th quadrant while half the first three still sit empty
    # just creates more unreachable dead land, not more income (confirmed
    # locally: an earlier version that bought on affordability alone ended
    # up with ~half of all unlocked tiles permanently empty). Require
    # decent utilization of the current footprint too, matching land growth
    # to demonstrated crew capacity instead of pure cash availability.
    n_unlocked_extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
    unlocked = info["unlocked"]
    utilization = info["occupied"] / unlocked if unlocked > 0 else 0
    land_phase_ok = not in_crisis and (
        not config["phase_gate_enabled"]
        or config["season_days"] - day >= config["land_min_days_left"]
    )
    if (
        not winding_down
        and land_phase_ok
        and day >= config["land_startup_days"]
        and utilization >= config["land_utilization_threshold"]
        and 0 <= n_unlocked_extra < len(LAND_PRICES)
    ):
        next_land_cost = LAND_PRICES[n_unlocked_extra]
        if config["roi_gate_enabled"]:
            land_ok = _land_roi_ok(next_land_cost, day, n_unlocked_extra + 1, unlocked, config, prices)
            land_cushion = _cushion(reserve_floor)
        else:
            land_ok = True
            # opening_scale, not scale -- found 2026-08-22 (audit): every other
            # cushion (animal, hire) already gets the opening-window discount,
            # land was the one missed. Currently a no-op for the live champion
            # (land_startup_days=6 > opening_days=1, so land structurally can't
            # fire during the opening window at all) but that relationship is
            # not enforced anywhere -- a future search sampling land_startup_days
            # low would have silently skipped the discount everything else gets.
            land_cushion = _cushion(max(reserve_floor, config["land_reserve_multiple"] * opening_scale * next_land_cost))
        if land_ok and remaining >= next_land_cost + land_cushion:
            orders.append(["BUY_LAND"])
            remaining -= next_land_cost

    # Top up fertilizer stock a little if we're not collecting enough from
    # animals yet and there's something worth fertilizing. Off by default --
    # animal-collected fertilizer is free, buying it is a marginal spend.
    if config["buy_fertilizer"] and started_up and info["fertilize"]:
        fert_total = shed.get("FERTILIZER", 0) + _carried_total(private, "FERTILIZER")
        if fert_total < 3 and remaining >= _base_price("FERTILIZER") + reserve_floor:
            orders.append(["BUY_PRODUCT", "FERTILIZER", 1])
            remaining -= _base_price("FERTILIZER")

    # Emergency wheat buy so animals don't starve if we're not growing any --
    # only when truly none is available anywhere (shed or carried), so this
    # can't re-trigger just because a unit is mid-transit with a full load.
    wheat_total = shed.get("WHEAT", 0) + _carried_total(private, "WHEAT")
    if info["feed"] and wheat_total == 0 and remaining >= _base_price("WHEAT") + reserve_floor:
        orders.append(["BUY_PRODUCT", "WHEAT", min(len(info["feed"]), 3)])
        remaining -= _base_price("WHEAT")

    return orders[:10]  # maxMarketOrdersPerTurn default; extras would be dropped anyway.


# --------------------------------------------------------------------------
# RL policy hook -- a trained network can re-decide these strategic knobs
# once per day (not per turn: 30 decisions/game is tractable for RL, 720
# isn't, and none of these need finer granularity than a day). Everything
# mechanical (movement, task priority, harvesting) stays the proven
# rule-based code above completely untouched -- the network only ever
# outputs values that get merged into `cfg` and read by the same functions
# Optuna already tunes. See train_rl.py for the training side; this module
# stays torch-free on purpose so the submission build never needs torch.
# --------------------------------------------------------------------------

# (key, low, high, is_int) -- ranges match optimize.py's Optuna search space
# so a trained policy and a searched static config are directly comparable.
KNOB_SPECS = [
    ("sell_fraction_base", 0.2, 0.8, False),
    ("sell_fraction_day_weight", -0.3, 0.3, False),
    ("sell_fraction_cash_weight", -0.3, 0.3, False),
    ("hire_reserve_multiple", 0.5, 6.0, False),
    ("max_hires_per_day", 0, 8, True),
    ("land_utilization_threshold", 0.4, 0.95, False),
    ("pasture_target_ratio", 0.0, 1.2, False),
    ("animal_reserve_multiple", 1.0, 5.0, False),
    ("land_reserve_multiple", 0.2, 4.0, False),
    # ROI payback gate -- see DEFAULT_CONFIG's roi_gate_enabled comment. New,
    # unverified; when enabled this replaces the two multiples above.
    ("roi_margin", 1.0, 3.0, False),
    # Phase-window gates -- see DEFAULT_CONFIG's phase_gate_enabled comment.
    # New, unverified; additive on top of the existing gates.
    ("land_min_days_left", 0, 20, True),
    ("animal_purchase_last_day", 5, 27, True),
    ("crisis_backlog_ratio", 0.5, 6.0, False),
    ("hire_backlog_ratio", 0.3, 4.0, False),
    ("diversification_weight", 0.0, 1.0, False),
    # Shadow prices -- see DEFAULT_CONFIG's comment. Unverified defaults,
    # first thing this range should be searched to sanity-check.
    ("lambda_labor", 0.0, 15.0, False),
    ("lambda_land", 0.0, 80.0, False),
    # Cash-flow lookahead -- see DEFAULT_CONFIG's comment. Unverified.
    ("income_lookahead_days", 0, 5, True),
    ("income_discount", 0.0, 1.0, False),
    # Scripted opening window -- see DEFAULT_CONFIG's opening_enabled
    # comment. opening_enabled itself is a bool, not searched here (same
    # reason animal_enabled/enable_coop aren't -- this list is numeric-only,
    # toggled separately in optimize.py's sample_config).
    ("opening_days", 0, 5, True),
    ("opening_reserve_scale", 0.0, 1.0, False),
    ("opening_hires_day0", 0, 8, True),
    # Narrowed from (20, 500) -- money_reserve is now a small fixed survival
    # floor (see its DEFAULT_CONFIG comment), not a spending-policy buffer,
    # so its useful range is the same scale as hire_money_floor/
    # seed_money_floor, not the old value's magnitude.
    # Relative-wealth knobs, restored 2026-08-22 (see relative_wealth_enabled).
    ("wealth_margin_scale", 500.0, 10000.0, False),
    ("risk_sensitivity", 0.0, 2.0, False),
    ("risk_scale_min", 0.7, 1.5, False),
    ("risk_scale_max", 1.0, 4.0, False),
    ("money_reserve", 10, 150, True),
    ("max_sell_chunk", 3, 20, True),
    # sell_backlog_multiple deliberately NOT here -- it's genuinely unused by
    # _market_orders (the force-sell logic that read it was reverted, see
    # that revert's comment a few hundred lines up), so searching it just
    # burns a real dimension on a knob with zero behavioral effect. Stays in
    # DEFAULT_CONFIG in case a smarter version of the idea gets revisited.
    ("opponent_concentration_sensitivity", 0.0, 1.0, False),
    # Item-holding crash-sell mechanic -- see DEFAULT_CONFIG's
    # crash_sell_enabled comment. New, unverified.
    ("crash_sell_margin", 1, 20, True),
    ("crash_sell_multiplier", 1.0, 3.0, False),
    ("crash_sell_chunk", 10, 100, True),
    # Shed-distance tile-placement penalty -- see DEFAULT_CONFIG's
    # cluster_shed_weight comment. New, unverified.
    ("cluster_shed_weight", 0.0, 2.0, False),
] + [
    # One entry per dispatch tier (see DEFAULT_CONFIG's priority_weight_*
    # comment) -- generated from PRIORITY_TIER_NAMES so this list can't
    # silently drift out of sync with what _plan_units actually reads.
    (f"priority_weight_{name}", 0.0, 100.0, False) for name in PRIORITY_TIER_NAMES
]
KNOB_KEYS = [k for k, *_ in KNOB_SPECS]

# State features fed to the policy, in this fixed order. Kept small and
# hand-picked (not the raw board) since the mechanical layer already turns
# the board into a handful of meaningful numbers every turn (_scan_farm) --
# reusing that instead of learning grid perception from scratch is exactly
# what makes this tractable to train on a single GPU in hours, not days.
FEATURE_NAMES = [
    "day_frac", "days_left_frac", "cash_frac", "opp_cash_frac",
    "land_frac", "occupancy", "unit_frac", "opp_unit_frac", "opp_land_frac",
    "wheat_price", "carrot_price", "tomato_price", "strawberry_price", "melon_price",
    "egg_price", "milk_price", "wool_price",
]
# Deliberately separate from config["cash_scale"] (which normalizes OUR OWN
# sell-aggressiveness in _dynamic_sell_fraction, tuned around a few thousand)
# -- reusing it here saturated cash_frac/opp_cash_frac at 1.0 well before
# day 10 against real strong opponents (kawa reaches ~$15k by day 12, $150k+
# by day 29), making the RL policy unable to tell "modestly ahead/behind"
# from "astronomically ahead/behind" for most of any competitive game. See
# docs/tests/LOG.md.
RL_MONEY_SCALE = 50000.0


def extract_features(obs, cfg):
    """Pure function, obs -> fixed-length float list (see FEATURE_NAMES).
    No torch here -- the network lives in train_rl.py and calls this."""
    farms = obs.get("farms", []) or []
    player = obs.get("player", 0)
    if not farms or player >= len(farms):
        return [0.0] * len(FEATURE_NAMES)
    farm = farms[player]
    opponent_farm = next((f for i, f in enumerate(farms) if i != player), None)
    day = obs.get("day", 0)
    season_days = max(1, cfg["season_days"])
    prices = (obs.get("market", {}) or {}).get("prices", {}) or {}

    board_size = len(farm["tiles"])
    info = _scan_farm(farm, board_size, day)
    unit_count = 1 + len(farm.get("hands", []) or [])
    land_frac = len(farm.get("unlocked_quadrants", ["NW"])) / 4.0
    occupancy = info["occupied"] / info["unlocked"] if info["unlocked"] else 0.0

    if opponent_farm is not None:
        opp_money = opponent_farm.get("money", 0.0)
        opp_unit_count = 1 + len(opponent_farm.get("hands", []) or [])
        opp_land_frac = len(opponent_farm.get("unlocked_quadrants", ["NW"])) / 4.0
    else:
        opp_money, opp_unit_count, opp_land_frac = 0.0, 1, 0.25

    def price_ratio(item):
        base = _base_price(item)
        return (prices.get(item, base) / base) if base else 1.0

    return [
        day / season_days,
        max(0.0, season_days - day) / season_days,
        min(1.0, farm.get("money", 0.0) / RL_MONEY_SCALE),
        min(1.0, opp_money / RL_MONEY_SCALE),
        land_frac,
        occupancy,
        min(1.0, unit_count / 13.0),
        min(1.0, opp_unit_count / 13.0),
        opp_land_frac,
        price_ratio("WHEAT"), price_ratio("CARROT"), price_ratio("TOMATO"),
        price_ratio("STRAWBERRY"), price_ratio("MELON"),
        price_ratio("EGG"), price_ratio("MILK"), price_ratio("WOOL"),
    ]


def decode_knobs(raw_actions):
    """raw_actions: iterable of floats in [0, 1] (one per KNOB_SPECS entry,
    e.g. a sigmoid-squashed network output) -> {config_key: value} scaled
    into each knob's real range, rounded for the int-valued ones."""
    out = {}
    for (key, lo, hi, is_int), a in zip(KNOB_SPECS, raw_actions):
        a = min(1.0, max(0.0, a))
        value = lo + a * (hi - lo)
        out[key] = int(round(value)) if is_int else value
    return out


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def make_agent(config=None, policy_fn=None):
    """policy_fn, if given, is called as policy_fn(features) -> {config_key:
    value} once at the start of each in-game day (features = extract_features
    output); the returned dict is merged over cfg for that whole day. Used
    by train_rl.py during training and can be used for a trained policy's
    submission agent too."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    state = {"last_day": None, "unit_targets": {}}

    def agent(obs):
        try:
            farms = obs.get("farms", [])
            player = obs.get("player", 0)
            if not farms or player >= len(farms):
                return dict(SAFE_FALLBACK)

            farm = farms[player]
            opponent_farm = next((f for i, f in enumerate(farms) if i != player), None)
            private = obs.get("private", {}) or {}
            day = obs.get("day", 0)
            board_size = len(farm["tiles"])
            prices = (obs.get("market", {}) or {}).get("prices", {}) or {}

            turn_cfg = cfg
            if policy_fn is not None and day != state["last_day"]:
                state["last_day"] = day
                features = extract_features(obs, cfg)
                overrides = policy_fn(features)
                if overrides:
                    turn_cfg = dict(cfg)
                    turn_cfg.update(overrides)
                    state["cfg"] = turn_cfg
            elif policy_fn is not None and "cfg" in state:
                turn_cfg = state["cfg"]

            info = _scan_farm(farm, board_size, day)
            opponent_supply, opponent_concentration, opponent_scale = _opponent_profile(
                opponent_farm, board_size, day, turn_cfg["opponent_lookahead_days"]
            )
            our_supply, _, _ = _opponent_profile(farm, board_size, day, turn_cfg["income_lookahead_days"])
            market_orders = _market_orders(
                farm, private, info, turn_cfg, prices, day,
                opponent_supply, opponent_concentration, opponent_scale,
                opponent_farm, our_supply,
            )
            farmer_action, hands_actions = _plan_units(
                farm, private, board_size, day, info, turn_cfg, prices, state["unit_targets"]
            )

            return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}
        except Exception as exc:  # noqa: BLE001 -- deliberate catch-all safety net
            print(f"robust_agent: swallowed exception, falling back to PASS: {exc!r}", file=sys.stderr)
            return dict(SAFE_FALLBACK)

    return agent


robust_agent = make_agent()
