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
    # Never spend below this bank balance, except cheap seed top-ups. Kept
    # low deliberately: the reigning #1 player runs as low as $73-760 for
    # the first ~10 days, pouring nearly everything into hiring/land/animals
    # while it still compounds for the rest of the season. Not near-zero,
    # though -- a $3000 start that's fully drained on day 0 (before any
    # crop has had time to mature) has no buffer to survive the multi-day
    # gap before first income, which is a real failure mode we hit tuning
    # this (see docs/tests/LOG.md).
    "money_reserve": 100,
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
    # Extra reserve multiple required specifically before BUY_ANIMAL for a
    # pasture ($400-500) -- much pricier than a hire, so it needs a
    # stronger brake than hire_reserve_multiple or it drains day-0 cash
    # before hiring/land get a turn at it.
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
    "opponent_awareness_enabled": True,
    "opponent_incoming_threshold": 3,
    "opponent_race_discount": 0.7,
    "opponent_lookahead_days": 2,
}

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


def _crop_score(crop, price):
    c = CROPS[crop]
    revenue = price * c["max_yield"]
    return (revenue - c["seed"]) / _crop_cycle_days(crop)


def _diversified_crop_score(crop, price, crop_counts, weight):
    # Discount by how much of this crop is already growing, so several units
    # deciding what to plant in the same turn don't all pile into whichever
    # single crop currently scores highest.
    return _crop_score(crop, price) / (1 + crop_counts.get(crop, 0) * weight)


def _animal_score(animal, price):
    a = ANIMALS[animal]
    # Steady-state $/day once producing; ignores the wheat feed cost and the
    # ramp-up to first_yield_day, both small relative to season length.
    return price / max(1, a["interval"])


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


def _plan_units(farm, private, board_size, day, info, config, prices, unit_targets=None):
    """Decide one action per unit (farmer + hands), consuming tiles from
    `info` so two units never chase the same tile in the same turn.

    `unit_targets`, if given, is a {unit_idx: (task_key, target)} dict
    mutated in place across calls (one call per turn) so a unit already
    walking toward a target keeps heading there instead of re-running the
    full priority ladder from scratch every turn. Without this, a unit
    several tiles into a walk toward (say) a water target gets pulled onto
    a newly-appeared, higher-priority-tier task the instant one shows up
    anywhere on the board, abandoning the walk and starting a new one --
    confirmed live via an action-type histogram showing ~74% of all
    unit-turns were pure movement, essentially unchanged from the original
    "76% movement" diagnosis despite later footprint-scaling fixes (see
    docs/tests/LOG.md). Committing to a target once picked, rather than
    re-litigating it every turn, is what actually brings that ratio down.
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

    def inv_of(idx):
        return inventories[idx] if idx < len(inventories) else {}

    # Target structure count scales with current labor so pasture capacity
    # builds out roughly in step with hands instead of all at once or only
    # once crop land runs out. Building itself is free (only the animal
    # costs money), so there's no cash reason to delay it.
    structures_committed = info["pasture_count"] + info["coop_count"]
    if config["season_days"] - day <= config["wind_down_days"]:
        target_structures = 0  # no time left for a new pasture to pay back
    else:
        target_structures = min(config["max_structures"], math.ceil(len(units) * config["pasture_target_ratio"]))

    farmer_action = ["PASS"]
    hands_actions = []

    def commit(idx, target, task_key, action):
        # Called after every pick below: remember it if the unit hasn't
        # arrived yet (still walking, needs to survive to next turn),
        # forget it once acted on (nothing left to remember).
        if pos == target:
            unit_targets.pop(idx, None)
        else:
            unit_targets[idx] = (task_key, target)
        return action

    for idx, pos in units:
        inv = inv_of(idx)
        action = None
        target = None

        # Sticky short-circuit: if this unit was already walking toward a
        # target last turn and that target is still valid (still present in
        # this turn's freshly-scanned info list), keep heading there instead
        # of re-running the whole priority ladder -- see docstring above.
        remembered = unit_targets.get(idx)
        if remembered is not None:
            task_key, remembered_target = remembered
            candidates = info.get(TASK_INFO_KEY.get(task_key, ""))
            if candidates is not None and remembered_target in candidates:
                target = remembered_target
                candidates.remove(target)
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
                    best_crop = max(
                        viable_crops,
                        key=lambda c: _diversified_crop_score(
                            c, prices.get(c, _base_price(c)), crop_counts, config["diversification_weight"]
                        ),
                    ) if viable_crops else None
                    if best_crop is not None:
                        action = act_or_move(pos, target, ["PLANT", best_crop])
                        if pos == target:
                            seeds[best_crop] = seeds.get(best_crop, 0) - 1
                            if seeds[best_crop] <= 0:
                                crop_pool = [c for c in crop_pool if c != best_crop]
                            crop_counts[best_crop] = crop_counts.get(best_crop, 0) + 1
                if pos == target:
                    unit_targets.pop(idx, None)
                else:
                    unit_targets[idx] = (task_key, target)
            else:
                unit_targets.pop(idx, None)

        if action is not None:
            pass
        elif inv.get("WHEAT", 0) > 0 and info["feed"]:
            target = closest(pos, info["feed"])
            action = commit(idx, target, "feed", act_or_move(pos, target, ["FEED"]))
            info["feed"].remove(target)
        elif info["harvest"]:
            target = closest(pos, info["harvest"])
            action = commit(idx, target, "harvest", act_or_move(pos, target, ["HARVEST"]))
            info["harvest"].remove(target)
        elif inv.get("FERTILIZER", 0) > 0 and info["fertilize"]:
            target = closest(pos, info["fertilize"])
            action = commit(idx, target, "fertilize", act_or_move(pos, target, ["FERTILIZE"]))
            info["fertilize"].remove(target)
        elif info["collect_fertilizer"]:
            target = closest(pos, info["collect_fertilizer"])
            action = commit(idx, target, "collect_fertilizer", act_or_move(pos, target, ["COLLECT_FERTILIZER"]))
            info["collect_fertilizer"].remove(target)
        elif info["water"]:
            target = closest(pos, info["water"])
            action = commit(idx, target, "water", act_or_move(pos, target, ["WATER"]))
            info["water"].remove(target)
        elif (
            inv.get("WHEAT", 0) == 0
            and info["feed"]
            and shed.get("WHEAT", 0) > 0
        ):
            target = closest(pos, shed_spots)
            action = act_or_move(pos, target, ["PICKUP", "WHEAT", min(shed["WHEAT"], 10)])
        elif (
            inv.get("FERTILIZER", 0) == 0
            and info["fertilize"]
            and shed.get("FERTILIZER", 0) > 0
        ):
            target = closest(pos, shed_spots)
            action = act_or_move(pos, target, ["PICKUP", "FERTILIZER", min(shed["FERTILIZER"], 10)])
        elif config["enable_coop"] and inv.get("GOOSE", 0) > 0 and info["empty_coop"]:
            target = closest(pos, info["empty_coop"])
            action = commit(idx, target, "empty_coop_place", act_or_move(pos, target, ["PLACE", "GOOSE"]))
            info["empty_coop"].remove(target)
        elif config["animal_enabled"] and (inv.get("COW", 0) > 0 or inv.get("SHEEP", 0) > 0) and info["empty_pasture"]:
            animal = "COW" if inv.get("COW", 0) > 0 else "SHEEP"
            target = closest(pos, info["empty_pasture"])
            action = commit(idx, target, "empty_pasture_place", act_or_move(pos, target, ["PLACE", animal]))
            info["empty_pasture"].remove(target)
        elif config["enable_coop"] and info["empty_coop"] and shed.get("GOOSE", 0) > 0:
            target = closest(pos, shed_spots)
            action = act_or_move(pos, target, ["PICKUP", "GOOSE", 1])
        elif (
            config["animal_enabled"]
            and info["empty_pasture"]
            and (shed.get("COW", 0) > 0 or shed.get("SHEEP", 0) > 0)
        ):
            animal = "COW" if shed.get("COW", 0) > 0 else "SHEEP"
            target = closest(pos, shed_spots)
            action = act_or_move(pos, target, ["PICKUP", animal, 1])
        elif info["care"]:
            target = closest(pos, info["care"])
            action = commit(idx, target, "care", act_or_move(pos, target, ["CARE"]))
            info["care"].remove(target)
        elif info["weeds"]:
            target = closest(pos, info["weeds"])
            action = commit(idx, target, "weeds", act_or_move(pos, target, ["DIG"]))
            info["weeds"].remove(target)
        elif (
            config["animal_enabled"]
            and info["empty"]
            and structures_committed < target_structures
        ):
            # Build pasture (or coop, if enabled) capacity in step with
            # current labor. Ranked above planting: BUILD_PASTURE is free,
            # so there's no cash trade-off, only a tile-allocation one, and
            # the #1 player builds pastures on day 0 alongside its first
            # crops rather than treating them as leftover-slack spending.
            target = closest(pos, info["empty"])
            if config["enable_coop"] and info["coop_count"] <= info["pasture_count"]:
                build = "BUILD_COOP"
            else:
                build = "BUILD_PASTURE"
            action = commit(idx, target, "empty_build", act_or_move(pos, target, [build]))
            info["empty"].remove(target)
            # Count this tile toward the target immediately (not only once
            # actually built) -- otherwise every unit still mid-walk this
            # turn would see the same unmet target and all pile onto
            # BUILD_PASTURE, overshooting it in a single turn.
            structures_committed += 1
            if pos == target:
                if build == "BUILD_COOP":
                    info["coop_count"] += 1
                else:
                    info["pasture_count"] += 1
        elif info["empty"] and (viable_crops := [c for c in crop_pool if season_ok.get(c, True)]):
            target = closest(pos, info["empty"])
            best_crop = max(
                viable_crops,
                key=lambda c: _diversified_crop_score(
                    c, prices.get(c, _base_price(c)), crop_counts, config["diversification_weight"]
                ),
            )
            action = commit(idx, target, "empty_plant", act_or_move(pos, target, ["PLANT", best_crop]))
            info["empty"].remove(target)
            if pos == target:
                seeds[best_crop] = seeds.get(best_crop, 0) - 1
                if seeds[best_crop] <= 0:
                    crop_pool = [c for c in crop_pool if c != best_crop]
                crop_counts[best_crop] = crop_counts.get(best_crop, 0) + 1
        elif inv:
            target = closest(pos, shed_spots)
            action = act_or_move(pos, target, ["DROP"])
        else:
            action = ["PASS"]

        if idx == 0:
            farmer_action = action
        else:
            hands_actions.append(action)

    return farmer_action, hands_actions


# --------------------------------------------------------------------------
# Market orders
# --------------------------------------------------------------------------

def _carried_total(private, item):
    return sum(inv.get(item, 0) for inv in (private.get("inventories", []) or []))


def _opponent_incoming_supply(opponent_farm, board_size, day, lookahead_days):
    """Estimate how much of each product the opponent is likely to dump on
    the market soon, from their public tiles: already-ripe crops/animal
    products, plus one-time crops that will mature within `lookahead_days`.
    A rough signal, not a guarantee they'll actually sell -- but a rational
    opponent usually does once something's ripe."""
    supply = {}
    if opponent_farm is None:
        return supply
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
                if tile["yield_units"] > 0 and _plant_harvest_ready(tile, day):
                    supply[crop] = supply.get(crop, 0) + tile["yield_units"]
                elif not c["ongoing"]:
                    days_to_ready = c["max_yield_day"] - (day - tile["planted_day"])
                    if 0 <= days_to_ready <= lookahead_days:
                        supply[crop] = supply.get(crop, 0) + c["max_yield"]
            elif kind in ("COOP", "PASTURE") and "animal" in tile and tile["yield_units"] > 0:
                product = ANIMALS[tile["animal"]]["product"]
                supply[product] = supply.get(product, 0) + tile["yield_units"]
    return supply


def _market_orders(farm, private, info, config, prices, day, opponent_supply=None):
    orders = []
    money = farm["money"]
    reserve = config["money_reserve"]
    shed = private.get("shed", {}) or {}
    seeds = private.get("seeds", {}) or {}
    started_up = day >= config["startup_days"]
    opponent_supply = opponent_supply or {}
    # Reward is money at game end, full stop -- unsold shed inventory and
    # freshly-hired hands with no time left to earn back their cost are pure
    # waste in the closing days. The #1 player visibly winds crop mix back
    # down in the final ~5 days rather than planting things that won't
    # mature; this is the same idea applied to market orders: stop paying
    # for anything that can't pay itself back, and dump inventory for
    # whatever it fetches instead of holding out for a better price that
    # will never be realized.
    winding_down = config["season_days"] - day <= config["wind_down_days"]

    # Sell everything sellable that's above threshold, in bounded chunks.
    # If the opponent's about to dump a lot of this item on the market
    # (visible from their public tiles), lower our own bar so we sell into
    # the current, still-healthy price instead of after their sale craters it.
    # In the wind-down window, ignore the threshold entirely -- any price
    # beats letting it sit unsold in the shed when the season ends.
    sellable = list(config["crops"]) + [a["product"] for a in ANIMALS.values()]
    sell_fraction = _dynamic_sell_fraction(config, day, money)
    for item in sellable:
        qty = shed.get(item, 0)
        if qty <= 0:
            continue
        if winding_down:
            orders.append(["SELL", item, min(qty, config["max_sell_chunk"])])
            continue
        price = prices.get(item, _base_price(item))
        threshold = sell_fraction * _base_price(item)
        if config["opponent_awareness_enabled"] and opponent_supply.get(item, 0) >= config["opponent_incoming_threshold"]:
            threshold *= config["opponent_race_discount"]
        if price >= threshold:
            orders.append(["SELL", item, min(qty, config["max_sell_chunk"])])

    # Keep at least one seed on hand per grown crop so a unit can always
    # plant whatever the scoring function currently prefers.
    for crop in config["crops"]:
        if seeds.get(crop, 0) == 0 and money >= max(CROPS[crop]["seed"], config["seed_money_floor"]):
            orders.append(["BUY_SEED", crop, 1])

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
    if (
        not winding_down
        and day >= config["land_startup_days"]
        and utilization >= config["land_utilization_threshold"]
        and 0 <= n_unlocked_extra < len(LAND_PRICES)
    ):
        next_land_cost = LAND_PRICES[n_unlocked_extra]
        if money - reserve >= next_land_cost:
            orders.append(["BUY_LAND"])

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
        if (
            backlog > unit_count * config["hire_backlog_ratio"]
            and money - config["hire_money_floor"] >= hire_cost * config["hire_reserve_multiple"]
        ):
            orders.append(["HIRE"])

    if config["animal_enabled"] and started_up and not winding_down:
        # Only buy up to the number of slots that are actually empty, minus
        # whatever's already bought-but-not-placed (shed + carried) -- a
        # slot being "empty" doesn't mean nothing is already en route to it.
        if config["enable_coop"]:
            goose_pending = shed.get("GOOSE", 0) + _carried_total(private, "GOOSE")
            if len(info["empty_coop"]) > goose_pending and money - reserve >= ANIMALS["GOOSE"]["cost"]:
                orders.append(["BUY_ANIMAL", "GOOSE", 1])
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
            if (
                len(info["empty_pasture"]) > pasture_pending
                and money - reserve >= ANIMALS[best]["cost"] * config["animal_reserve_multiple"]
            ):
                orders.append(["BUY_ANIMAL", best, 1])

    # Top up fertilizer stock a little if we're not collecting enough from
    # animals yet and there's something worth fertilizing. Off by default --
    # animal-collected fertilizer is free, buying it is a marginal spend.
    if config["buy_fertilizer"] and started_up and info["fertilize"]:
        fert_total = shed.get("FERTILIZER", 0) + _carried_total(private, "FERTILIZER")
        if fert_total < 3 and money - reserve >= _base_price("FERTILIZER"):
            orders.append(["BUY_PRODUCT", "FERTILIZER", 1])

    # Emergency wheat buy so animals don't starve if we're not growing any --
    # only when truly none is available anywhere (shed or carried), so this
    # can't re-trigger just because a unit is mid-transit with a full load.
    wheat_total = shed.get("WHEAT", 0) + _carried_total(private, "WHEAT")
    if info["feed"] and wheat_total == 0 and money - reserve >= _base_price("WHEAT"):
        orders.append(["BUY_PRODUCT", "WHEAT", min(len(info["feed"]), 3)])

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
    ("hire_backlog_ratio", 0.3, 4.0, False),
    ("diversification_weight", 0.0, 1.0, False),
    ("money_reserve", 20, 500, True),
    ("max_sell_chunk", 3, 20, True),
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
            opponent_supply = _opponent_incoming_supply(opponent_farm, board_size, day, turn_cfg["opponent_lookahead_days"])
            market_orders = _market_orders(farm, private, info, turn_cfg, prices, day, opponent_supply)
            farmer_action, hands_actions = _plan_units(
                farm, private, board_size, day, info, turn_cfg, prices, state["unit_targets"]
            )

            return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}
        except Exception as exc:  # noqa: BLE001 -- deliberate catch-all safety net
            print(f"robust_agent: swallowed exception, falling back to PASS: {exc!r}", file=sys.stderr)
            return dict(SAFE_FALLBACK)

    return agent


robust_agent = make_agent()
