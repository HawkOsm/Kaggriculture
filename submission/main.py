"""Kaggriculture competition submission.

Self-contained build of src/robust_agent.py + src/farm_utils.py, with the
tuned config from src/best_config.json baked in directly (merged over
DEFAULT_CONFIG for the keys best_config.json doesn't have yet -- see
docs/tests/LOG.md, "best_config.json verification" and later entries, for
how this config was found and verified: 6/6 vs the untuned default, and it
held up against a second search round's attempted replacement).

Single-file on purpose: no sibling-file imports to get wrong on a real
submission with a 5/day limit. If you change src/robust_agent.py,
src/farm_utils.py, or src/best_config.json, regenerate this file rather than
hand-editing it out of sync.

Verified locally before submission:
  python src/run_match.py robust_agent:robust_agent melon_maxxer:melon_maxxer
See docs/tests/LOG.md for the full verification history.
"""

import sys

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    MARKET_PARAMS,
)

CONFIG = {
    "sell_fraction": 0.5268146257913258,
    "max_sell_chunk": 16,
    "money_reserve": 150,
    "seed_money_floor": 29,
    "hire_reserve_multiple": 5.936052940507935,
    "max_hires_per_day": 1,
    "land_utilization_threshold": 0.8728656897907402,
    "animal_enabled": False,
    "max_structures": 5,
    "startup_days": 4,
    "structure_money_threshold": 450,
    "buy_fertilizer": False,
    "crops": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"],
    "hire_backlog_ratio": 1.2,
    "diversification_weight": 0.15,
    "opponent_awareness_enabled": True,
    "opponent_incoming_threshold": 3,
    "opponent_race_discount": 0.7,
    "opponent_lookahead_days": 2,
}

SAFE_FALLBACK = {"farmer": ["PASS"], "hands": [], "market": []}


# --------------------------------------------------------------------------
# farm_utils.py -- movement/geometry helpers
# --------------------------------------------------------------------------

def step_toward(pos, target):
    fx, fy = pos
    tx, ty = target
    if fx > tx:
        return "WEST"
    if fx < tx:
        return "EAST"
    if fy > ty:
        return "NORTH"
    if fy < ty:
        return "SOUTH"
    return None


def closest(pos, coords):
    if not coords:
        return None
    fx, fy = pos
    return min(coords, key=lambda c: abs(c[0] - fx) + abs(c[1] - fy))


def act_or_move(pos, target, act_action):
    if pos == target:
        return act_action
    step = step_toward(pos, target)
    return [step] if step else ["PASS"]


def shed_tiles(board_size):
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


# --------------------------------------------------------------------------
# robust_agent.py -- scoring
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
    return _crop_score(crop, price) / (1 + crop_counts.get(crop, 0) * weight)


def _animal_score(animal, price):
    a = ANIMALS[animal]
    return price / max(1, a["interval"])


def _base_price(item):
    return MARKET_PARAMS.get(item, {}).get("base", 0)


# --------------------------------------------------------------------------
# robust_agent.py -- farm scanning
# --------------------------------------------------------------------------

def _plant_harvest_ready(tile, day):
    crop = tile["crop"]
    c = CROPS[crop]
    age = day - tile["planted_day"]
    if age < c["first_yield_day"]:
        return False
    if not c["ongoing"]:
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
# robust_agent.py -- task assignment
# --------------------------------------------------------------------------

def _plan_units(farm, private, board_size, day, info, config, prices):
    units = [(0, tuple(farm["farmer"]))]
    for i, pos in enumerate(farm.get("hands", []) or []):
        units.append((i + 1, tuple(pos)))

    inventories = private.get("inventories", []) or []
    shed = private.get("shed", {}) or {}

    shed_spots = shed_tiles(board_size)
    seeds = private.get("seeds", {}) or {}

    crop_pool = [c for c in config["crops"] if seeds.get(c, 0) > 0]
    crop_counts = dict(info.get("crop_counts", {}))

    def inv_of(idx):
        return inventories[idx] if idx < len(inventories) else {}

    farmer_action = ["PASS"]
    hands_actions = []

    for idx, pos in units:
        inv = inv_of(idx)
        action = None
        target = None

        if inv.get("WHEAT", 0) > 0 and info["feed"]:
            target = closest(pos, info["feed"])
            action = act_or_move(pos, target, ["FEED"])
            info["feed"].remove(target)
        elif info["harvest"]:
            target = closest(pos, info["harvest"])
            action = act_or_move(pos, target, ["HARVEST"])
            info["harvest"].remove(target)
        elif inv.get("FERTILIZER", 0) > 0 and info["fertilize"]:
            target = closest(pos, info["fertilize"])
            action = act_or_move(pos, target, ["FERTILIZE"])
            info["fertilize"].remove(target)
        elif info["collect_fertilizer"]:
            target = closest(pos, info["collect_fertilizer"])
            action = act_or_move(pos, target, ["COLLECT_FERTILIZER"])
            info["collect_fertilizer"].remove(target)
        elif info["water"]:
            target = closest(pos, info["water"])
            action = act_or_move(pos, target, ["WATER"])
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
        elif config["animal_enabled"] and inv.get("GOOSE", 0) > 0 and info["empty_coop"]:
            target = closest(pos, info["empty_coop"])
            action = act_or_move(pos, target, ["PLACE", "GOOSE"])
            info["empty_coop"].remove(target)
        elif config["animal_enabled"] and (inv.get("COW", 0) > 0 or inv.get("SHEEP", 0) > 0) and info["empty_pasture"]:
            animal = "COW" if inv.get("COW", 0) > 0 else "SHEEP"
            target = closest(pos, info["empty_pasture"])
            action = act_or_move(pos, target, ["PLACE", animal])
            info["empty_pasture"].remove(target)
        elif config["animal_enabled"] and info["empty_coop"] and shed.get("GOOSE", 0) > 0:
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
            action = act_or_move(pos, target, ["CARE"])
            info["care"].remove(target)
        elif info["weeds"]:
            target = closest(pos, info["weeds"])
            action = act_or_move(pos, target, ["DIG"])
            info["weeds"].remove(target)
        elif info["empty"] and crop_pool:
            target = closest(pos, info["empty"])
            best_crop = max(
                crop_pool,
                key=lambda c: _diversified_crop_score(
                    c, prices.get(c, _base_price(c)), crop_counts, config["diversification_weight"]
                ),
            )
            action = act_or_move(pos, target, ["PLANT", best_crop])
            info["empty"].remove(target)
            if pos == target:
                seeds[best_crop] = seeds.get(best_crop, 0) - 1
                if seeds[best_crop] <= 0:
                    crop_pool = [c for c in crop_pool if c != best_crop]
                crop_counts[best_crop] = crop_counts.get(best_crop, 0) + 1
        elif (
            config["animal_enabled"]
            and day >= config["startup_days"]
            and farm["money"] - config["money_reserve"] >= config["structure_money_threshold"]
            and info["empty"]
            and (len(info["empty_coop"]) + len(info["empty_pasture"])) < config["max_structures"]
        ):
            target = closest(pos, info["empty"])
            build = "BUILD_COOP" if len(info["empty_coop"]) <= len(info["empty_pasture"]) else "BUILD_PASTURE"
            action = act_or_move(pos, target, [build])
            info["empty"].remove(target)
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
# robust_agent.py -- market orders
# --------------------------------------------------------------------------

def _carried_total(private, item):
    return sum(inv.get(item, 0) for inv in (private.get("inventories", []) or []))


def _opponent_incoming_supply(opponent_farm, board_size, day, lookahead_days):
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

    sellable = list(config["crops"]) + [a["product"] for a in ANIMALS.values()]
    for item in sellable:
        qty = shed.get(item, 0)
        if qty <= 0:
            continue
        price = prices.get(item, _base_price(item))
        threshold = config["sell_fraction"] * _base_price(item)
        if config["opponent_awareness_enabled"] and opponent_supply.get(item, 0) >= config["opponent_incoming_threshold"]:
            threshold *= config["opponent_race_discount"]
        if price >= threshold:
            orders.append(["SELL", item, min(qty, config["max_sell_chunk"])])

    for crop in config["crops"]:
        if seeds.get(crop, 0) == 0 and money >= max(CROPS[crop]["seed"], config["seed_money_floor"]):
            orders.append(["BUY_SEED", crop, 1])

    unlocked = info["unlocked"]
    if started_up and unlocked > 0:
        utilization = info["occupied"] / unlocked
        if utilization >= config["land_utilization_threshold"]:
            orders.append(["BUY_LAND"])

    hires_today = farm.get("hires_today", 0)
    if hires_today < config["max_hires_per_day"]:
        unit_count = 1 + len(farm.get("hands", []) or [])
        backlog = (
            len(info["harvest"]) + len(info["water"]) + len(info["feed"])
            + len(info["weeds"]) + len(info["fertilize"]) + len(info["empty"])
        )
        a, b = 1, 1
        for _ in range(hires_today):
            a, b = b, a + b
        hire_cost = a
        if (
            backlog > unit_count * config["hire_backlog_ratio"]
            and money - reserve >= hire_cost * config["hire_reserve_multiple"]
        ):
            orders.append(["HIRE"])

    if config["animal_enabled"] and started_up:
        goose_pending = shed.get("GOOSE", 0) + _carried_total(private, "GOOSE")
        if len(info["empty_coop"]) > goose_pending and money - reserve >= ANIMALS["GOOSE"]["cost"]:
            orders.append(["BUY_ANIMAL", "GOOSE", 1])
        if info["empty_pasture"]:
            best = max(
                ("COW", "SHEEP"),
                key=lambda a: _animal_score(a, prices.get(ANIMALS[a]["product"], _base_price(ANIMALS[a]["product"]))),
            )
            pasture_pending = shed.get("COW", 0) + shed.get("SHEEP", 0) + _carried_total(private, "COW") + _carried_total(private, "SHEEP")
            if len(info["empty_pasture"]) > pasture_pending and money - reserve >= ANIMALS[best]["cost"]:
                orders.append(["BUY_ANIMAL", best, 1])

    if config["buy_fertilizer"] and started_up and info["fertilize"]:
        fert_total = shed.get("FERTILIZER", 0) + _carried_total(private, "FERTILIZER")
        if fert_total < 3 and money - reserve >= _base_price("FERTILIZER"):
            orders.append(["BUY_PRODUCT", "FERTILIZER", 1])

    wheat_total = shed.get("WHEAT", 0) + _carried_total(private, "WHEAT")
    if info["feed"] and wheat_total == 0 and money - reserve >= _base_price("WHEAT"):
        orders.append(["BUY_PRODUCT", "WHEAT", min(len(info["feed"]), 3)])

    return orders[:10]


# --------------------------------------------------------------------------
# Entry point -- must be the last callable defined in this file.
# kaggle_environments loads a file agent by exec'ing it and taking the LAST
# callable bound in the resulting module namespace (see
# kaggle_environments.agent.get_last_callable), not a specifically-named
# function. Do not add any code after the `agent = make_agent()` line below.
# --------------------------------------------------------------------------

def make_agent(config):
    cfg = dict(config)

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

            info = _scan_farm(farm, board_size, day)
            opponent_supply = _opponent_incoming_supply(opponent_farm, board_size, day, cfg["opponent_lookahead_days"])
            market_orders = _market_orders(farm, private, info, cfg, prices, day, opponent_supply)
            farmer_action, hands_actions = _plan_units(farm, private, board_size, day, info, cfg, prices)

            return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}
        except Exception as exc:  # noqa: BLE001 -- deliberate catch-all safety net
            print(f"kaggriculture submission: swallowed exception, falling back to PASS: {exc!r}", file=sys.stderr)
            return dict(SAFE_FALLBACK)

    return agent


agent = make_agent(CONFIG)
