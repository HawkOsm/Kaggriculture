"""Kaggriculture multi-crop diversification agent.

Same walk / plant / water / harvest / sell loop as melon_maxxer.py, but
grows a rotation of crops instead of just melon. Melon has the best price
but a 10-12 day growth cycle -- for most of the season a melon-only farm has
nothing to sell and is fully exposed to one commodity's price. Mixing in
wheat and carrot (2-3 day cycles) keeps cash flowing and spreads price risk
across crops that don't glut or spike together.

Deliberately out of scope here (see TUTORIAL.md "What's wrong with this
agent?" for melon_maxxer, most of which still applies): farm hands, land
expansion, fertilizing, animals. This is a diversification-only step.
"""

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, MARKET_PARAMS

from farm_utils import act_or_move

# Fixed rotation: three one-time crops with short-to-long cycles, plus one
# ongoing crop for steady mid-season income. Strawberry/animals are left for
# a later iteration.
GROWN_CROPS = ["WHEAT", "CARROT", "TOMATO", "MELON"]

SEED_COST = {c: CROPS[c]["seed"] for c in GROWN_CROPS}
MAX_YIELD_DAY = {c: CROPS[c]["max_yield_day"] for c in GROWN_CROPS}
BASE_PRICE = {c: MARKET_PARAMS[c]["base"] for c in GROWN_CROPS}

# Only sell once the market is paying at least this fraction of base price --
# keeps us from fire-selling into a glut, ours or the opponent's.
SELL_THRESHOLD = {c: round(BASE_PRICE[c] * 0.8) for c in GROWN_CROPS}

# Cap how much of a single crop we sell in one order. Selling is processed
# one unit at a time and the price moves as inventory grows, so dumping an
# entire shed's worth of one crop in one order (melon_maxxer's approach)
# can crash its own price mid-sale. A smaller chunk each turn lets the price
# partially recover between sells.
MAX_SELL_CHUNK = 8


def _tile_purpose(tile, day):
    """Return 'harvest', 'water', or None for an occupied plant tile."""
    if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
        return None
    crop = tile["crop"]
    purpose = None
    if tile["yield_units"] > 0:
        if CROPS[crop]["ongoing"]:
            # Ongoing crops (tomato) pay out on every scheduled tick, so
            # harvest as soon as there's anything sitting on the tile.
            purpose = "harvest"
        else:
            # One-time crops: wait for full maturity before harvesting so we
            # get the full watering-bonus yield in a single harvest -- an
            # early HARVEST is invalid before first_yield_day and just
            # wastes the turn.
            age = day - tile["planted_day"]
            max_day = MAX_YIELD_DAY.get(crop)
            if max_day is not None and age >= max_day:
                purpose = "harvest"
    if not tile["watered_today"]:
        purpose = "water" if purpose is None else purpose
    return purpose


def _next_crop_to_plant(seeds):
    """Pick whichever grown crop we're currently most short on seed for,
    among the crops we can actually afford more of -- keeps planting spread
    across the rotation instead of piling into whichever seed we bought
    first."""
    candidates = [c for c in GROWN_CROPS if seeds.get(c, 0) > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda c: seeds.get(c, 0))


def _find_target_tile(farm, board_size, day, seeds):
    fx, fy = farm["farmer"]
    candidates = []
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            purpose = _tile_purpose(tile, day)
            if purpose:
                candidates.append((x, y, purpose))
            elif tile is None and _next_crop_to_plant(seeds):
                candidates.append((x, y, "plant"))

    if not candidates:
        return None

    priority = {"harvest": 0, "water": 1, "plant": 2}
    candidates.sort(key=lambda c: (priority[c[2]], abs(c[0] - fx) + abs(c[1] - fy)))
    return candidates[0]


def multi_crop(obs):
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farm = farms[player]
    board_size = len(farm["tiles"])
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]
    day = obs.get("day", 0)

    seeds = private.get("seeds", {})
    shed = private.get("shed", {})
    market_prices = (obs.get("market", {}) or {}).get("prices", {})

    market = []

    # Sell each crop independently: only above its own threshold, and only
    # a bounded chunk per turn so one order doesn't tank the price for the
    # rest of what's sitting in the shed.
    for crop in GROWN_CROPS:
        in_shed = shed.get(crop, 0)
        price = market_prices.get(crop, 0)
        if in_shed > 0 and price >= SELL_THRESHOLD[crop]:
            market.append(["SELL", crop, min(in_shed, MAX_SELL_CHUNK)])

    # Keep at least one seed of each crop on hand so we can always plant
    # whatever's cheapest to top up, without over-buying seeds we won't
    # plant for a while.
    for crop in GROWN_CROPS:
        if seeds.get(crop, 0) == 0 and farm["money"] >= SEED_COST[crop]:
            market.append(["BUY_SEED", crop, 1])

    farmer = ["PASS"]
    purpose = _tile_purpose(tile, day)

    if purpose == "harvest":
        farmer = ["HARVEST"]
    elif purpose == "water":
        farmer = ["WATER"]
    elif tile is None and _next_crop_to_plant(seeds):
        farmer = ["PLANT", _next_crop_to_plant(seeds)]
    else:
        target = _find_target_tile(farm, board_size, day, seeds)
        if target:
            farmer = act_or_move((fx, fy), (target[0], target[1]), ["PASS"])

    return {"farmer": farmer, "hands": [], "market": market}
