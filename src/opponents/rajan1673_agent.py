"""rajan1673_agent: local benchmark opponent, NOT original work.

Pulled from a public Kaggle notebook (rajan jha, "kagriculture"):
https://www.kaggle.com/code/rajan1673/kagriculture
Extracted from its %%writefile main.py cell on 2026-08-21. Near our own
current live rating tier (rank ~4478/5684, score 434.7 at pull time) --
deliberately NOT top-tier, added to give a locally testable opponent
representative of what the live ladder actually matches us against right
now (per community discussion, matchmaking is near-rating, not
strongest-vs-all). See CREDITS.md for full attribution.
"""
# Kaggriculture Agent v5
# History: v1 (basic wheat/carrot loop) -> v2 (hands + land + more crops)
# -> v3 (cows/caretaker) -> v4 (terminal-day sell-everything) -> v5 (this):
#  - Buys/builds a PASTURE and raises COWs (milk income)
#  - One dedicated "caretaker" hand manages the cows (feed/care/harvest/fertilizer)
#  - Buys enough seed for EVERY crop-worker to plant on the same turn
#  - Lower money buffers so we actually spend money instead of hoarding it
#  - Per-item sell caps based on price-crash sensitivity (wheat/egg sell in
#    bulk safely; melon/wool/strawberry/milk sold in small batches)
#  - Terminal-day liquidation: on the final day, dump all shed inventory
#    since unsold items don't count toward the score
#  - Fertilizer: crop workers opportunistically grab fertilizer from the
#    shed and apply FERTILIZE on newly-planted one-time crops when carrying
#    it, without ever blocking normal watering/harvesting

FIRST_YIELD_DAY = {"WHEAT": 2, "CARROT": 2, "MELON": 10, "TOMATO": 8}
SEED_COST = {"WHEAT": 10, "CARROT": 20, "MELON": 80, "TOMATO": 50}
ONE_TIME_CROPS = {"WHEAT", "CARROT", "MELON"}
CROP_PRIORITY = ["MELON", "TOMATO", "CARROT", "WHEAT"]

SELL_CAP_PER_ITEM = {
    "WHEAT": 60,
    "EGG": 60,
    "FERTILIZER": 40,
    "TOMATO": 25,
    "CARROT": 20,
    "MILK": 10,
    "STRAWBERRY": 10,
    "WOOL": 6,
    "MELON": 6,
}
DEFAULT_SELL_CAP = 15

MAX_SELL_PER_TURN = 25
TERMINAL_RELAY_START_DAY = 29
SEED_MONEY_BUFFER = 200
LAND_MONEY_BUFFER = 800
HIRE_MONEY_BUFFER = 50
MAX_HANDS = 6

COW_COST = 400
COW_STRUCTURE = "PASTURE"
NUM_COWS = 4
ANIMAL_TILES = [(0, 4), (1, 4), (2, 4), (3, 4)]

SHED_ADJ = {(4, 4), (5, 4), (4, 5), (5, 5)}

QUADRANT_BOUNDS = {
    "NW": (0, 4, 0, 4),
    "NE": (5, 9, 0, 4),
    "SW": (0, 4, 5, 9),
    "SE": (5, 9, 5, 9),
}


def unlocked_tile_coords(unlocked_quadrants):
    coords = []
    for q in unlocked_quadrants:
        x0, x1, y0, y1 = QUADRANT_BOUNDS[q]
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                coords.append((x, y))
    return coords


def step_towards(fx, fy, target):
    tx, ty = target
    if fx < tx:
        return "EAST"
    if fx > tx:
        return "WEST"
    if fy < ty:
        return "SOUTH"
    if fy > ty:
        return "NORTH"
    return None


def classify_crop_tile(tile, day):
    if tile is None:
        return "empty"
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile["crop"]
        age = day - tile["planted_day"]
        ready_day = FIRST_YIELD_DAY.get(crop, 2)
        if tile["yield_units"] > 0 and age >= ready_day:
            return "harvest"
        if not tile["watered_today"]:
            return "water"
    return None


def find_crop_target(tiles, coords, day, claimed):
    best = {"harvest": None, "water": None, "empty": None}
    for (x, y) in coords:
        if (x, y) in claimed:
            continue
        reason = classify_crop_tile(tiles[y][x], day)
        if reason and best[reason] is None:
            best[reason] = (x, y)
    for reason in ("harvest", "water", "empty"):
        if best[reason]:
            return best[reason][0], best[reason][1], reason
    return None


def crop_worker_action(x, y, tiles, day, seeds, claimed, coords, inv=None, shed_fertilizer=0):
    inv = inv or {}
    if (x, y) in ANIMAL_TILES:
        target = find_crop_target(tiles, coords, day, claimed)
        if target:
            tx, ty, _ = target
            claimed.add((tx, ty))
            move = step_towards(x, y, (tx, ty))
            return [move] if move else ["PASS"]
        return ["PASS"]

    tile = tiles[y][x]
    reason = classify_crop_tile(tile, day)

    if reason == "harvest":
        claimed.add((x, y))
        return ["HARVEST"]

    if (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile["crop"] in ONE_TIME_CROPS
        and tile.get("fertilized_until_day", -1) < day
        and inv.get("FERTILIZER", 0) > 0
    ):
        claimed.add((x, y))
        return ["FERTILIZE"]

    if reason == "water":
        claimed.add((x, y))
        return ["WATER"]
    if reason == "empty":
        for crop in CROP_PRIORITY:
            if seeds.get(crop, 0) > 0:
                claimed.add((x, y))
                return ["PLANT", crop]

    target = find_crop_target(tiles, coords, day, claimed)
    if target:
        tx, ty, _ = target
        claimed.add((tx, ty))
        move = step_towards(x, y, (tx, ty))
        return [move] if move else ["PASS"]

    if (x, y) in SHED_ADJ and inv.get("FERTILIZER", 0) == 0 and shed_fertilizer > 0:
        return ["PICKUP", "FERTILIZER", 5]

    return ["PASS"]


def find_animal_need(tiles):
    priority = {"feed": 0, "harvest": 1, "fertilizer": 2, "care": 3, "cow": 4, "build": 5}
    best = None
    for (x, y) in ANIMAL_TILES:
        tile = tiles[y][x]
        need = None
        if tile is None:
            need = "build"
        elif isinstance(tile, dict) and tile.get("kind") == COW_STRUCTURE:
            if tile.get("animal") is None:
                need = "cow"
            elif not tile["fed_today"]:
                need = "feed"
            elif tile["yield_units"] > 0:
                need = "harvest"
            elif tile.get("fertilizer_available"):
                need = "fertilizer"
            elif not tile["cared_today"]:
                need = "care"
        if need and (best is None or priority[need] < priority[best[2]]):
            best = (x, y, need)
    return best


def caretaker_action(x, y, tiles, inv):
    need_info = find_animal_need(tiles)
    tile_here = tiles[y][x]

    if (x, y) in ANIMAL_TILES:
        if tile_here is None:
            return ["BUILD_PASTURE"]
        if isinstance(tile_here, dict) and tile_here.get("kind") == COW_STRUCTURE:
            if tile_here.get("animal") is None:
                if inv.get("COW", 0) > 0:
                    return ["PLACE", "COW"]
            else:
                if not tile_here["fed_today"]:
                    if inv.get("WHEAT", 0) > 0:
                        return ["FEED"]
                elif tile_here["yield_units"] > 0:
                    return ["HARVEST"]
                elif tile_here.get("fertilizer_available"):
                    return ["COLLECT_FERTILIZER"]
                elif not tile_here["cared_today"]:
                    return ["CARE"]

    if (x, y) in SHED_ADJ:
        if need_info and need_info[2] == "feed" and inv.get("WHEAT", 0) == 0:
            return ["PICKUP", "WHEAT", 20]
        if need_info and need_info[2] == "cow" and inv.get("COW", 0) == 0:
            return ["PICKUP", "COW", 1]

    if need_info:
        tx, ty, need = need_info
        if need == "feed" and inv.get("WHEAT", 0) == 0:
            move = step_towards(x, y, (4, 4))
        elif need == "cow" and inv.get("COW", 0) == 0:
            move = step_towards(x, y, (4, 4))
        else:
            move = step_towards(x, y, (tx, ty))
        return [move] if move else ["PASS"]

    return ["PASS"]


def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    day = obs["day"]
    tiles = me["tiles"]
    money = me["money"]
    seeds = private["seeds"]
    shed = private["shed"]
    unlocked = me["unlocked_quadrants"]
    all_coords = unlocked_tile_coords(unlocked)
    crop_coords = [c for c in all_coords if c not in ANIMAL_TILES]

    inventories = private.get("inventories", [{}])
    num_workers = 1 + len(me["hands"])

    market = []

    crop_workers = max(1, num_workers - 1)
    for crop in CROP_PRIORITY:
        cost = SEED_COST[crop]
        have = seeds.get(crop, 0)
        needed = crop_workers - have
        if needed > 0 and money >= cost * needed + SEED_MONEY_BUFFER:
            market.append(["BUY_SEED", crop, needed])

    is_terminal_phase = day >= TERMINAL_RELAY_START_DAY
    for item, qty in shed.items():
        if qty > 0:
            cap = SELL_CAP_PER_ITEM.get(item, DEFAULT_SELL_CAP)
            sell_qty = qty if is_terminal_phase else min(qty, cap)
            market.append(["SELL", item, sell_qty])

    quadrant_order = ["NE", "SW", "SE"]
    next_quadrant = next((q for q in quadrant_order if q not in unlocked), None)
    if next_quadrant and money >= LAND_MONEY_BUFFER + 1000:
        market.append(["BUY_LAND"])

    hands_today = me.get("hires_today", 0)
    if hands_today < MAX_HANDS and money >= HIRE_MONEY_BUFFER:
        market.append(["HIRE"])

    cow_in_shed = shed.get("COW", 0)
    cow_in_hand = sum(inv.get("COW", 0) for inv in inventories)
    if cow_in_shed + cow_in_hand == 0 and money >= COW_COST + LAND_MONEY_BUFFER:
        market.append(["BUY_ANIMAL", "COW", 1])

    if shed.get("WHEAT", 0) < 5 and money >= 200:
        market.append(["BUY_PRODUCT", "WHEAT", 10])

    if shed.get("FERTILIZER", 0) < 5 and money >= 300:
        market.append(["BUY_PRODUCT", "FERTILIZER", 5])

    market = market[:10]

    claimed = set()

    fx, fy = me["farmer"]
    hand_positions = me["hands"]

    if hand_positions:
        caretaker_pos = hand_positions[-1]
        crop_hand_positions = hand_positions[:-1]
    else:
        caretaker_pos = None
        crop_hand_positions = []

    farmer_inv = inventories[0] if len(inventories) > 0 else {}
    shed_fert = shed.get("FERTILIZER", 0)
    farmer_action = crop_worker_action(fx, fy, tiles, day, seeds, claimed, crop_coords, farmer_inv, shed_fert)

    hand_actions = []
    for i, (hx, hy) in enumerate(crop_hand_positions):
        h_inv = inventories[i + 1] if len(inventories) > i + 1 else {}
        hand_actions.append(crop_worker_action(hx, hy, tiles, day, seeds, claimed, crop_coords, h_inv, shed_fert))

    if caretaker_pos is not None:
        cx, cy = caretaker_pos
        c_inv = inventories[len(crop_hand_positions) + 1] if len(inventories) > len(crop_hand_positions) + 1 else {}
        hand_actions.append(caretaker_action(cx, cy, tiles, c_inv))

    return {"farmer": farmer_action, "hands": hand_actions, "market": market}
rajan1673_agent = agent
