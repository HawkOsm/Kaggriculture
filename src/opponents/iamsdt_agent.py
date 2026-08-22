"""iamsdt_agent: local benchmark opponent.
"""
"""Sheep build for Kaggriculture. 16 pastures in the NW quadrant, 6 hands a day.

Config comes from EXP-01: sheep at k=16, h=6 was the highest measured net
($46,887 against a passing opponent), and 6 hands cost $600 a season against the
$11,280 that 12 hands cost. Everything past 8 hands is negative marginal value.
"""

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS

LINE = "SHEEP"
TILES = 16
HANDS = 6

NW_SHED = (4, 4)  # only shed-access tile inside the starting NW quadrant
TURNS_PER_DAY = 24

MOVE_OF = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}

PRODUCT = ANIMALS[LINE]["product"]
STRUCTURE = ANIMALS[LINE]["structure"]
BUILD_OP = "BUILD_" + STRUCTURE


def g(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def nw_tiles_by_distance():
    """NW-quadrant tiles ordered nearest-first from the shed access tile."""
    tiles = [(x, y) for y in range(5) for x in range(5)]
    tiles.sort(key=lambda t: (abs(t[0] - NW_SHED[0]) + abs(t[1] - NW_SHED[1]), t[1], t[0]))
    return tiles


def step_toward(pos, target):
    px, py = pos
    tx, ty = target
    if px != tx:
        return MOVE_OF[(1, 0) if tx > px else (-1, 0)]
    if py != ty:
        return MOVE_OF[(0, 1) if ty > py else (0, -1)]
    return None


PLOTS = nw_tiles_by_distance()[:TILES]


def agent(obs):
    me = g(obs, "player", 0)
    farm = g(obs, "farms")[me]
    priv = g(obs, "private")
    shed = dict(g(priv, "shed", {}) or {})
    invs = g(priv, "inventories", [{}]) or [{}]
    tiles = farm["tiles"]
    money = farm["money"]
    hour = g(obs, "hour", 0)

    positions = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
    n_units = len(positions)
    market = []

    # Hiring, front-loaded at the start of the day.
    if hour < 2:
        for _ in range(min(HANDS - farm["hires_today"], 10)):
            market.append(["HIRE"])

    # Restock: one animal at a time onto a ready pasture, keeping a feed buffer.
    placed = sum(
        1 for (x, y) in PLOTS
        if isinstance(tiles[y][x], dict) and "animal" in tiles[y][x]
    )
    in_hand = shed.get(LINE, 0) + sum(inv.get(LINE, 0) for inv in invs)
    structures_ready = sum(
        1 for (x, y) in PLOTS
        if isinstance(tiles[y][x], dict)
        and tiles[y][x].get("kind") == STRUCTURE
        and "animal" not in tiles[y][x]
    )
    if min(TILES - placed - in_hand, structures_ready) > 0 and money > ANIMALS[LINE]["cost"] + 600:
        market.append(["BUY_ANIMAL", LINE, 1])

    # Feed: one wheat per animal per day, bought into the shed.
    need_wheat = placed + 1 - shed.get("WHEAT", 0)
    if placed and need_wheat > 0 and money > 400:
        market.append(["BUY_PRODUCT", "WHEAT", max(1, need_wheat)])

    # Sell the product as it lands. Wheat in the shed is feed, never stock.
    if shed.get(PRODUCT, 0) > 0:
        market.append(["SELL", PRODUCT, shed[PRODUCT]])

    market = market[:10]

    # Task list, lowest priority number first.
    tasks = []
    for (x, y) in PLOTS:
        t = tiles[y][x]
        if t is None:
            tasks.append((3, (x, y), [BUILD_OP]))
            continue
        if t == "LOCKED":
            continue
        kind = t.get("kind")
        if kind == "WEED":
            tasks.append((2, (x, y), ["DIG"]))
        elif kind == STRUCTURE:
            if "animal" not in t:
                tasks.append((3, (x, y), ["PLACE", LINE]))
            else:
                if t.get("yield_units", 0) > 0:
                    tasks.append((0, (x, y), ["HARVEST"]))
                if not t["fed_today"]:
                    tasks.append((1, (x, y), ["FEED"]))
                if not t["cared_today"]:
                    tasks.append((4, (x, y), ["CARE"]))
    tasks.sort(key=lambda t: t[0])

    ops = [["PASS"] for _ in range(n_units)]
    taken = [False] * n_units
    end_of_day = hour >= TURNS_PER_DAY - 2

    # Deposit produce before the day ends, since the auto-drop can overflow.
    for u in range(n_units):
        inv = invs[u] if u < len(invs) else {}
        if end_of_day and sum(inv.values()) > 0:
            if positions[u] == NW_SHED:
                ops[u] = ["DROP"]
            else:
                mv = step_toward(positions[u], NW_SHED)
                ops[u] = [mv] if mv else ["PASS"]
            taken[u] = True

    # FEED and PLACE both draw from the unit's own inventory, so a unit has to
    # walk to the shed and pick up before it can do either.
    needs_feed = any(t[2][0] == "FEED" for t in tasks)
    needs_place = any(t[2][0] == "PLACE" for t in tasks)
    for u in range(n_units):
        if taken[u]:
            continue
        inv = invs[u] if u < len(invs) else {}
        if needs_place and inv.get(LINE, 0) == 0 and shed.get(LINE, 0) > 0:
            pick = ["PICKUP", LINE, 1]
        elif needs_feed and inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
            pick = ["PICKUP", "WHEAT", min(4, shed.get("WHEAT", 0))]
        else:
            continue
        if positions[u] == NW_SHED:
            ops[u] = pick
        else:
            mv = step_toward(positions[u], NW_SHED)
            ops[u] = [mv] if mv else ["PASS"]
        taken[u] = True

    # Assign each task to the nearest free unit that can actually perform it.
    used_tiles = set()
    for prio, tile, op in tasks:
        if tile in used_tiles:
            continue
        best, best_d = None, None
        for u in range(n_units):
            if taken[u]:
                continue
            inv = invs[u] if u < len(invs) else {}
            if op[0] == "FEED" and inv.get("WHEAT", 0) == 0:
                continue
            if op[0] == "PLACE" and inv.get(LINE, 0) == 0:
                continue
            d = abs(positions[u][0] - tile[0]) + abs(positions[u][1] - tile[1])
            if best_d is None or d < best_d:
                best, best_d = u, d
        if best is None:
            continue
        if best_d == 0:
            ops[best] = op
        else:
            mv = step_toward(positions[best], tile)
            ops[best] = [mv] if mv else ["PASS"]
        taken[best] = True
        used_tiles.add(tile)

    return {"farmer": ops[0], "hands": ops[1:], "market": market}

iamsdt_agent = agent

