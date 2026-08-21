"""chaitanyajamble_agent: local benchmark opponent, NOT original work.

Pulled from a public Kaggle notebook (Chaitanya Jamble, "Kaggriculture"):
https://www.kaggle.com/code/chaitanyajamble/kaggriculture
Extracted from its %%writefile submission.py cell on 2026-08-21. Near our
own current live rating tier (rank ~4058/5684, score 507.2 at pull time) --
deliberately NOT top-tier, added to give a locally testable opponent
representative of what the live ladder actually matches us against right
now (per community discussion, matchmaking is near-rating, not
strongest-vs-all). See CREDITS.md for full attribution.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------

# crop -> (seed_cost, first_harvest_day, target_harvest_day, last_plant_day, ongoing)
# "ongoing" crops (TOMATO, STRAWBERRY) regrow after harvest instead of being
# replanted; "target_harvest_day" of 0 means "harvest as soon as it's ready".
CROP_RULES = {
    "WHEAT": (10, 2, 4, 24, False),
    "CARROT": (20, 2, 3, 25, False),
    "TOMATO": (50, 8, 0, 20, True),
    "STRAWBERRY": (100, 10, 0, 18, True),
    "MELON": (80, 10, 12, 16, False),
}

# Initial core-plot crop mix (25 plots), highest-value crop weighted heaviest.
CROP_MIX = (
    ("MELON", 13),
    ("CARROT", 4),
    ("WHEAT", 4),
    ("TOMATO", 2),
    ("STRAWBERRY", 2),
)

# Sell priority: highest unit value first so the best goods clear the market
# before lower-value ones compete for the same daily demand.
SELL_ORDER = ("MELON", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT", "EGG", "MILK", "WOOL")

# Per-tick sell cap per item. High-value/low-volume goods are throttled hard
# so we don't crash our own sale price; bulk/low-value goods can move faster.
SELL_THROTTLE = {
    "MELON": 3,
    "STRAWBERRY": 4,
    "TOMATO": 4,
    "CARROT": 8,
    "WHEAT": 8,
    "EGG": 6,
    "MILK": 6,
    "WOOL": 6,
}

TARGET_HANDS = 7
HAND_HIRE_CASH_RESERVE = 150.0   # never hire below this cash balance
EXTRA_WHEAT_TILES = 15           # wheat filler on the expanded NE quadrant

LAND_MIN_DAY = 11                # earliest day the NE quadrant unlocks
LAND_MAX_DAY = 20                # buying later leaves too little payback time
LAND_COST = 1000.0
LAND_CASH_RESERVE = 300.0        # cash to keep on hand after buying land

SEED_CASH_RESERVE = 100.0        # cash to keep on hand after buying seeds
FINAL_DAY = 29                   # season end: liquidate only, no more work
MAX_MARKET_ORDERS = 10           # engine-imposed cap on orders per turn

SAFE_RESULT = {"farmer": ["PASS"], "hands": [], "market": []}


# ---------------------------------------------------------------------------
# Farm layout planning
# ---------------------------------------------------------------------------

def build_plan(board_size: int, expanded: bool) -> dict:
    """Map each owned tile position to the crop that belongs there.

    The core (SW) quadrant is planted with the fixed CROP_MIX, positioned
    so the crops closest to the shed are worked first. Once the NE
    quadrant is unlocked, its tiles nearest the shed are filled with wheat
    (cheap, fast, low-maintenance) up to EXTRA_WHEAT_TILES.
    """
    half = max(1, board_size // 2)
    shed = (half - 1, half - 1)
    core_cells = [(x, y) for y in range(half) for x in range(half)]
    core_cells.sort(key=lambda p: (-(abs(p[0] - shed[0]) + abs(p[1] - shed[1])), p[1], p[0]))

    crops = []
    for crop, count in CROP_MIX:
        crops.extend([crop] * count)
    plan = dict(zip(core_cells, crops))

    if expanded:
        ne_shed = (half, half - 1)
        ne_cells = [(x, y) for y in range(half) for x in range(half, board_size)]
        ne_cells.sort(key=lambda p: (abs(p[0] - ne_shed[0]) + abs(p[1] - ne_shed[1]), p[1], p[0]))
        plan.update({pos: "WHEAT" for pos in ne_cells[:EXTRA_WHEAT_TILES]})

    return plan


def plan_tile_action(tile, planned_crop: str, day: int, hour: int, seeds: dict):
    """Decide what a single planned tile needs done, if anything.

    Returns (priority, action) where lower priority is more urgent, or
    None if the tile needs no attention this turn.
    """
    if tile == "LOCKED":
        return None

    if tile is None:
        rules = CROP_RULES.get(planned_crop)
        if rules is None:
            return None
        last_plant_day = rules[3]
        if hour <= 21 and day <= last_plant_day and seeds.get(planned_crop, 0) > 0:
            return 3, ["PLANT", planned_crop]
        return None

    if not isinstance(tile, dict):
        return None

    if tile.get("kind") == "WEED":
        return 2, ["DIG"]
    if tile.get("kind") != "PLANT":
        return None

    crop = tile.get("crop", planned_crop)
    rules = CROP_RULES.get(crop, CROP_RULES.get(planned_crop))
    if rules is None:
        return None
    _, first_day, target_harvest_day, _, ongoing = rules

    age = day - tile.get("planted_day", day)
    amount = tile.get("yield_units", 0)
    watered_today = tile.get("watered_today", False)

    # Melons: harvest early once a solid batch has accumulated rather than
    # waiting for the full target, to free the tile for the next cycle.
    if crop == "MELON" and age >= 10 and amount >= 6:
        return 0, ["HARVEST"]

    if not ongoing:
        if age >= target_harvest_day and amount > 0:
            # For crops that still benefit from one more watering before
            # the yield caps out, water first; otherwise harvest now.
            cap = {"WHEAT": 4, "CARROT": 3}.get(crop, 0)
            if cap and amount < cap and not watered_today:
                return 0, ["WATER"]
            return 0, ["HARVEST"]
        if not watered_today:
            return 1, ["WATER"]
        return None

    # Ongoing (regrowing) crops: keep watering, harvest whenever ripe.
    if not watered_today:
        return 1, ["WATER"]
    if age >= first_day and amount > 0:
        return 0, ["HARVEST"]
    return None


def move_toward(source, target):
    """Single-step Manhattan move from source toward target."""
    sx, sy = source
    tx, ty = target
    if sx < tx:
        return ["EAST"]
    if sx > tx:
        return ["WEST"]
    if sy < ty:
        return ["SOUTH"]
    if sy > ty:
        return ["NORTH"]
    return ["PASS"]


# ---------------------------------------------------------------------------
# Worker task assignment
# ---------------------------------------------------------------------------

def assign_unit_actions(obs, farm, private, plan, core_first):
    """Assign every worker (farmer + hands) an action for this turn.

    Builds the full list of pending tile tasks, then greedily matches
    workers to tasks in ascending (region, priority, distance) order --
    a cheap approximation of optimal assignment that avoids sending two
    workers to the same tile and prefers the closest capable worker for
    each urgent task.
    """
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    seeds = private.get("seeds", {}) or {}
    tiles = farm.get("tiles") or []
    if not tiles:
        workers = [farm.get("farmer")] + list(farm.get("hands", []) or [])
        return [["PASS"] for _ in workers]

    half = max(1, len(tiles) // 2)
    tasks = []  # (region, priority, position, action)
    for position, crop in plan.items():
        x, y = position
        if y >= len(tiles) or x >= len(tiles[y]):
            continue
        result = plan_tile_action(tiles[y][x], crop, day, hour, seeds)
        if result is not None:
            priority, action = result
            region = int(core_first and x >= half)
            tasks.append((region, priority, position, action))

    workers = [farm.get("farmer")] + list(farm.get("hands", []) or [])
    workers = [w for w in workers if w is not None]

    # Build all worker/task pairs and sort by assignment cost, cheapest first.
    pairs = []
    for w_idx, w_pos in enumerate(workers):
        for t_idx, (region, priority, target, _action) in enumerate(tasks):
            distance = abs(w_pos[0] - target[0]) + abs(w_pos[1] - target[1])
            cost = region * 1000 + priority * 100 + distance
            pairs.append((cost, distance, w_idx, t_idx))
    pairs.sort(key=lambda p: (p[0], p[1]))

    seed_budget = dict(seeds)
    assigned_worker = set()
    assigned_task = set()
    result = [None] * len(workers)

    for _cost, distance, w_idx, t_idx in pairs:
        if w_idx in assigned_worker or t_idx in assigned_task:
            continue
        _region, _priority, target, action = tasks[t_idx]
        if distance == 0 and action[0] == "PLANT" and seed_budget.get(action[1], 0) <= 0:
            continue
        assigned_worker.add(w_idx)
        assigned_task.add(t_idx)
        if distance:
            result[w_idx] = move_toward(workers[w_idx], target)
        else:
            if action[0] == "PLANT":
                seed_budget[action[1]] = seed_budget.get(action[1], 0) - 1
            result[w_idx] = action

    for i in range(len(result)):
        if result[i] is None:
            result[i] = ["PASS"]
    return result


# ---------------------------------------------------------------------------
# Market decisions
# ---------------------------------------------------------------------------

def build_market_orders(obs, farm, private, plan):
    """Decide sell/buy/hire orders for this turn, respecting the engine's
    per-turn order cap and the farm's available cash."""
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    shed = private.get("shed", {}) or {}
    seeds = private.get("seeds", {}) or {}
    orders = []

    liquidating = day >= FINAL_DAY

    # --- Selling: throttled normally, full liquidation on the final day ---
    for item in SELL_ORDER:
        if len(orders) >= MAX_MARKET_ORDERS:
            break
        amount = int(shed.get(item, 0) or 0)
        if amount <= 0:
            continue
        if liquidating:
            quantity = amount
        else:
            quantity = min(amount, SELL_THROTTLE.get(item, amount))
        if quantity > 0:
            orders.append(["SELL", item, quantity])

    if liquidating:
        # No more capital investment once the season is ending.
        return orders

    money = float(farm.get("money", 0) or 0)

    # --- Land purchase: only within a payback-viable window and with a
    # cash reserve left afterward ---
    owned_quadrants = farm.get("unlocked_quadrants", []) or []
    if (
        LAND_MIN_DAY <= day <= LAND_MAX_DAY
        and "NE" not in owned_quadrants
        and money >= LAND_COST + LAND_CASH_RESERVE
        and len(orders) < MAX_MARKET_ORDERS
    ):
        orders.append(["BUY_LAND"])
        money -= LAND_COST

    # --- Seed purchase: only what's needed to fill unplanted planned tiles,
    # net of seeds already on hand, capped by remaining cash reserve ---
    needs = {}
    tiles = farm.get("tiles") or []
    for (x, y), crop in plan.items():
        rules = CROP_RULES.get(crop)
        if rules is None:
            continue
        if y < len(tiles) and x < len(tiles[y]) and tiles[y][x] is None and day <= rules[3]:
            needs[crop] = needs.get(crop, 0) + 1

    for crop in ("MELON", "CARROT", "WHEAT", "TOMATO", "STRAWBERRY"):
        if len(orders) >= MAX_MARKET_ORDERS:
            break
        wanted = max(0, needs.get(crop, 0) - int(seeds.get(crop, 0) or 0))
        if wanted <= 0:
            continue
        cost_per_seed = CROP_RULES[crop][0]
        affordable = int(max(0.0, money - SEED_CASH_RESERVE) // cost_per_seed)
        quantity = min(wanted, affordable)
        if quantity > 0:
            orders.append(["BUY_SEED", crop, quantity])
            money -= quantity * cost_per_seed

    # --- Hiring: top up to TARGET_HANDS once per morning, only if a cash
    # buffer remains afterward ---
    if hour <= 5:
        current_hands = len(farm.get("hands", []) or [])
        missing = max(0, TARGET_HANDS - current_hands)
        while missing > 0 and len(orders) < MAX_MARKET_ORDERS and money >= HAND_HIRE_CASH_RESERVE:
            orders.append(["HIRE"])
            missing -= 1
            money -= HAND_HIRE_CASH_RESERVE  # conservative placeholder cost buffer

    return orders


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _agent_impl(obs):
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0) or 0)
    if player >= len(farms) or player < 0:
        return dict(SAFE_RESULT)

    farm = farms[player] or {}
    private = obs.get("private", {}) or {}
    owned_quadrants = farm.get("unlocked_quadrants", []) or []
    expanded = "NE" in owned_quadrants
    board_size = len(farm.get("tiles", []) or []) or 10
    plan = build_plan(board_size, expanded)

    day = int(obs.get("day", 0) or 0)
    num_hands = len(farm.get("hands", []) or [])

    if day >= FINAL_DAY:
        # Season is wrapping up: stop working the fields, just sell out.
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in range(num_hands)],
            "market": build_market_orders(obs, farm, private, plan),
        }

    unit_actions = assign_unit_actions(obs, farm, private, plan, expanded)
    farmer_action = unit_actions[0] if unit_actions else ["PASS"]
    hand_actions = unit_actions[1:] if len(unit_actions) > 1 else [["PASS"] for _ in range(num_hands)]

    return {
        "farmer": farmer_action,
        "hands": hand_actions,
        "market": build_market_orders(obs, farm, private, plan),
    }


def agent(obs):
    """Public entry point required by the game engine.

    Wrapped so that any unexpected observation shape or runtime error
    degrades to a safe no-op turn instead of failing the episode.
    """
    try:
        return _agent_impl(obs)
    except Exception:
        try:
            farms = obs.get("farms", []) or []
            player = int(obs.get("player", 0) or 0)
            num_hands = len(farms[player].get("hands", []) or []) if 0 <= player < len(farms) else 0
        except Exception:
            num_hands = 0
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(num_hands)], "market": []}
chaitanyajamble_agent = agent
