"""premaananda108_agent: local benchmark opponent, NOT original work.

Pulled from a public Kaggle notebook (Prema Ananda, "Economics-Driven Rule
Agent (EcoBot v2)"):
https://www.kaggle.com/code/premaananda108/economics-driven-rule-agent-ecobot-v2
Extracted from its `%%writefile main.py` cell on 2026-08-22. A distinct,
independently-authored rule-based agent (workload-adaptive crew sizing,
non-linear per-item market pricing model, task-auction unit dispatcher) --
checked against the rest of this pool via AST def-name-set diffing with
zero meaningful overlap. Rank ~2604/5684, score 850.4 on the public
leaderboard at pull time. See CREDITS.md for full attribution.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Game Constants
# ---------------------------------------------------------------------------

TOTAL_DAYS: int = 30
TURNS_PER_DAY: int = 24
SHED_CAPACITY: int = 100
MAX_MARKET_ORDERS: int = 10
BOARD_SIZE: int = 10
I0: int = 10_000

PASTURE_CLUSTER: list[tuple[int, int]] = [
    # NW quadrant (Day 0)
    (4, 4), (4, 3), (3, 4), (3, 3), (4, 2), (2, 4),
    # NE quadrant (Day 5)
    (5, 4), (5, 3), (6, 4), (6, 3), (5, 2), (7, 4),
    # SW quadrant (Day 9)
    (4, 5), (3, 5), (2, 5), (3, 6),
    # SE quadrant (Day 13+ if unlocked)
    (5, 5), (6, 5), (7, 5), (5, 6),
]

_RESERVED_PASTURE_TILES: frozenset[tuple[int, int]] = frozenset(PASTURE_CLUSTER)
SHED_TILES: list[tuple[int, int]] = [(4, 4), (5, 4), (4, 5), (5, 5)]
_SHED_TILES_SET: frozenset[tuple[int, int]] = frozenset(SHED_TILES)

_MILK_SUPPORT_SHOPS: tuple[str, ...] = ("PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP")

_SELLABLE_ITEMS: tuple[str, ...] = (
    "FERTILIZER", "MILK", "WOOL", "EGG",
    "MELON", "STRAWBERRY", "CARROT", "TOMATO", "WHEAT",
)


def get_target_animals(unlocked_shops: list[str]) -> tuple[int, int, int]:
    """Return (target_cows, target_sheep, target_quadrants)."""
    if "YARN_STORE" in unlocked_shops[:3]:
        return 6, 8, 3
    if any(s in _MILK_SUPPORT_SHOPS for s in unlocked_shops[:3]):
        return 10, 4, 3
    return 8, 4, 3


SELL_FLOOR_PCT: float = 0.30
SHED_SELL_PRESSURE: int = 85
LIQUIDATION_DAY: int = 28
STRAWBERRY_LAST_PLANT_DAY: int = 14
WHEAT_LAST_PLANT_DAY: int = 26
SEED_BUY_FLOOR_PCT: float = 0.25
N_FERTILIZER_HANDS: int = 2
FERTILIZER_PICKUP_BATCH: int = 4

CROPS: dict[str, dict[str, Any]] = {
    "WHEAT": {
        "seed_cost": 10, "base_price": 25,
        "first_yield_day": 2, "max_yield_day": 4,
        "max_units_base": 4, "max_units_fert": 6,
        "bonus_start_day": 2, "is_ongoing": False,
    },
    "CARROT": {
        "seed_cost": 20, "base_price": 35,
        "first_yield_day": 2, "max_yield_day": 3,
        "max_units_base": 3, "max_units_fert": 4,
        "bonus_start_day": 2, "is_ongoing": False,
    },
    "TOMATO": {
        "seed_cost": 50, "base_price": 60,
        "first_yield_day": 8, "max_yield_day": 11,
        "is_ongoing": True,
        "bonus_start_day": 8, "yield_days": [8, 9, 10, 11],
    },
    "STRAWBERRY": {
        "seed_cost": 100, "base_price": 120,
        "first_yield_day": 10, "max_yield_day": 16,
        "is_ongoing": True,
        "bonus_start_day": 10, "yield_days": [10, 12, 14, 16],
    },
    "MELON": {
        "seed_cost": 80, "base_price": 250,
        "first_yield_day": 10, "max_yield_day": 10,
        "is_ongoing": False,
        "bonus_start_day": 6, "max_units_base": 6, "max_units_fert": 6,
    },
}

ANIMALS: dict[str, dict[str, Any]] = {
    "GOOSE": {"cost": 300, "base_price": 50, "structure": "COOP", "first_yield_day": 4, "interval": 1, "product": "EGG"},
    "COW":   {"cost": 400, "base_price": 160, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "product": "MILK"},
    "SHEEP": {"cost": 500, "base_price": 200, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "product": "WOOL"},
}

MARKET_PARAMS: dict[str, dict[str, Any]] = {
    "WHEAT":       {"base": 25,  "T": 400, "below_func": "sqrt",  "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":      {"base": 35,  "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":      {"base": 60,  "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY":  {"base": 120, "T": 100, "below_func": "sqrt",  "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":       {"base": 250, "T": 300, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":         {"base": 50,  "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":        {"base": 160, "T": 122, "below_func": "sqrt",  "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":        {"base": 200, "T": 105, "below_func": "log",   "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER":  {"base": 100, "T": 200, "below_func": "linear","below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

_FIB: list[int] = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


def fib_cost(n: int) -> int:
    if n < len(_FIB):
        return _FIB[n]
    a, b = _FIB[-2], _FIB[-1]
    for _ in range(n - len(_FIB) + 1):
        a, b = b, a + b
    return b


def total_hire_cost(count: int, already_hired: int) -> int:
    return sum(fib_cost(already_hired + i) for i in range(count))


# ---------------------------------------------------------------------------
# Animal Census (DRY: computed once per turn, used by market + tasks + agent)
# ---------------------------------------------------------------------------


@dataclass
class AnimalCensus:
    field_cows: int
    field_sheep: int
    shed_cows: int
    shed_sheep: int
    carried_cows: int
    carried_sheep: int

    @property
    def total_cows(self) -> int:
        return self.field_cows + self.shed_cows + self.carried_cows

    @property
    def total_sheep(self) -> int:
        return self.field_sheep + self.shed_sheep + self.carried_sheep

    @property
    def total(self) -> int:
        return self.total_cows + self.total_sheep

    @property
    def on_field(self) -> int:
        return self.field_cows + self.field_sheep

    @property
    def in_shed(self) -> int:
        return self.shed_cows + self.shed_sheep

    @property
    def carried(self) -> int:
        return self.carried_cows + self.carried_sheep


def _count_animal_census(
    farm_state: dict[str, Any],
    shed: dict[str, int],
    inventories: list[dict[str, int]],
) -> AnimalCensus:
    return AnimalCensus(
        field_cows=sum(1 for a in farm_state["animals"] if a["animal"] == "COW"),
        field_sheep=sum(1 for a in farm_state["animals"] if a["animal"] == "SHEEP"),
        shed_cows=shed.get("COW", 0),
        shed_sheep=shed.get("SHEEP", 0),
        carried_cows=sum(inv.get("COW", 0) for inv in inventories),
        carried_sheep=sum(inv.get("SHEEP", 0) for inv in inventories),
    )


# ---------------------------------------------------------------------------
# Market Pricing
# ---------------------------------------------------------------------------


def _shape(func: str, x: float, t: float) -> float:
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(max(0.0, x))
    if func == "log":
        return math.log(1.0 + max(0.0, x))
    if func == "hinge":
        u = x / max(1.0, t)
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    raise ValueError(f"Unknown shape function: {func!r}")


def market_price(item: str, inv: int) -> int:
    p = MARKET_PARAMS[item]
    base = float(p["base"])
    t = float(p["T"])
    if inv == I0:
        return int(round(base))
    if inv < I0:
        x = float(I0 - inv)
        denom = _shape(p["below_func"], t, t)
        amp = (p["below_target"] * base) / denom if denom > 0 else 0.0
        return max(1, int(round(base + amp * _shape(p["below_func"], x, t))))
    x = float(inv - I0)
    denom = _shape(p["above_func"], t, t)
    amp = (p["above_target"] * base) / denom if denom > 0 else 0.0
    return max(1, int(round(base - amp * _shape(p["above_func"], x, t))))


def _get_price(item: str, market_inv: dict[str, int]) -> int:
    """DRY wrapper: market_price + default inventory lookup."""
    return market_price(item, market_inv.get(item, I0))


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

_DIRS = ((0, -1, "NORTH"), (0, 1, "SOUTH"), (1, 0, "EAST"), (-1, 0, "WEST"))


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_on_shed_tile(pos: tuple[int, int]) -> bool:
    return pos in _SHED_TILES_SET


def closest_shed_pos(pos: tuple[int, int]) -> tuple[int, int]:
    return min(SHED_TILES, key=lambda s: manhattan(pos, s))


def bfs_step(start: tuple[int, int], goal: tuple[int, int]) -> str | None:
    if start == goal:
        return None
    visited: set[tuple[int, int]] = {start}
    q: deque[tuple[tuple[int, int], str]] = deque()
    for dx, dy, move in _DIRS:
        nx, ny = start[0] + dx, start[1] + dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if (nx, ny) == goal:
                return move
            visited.add((nx, ny))
            q.append(((nx, ny), move))
    while q:
        (cx, cy), first = q.popleft()
        for dx, dy, _ in _DIRS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and (nx, ny) not in visited:
                if (nx, ny) == goal:
                    return first
                visited.add((nx, ny))
                q.append(((nx, ny), first))
    return None


def _go_shed_or_drop(pos: tuple[int, int]) -> list[Any]:
    """DRY: navigate toward shed or DROP if already on a shed tile."""
    if is_on_shed_tile(pos):
        return ["DROP"]
    return [bfs_step(pos, closest_shed_pos(pos)) or "PASS"]


# ---------------------------------------------------------------------------
# Farm State Analysis
# ---------------------------------------------------------------------------


def parse_farm_state(tiles: list[list[Any]], day: int) -> dict[str, Any]:
    state: dict[str, Any] = {
        "animals": [],          # list of dicts: {pos, animal, fed, cared, fert, yu}
        "empty_pastures": [],   # (x, y) empty pasture tiles
        "plants": [],           # list of dicts: {pos, crop, age, watered, yu, fert_due, is_expired}
        "weeds": [],            # (x, y) weed positions
        "empty_tiles": [],      # (x, y) empty unlocked tiles
        "unlocked_count": 0,
    }

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            tile = tiles[y][x]
            pos = (x, y)

            if tile == "LOCKED":
                continue
            state["unlocked_count"] += 1

            if tile is None:
                state["empty_tiles"].append(pos)
                continue

            if not isinstance(tile, dict):
                continue

            kind = tile.get("kind")
            if kind == "WEED":
                state["weeds"].append(pos)
            elif kind == "PASTURE" or kind == "COOP":
                animal = tile.get("animal")
                if animal is None:
                    state["empty_pastures"].append(pos)
                else:
                    state["animals"].append({
                        "pos": pos,
                        "animal": animal,
                        "fed_today": tile.get("fed_today", False),
                        "cared_today": tile.get("cared_today", False),
                        "fertilizer_available": tile.get("fertilizer_available", False),
                        "yield_units": tile.get("yield_units", 0),
                    })
            elif kind == "PLANT":
                crop = tile.get("crop")
                if crop is None or crop not in CROPS:
                    raise ValueError(f"Unknown or missing crop {crop!r} at {pos}: {tile}")
                planted_day = tile.get("planted_day", day)
                age = day - planted_day
                spec = CROPS[crop]
                yu = tile.get("yield_units", 0)
                watered = tile.get("watered_today", False)
                fert_until = tile.get("fertilized_until_day", -1)

                is_expired = False
                if spec["is_ongoing"]:
                    last_yield_age = spec["yield_days"][-1]
                    if age >= last_yield_age + 1:
                        is_expired = True

                if spec["is_ongoing"]:
                    upcoming_ages = [d for d in spec["yield_days"] if d >= age]
                    if upcoming_ages:
                        days_to_yield = upcoming_ages[0] - age
                        yield_day = day + days_to_yield
                        fert_due = days_to_yield <= 2 and fert_until < yield_day
                    else:
                        fert_due = False
                else:
                    no_fert_benefit = spec["max_units_fert"] <= spec["max_units_base"]
                    fert_due = (
                        not no_fert_benefit
                        and spec["bonus_start_day"] <= age <= spec["max_yield_day"]
                        and fert_until < day
                    )

                state["plants"].append({
                    "pos": pos,
                    "crop": crop,
                    "age": age,
                    "watered_today": watered,
                    "yield_units": yu,
                    "fertilize_due": fert_due,
                    "is_expired": is_expired,
                    "is_ongoing": spec["is_ongoing"],
                    "first_yield_day": spec["first_yield_day"],
                })
            else:
                raise ValueError(f"Unknown tile kind {kind!r} at {pos}: {tile}")

    return state


# ---------------------------------------------------------------------------
# Market Strategy
# ---------------------------------------------------------------------------

# --- Dynamic hire sizing: physical workload model ---
_WORK_PER_ANIMAL: float = 4.0           # FEED + CARE + COLLECT_FERTILIZER + HARVEST + DROP
_WORK_PER_PLANT: float = 2.5            # WATER + FERTILIZE + HARVEST + travel
_WORK_PER_PLANTING: float = 3.5         # PLANT + WATER on planting day
_WORK_PER_DEPLOY: float = 7.0           # BUILD_PASTURE + PICKUP from shed + PLACE
_WORK_PER_PENDING_HARVEST: float = 1.5  # additional weight per ready crop/animal tile


def get_desired_hires(
    day: int,
    n_animals: int,
    n_plants: int,
    seeds_to_plant: int,
    animals_to_deploy: int,
    ready_harvest_tiles: int,
    unlocked_quads_count: int,
) -> int:
    """Calculate workforce needed today from physical task volume and farm layout size."""
    if day == 0:
        return 5

    work = (
        _WORK_PER_ANIMAL * n_animals
        + _WORK_PER_PLANT * n_plants
        + _WORK_PER_PLANTING * seeds_to_plant
        + _WORK_PER_DEPLOY * animals_to_deploy
        + _WORK_PER_PENDING_HARVEST * ready_harvest_tiles
    )

    efficiency = 11.0 if unlocked_quads_count == 1 else (9.0 if unlocked_quads_count == 2 else 7.5)
    desired = int(math.ceil(work / efficiency))

    if day <= 2:
        desired = max(2, min(desired, 4))
    elif day == LIQUIDATION_DAY:
        desired = min(desired, 9)
    elif day > LIQUIDATION_DAY:
        desired = min(desired, 8)

    return max(2, min(12, desired))


def _plan_seed_topup(
    crop: str, empty_count: int, curr_seeds: int, budget: float
) -> tuple[list[Any] | None, float]:
    seed_cost = CROPS[crop]["seed_cost"]
    need = max(0, empty_count - curr_seeds)
    if need <= 0 or budget < seed_cost:
        return None, budget
    can_buy = min(need, int(budget // seed_cost))
    if can_buy <= 0:
        return None, budget
    return ["BUY_SEED", crop, can_buy], budget - can_buy * seed_cost


# --- Market order sub-functions (Single Responsibility) ---


def _plan_sells(
    day: int,
    hour: int,
    shed: dict[str, int],
    farm_state: dict[str, Any],
    market_inv: dict[str, int],
    inventories: list[dict[str, int]],
    census: AnimalCensus,
) -> tuple[list[list[Any]], float]:
    """Plan all sell orders. Returns (orders, total_revenue)."""
    orders: list[list[Any]] = []
    revenue: float = 0.0

    # End-of-season: include items carried in inventories
    carried_sells: dict[str, int] = defaultdict(int)
    if day >= 29 and hour >= 21:
        for inv in inventories:
            for item, qty in inv.items():
                if qty > 0 and item not in ("COW", "SHEEP", "GOOSE"):
                    carried_sells[item] += qty

    # Fertilizer: keep reserve for pending fertilize tasks
    fert_due_count = sum(1 for p in farm_state["plants"] if p["fertilize_due"])
    fert_in_shed = shed.get("FERTILIZER", 0) + carried_sells.get("FERTILIZER", 0)
    fert_capacity = N_FERTILIZER_HANDS * FERTILIZER_PICKUP_BATCH
    fert_reserve = 0 if day >= LIQUIDATION_DAY else min(fert_due_count, fert_capacity)
    fert_to_sell = max(0, fert_in_shed - fert_reserve)
    if fert_to_sell > 0:
        orders.append(["SELL", "FERTILIZER", fert_to_sell])
        revenue += fert_to_sell * _get_price("FERTILIZER", market_inv)

    # Produce: sell everything immediately
    for item in ("MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "TOMATO"):
        qty = shed.get(item, 0) + carried_sells.get(item, 0)
        if qty > 0:
            orders.append(["SELL", item, qty])
            revenue += qty * _get_price(item, market_inv)

    # Wheat: sell excess over feed buffer
    wheat_in_shed = shed.get("WHEAT", 0) + carried_sells.get("WHEAT", 0)
    n_feedable = census.on_field + census.in_shed
    feed_buffer = 0 if day >= LIQUIDATION_DAY else max(4, n_feedable * 2)
    if wheat_in_shed > feed_buffer and day > 0:
        excess = wheat_in_shed - feed_buffer
        orders.append(["SELL", "WHEAT", excess])
        revenue += excess * _get_price("WHEAT", market_inv)

    return orders, revenue


def _plan_hires(
    budget: float,
    day: int,
    farm_state: dict[str, Any],
    seeds: dict[str, int],
    hires_today: int,
    unlocked_quads: list[str],
    needed_pastures: list[tuple[int, int]],
    census: AnimalCensus,
) -> tuple[list[list[Any]], float]:
    """Plan hire orders. Returns (orders, total_cost)."""
    ready_harvest = sum(1 for p in farm_state["plants"] if p["yield_units"] > 0)
    ready_harvest += sum(1 for a in farm_state["animals"] if a["yield_units"] > 0)

    target_crop_today = _select_target_crop(day, seeds)
    seeds_to_plant = min(seeds.get(target_crop_today, 0), len(farm_state["empty_tiles"]))

    desired = get_desired_hires(
        day,
        len(farm_state["animals"]),
        len(farm_state["plants"]),
        seeds_to_plant,
        census.in_shed + len(needed_pastures),
        ready_harvest,
        len(unlocked_quads),
    )
    new_hires = max(0, desired - hires_today)
    if new_hires <= 0:
        return [], 0.0

    cost = total_hire_cost(new_hires, hires_today)
    if budget >= cost + 50 or day <= 10:
        return [["HIRE"] for _ in range(new_hires)], float(cost)
    return [], 0.0


def _plan_land_expansion(
    budget: float,
    day: int,
    unlocked_quads: list[str],
    target_quadrants: int,
) -> tuple[list[list[Any]], float]:
    """Plan land purchase orders. Returns (orders, total_cost)."""
    orders: list[list[Any]] = []
    cost: float = 0.0

    if "NE" not in unlocked_quads and day >= 5 and budget >= 1000 + 100:
        orders.append(["BUY_LAND"])
        cost += 1000
        budget -= 1000

    if "SW" not in unlocked_quads and "NE" in unlocked_quads and day >= 9 and budget >= 2000 + 200:
        orders.append(["BUY_LAND"])
        cost += 2000
        budget -= 2000

    if (
        target_quadrants >= 4
        and "SE" not in unlocked_quads
        and "SW" in unlocked_quads
        and day >= 12
        and budget >= 4000 + 400
    ):
        orders.append(["BUY_LAND"])
        cost += 4000

    return orders, cost


def _plan_animal_expansion(
    budget: float,
    day: int,
    census: AnimalCensus,
    max_pasture_animals: int,
    target_cows: int,
    target_sheep: int,
) -> tuple[list[list[Any]], float]:
    """Plan animal purchase orders. Returns (orders, total_cost)."""
    if not (5 <= day <= 16 and census.total < max_pasture_animals):
        return [], 0.0

    orders: list[list[Any]] = []
    cost: float = 0.0
    sheep_price = ANIMALS["SHEEP"]["cost"]
    cow_price = ANIMALS["COW"]["cost"]

    if census.total_sheep < target_sheep and budget >= sheep_price + 100:
        want = min(2, target_sheep - census.total_sheep)
        if budget >= want * sheep_price + 100:
            orders.append(["BUY_ANIMAL", "SHEEP", want])
            cost += want * sheep_price
            budget -= want * sheep_price

    if census.total_cows < target_cows and budget >= cow_price + 100:
        want = min(2, target_cows - census.total_cows)
        if budget >= want * cow_price + 100:
            orders.append(["BUY_ANIMAL", "COW", want])
            cost += want * cow_price

    return orders, cost


def _plan_feed_backup(
    budget: float,
    day: int,
    shed: dict[str, int],
    census: AnimalCensus,
    market_inv: dict[str, int],
) -> tuple[list[Any] | None, float]:
    """Plan emergency wheat purchase if animals risk starvation."""
    n_feedable = census.on_field + census.in_shed
    wheat_in_shed = shed.get("WHEAT", 0)
    if day >= LIQUIDATION_DAY or wheat_in_shed >= n_feedable * 2:
        return None, 0.0

    wp = _get_price("WHEAT", market_inv)
    buy_n = min(15, max(4, n_feedable * 2 - wheat_in_shed))
    cost = wp * buy_n
    if budget >= cost + 50:
        return ["BUY_PRODUCT", "WHEAT", buy_n], float(cost)
    return None, 0.0


def _plan_seed_purchases(
    budget: float,
    day: int,
    seeds: dict[str, int],
    farm_state: dict[str, Any],
) -> tuple[list[list[Any]], float]:
    """Plan seed purchase orders by game phase."""
    orders: list[list[Any]] = []
    cost: float = 0.0
    empty_count = len(farm_state["empty_tiles"])
    wheat_seed_cost = CROPS["WHEAT"]["seed_cost"]
    straw_seed_cost = CROPS["STRAWBERRY"]["seed_cost"]

    if 21 <= day <= WHEAT_LAST_PLANT_DAY:
        curr_wheat_seeds = seeds.get("WHEAT", 0)
        need = max(0, empty_count + 35 - curr_wheat_seeds)
        if need > 0 and budget >= wheat_seed_cost:
            can_buy = min(need, int(budget // wheat_seed_cost))
            if can_buy > 0:
                orders.append(["BUY_SEED", "WHEAT", can_buy])
                cost += can_buy * wheat_seed_cost

    elif 5 <= day <= 20:
        curr_wheat = sum(1 for p in farm_state["plants"] if p["crop"] == "WHEAT") + seeds.get("WHEAT", 0)
        if curr_wheat < 14 and budget >= wheat_seed_cost:
            need_w = min(14 - curr_wheat, int(budget // wheat_seed_cost))
            if need_w > 0:
                orders.append(["BUY_SEED", "WHEAT", need_w])
                cost += need_w * wheat_seed_cost
                budget -= need_w * wheat_seed_cost
                empty_count = max(0, empty_count - need_w)

        curr_straw = seeds.get("STRAWBERRY", 0)
        total_straw = sum(1 for p in farm_state["plants"] if p["crop"] == "STRAWBERRY") + curr_straw
        buy_s = 0
        if day <= STRAWBERRY_LAST_PLANT_DAY and total_straw < 42 and empty_count > 0:
            buy_s = min(42 - total_straw, empty_count, int(budget // straw_seed_cost))
            if buy_s > 0:
                orders.append(["BUY_SEED", "STRAWBERRY", buy_s])
                cost += buy_s * straw_seed_cost
                budget -= buy_s * straw_seed_cost

        remaining_empty = max(0, empty_count - buy_s)
        order, budget = _plan_seed_topup("WHEAT", remaining_empty, seeds.get("WHEAT", 0), budget)
        if order is not None:
            orders.append(order)
            cost += wheat_seed_cost * order[2]

    elif 1 <= day <= 4:
        order, budget = _plan_seed_topup("WHEAT", empty_count, seeds.get("WHEAT", 0), budget)
        if order is not None:
            orders.append(order)
            cost += wheat_seed_cost * order[2]

    return orders, cost


def plan_market_orders(
    day: int,
    hour: int,
    money: float,
    shed: dict[str, int],
    seeds: dict[str, int],
    market_inv: dict[str, int],
    unlocked_quads: list[str],
    unlocked_shops: list[str],
    farm_state: dict[str, Any],
    hires_today: int,
    inventories: list[dict[str, int]],
    needed_pastures: list[tuple[int, int]],
    census: AnimalCensus,
) -> list[list[Any]]:
    orders: list[list[Any]] = []
    budget = money

    shops_stable = len(unlocked_shops) >= 3
    target_cows, target_sheep, target_quadrants = (
        get_target_animals(unlocked_shops) if shops_stable else (0, 0, 3)
    )
    cluster_capacity = 20 if "SE" in unlocked_quads else 16
    max_pasture_animals = min(target_cows + target_sheep, cluster_capacity)

    # Day 0 special opening
    if day == 0 and hour <= 1:
        if hires_today < 5:
            for _ in range(5 - hires_today):
                orders.append(["HIRE"])
        if shed.get("SHEEP", 0) == 0 and len(farm_state["animals"]) == 0:
            orders.append(["BUY_ANIMAL", "SHEEP", 2])
            orders.append(["BUY_ANIMAL", "COW", 2])
            orders.append(["BUY_SEED", "MELON", 12])
            orders.append(["BUY_SEED", "WHEAT", 7])
            orders.append(["BUY_PRODUCT", "WHEAT", 4])
        return orders[:MAX_MARKET_ORDERS]

    sell_orders, revenue = _plan_sells(day, hour, shed, farm_state, market_inv, inventories, census)
    orders.extend(sell_orders)
    budget += revenue

    hire_orders, hire_cost = _plan_hires(
        budget, day, farm_state, seeds, hires_today, unlocked_quads, needed_pastures, census,
    )
    orders.extend(hire_orders)
    budget -= hire_cost

    land_orders, land_cost = _plan_land_expansion(budget, day, unlocked_quads, target_quadrants)
    orders.extend(land_orders)
    budget -= land_cost

    animal_orders, animal_cost = _plan_animal_expansion(
        budget, day, census, max_pasture_animals, target_cows, target_sheep,
    )
    orders.extend(animal_orders)
    budget -= animal_cost

    feed_order, feed_cost = _plan_feed_backup(budget, day, shed, census, market_inv)
    if feed_order is not None:
        orders.append(feed_order)
        budget -= feed_cost

    seed_orders, _ = _plan_seed_purchases(budget, day, seeds, farm_state)
    orders.extend(seed_orders)

    return orders[:MAX_MARKET_ORDERS]


# ---------------------------------------------------------------------------
# Task Auction Unit Dispatcher
# ---------------------------------------------------------------------------

STANDING_BONUS: int = 20


@dataclass
class Task:
    priority: int
    pos: tuple[int, int]
    action: list[Any]
    need: tuple[str, ...] | None = None


def _select_target_crop(day: int, seeds_stock: dict[str, int]) -> str:
    if day > WHEAT_LAST_PLANT_DAY:
        return "NONE"
    if 21 <= day <= WHEAT_LAST_PLANT_DAY:
        return "WHEAT"
    if seeds_stock.get("MELON", 0) > 0 and day <= 12:
        return "MELON"
    if seeds_stock.get("STRAWBERRY", 0) > 0 and day <= STRAWBERRY_LAST_PLANT_DAY:
        return "STRAWBERRY"
    return "WHEAT"


def _compute_needed_pastures(
    farm_state: dict[str, Any],
    census: AnimalCensus,
) -> list[tuple[int, int]]:
    needed: list[tuple[int, int]] = []
    for p_pos in PASTURE_CLUSTER:
        current = len(needed) + len(farm_state["animals"]) + len(farm_state["empty_pastures"])
        if current < census.total and (p_pos in farm_state["empty_tiles"] or p_pos in farm_state["weeds"]):
            needed.append(p_pos)
    return needed


def _build_tasks(
    farm_state: dict[str, Any],
    shed: dict[str, int],
    seeds_stock: dict[str, int],
    target_crop: str,
    needed_pastures: list[tuple[int, int]],
    all_units: list[tuple[tuple[int, int], dict[str, int]]],
    day: int,
) -> list[Task]:
    tasks: list[Task] = []

    # 1. Animal care
    for a in farm_state["animals"]:
        pos = a["pos"]
        if not a["fed_today"]:
            tasks.append(Task(260, pos, ["FEED"], need=("WHEAT",)))
        elif not a["cared_today"]:
            tasks.append(Task(230, pos, ["CARE"]))
        if a["fertilizer_available"]:
            tasks.append(Task(210, pos, ["COLLECT_FERTILIZER"]))
        if a["yield_units"] > 0:
            tasks.append(Task(205, pos, ["HARVEST"]))

    # 2. Animal placement & pickups
    n_cow_shed, n_sheep_shed = shed.get("COW", 0), shed.get("SHEEP", 0)
    has_carried_cow = any(u[1].get("COW", 0) > 0 for u in all_units)
    has_carried_sheep = any(u[1].get("SHEEP", 0) > 0 for u in all_units)
    has_animals_to_place = (n_cow_shed > 0) or (n_sheep_shed > 0) or has_carried_cow or has_carried_sheep

    if has_animals_to_place and farm_state["empty_pastures"]:
        for pos in farm_state["empty_pastures"]:
            tasks.append(Task(300, pos, ["PLACE"], need=("COW", "SHEEP")))

    if (n_cow_shed > 0 or n_sheep_shed > 0) and farm_state["empty_pastures"]:
        species = "COW" if n_cow_shed > 0 else "SHEEP"
        for shed_pos in SHED_TILES:
            tasks.append(Task(185, shed_pos, ["PICKUP", species, 1]))

    # 3. Crops (Harvest, Water, Fertilize, Reclaim expired)
    for p in farm_state["plants"]:
        if p["is_expired"]:
            tasks.append(Task(150, p["pos"], ["DIG"]))
        if not p["watered_today"] and not p["is_expired"]:
            tasks.append(Task(200, p["pos"], ["WATER"]))
        if p["yield_units"] > 0 and (p["is_ongoing"] or p["age"] >= p["first_yield_day"]):
            tasks.append(Task(195, p["pos"], ["HARVEST"]))
        if p["fertilize_due"]:
            tasks.append(Task(180, p["pos"], ["FERTILIZE"], need=("FERTILIZER",)))

    # 4. Fertilizer Pickup
    fert_due = sum(1 for p in farm_state["plants"] if p["fertilize_due"])
    if fert_due > 0 and shed.get("FERTILIZER", 0) > 0:
        for shed_pos in SHED_TILES:
            tasks.append(Task(178, shed_pos, ["PICKUP", "FERTILIZER", FERTILIZER_PICKUP_BATCH]))

    # 5. Weeds (restricted to planting season)
    if day <= WHEAT_LAST_PLANT_DAY:
        for pos in farm_state["weeds"]:
            tasks.append(Task(150, pos, ["DIG"]))

    # 6. Crop planting (concentric wavefront sorted by distance to (4, 4))
    if target_crop != "NONE":
        plantable = [pos for pos in farm_state["empty_tiles"] if pos not in _RESERVED_PASTURE_TILES]
        plantable.sort(key=lambda p: (manhattan(p, (4, 4)), p[1], p[0]))
        n_plant = min(len(plantable), seeds_stock.get(target_crop, 0))
        for pos in plantable[:n_plant]:
            tasks.append(Task(140, pos, ["PLANT", target_crop]))

    # 7. Build Pastures (High priority if animals are waiting in shed)
    pasture_priority = 200 if (n_cow_shed > 0 or n_sheep_shed > 0) else 120
    for pos in needed_pastures:
        act = ["DIG"] if pos in farm_state["weeds"] else ["BUILD_PASTURE"]
        tasks.append(Task(pasture_priority, pos, act))

    # 8. Wheat Feed Pickup (High priority so animals are fed before other field work)
    unfed_count = sum(1 for a in farm_state["animals"] if not a["fed_today"])
    if unfed_count > 0 and shed.get("WHEAT", 0) > 0:
        carried_feeders = sum(1 for u in all_units if u[1].get("WHEAT", 0) > 0)
        needed_feeders = min(4, max(1, math.ceil(unfed_count / 4.0)))
        if carried_feeders < needed_feeders:
            for shed_pos in SHED_TILES[: (needed_feeders - carried_feeders)]:
                tasks.append(Task(250, shed_pos, ["PICKUP", "WHEAT", 4]))

    return tasks


def _assign_tasks(
    units: list[tuple[tuple[int, int], dict[str, int]]],
    tasks: list[Task],
) -> dict[int, Task]:
    def eligible(inv: dict[str, int], task: Task) -> bool:
        # Don't pick up more animals if already carrying one — go PLACE first
        if (task.action[0] == "PICKUP"
                and len(task.action) >= 2
                and task.action[1] in ("COW", "SHEEP")
                and (inv.get("COW", 0) > 0 or inv.get("SHEEP", 0) > 0)):
            return False
        return task.need is None or any(inv.get(item, 0) > 0 for item in task.need)

    assigned: dict[int, Task] = {}
    claimed_units: set[int] = set()
    claimed_pos: set[tuple[int, int]] = set()

    # Pass 1: Immediate standing actions (zero travel overhead)
    for ui, (pos, inv) in enumerate(units):
        standing_tasks = [
            t for t in tasks
            if t.pos == pos and t.pos not in claimed_pos and eligible(inv, t)
        ]
        if standing_tasks:
            best_t = max(standing_tasks, key=lambda t: t.priority)
            assigned[ui] = best_t
            claimed_units.add(ui)
            claimed_pos.add(pos)

    # Pass 2: Unit-centric dispatch to nearest high-priority task
    unassigned_units = [ui for ui in range(len(units)) if ui not in claimed_units]
    for ui in unassigned_units:
        pos, inv = units[ui]
        available_tasks = [
            t for t in tasks
            if t.pos not in claimed_pos and eligible(inv, t)
        ]
        if not available_tasks:
            continue

        # Score tasks: prioritize high priority, then minimize travel distance
        # Priority tiers: tier = priority // 20 so close tasks within same priority tier win
        best_t = max(
            available_tasks,
            key=lambda t: (t.priority // 20, -manhattan(pos, t.pos), t.priority),
        )
        assigned[ui] = best_t
        claimed_units.add(ui)
        claimed_pos.add(best_t.pos)

    return assigned


def _to_action(
    pos: tuple[int, int],
    inv: dict[str, int],
    task: Task,
    step_to: Callable[[tuple[int, int]], list[Any]],
) -> list[Any]:
    if pos != task.pos:
        return step_to(task.pos)
    if task.action[0] == "PLACE":
        animal_to_place = "COW" if inv.get("COW", 0) > 0 else "SHEEP"
        return ["PLACE", animal_to_place]
    return task.action


def dispatch_units(
    all_units: list[tuple[tuple[int, int], dict[str, int]]],
    farm_state: dict[str, Any],
    shed: dict[str, int],
    seeds: dict[str, int],
    day: int,
    hour: int,
    needed_pastures: list[tuple[int, int]],
) -> list[list[Any]]:
    # Day 29 Final Evacuation: hour >= 21
    evac_actions: dict[int, list[Any]] = {}
    if day >= 29 and hour >= 21:
        for ui, (pos, inv) in enumerate(all_units):
            carried_sellable = sum(inv.get(item, 0) for item in _SELLABLE_ITEMS)
            if carried_sellable > 0:
                evac_actions[ui] = _go_shed_or_drop(pos)

    seeds_stock = dict(seeds)
    target_crop = _select_target_crop(day, seeds_stock)
    tasks = _build_tasks(farm_state, shed, seeds_stock, target_crop, needed_pastures, all_units, day)
    assigned = _assign_tasks(all_units, tasks)

    actions: list[list[Any]] = []
    for ui, (pos, inv) in enumerate(all_units):
        if ui in evac_actions:
            actions.append(evac_actions[ui])
            continue

        task = assigned.get(ui)
        if task is not None:
            step_to = lambda target, _p=pos: [bfs_step(_p, target) or "PASS"]
            actions.append(_to_action(pos, inv, task, step_to))
            continue

        # Unassigned fallback: return items to shed
        inv_total = sum(inv.values())
        if inv_total > 0:
            carrying_feed_only = inv.get("WHEAT", 0) == inv_total
            if not (carrying_feed_only and any(not a["fed_today"] for a in farm_state["animals"])):
                actions.append(_go_shed_or_drop(pos))
                continue
        actions.append(["PASS"])

    return actions


# ---------------------------------------------------------------------------
# Main Agent Entry Point
# ---------------------------------------------------------------------------


def agent(obs: dict[str, Any]) -> dict[str, Any]:
    player = obs["player"]
    day = obs["day"]
    hour = obs["hour"]

    me = obs["farms"][player]
    private = obs["private"]
    mkt = obs["market"]
    unlocked_shops = obs["town"]["unlocked_shops"]

    money = float(me["money"])
    tiles = me["tiles"]
    unlocked_quads = me["unlocked_quadrants"]
    shed = private["shed"]
    seeds = private["seeds"]
    market_inv = mkt["inventory"]
    inventories = private["inventories"]
    hires_today = me["hires_today"]

    # 1. Parse current farm grid
    farm_state = parse_farm_state(tiles, day)

    # 2. Animal census — computed once, used by market orders and task auction
    census = _count_animal_census(farm_state, shed, inventories)

    # 3. Structures currently needed
    needed_pastures = _compute_needed_pastures(farm_state, census)

    # 4. Plan Market Orders
    market_orders = plan_market_orders(
        day, hour, money, shed, seeds, market_inv,
        unlocked_quads, unlocked_shops, farm_state, hires_today, inventories,
        needed_pastures, census,
    )

    # 5. Assemble all units (Farmer + Hired Hands)
    farmer_pos = (me["farmer"][0], me["farmer"][1])
    hands_pos = [(h[0], h[1]) for h in me["hands"]]

    all_units: list[tuple[tuple[int, int], dict[str, int]]] = [
        (farmer_pos, inventories[0])
    ]
    for idx, hp in enumerate(hands_pos):
        all_units.append((hp, inventories[idx + 1]))

    # 6. Dispatch unit movement & field actions via Task Auction
    actions = dispatch_units(all_units, farm_state, shed, seeds, day, hour, needed_pastures)

    return {
        "farmer": actions[0],
        "hands": actions[1:],
        "market": market_orders,
    }


premaananda108_agent = agent
