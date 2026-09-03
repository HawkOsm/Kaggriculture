from .config import get_config as _cfg
from .constants import (
    MARKET, MAX_MARKET_ORDERS, SCHEDULE_WHEAT_CAP, TOTAL_DAYS, SHED_CAPACITY,
    PRODUCTS, MARKET_I0, FEED_STOCK_DAYS, CORE_HERD, ANIMALS, ANIMAL_PURCHASE_LAST_DAY,
    MAX_EXTRA_LAND, LAND_OPEN_DAYS, LAND_PRICES, CROPS
)
from .state import _survey, _policy_phase, _private_item_total
from .market_forecast import _sell_quantity, _price_at
from .farm_plan import _farm_animal_counts, _opponent_animal_counts, _livestock_score, _target_hands
from .dispatch import _fib, _post_field_storage, _market_commitment_cost, _append_schedule_order, _field_jobs

_SIGNATURE_ACTIVE = True
_SIGNATURE_LAST_STEP = -1
_SCHEDULE_WHEAT_REQUESTED = 0

def _public_type_counts(farm):
    counts = {}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                name = str(tile.get("crop", "UNKNOWN"))
            elif tile.get("animal"):
                name = str(tile["animal"])
            else:
                name = str(tile.get("kind", "STRUCTURE"))
            counts[name] = counts.get(name, 0) + 1
    return counts

def _schedule_signature(obs):
    global _SIGNATURE_ACTIVE, _SIGNATURE_LAST_STEP
    raw_step = obs.get("step")
    step = int(
        raw_step
        if raw_step is not None
        else 24 * int(obs.get("day", 0) or 0) + int(obs.get("hour", 0) or 0)
    )
    if step == 0 or step <= _SIGNATURE_LAST_STEP:
        _SIGNATURE_ACTIVE = False
    _SIGNATURE_LAST_STEP = step

    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0) or 0)
    if len(farms) != 2 or not (0 <= player < 2):
        return False
    opponent = farms[1 - player]
    counts = _public_type_counts(opponent)

    if step == 24:
        hard_negative = any(
            counts.get(item, 0) for item in ("GOOSE", "CARROT", "TOMATO")
        )
        _SIGNATURE_ACTIVE = (
            not hard_negative
            and len(opponent.get("unlocked_quadrants", []) or []) == 1
            and counts.get("COW", 0) == 3
            and counts.get("SHEEP", 0) == 1
            and counts.get("MELON", 0) == 6
            and counts.get("STRAWBERRY", 0) == 0
        )
    if step == 192 and _SIGNATURE_ACTIVE:
        _SIGNATURE_ACTIVE = (
            len(opponent.get("unlocked_quadrants", []) or []) == 1
            and 3 <= counts.get("COW", 0) <= 5
            and counts.get("SHEEP", 0) == 1
            and 9 <= counts.get("MELON", 0) <= 10
            and 6 <= counts.get("STRAWBERRY", 0) <= 7
        )
    if step == 264 and _SIGNATURE_ACTIVE:
        hard_negative = any(
            counts.get(item, 0) for item in ("GOOSE", "CARROT", "TOMATO")
        )
        _SIGNATURE_ACTIVE = (
            not hard_negative
            and len(opponent.get("unlocked_quadrants", []) or []) == 2
            and 3 <= counts.get("COW", 0) <= 5
            and counts.get("SHEEP", 0) == 5
            and counts.get("MELON", 0) == 6
            and 15 <= counts.get("STRAWBERRY", 0) <= 16
            and 4 <= counts.get("PASTURE", 0) <= 5
        )
    return _SIGNATURE_ACTIVE

def _schedule_market_adjustment(obs, config, farm, private, orders):
    global _SCHEDULE_WHEAT_REQUESTED
    active = _schedule_signature(obs)
    raw_step = obs.get("step")
    step = int(
        raw_step
        if raw_step is not None
        else 24 * int(obs.get("day", 0) or 0) + int(obs.get("hour", 0) or 0)
    )
    if step == 0 or not active:
        _SCHEDULE_WHEAT_REQUESTED = 0
        return orders
    day = int(obs.get("day", step // 24) or 0)
    hour = int(obs.get("hour", step % 24) or 0)
    if day != 11:
        _SCHEDULE_WHEAT_REQUESTED = 0
        return orders

    max_orders = int(_cfg(config, "maxMarketOrdersPerTurn", MAX_MARKET_ORDERS))
    result = [list(order) for order in orders[:max_orders]]
    if hour == 2 and _SCHEDULE_WHEAT_REQUESTED == 0:
        wheat_price = float(
            (((obs.get("market", {}) or {}).get("prices", {}) or {}).get(
                "WHEAT", MARKET["WHEAT"][0]
            )
            or MARKET["WHEAT"][0])
        )
        money = float(farm.get("money", 0) or 0)
        committed = _market_commitment_cost(obs, farm, result)
        affordable = int(
            max(0.0, money - committed - 750.0)
            // max(1.0, wheat_price + 25.0)
        )
        quantity = min(SCHEDULE_WHEAT_CAP, affordable)
        if quantity > 0:
            result, inserted = _append_schedule_order(
                result,
                ["BUY_PRODUCT", "WHEAT", quantity],
                max_orders,
            )
            if inserted:
                _SCHEDULE_WHEAT_REQUESTED = quantity
                return result

    if hour == 19 and _SCHEDULE_WHEAT_REQUESTED > 0:
        shed_wheat = int((private.get("shed", {}) or {}).get("WHEAT", 0) or 0)
        animals = sum(_farm_animal_counts(farm).values())
        quantity = min(
            _SCHEDULE_WHEAT_REQUESTED,
            max(0, shed_wheat - 3 * animals),
        )
        _SCHEDULE_WHEAT_REQUESTED = 0
        if quantity > 0:
            result, inserted = _append_schedule_order(
                result,
                ["SELL", "WHEAT", quantity],
                max_orders,
            )
            if inserted:
                return result
    return orders

def _seed_needs(obs, farm, private, roles):
    day = int(obs.get("day", 0) or 0)
    seeds = private.get("seeds", {}) or {}
    needs = {}
    for (x, y), (kind, item) in roles.items():
        if (
            kind == "CROP"
            and (
                farm["tiles"][y][x] is None
                or (
                    isinstance(farm["tiles"][y][x], dict)
                    and farm["tiles"][y][x].get("kind") == "WEED"
                )
            )
            and day <= CROPS[item]["last_plant"]
        ):
            needs[item] = needs.get(item, 0) + 1
    return {
        crop: max(0, count - int(seeds.get(crop, 0) or 0))
        for crop, count in needs.items()
    }

def _market_actions(obs, config, farm, private, roles, field):
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    left = TOTAL_DAYS - day
    money = float(farm.get("money", 0) or 0)
    shed_capacity = int(_cfg(config, "shedCapacity", SHED_CAPACITY))
    shed, post_field_inventories = _post_field_storage(
        private, field, shed_capacity
    )
    market_inventory = dict(
        ((obs.get("market", {}) or {}).get("inventory", {}) or {})
    )
    max_orders = int(_cfg(config, "maxMarketOrdersPerTurn", MAX_MARKET_ORDERS))
    summary = _survey(farm, private, roles, day)
    phase = _policy_phase(obs, config, farm, private, summary)
    orders = []
    occupancy = sum(max(0, int(value or 0)) for value in shed.values())
    shed_load = occupancy / float(max(1, shed_capacity))
    animal_pipeline = summary["animals"] + sum(
        summary["animal_stock"].values()
    )
    feed_floor = animal_pipeline * FEED_STOCK_DAYS
    total_wheat = int(shed.get("WHEAT", 0) or 0) + sum(
        int(inventory.get("WHEAT", 0) or 0)
        for inventory in post_field_inventories
    )

    sells = []
    for item in PRODUCTS:
        have = int(shed.get(item, 0) or 0)
        if item == "WHEAT" and left > 2:
            have = min(have, max(0, total_wheat - feed_floor))
        if have <= 0:
            continue
        raw_inventory = market_inventory.get(item)
        inventory = MARKET_I0 if raw_inventory is None else int(raw_inventory)
        quantity = _sell_quantity(
            item, have, inventory, day, shed_load, config, obs
        )
        if quantity <= 0:
            continue
        proceeds = sum(
            _price_at(item, inventory + offset, obs)
            for offset in range(quantity)
        )
        sells.append((proceeds, item, quantity))
    # Price-impact SELL ordering: adapted from raykkretzschmar's public
    # meta-notebook (Code-tab-shared, OSI-licensed per RULES.md 3.6; see
    # CREDITS.md for the same reimplementation-not-copy standard already
    # applied to pilkwang's architecture) -- "C71 Giovanni Impact"/C94/C95
    # sections describe racing steep-decay premium products (melon,
    # strawberry, milk, wool) ahead of an anticipated shared-market dump,
    # while leaving thinner two-sided markets (wheat, fertilizer -- ones
    # opponents also buy) on ordinary proceeds-based timing. Config-driven,
    # default () so this is a no-op until searched/enabled.
    priority_items = set(_cfg(config, "PRICE_IMPACT_PRODUCTS", ()))
    sells.sort(key=lambda entry: (entry[1] in priority_items, entry[0]), reverse=True)
    for proceeds, item, quantity in sells:
        if len(orders) >= max_orders:
            break
        orders.append(["SELL", item, quantity])
        money += 0.85 * proceeds
        occupancy = max(0, occupancy - quantity)
        shed[item] = max(0, int(shed.get(item, 0) or 0) - quantity)
        raw_inventory = market_inventory.get(item)
        inventory = MARKET_I0 if raw_inventory is None else int(raw_inventory)
        market_inventory[item] = inventory + quantity

    if field["liquidation"] or left <= 1:
        if day >= 29 and hour <= 1:
            terminal_jobs = _field_jobs(
                obs, config, farm, private, roles, liquidation=True
            )
            target = min(8, len(terminal_jobs))
            hires = int(farm.get("hires_today", 0) or 0)
            while hires < target and len(orders) < max_orders:
                cost = _fib(hires)
                if money < cost + 20:
                    break
                orders.append(["HIRE"])
                money -= cost
                hires += 1
        return orders[:max_orders]

    placed = _farm_animal_counts(farm)
    role_targets = {animal: 0 for animal in ANIMALS}
    for kind, item in roles.values():
        if kind == "ANIMAL" and item in role_targets:
            role_targets[item] += 1
    owned = {
        animal: placed.get(animal, 0) + _private_item_total(private, animal)
        for animal in ANIMALS
    }
    animal_capital_open = phase in {"BOOTSTRAP", "COMPOUND"} or (
        phase == "CRISIS"
        and summary["at_risk_animals"] == 0
        and summary["shed_load"] + summary["carried_load"] < 95
        and summary["open_structures"] > 0
        and animal_pipeline
        < summary["animals"] + summary["open_structures"]
    )
    if (
        animal_capital_open
        and day <= ANIMAL_PURCHASE_LAST_DAY
        and left >= 8
    ):
        purchase_order = sorted(
            ("COW", "SHEEP"),
            key=lambda animal: (
                _livestock_score(
                    obs,
                    animal,
                    owned[animal],
                    _opponent_animal_counts(obs).get(animal, 0),
                ),
                role_targets[animal] - owned[animal],
                animal == "COW",
            ),
            reverse=True,
        )
        for animal in purchase_order:
            if len(orders) >= max_orders:
                break
            missing = max(0, role_targets[animal] - owned[animal])
            if missing <= 0:
                continue
            operating_reserve = 80 if sum(owned.values()) < CORE_HERD else 220
            quantity = min(
                missing,
                2,
                max(0, shed_capacity - occupancy),
                max(
                    0,
                    int(
                        (money - operating_reserve)
                        // ANIMALS[animal]["cost"]
                    ),
                ),
            )
            if quantity > 0:
                orders.append(["BUY_ANIMAL", animal, quantity])
                money -= quantity * ANIMALS[animal]["cost"]
                occupancy += quantity
                owned[animal] += quantity

    total_wheat = int(shed.get("WHEAT", 0) or 0) + sum(
        int(inventory.get("WHEAT", 0) or 0)
        for inventory in post_field_inventories
    )
    planned_herd = sum(owned.values())
    desired_wheat = max(
        planned_herd * FEED_STOCK_DAYS,
        8 if planned_herd > 0 else 0,
    )
    if (
        desired_wheat > total_wheat
        and len(orders) < max_orders
        and planned_herd > 0
    ):
        raw_inventory = market_inventory.get("WHEAT")
        inventory = MARKET_I0 if raw_inventory is None else int(raw_inventory)
        emergency_reserve = 0 if summary["at_risk_animals"] else 80
        quantity = 0
        cost = 0
        limit = min(
            desired_wheat - total_wheat,
            max(0, shed_capacity - occupancy),
        )
        for offset in range(limit):
            unit = _price_at("WHEAT", inventory - offset - 1, obs)
            if money - cost - unit < emergency_reserve:
                break
            cost += unit
            quantity += 1
        if quantity > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", quantity])
            money -= cost
            occupancy += quantity

    extra_land = max(0, len(farm.get("unlocked_quadrants", ["NW"])) - 1)
    if (
        phase in {"BOOTSTRAP", "COMPOUND"}
        and extra_land < _cfg(config, "MAX_EXTRA_LAND", MAX_EXTRA_LAND)
        and day >= _cfg(config, "LAND_OPEN_DAYS", LAND_OPEN_DAYS)[extra_land]
        and left >= 12
        and len(orders) < max_orders
    ):
        cost = LAND_PRICES[extra_land]
        reserve = 300 if extra_land == 0 else 500
        if money >= cost + reserve:
            orders.append(["BUY_LAND"])
            money -= cost

    needs = _seed_needs(obs, farm, private, roles)
    seed_reserve = 80 if day <= 4 else 150
    seed_order = (
        ("MELON", "WHEAT")
        if day == 0
        else ("MELON", "WHEAT", "STRAWBERRY", "CARROT", "TOMATO")
    )
    for crop in seed_order:
        if len(orders) >= max_orders or needs.get(crop, 0) <= 0:
            continue
        cost = CROPS[crop]["seed"]
        quantity = min(
            needs[crop],
            25,
            max(0, int((money - seed_reserve) // cost)),
        )
        if quantity > 0:
            orders.append(["BUY_SEED", crop, quantity])
            money -= quantity * cost

    if hour <= 2:
        target_hands = _target_hands(obs, config, farm, private, roles)
        hires = int(farm.get("hires_today", 0) or 0)
        while hires < target_hands and len(orders) < max_orders:
            cost = _fib(hires)
            if money < max(20, 3 * cost):
                break
            orders.append(["HIRE"])
            money -= cost
            hires += 1

    return orders[:max_orders]

