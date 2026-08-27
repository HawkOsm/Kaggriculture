import math
from collections import deque
from .constants import (
    TOTAL_DAYS, CROPS, ANIMALS, MARKET, PRODUCTS, MOVES,
    LIQUIDATION_TURNS, PRIORITY_BONUS, SHED_CAPACITY, TRAVEL_COST, LAND_PRICES
)
from .config import get_config as _cfg
from .state import _distance, _shed_tiles, _survey, _inventory_total

def _nearest_shed(position, board_size, tiles=None):
    return min(
        _shed_tiles(board_size, tiles),
        key=lambda target: (_distance(position, target), target[1], target[0]),
    )

def _bfs_first_step(tiles, source, target):
    source = (int(source[0]), int(source[1]))
    target = (int(target[0]), int(target[1]))
    if source == target:
        return ["PASS"]
    board_size = len(tiles)
    queue = deque([source])
    parent = {source: None}
    parent_move = {}
    while queue:
        current = queue.popleft()
        if current == target:
            break
        for name, dx, dy in MOVES:
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < board_size and 0 <= nxt[1] < board_size):
                continue
            if nxt in parent or tiles[nxt[1]][nxt[0]] == "LOCKED":
                continue
            parent[nxt] = current
            parent_move[nxt] = name
            queue.append(nxt)
    if target not in parent:
        return ["PASS"]
    current = target
    while parent[current] != source:
        current = parent[current]
        if current is None:
            return ["PASS"]
    return [parent_move[current]]

def _animal_produces_tonight(tile, rule, day):
    next_day = int(day) + 1
    placed_day = int(tile.get("placed_day", day) or 0)
    days_since_first = next_day - placed_day - int(rule["first"])
    return (
        days_since_first >= 0
        and days_since_first % int(rule["interval"]) == 0
    )

def _add_job(
    jobs,
    priority,
    value,
    target,
    action,
    need=None,
    reason="",
    latest_hour=23,
):
    jobs.append(
        {
            "priority": int(priority),
            "value": float(value),
            "target": tuple(target),
            "action": list(action),
            "need": need,
            "reason": str(reason),
            "latest_hour": int(latest_hour),
        }
    )

def _crop_jobs(obs, jobs, tile, target, day, liquidation):
    crop = tile.get("crop")
    rule = CROPS.get(crop)
    if rule is None:
        return
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    price = float(prices.get(crop, MARKET[crop][0]) or MARKET[crop][0])
    age = day - int(tile.get("planted_day", day))
    amount = int(tile.get("yield_units", 0) or 0)
    watered = bool(tile.get("watered_today", False))
    drought = int(tile.get("consecutive_unwatered", 0) or 0)
    critical = drought >= 1

    if liquidation:
        if amount > 0 and age >= rule["first"]:
            _add_job(
                jobs,
                0,
                amount * price,
                target,
                ("HARVEST",),
                reason="terminal_harvest",
            )
        return

    if critical and not watered:
        protected = max(amount, rule["max_yield"] * 0.7) * price
        _add_job(
            jobs,
            0,
            protected,
            target,
            ("WATER",),
            reason="critical_water",
        )
        return

    if rule["ongoing"]:
        next_day = day + 1
        interval = max(1, int(rule.get("interval", 1)))
        days_since_first = (
            next_day - int(tile.get("planted_day", day)) - rule["first"]
        )
        production_index = (
            days_since_first // interval + 1 if days_since_first >= 0 else 0
        )
        produces_tonight = (
            days_since_first >= 0
            and days_since_first % interval == 0
            and production_index <= rule["max_yield"]
        )
        fertilized_until = tile.get("fertilized_until_day", -1)
        fertilized_until = (
            -1 if fertilized_until is None else int(fertilized_until)
        )
        if (
            crop == "STRAWBERRY"
            and produces_tonight
            and fertilized_until < day
        ):
            _add_job(
                jobs,
                2,
                max(100.0, price),
                target,
                ("FERTILIZE",),
                need="FERTILIZER",
                reason="strawberry_production_fertilizer",
            )
        if amount >= rule["max_yield"] - 1 or (amount > 0 and day >= 27):
            _add_job(
                jobs,
                2,
                amount * price,
                target,
                ("HARVEST",),
                reason="ongoing_harvest",
            )
        elif not watered and age >= rule["first"] - 1:
            _add_job(
                jobs,
                3,
                price,
                target,
                ("WATER",),
                reason="ongoing_water",
            )
        return

    ripe = age >= rule["ripe"] and amount > 0
    in_growth_window = (rule["max_day"] + 1) // 2 <= age <= rule["max_day"]
    if ripe:
        if in_growth_window and not watered and amount < rule["max_yield"]:
            _add_job(
                jobs,
                1,
                price,
                target,
                ("WATER",),
                reason="final_growth_water",
            )
        else:
            _add_job(
                jobs,
                2,
                amount * price,
                target,
                ("HARVEST",),
                reason="ripe_harvest",
            )
    elif in_growth_window and not watered:
        _add_job(
            jobs,
            3,
            price,
            target,
            ("WATER",),
            reason="yield_water",
        )

def _terminal_feasible(position, target, tiles, actions_left):
    board_size = len(tiles)
    return (
        _distance(position, target)
        + 1
        + min(
            _distance(target, shed)
            for shed in _shed_tiles(board_size, tiles)
        )
        + 1
        <= actions_left
    )

def _field_jobs(obs, config, farm, private, roles, liquidation):
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    left = TOTAL_DAYS - day
    tiles = farm["tiles"]
    seeds = dict(private.get("seeds", {}) or {})
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    jobs = []
    planned_roles = dict(roles)
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict) or (x, y) in planned_roles:
                continue
            if tile.get("kind") == "PLANT" and tile.get("crop") in CROPS:
                planned_roles[(x, y)] = ("CROP", tile["crop"])
            elif tile.get("animal") in ANIMALS:
                planned_roles[(x, y)] = ("ANIMAL", tile["animal"])

    for target, role in planned_roles.items():
        x, y = target
        tile = tiles[y][x]
        role_kind, role_item = role

        if tile is None:
            if liquidation:
                continue
            if role_kind == "ANIMAL":
                animal_rule = ANIMALS[role_item]
                if left >= animal_rule["first"] + 2:
                    _add_job(
                        jobs,
                        3,
                        420,
                        target,
                        ("BUILD_" + animal_rule["structure"],),
                        reason="build_" + animal_rule["structure"].lower(),
                        latest_hour=22,
                    )
            elif (
                hour <= 22
                and day <= CROPS[role_item]["last_plant"]
                and seeds.get(role_item, 0) > 0
            ):
                rule = CROPS[role_item]
                expected = 4 if role_item == "WHEAT" else rule["max_yield"]
                price = float(
                    prices.get(role_item, MARKET[role_item][0])
                    or MARKET[role_item][0]
                )
                value = max(40, 0.65 * expected * price - rule["seed"])
                _add_job(
                    jobs,
                    4,
                    value,
                    target,
                    ("PLANT", role_item),
                    reason="plant_" + role_item,
                    latest_hour=22,
                )
            continue

        if not isinstance(tile, dict):
            continue
        kind = tile.get("kind")
        if role_kind == "ANIMAL" and kind in {"WEED", "PLANT"}:
            if not liquidation:
                _add_job(
                    jobs,
                    2,
                    500,
                    target,
                    ("DIG",),
                    reason="clear_animal_slot",
                    latest_hour=22,
                )
            continue
        if kind == "WEED":
            if not liquidation:
                _add_job(
                    jobs,
                    4,
                    120 if left > 5 else 10,
                    target,
                    ("DIG",),
                    reason="dig_weed",
                    latest_hour=22,
                )
            continue
        if kind == "PLANT":
            _crop_jobs(obs, jobs, tile, target, day, liquidation)
            continue
        if (
            role_kind == "ANIMAL"
            and kind == ANIMALS[role_item]["structure"]
            and "animal" not in tile
        ):
            if not liquidation:
                _add_job(
                    jobs,
                    1,
                    900,
                    target,
                    ("PLACE", role_item),
                    need=role_item,
                    reason="place_" + role_item.lower(),
                )
            continue
        if (
            role_kind == "ANIMAL"
            and kind in {"COOP", "PASTURE"}
            and kind != ANIMALS[role_item]["structure"]
            and "animal" not in tile
        ):
            if not liquidation:
                _add_job(
                    jobs,
                    3,
                    250,
                    target,
                    ("DIG",),
                    reason="replace_incompatible_structure",
                )
            continue
        if "animal" not in tile:
            continue
        if liquidation:
            if int(tile.get("yield_units", 0) or 0) > 0:
                product = ANIMALS.get(
                    tile.get("animal"), ANIMALS["GOOSE"]
                )["product"]
                value = int(tile.get("yield_units", 0) or 0) * float(
                    prices.get(product, MARKET[product][0])
                    or MARKET[product][0]
                )
                _add_job(
                    jobs,
                    0,
                    value,
                    target,
                    ("HARVEST",),
                    reason="terminal_animal",
                )
            if tile.get("fertilizer_available", False):
                _add_job(
                    jobs,
                    0,
                    float(
                        prices.get("FERTILIZER", MARKET["FERTILIZER"][0])
                        or MARKET["FERTILIZER"][0]
                    ),
                    target,
                    ("COLLECT_FERTILIZER",),
                    reason="terminal_fertilizer",
                )
            continue
        if not tile.get("fed_today", False):
            risk = int(tile.get("consecutive_unfed", 0) or 0) >= 1
            _add_job(
                jobs,
                0 if risk else 1,
                900 if risk else 260,
                target,
                ("FEED",),
                need="WHEAT",
                reason="critical_feed" if risk else "feed",
            )
        held = int(tile.get("yield_units", 0) or 0)
        animal_rule = ANIMALS.get(tile.get("animal"), ANIMALS["GOOSE"])
        product = animal_rule["product"]
        pending_care = int(tile.get("pending_care_bonus", 0) or 0)
        produces_tonight = _animal_produces_tonight(
            tile, animal_rule, day
        )
        production_gain = 1 + pending_care if produces_tonight else 0
        if held > 0 and (
            held >= 3
            or held + production_gain >= animal_rule["max_held"]
            or day >= 27
        ):
            _add_job(
                jobs,
                2,
                held
                * float(
                    prices.get(product, MARKET[product][0])
                    or MARKET[product][0]
                ),
                target,
                ("HARVEST",),
                reason="animal_harvest",
            )
        if tile.get("fertilizer_available", False):
            _add_job(
                jobs,
                2,
                float(
                    prices.get("FERTILIZER", MARKET["FERTILIZER"][0])
                    or MARKET["FERTILIZER"][0]
                ),
                target,
                ("COLLECT_FERTILIZER",),
                reason="fertilizer",
            )
        if (
            not tile.get("cared_today", False)
            and day <= 27
            and held
            + (0 if produces_tonight else pending_care)
            + 1
            < animal_rule["max_held"]
            and float(
                prices.get(product, MARKET[product][0])
                or MARKET[product][0]
            ) >= 20
        ):
            _add_job(
                jobs,
                3,
                float(
                    prices.get(product, MARKET[product][0])
                    or MARKET[product][0]
                ),
                target,
                ("CARE",),
                reason="care",
            )
    return jobs



def _unit_actions(obs, config, farm, private, roles):
    tiles = farm["tiles"]
    board_size = len(tiles)
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    step = int(obs.get("step", day * 24 + hour) or 0)
    final_step = int(_cfg(config, "episodeSteps", 720)) - 2
    actions_left = max(0, final_step - step + 1)
    liquidation = actions_left <= _cfg(config, "LIQUIDATION_TURNS", LIQUIDATION_TURNS)

    positions = [farm["farmer"], *(farm.get("hands", []) or [])]
    inventories = [dict(inv or {}) for inv in private.get("inventories", []) or []]
    while len(inventories) < len(positions):
        inventories.append({})

    summary = _survey(farm, private, roles, day)
    jobs = _field_jobs(obs, config, farm, private, roles, liquidation)
    seed_budget = dict(private.get("seeds", {}) or {})
    actions = [["PASS"] for _ in positions]

    feed_jobs = [job for job in jobs if job["need"] == "WHEAT"]
    shed_wheat = int((private.get("shed", {}) or {}).get("WHEAT", 0) or 0)
    carried_wheat = sum(int(inv.get("WHEAT", 0) or 0) for inv in inventories)

    missions = []
    for job in jobs:
        mission = dict(job)
        mission.update({"kind": "FIELD"})
        missions.append(mission)

    wheat_missing = max(0, len(feed_jobs) - carried_wheat)
    wheat_pickups = min(
        len(positions),
        int(math.ceil(min(wheat_missing, shed_wheat) / 6.0)),
    )
    wheat_remaining = min(wheat_missing, shed_wheat)
    critical_feed = any(job["priority"] == 0 for job in feed_jobs)
    for _ in range(wheat_pickups):
        amount = min(6, wheat_remaining)
        wheat_remaining -= amount
        missions.append(
            {
                "kind": "PICKUP",
                "item": "WHEAT",
                "amount": amount,
                "priority": 0 if critical_feed else 1,
                "value": 900 if critical_feed else 500,
                "target": None,
            }
        )

    fertilizer_jobs = [
        job for job in jobs if job.get("need") == "FERTILIZER"
    ]
    shed_fertilizer = int(
        (private.get("shed", {}) or {}).get("FERTILIZER", 0) or 0
    )
    carried_fertilizer = sum(
        int(inventory.get("FERTILIZER", 0) or 0)
        for inventory in inventories
    )
    fertilizer_missing = max(
        0, len(fertilizer_jobs) - carried_fertilizer
    )
    fertilizer_remaining = min(fertilizer_missing, shed_fertilizer)
    fertilizer_pickups = min(
        len(positions),
        int(math.ceil(fertilizer_remaining / 4.0)),
    )
    for _ in range(fertilizer_pickups):
        amount = min(4, fertilizer_remaining)
        fertilizer_remaining -= amount
        missions.append(
            {
                "kind": "PICKUP",
                "item": "FERTILIZER",
                "amount": amount,
                "priority": 1,
                "value": 700,
                "target": None,
            }
        )

    for animal in ANIMALS:
        place_jobs = [job for job in jobs if job["need"] == animal]
        shed_animals = int(
            (private.get("shed", {}) or {}).get(animal, 0) or 0
        )
        carried_animals = sum(
            int(inv.get(animal, 0) or 0) for inv in inventories
        )
        pickup_count = min(
            max(0, len(place_jobs) - carried_animals),
            shed_animals,
            2,
            len(positions),
        )
        for _ in range(pickup_count):
            missions.append(
                {
                    "kind": "PICKUP",
                    "item": animal,
                    "amount": 1,
                    "priority": 1,
                    "value": 900,
                    "target": None,
                }
            )

    market_prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    pressure = summary["shed_load"] + summary["carried_load"]
    cash_needed = (
        day < 22
        and float(farm.get("money", 0) or 0) < 500
    )
    for index, inventory in enumerate(inventories):
        cash_units = sum(
            max(0, int(inventory.get(item, 0) or 0)) for item in PRODUCTS
        )
        if cash_units <= 0:
            continue
        cash_value = sum(
            max(0, int(inventory.get(item, 0) or 0))
            * float(market_prices.get(item, MARKET[item][0]) or MARKET[item][0])
            for item in PRODUCTS
        )
        has_feed_mission = (
            int(inventory.get("WHEAT", 0) or 0) > 0 and bool(feed_jobs)
        )
        should_drop = (
            liquidation
            or pressure >= 80
            or cash_units >= 20
            or cash_value >= 2500
            or (cash_needed and cash_value >= 400)
            or (
                tuple(positions[index]) in _shed_tiles(board_size, tiles)
                and not has_feed_mission
                and cash_needed
            )
        )
        if should_drop:
            missions.append(
                {
                    "kind": "DROP",
                    "eligible": index,
                    "priority": -1 if liquidation else 2,
                    "value": max(120.0, 0.22 * cash_value),
                    "target": None,
                }
            )

    pairs = []
    for worker_index, raw_position in enumerate(positions):
        position = (int(raw_position[0]), int(raw_position[1]))
        inventory = inventories[worker_index]
        for mission_index, mission in enumerate(missions):
            kind = mission["kind"]
            if kind == "DROP" and mission["eligible"] != worker_index:
                continue
            if kind == "FIELD":
                need = mission.get("need")
                if need is not None and int(inventory.get(need, 0) or 0) <= 0:
                    continue
                target = mission["target"]
                distance = _distance(position, target)
                if hour + distance > mission.get("latest_hour", 23):
                    continue
                if liquidation and not _terminal_feasible(
                    position, target, tiles, actions_left
                ):
                    continue
            else:
                target = _nearest_shed(position, board_size, tiles)
                distance = _distance(position, target)
                if kind == "PICKUP" and int(
                    inventory.get(mission["item"], 0) or 0
                ) > 0:
                    continue

            priority = int(mission["priority"])
            bonus_map = _cfg(config, "PRIORITY_BONUS", PRIORITY_BONUS)
            # JSON round-trips int dict keys to strings (json.dump/json.load) --
            # a config loaded from a saved file has "PRIORITY_BONUS" keyed by
            # str(priority), not priority itself, so a plain .get(priority, ...)
            # silently misses every entry and falls back to the much weaker
            # -1000.0 * priority approximation. Try both key forms.
            bonus = bonus_map.get(priority, bonus_map.get(str(priority), -1000.0 * priority))
            score = (
                bonus
                + float(mission["value"])
                - _cfg(config, "TRAVEL_COST", TRAVEL_COST) * distance
            )
            pairs.append(
                (
                    -score,
                    distance,
                    worker_index,
                    mission_index,
                    target[1],
                    target[0],
                    target,
                )
            )

    used_workers = set()
    used_missions = set()
    used_targets = set()
    shed_capacity = int(_cfg(config, "shedCapacity", SHED_CAPACITY))
    drop_room = max(0, shed_capacity - summary["shed_load"])
    for _, distance, worker_index, mission_index, _, _, target in sorted(pairs):
        if worker_index in used_workers or mission_index in used_missions:
            continue
        mission = missions[mission_index]
        target_key = target
        if mission["kind"] == "FIELD":
            operation = mission["action"][0]
            if (
                liquidation
                and operation in {"HARVEST", "COLLECT_FERTILIZER"}
            ) or operation == "FERTILIZE":
                target_key = (target, operation)
            if target_key in used_targets:
                continue

        action = None
        plant_crop = None
        if mission["kind"] == "FIELD":
            planned = mission["action"]
            if planned[0] == "PLANT":
                plant_crop = planned[1]
                if int(seed_budget.get(plant_crop, 0) or 0) <= 0:
                    continue
            action = (
                list(planned)
                if distance == 0
                else _bfs_first_step(tiles, positions[worker_index], target)
            )
        elif mission["kind"] == "PICKUP":
            action = (
                ["PICKUP", mission["item"], int(mission["amount"])]
                if distance == 0
                else _bfs_first_step(tiles, positions[worker_index], target)
            )
        else:
            inventory = inventories[worker_index]
            product_counts = {
                item: max(0, int(inventory.get(item, 0) or 0))
                for item in PRODUCTS
                if int(inventory.get(item, 0) or 0) > 0
            }
            if distance:
                action = _bfs_first_step(
                    tiles, positions[worker_index], target
                )
            else:
                cash_units = sum(product_counts.values())
                noncash = _inventory_total(inventory) - cash_units
                if cash_units <= 0 or drop_room <= 0:
                    continue
                if noncash <= 0 and cash_units <= drop_room:
                    action = ["DROP"]
                    drop_room -= cash_units
                else:
                    item = max(
                        product_counts,
                        key=lambda name: (
                            float(
                                market_prices.get(name, MARKET[name][0])
                                or MARKET[name][0]
                            ),
                            name,
                        ),
                    )
                    quantity = min(product_counts[item], drop_room)
                    if quantity <= 0:
                        continue
                    action = ["PLACE", item, quantity]
                    drop_room -= quantity

        if not action or (action == ["PASS"] and distance > 0):
            continue
        if plant_crop is not None:
            seed_budget[plant_crop] -= 1
        if mission["kind"] == "FIELD":
            used_targets.add(target_key)
        actions[worker_index] = action
        used_workers.add(worker_index)
        used_missions.add(mission_index)

    return {
        "farmer": actions[0] if actions else ["PASS"],
        "hands": actions[1:],
        "liquidation": liquidation,
    }


# Market and capital allocation

def _fib(index):
    a, b = 1, 1
    for _ in range(index):
        a, b = b, a + b
    return a


def _pending_drop(private, field, capacity=SHED_CAPACITY):
    pending = {}
    inventories = list(private.get("inventories", []) or [])
    actions = [field["farmer"], *field["hands"]]
    room = max(
        0,
        int(capacity)
        - sum(
            max(0, int(value or 0))
            for value in (private.get("shed", {}) or {}).values()
        ),
    )
    for index, action in enumerate(actions):
        if index >= len(inventories) or not action or room <= 0:
            continue
        inventory = inventories[index] or {}
        if action[0] == "DROP":
            for item, count in inventory.items():
                accepted = min(max(0, int(count or 0)), room)
                if item in PRODUCTS and accepted > 0:
                    pending[item] = pending.get(item, 0) + accepted
                room -= accepted
                if room <= 0:
                    break
        elif action[0] == "PLACE" and len(action) >= 2:
            item = action[1]
            requested = int(action[2]) if len(action) >= 3 else 1
            accepted = min(
                max(0, requested),
                max(0, int(inventory.get(item, 0) or 0)),
                room,
            )
            if item in PRODUCTS and accepted > 0:
                pending[item] = pending.get(item, 0) + accepted
            room -= accepted
    return pending


def _post_field_storage(private, field, capacity=SHED_CAPACITY):
    shed = {
        item: max(0, int(count or 0))
        for item, count in (private.get("shed", {}) or {}).items()
    }
    inventories = [
        {
            item: max(0, int(count or 0))
            for item, count in (inventory or {}).items()
            if int(count or 0) > 0
        }
        for inventory in (private.get("inventories", []) or [])
    ]
    actions = [field.get("farmer", ["PASS"]), *field.get("hands", [])]

    while len(inventories) < len(actions):
        inventories.append({})

    for index, action in enumerate(actions):
        if not action:
            continue
        inventory = inventories[index]
        operation = action[0]
        if operation == "DROP":
            for item, count in list(inventory.items()):
                room = max(0, int(capacity) - sum(shed.values()))
                accepted = min(max(0, int(count or 0)), room)
                if accepted > 0:
                    shed[item] = shed.get(item, 0) + accepted
                del inventory[item]
        elif operation == "PLACE" and len(action) >= 2:
            item = action[1]
            if item not in PRODUCTS:
                continue
            requested = int(action[2]) if len(action) >= 3 else 1
            accepted = min(
                max(0, requested),
                max(0, int(inventory.get(item, 0) or 0)),
                max(0, int(capacity) - sum(shed.values())),
            )
            if accepted > 0:
                inventory[item] -= accepted
                if inventory[item] == 0:
                    del inventory[item]
                shed[item] = shed.get(item, 0) + accepted
        elif operation == "PICKUP" and len(action) >= 2:
            item = action[1]
            requested = int(action[2]) if len(action) >= 3 else 1
            picked = min(
                max(0, requested),
                max(0, int(shed.get(item, 0) or 0)),
            )
            if picked > 0:
                shed[item] -= picked
                inventory[item] = inventory.get(item, 0) + picked
        elif (
            operation == "FEED"
            and int(inventory.get("WHEAT", 0) or 0) > 0
        ):
            inventory["WHEAT"] -= 1
            if inventory["WHEAT"] == 0:
                del inventory["WHEAT"]
        elif (
            operation == "FERTILIZE"
            and int(inventory.get("FERTILIZER", 0) or 0) > 0
        ):
            inventory["FERTILIZER"] -= 1
            if inventory["FERTILIZER"] == 0:
                del inventory["FERTILIZER"]

    return shed, inventories


def _market_commitment_cost(obs, farm, orders):
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    hires = int(farm.get("hires_today", 0) or 0)
    land_index = max(0, len(farm.get("unlocked_quadrants", []) or []) - 1)
    cost = 0.0
    for order in orders:
        if not order:
            continue
        operation = order[0]
        quantity = int(order[2]) if len(order) > 2 else 1
        if operation == "HIRE":
            cost += _fib(hires)
            hires += 1
        elif operation == "BUY_LAND" and land_index < len(LAND_PRICES):
            cost += LAND_PRICES[land_index]
            land_index += 1
        elif operation == "BUY_ANIMAL" and len(order) > 1:
            cost += ANIMALS.get(order[1], {"cost": 10_000})["cost"] * quantity
        elif operation == "BUY_SEED" and len(order) > 1:
            cost += CROPS.get(order[1], {"seed": 10_000})["seed"] * quantity
        elif operation == "BUY_PRODUCT" and len(order) > 1:
            cost += float(prices.get(order[1], 100) or 100) * quantity * 1.25
    return cost


def _append_schedule_order(orders, order, max_orders):
    result = [list(value) for value in orders[:max_orders]]
    if len(result) < max_orders:
        result.append(order)
        return result, True
    for index in range(len(result) - 1, -1, -1):
        if result[index] and result[index][0] == "HIRE":
            result[index] = order
            return result, True
    return result, False


