"""kiykhoi_agent: local benchmark opponent.
"""
"""Kaggle entry point for the Kaggriculture competition."""

CROP_DATA = {
    "WHEAT": (10, 4, 4, False),
    "CARROT": (20, 3, 3, False),
    "TOMATO": (50, 4, 11, True),
    "STRAWBERRY": (100, 4, 16, True),
    "MELON": (80, 6, 10, False),
}
FIRST_YIELD_DAY = {"WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10}
HARVEST_AGE = {"WHEAT": 4, "CARROT": 3, "MELON": 10}
SELLABLE_ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
MAX_MELON_PLOTS = 4
MAX_HANDS = 3
MAX_MARKET_ORDERS = 10
WORK_TILES = ((4, 4), (3, 4), (4, 3), (3, 3))
QUADRANT_WORK_TILES = {
    "NW": WORK_TILES,
    "NE": ((5, 4), (6, 4), (5, 3), (6, 3)),
    "SW": ((4, 5), (3, 5), (4, 6), (3, 6)),
    "SE": ((5, 5), (6, 5), (5, 6), (6, 6)),
}
ANIMAL_DATA = {
    "GOOSE": (300, "EGG", "COOP", (2, 4), 1),
    "COW": (400, "MILK", "PASTURE", (2, 3), 2),
    "SHEEP": (500, "WOOL", "PASTURE", (2, 2), 3),
}
SHOP_DEMAND = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL", "WOOL"),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT", "CARROT"),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def _move_toward(position, target):
    x, y = position
    target_x, target_y = target
    if x != target_x:
        return ["EAST" if x < target_x else "WEST"]
    return ["SOUTH" if y < target_y else "NORTH"]


def _work_tiles(farm):
    return tuple(tile for quadrant in farm.get("unlocked_quadrants", ["NW"]) for tile in QUADRANT_WORK_TILES[quadrant])


def _crop_order(prices, seeds):
    ranked = sorted(
        CROP_DATA,
        key=lambda crop: (prices.get(crop, 0) * CROP_DATA[crop][1] - CROP_DATA[crop][0]) / CROP_DATA[crop][2],
        reverse=True,
    )
    return [crop for crop in ranked if seeds.get(crop, 0)]


def _fertilizer_worthwhile(crop, prices):
    return CROP_DATA[crop][3] and prices.get(crop, 0) * 4 > prices.get("FERTILIZER", 100)


def _needs_fertilizer(tile, day, prices):
    crop = tile["crop"]
    age = day - tile["planted_day"]
    return _fertilizer_worthwhile(crop, prices) and FIRST_YIELD_DAY[crop] <= age <= CROP_DATA[crop][2] and tile.get("fertilized_until_day", -1) < day


def _is_shed_access(position, board_size):
    half = board_size // 2
    return position in ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _preferred_animal(prices, money):
    if money < 5000:
        return None
    animal = max(ANIMAL_DATA, key=lambda candidate: prices.get(ANIMAL_DATA[candidate][1], 0) / ANIMAL_DATA[candidate][4])
    return animal if prices.get(ANIMAL_DATA[animal][1], 0) else None


def _unit_action(farm, position, day, seeds, claimed, crop_order, inventory, shed, prices, animal, work_tiles):
    """Care for a work tile, then move toward the next unclaimed one."""
    x, y = position
    tile = farm["tiles"][y][x]
    crop_to_plant = next(iter(crop_order), None)
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        claimed.add(position)
        return ["DIG"]
    if animal:
        animal_target = ANIMAL_DATA[animal][3]
        structure = ANIMAL_DATA[animal][2]
        target_tile = farm["tiles"][animal_target[1]][animal_target[0]]
        if target_tile is None:
            if position == animal_target:
                claimed.add(position)
                return ["BUILD_" + structure]
            return _move_toward(position, animal_target)
        if isinstance(target_tile, dict) and target_tile.get("kind") == structure and target_tile.get("animal") is None:
            if inventory.get(animal, 0):
                if position == animal_target:
                    claimed.add(position)
                    return ["PLACE", animal]
                return _move_toward(position, animal_target)
            if shed.get(animal, 0):
                if _is_shed_access(position, len(farm["tiles"])):
                    claimed.add(position)
                    return ["PICKUP", animal, 1]
                return _move_toward(position, (len(farm["tiles"]) // 2 - 1,) * 2)
        if isinstance(target_tile, dict) and target_tile.get("animal") == animal:
            if not target_tile.get("fed_today", False) and not inventory.get("WHEAT", 0) and shed.get("WHEAT", 0) and _is_shed_access(position, len(farm["tiles"])):
                claimed.add(position)
                return ["PICKUP", "WHEAT", 1]
            if position != animal_target:
                return _move_toward(position, animal_target)
            if not target_tile.get("fed_today", False):
                if inventory.get("WHEAT", 0):
                    claimed.add(position)
                    return ["FEED"]
                if shed.get("WHEAT", 0):
                    return _move_toward(position, (len(farm["tiles"]) // 2 - 1,) * 2)
            elif not target_tile.get("cared_today", False):
                claimed.add(position)
                return ["CARE"]
            elif target_tile.get("yield_units", 0):
                claimed.add(position)
                return ["HARVEST"]
            elif target_tile.get("fertilizer_available", False):
                claimed.add(position)
                return ["COLLECT_FERTILIZER"]
    if tile is None and position in work_tiles and position not in claimed and crop_to_plant:
        seeds[crop_to_plant] -= 1
        claimed.add(position)
        return ["PLANT", crop_to_plant]
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile.get("crop")
        age = day - tile["planted_day"]
        ongoing = CROP_DATA[crop][3]
        if _needs_fertilizer(tile, day, prices) and inventory.get("FERTILIZER", 0):
            claimed.add(position)
            return ["FERTILIZE"]
        if _needs_fertilizer(tile, day, prices) and shed.get("FERTILIZER", 0) and _is_shed_access(position, len(farm["tiles"])):
            claimed.add(position)
            return ["PICKUP", "FERTILIZER", 1]
        if not tile.get("watered_today", False):
            claimed.add(position)
            return ["WATER"]
        elif ongoing and tile.get("yield_units", 0) and age >= FIRST_YIELD_DAY[crop]:
            claimed.add(position)
            return ["HARVEST"]
        elif not ongoing and age >= HARVEST_AGE[crop]:
            claimed.add(position)
            return ["HARVEST"]

    needs_care = []
    empty_tiles = []
    for target in work_tiles:
        target_x, target_y = target
        target_tile = farm["tiles"][target_y][target_x]
        if isinstance(target_tile, dict) and target_tile.get("kind") == "PLANT":
            age = day - target_tile["planted_day"]
            crop = target_tile["crop"]
            ready = (
                CROP_DATA[crop][3] and target_tile.get("yield_units", 0) and age >= FIRST_YIELD_DAY[crop]
            ) or (not CROP_DATA[crop][3] and age >= HARVEST_AGE[crop])
            if not target_tile.get("watered_today", False) or ready:
                needs_care.append(target)
        elif target_tile is None and crop_to_plant:
            empty_tiles.append(target)

    targets = [target for target in needs_care + empty_tiles if target not in claimed]
    if targets:
        target = min(targets, key=lambda candidate: abs(x - candidate[0]) + abs(y - candidate[1]))
        claimed.add(target)
        return _move_toward(position, target)
    return ["PASS"]


def agent(obs):
    """Return one legal, stateless action for the current observation."""
    player = obs["player"]
    farm = obs["farms"][player]
    private = obs["private"]
    seeds = dict(private["seeds"])
    claimed = set()
    inventories = private["inventories"]
    crop_order = _crop_order(obs["market"]["prices"], seeds)
    animal = _preferred_animal(obs["market"]["prices"], farm["money"])
    work_tiles = _work_tiles(farm)

    x, y = farm["farmer"]
    farmer = _unit_action(farm, (x, y), obs["day"], seeds, claimed, crop_order, inventories[0], private["shed"], obs["market"]["prices"], animal, work_tiles)
    hands = []
    for index, (x, y) in enumerate(farm["hands"], start=1):
        inventory = inventories[index] if index < len(inventories) else {}
        hands.append(_unit_action(farm, (x, y), obs["day"], seeds, claimed, _crop_order(obs["market"]["prices"], seeds), inventory, private["shed"], obs["market"]["prices"], None, work_tiles))

    market = []
    for item in SELLABLE_ITEMS:
        if private["shed"].get(item, 0):
            market.append(["SELL", item, private["shed"][item]])
    wants_fertilizer = any(
        isinstance(tile, dict) and tile.get("kind") == "PLANT" and _needs_fertilizer(tile, obs["day"], obs["market"]["prices"])
        for row in farm["tiles"] for tile in row
    )
    has_fertilizer = private["shed"].get("FERTILIZER", 0) + sum(inventory.get("FERTILIZER", 0) for inventory in inventories)
    if wants_fertilizer and not has_fertilizer and farm["money"] >= obs["market"]["prices"].get("FERTILIZER", 100):
        market.append(["BUY_PRODUCT", "FERTILIZER", 1])
    if animal:
        cost = ANIMAL_DATA[animal][0]
        has_animal = any(
            isinstance(tile, dict) and tile.get("animal") == animal
            for row in farm["tiles"] for tile in row
        ) or private["shed"].get(animal, 0) or any(inventory.get(animal, 0) for inventory in inventories)
        if not has_animal and farm["money"] >= cost:
            market.append(["BUY_ANIMAL", animal, 1])
        needs_wheat = any(
            isinstance(tile, dict) and tile.get("animal") == animal and not tile.get("fed_today", False)
            for row in farm["tiles"] for tile in row
        )
        has_wheat = private["shed"].get("WHEAT", 0) + sum(inventory.get("WHEAT", 0) for inventory in inventories)
        if needs_wheat and not has_wheat and farm["money"] >= obs["market"]["prices"].get("WHEAT", 25):
            market.append(["BUY_PRODUCT", "WHEAT", 1])
    crop_count = sum(
        tile.get("kind") == "PLANT"
        for row in farm["tiles"] for tile in row
        if isinstance(tile, dict)
    )
    target_crop = _crop_order(obs["market"]["prices"], {crop: 1 for crop in CROP_DATA})[0]
    seed_cost = CROP_DATA[target_crop][0]
    needed_seeds = max(0, len(work_tiles) - crop_count - private["seeds"].get(target_crop, 0))
    if needed_seeds and farm["money"] >= needed_seeds * seed_cost:
        market.append(["BUY_SEED", target_crop, needed_seeds])

    if crop_count and len(farm["hands"]) < MAX_HANDS:
        market.extend([["HIRE"]] * (MAX_HANDS - len(farm["hands"])))
    unlocked = farm.get("unlocked_quadrants", ["NW"])
    land_costs = (1000, 2000, 4000)
    if crop_count >= len(work_tiles) and len(unlocked) < 4 and farm["money"] >= land_costs[len(unlocked) - 1]:
        market.append(["BUY_LAND"])

    town_demand = {}
    for shop in obs["town"].get("unlocked_shops", []):
        for item in SHOP_DEMAND.get(shop, ()):
            town_demand[item] = town_demand.get(item, 0) + 1

    def market_priority(order):
        if order[0] == "BUY_PRODUCT":
            return (0, 0)
        if order[0] == "SELL":
            return (1, -obs["market"]["prices"].get(order[1], 0), -town_demand.get(order[1], 0))
        return (2, 0)

    return {"farmer": farmer, "hands": hands, "market": sorted(market, key=market_priority)[:MAX_MARKET_ORDERS]}

kiykhoi_agent = agent

