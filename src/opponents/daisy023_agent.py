"""daisy023_agent: local benchmark opponent.
"""
import inspect

def my_agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    market_data = obs["market"]

    money = me["money"]
    tiles = me["tiles"]
    day = obs["day"]

    market = []

    # Get market prices
    prices = market_data.get("prices", {})

    wheat_price = prices.get("WHEAT", 25)
    carrot_price = prices.get("CARROT", 35)
    melon_price = prices.get("MELON", 250)

    # Sell crops
    for item in ["WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY"]:
        amount = private["shed"].get(item, 0)

        if amount > 0:
            market.append(["SELL", item, amount])

    # Choose crop based on price and remaining days
    if day <= 12 and melon_price >= 180 and money >= 80:
        crop = "MELON"
        seed_cost = 80

    elif day <= 20 and carrot_price >= 32 and money >= 20:
        crop = "CARROT"
        seed_cost = 20

    else:
        crop = "WHEAT"
        seed_cost = 10

    # Buy seeds
    seeds = private["seeds"].get(crop, 0)

    if seeds < 5 and money >= seed_cost:
        affordable = money // seed_cost
        amount = min(5 - seeds, affordable)

        if amount > 0:
            market.append(["BUY_SEED", crop, amount])

    # Hire workers gradually
    hands = me.get("hands", [])
    hand_count = len(hands)

    if day <= 8 and hand_count < 2 and money >= 3000:
        market.append(["HIRE"])

    elif day <= 16 and hand_count < 3 and money >= 6000:
        market.append(["HIRE"])

    # Find useful tiles
    targets = []

    for y in range(len(tiles)):
        for x in range(len(tiles[y])):

            tile = tiles[y][x]

            if tile == "LOCKED":
                continue

            if tile is None:
                targets.append((x, y, "PLANT"))
                continue

            if not isinstance(tile, dict):
                continue

            if tile.get("kind") != "PLANT":
                continue

            plant_crop = tile.get("crop")
            planted_day = tile.get("planted_day", day)
            age = day - planted_day

            # Water first
            if not tile.get("watered_today", False):
                targets.append((x, y, "WATER"))
                continue

            # Harvest when mature
            if plant_crop == "WHEAT" and age >= 4:
                targets.append((x, y, "HARVEST"))
                continue

            if plant_crop == "CARROT" and age >= 3:
                targets.append((x, y, "HARVEST"))
                continue

            if plant_crop == "MELON" and age >= 10:
                targets.append((x, y, "HARVEST"))
                continue

            # Harvest repeat-yield crops
            if plant_crop in ["TOMATO", "STRAWBERRY"]:
                if tile.get("yield_units", 0) > 0:
                    targets.append((x, y, "HARVEST"))

    # Find nearest target
    def choose_action(position, used):
        px, py = position

        choices = []

        for x, y, action in targets:

            if (x, y) in used:
                continue

            distance = abs(x - px) + abs(y - py)

            choices.append(
                (distance, x, y, action)
            )

        if not choices:
            return ["PASS"]

        choices.sort(key=lambda x: x[0])

        distance, tx, ty, action = choices[0]

        if px < tx:
            return ["EAST"]

        if px > tx:
            return ["WEST"]

        if py < ty:
            return ["SOUTH"]

        if py > ty:
            return ["NORTH"]

        used.add((tx, ty))

        if action == "PLANT":
            if private["seeds"].get(crop, 0) > 0:
                return ["PLANT", crop]

        if action == "WATER":
            return ["WATER"]

        if action == "HARVEST":
            return ["HARVEST"]

        return ["PASS"]

    # Assign different targets
    used = set()

    farmer_action = choose_action(
        me["farmer"],
        used
    )

    # Give hired hands their own targets
    hand_actions = []

    for hand in hands:
        hand_action = choose_action(
            hand,
            used
        )

        hand_actions.append(hand_action)

    # Give priority to the farmer's current tile
    fx, fy = me["farmer"]
    tile = tiles[fy][fx]

    if tile is None:

        if private["seeds"].get(crop, 0) > 0:
            farmer_action = ["PLANT", crop]

    elif isinstance(tile, dict):

        if tile.get("kind") == "PLANT":

            plant_crop = tile.get("crop")
            planted_day = tile.get("planted_day", day)
            age = day - planted_day

            if not tile.get("watered_today", False):
                farmer_action = ["WATER"]

            elif plant_crop == "WHEAT" and age >= 4:
                farmer_action = ["HARVEST"]

            elif plant_crop == "CARROT" and age >= 3:
                farmer_action = ["HARVEST"]

            elif plant_crop == "MELON" and age >= 10:
                farmer_action = ["HARVEST"]

            elif (
                plant_crop in ["TOMATO", "STRAWBERRY"]
                and tile.get("yield_units", 0) > 0
            ):
                farmer_action = ["HARVEST"]

    # Return actions
    return {
        "farmer": farmer_action,
        "hands": hand_actions,
        "market": market
    }


daisy023_agent = my_agent

