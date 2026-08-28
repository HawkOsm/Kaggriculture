import math
from .config import RESERVE_FRACTION, get_config
from .constants import MARKET, MARKET_I0, TOTAL_DAYS, SHOPS, HINGE_GAIN, CROPS, ANIMALS

def _shape(name, value, throughput=None):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    if name == "hinge":
        # Degenerates to linear when the throughput is missing or non-positive.
        if not throughput or throughput <= 0:
            return value
        unit = value / float(throughput)
        return unit + HINGE_GAIN * max(0.0, unit - 1.0) ** 2
    return value

def _market_parameters(obs, item):
    base, throughput, below_fn, below_move, above_fn, above_move = MARKET[item]
    custom = (((obs or {}).get("market", {}) or {}).get("params", {}) or {}).get(
        item, {}
    )
    return (
        float(custom.get("base", base)),
        float(custom.get("T", throughput)),
        str(custom.get("below_func", below_fn)),
        float(custom.get("below_target", below_move)),
        str(custom.get("above_func", above_fn)),
        float(custom.get("above_target", above_move)),
        float(custom.get("I0", MARKET_I0)),
    )

def _price_at(item, inventory, obs=None):
    (
        base,
        throughput,
        below_fn,
        below_move,
        above_fn,
        above_move,
        equilibrium,
    ) = _market_parameters(obs, item)
    if inventory < equilibrium:
        amplitude = below_move * base / max(
            1e-9, _shape(below_fn, throughput, throughput)
        )
        value = base + amplitude * _shape(
            below_fn, equilibrium - inventory, throughput
        )
    else:
        amplitude = above_move * base / max(
            1e-9, _shape(above_fn, throughput, throughput)
        )
        value = base - amplitude * _shape(
            above_fn, inventory - equilibrium, throughput
        )
    return max(1, int(round(value)))

def _town_demand_per_day(obs, item):
    # The town centre buys one of each product once per day and does not scale
    # with the calendar; shop instances are drawn with replacement, so the same
    # shop can appear several times and each copy consumes independently.
    center = 0 if item == "FERTILIZER" else 1
    shop = 0
    for name in (((obs or {}).get("town", {}) or {}).get(
        "unlocked_shops", []
    ) or []):
        products = SHOPS.get(name, ())
        if item in products:
            shop += 12 if len(products) == 1 else 6
    return center + shop

def _opponent_visible_supply(obs, item, horizon=1):
    player = int((obs or {}).get("player", 0) or 0)
    day = int((obs or {}).get("day", 0) or 0)
    animal_for = {"EGG": "GOOSE", "MILK": "COW", "WOOL": "SHEEP"}
    total = 0
    for index, farm in enumerate((obs or {}).get("farms", []) or []):
        if index == player:
            continue
        for row in farm.get("tiles", []) or []:
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                if (
                    item in CROPS
                    and tile.get("kind") == "PLANT"
                    and tile.get("crop") == item
                ):
                    rule = CROPS[item]
                    planted = tile.get("planted_day")
                    planted_day = day if planted is None else int(planted)
                    age = day - planted_day
                    held = int(tile.get("yield_units", 0) or 0)
                    if held > 0 and age >= rule["first"]:
                        total += held
                    elif age + horizon >= rule["ripe"]:
                        total += max(1, min(rule["max_yield"], held + horizon))
                elif item in animal_for and tile.get("animal") == animal_for[item]:
                    total += int(tile.get("yield_units", 0) or 0)
                    if horizon > 0:
                        total += min(2 * horizon, ANIMALS.get(
                            tile.get("animal"), {"max_held": 4}
                        )["max_held"])
    return total

def _sell_quantity(item, have, inventory, day, shed_load, config, obs=None):
    left = TOTAL_DAYS - day
    if left <= 1:
        return have
    base = _market_parameters(obs, item)[0]
    reserve = base * get_config(config, "RESERVE_FRACTION", RESERVE_FRACTION)[item]
    if left <= 7:
        reserve *= max(0.0, (left - 1) / 6.0)
    if shed_load >= 0.75:
        reserve *= 0.55

    opponent_supply = _opponent_visible_supply(obs, item, horizon=1)
    town_demand = _town_demand_per_day(obs, item)
    if opponent_supply > town_demand:
        reserve *= max(0.72, 1.0 - 0.015 * (opponent_supply - town_demand))
    projected_inventory = inventory + opponent_supply - town_demand
    future_price = _price_at(item, projected_inventory, obs)
    threshold = reserve
    if shed_load < 0.75 and left > 7:
        threshold = max(threshold, 0.88 * future_price)

    quantity = 0
    while (
        quantity < have
        and _price_at(item, inventory + quantity, obs) >= threshold
    ):
        quantity += 1
    if left <= 12:
        # Forcing an even sell-off starting a full 12 days out means we keep
        # dumping into a price some other seller (opponent or us) already
        # crashed, one unit an hour, with no chance for it to recover --
        # observed selling MILK at $1 for 8 straight hours on day18 this
        # way. Only override the price-conscious threshold above while
        # there's still real runway to wait for recovery (left > 3) if the
        # price isn't actually crashed; once genuinely out of time, force it
        # regardless, since unsold inventory is worthless at game end.
        price_now = _price_at(item, inventory, obs)
        crashed = price_now < 0.15 * base
        # A crash isn't always temporary -- if the market stays oversupplied
        # (e.g. an opponent that keeps dumping regardless of price), waiting
        # it out just lets held inventory snowball with the capital tied up
        # in it, for no better a price later. Stop waiting once the pile
        # itself gets large, even if the price hasn't recovered.
        overstocked = have > 20
        if not crashed or left <= 3 or overstocked:
            forced = int(math.ceil(have / float(max(1, left - 1))))
            quantity = max(quantity, min(have, forced))
    return quantity
