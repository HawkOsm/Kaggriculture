import math
from .constants import ANIMALS, MARKET, CORE_HERD_SEQUENCE
from .config import get_config as _cfg, CORE_HERD, MID_HERD, TARGET_HERD, HERD_EXPANSION_DAY, HERD_FINAL_DAY, ANIMAL_PURCHASE_LAST_DAY
from .constants import TOTAL_DAYS, MAX_HANDS
from .state import _survey, _private_item_total
from .market_forecast import _town_demand_per_day

# TEMPORARY SHIM for Stage 2: _target_hands needs _field_jobs, but dispatch.py is Stage 3.
from .dispatch import _field_jobs

def _farm_animal_counts(farm):
    counts = {}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and "animal" in tile:
                counts[tile["animal"]] = counts.get(tile["animal"], 0) + 1
    return counts

def _opponent_animal_counts(obs):
    counts = {}
    player = int(obs.get("player", 0) or 0)
    for index, farm in enumerate(obs.get("farms", []) or []):
        if index == player:
            continue
        for row in farm.get("tiles", []) or []:
            for tile in row:
                if isinstance(tile, dict) and "animal" in tile:
                    counts[tile["animal"]] = counts.get(tile["animal"], 0) + 1
    return counts

def _livestock_score(obs, animal, own_count, opponent_count):
    rule = ANIMALS[animal]
    product = rule["product"]
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    price = float(prices.get(product, MARKET[product][0]) or MARKET[product][0])
    normalized_price = price / float(MARKET[product][0])
    demand_support = 1.0 + 0.012 * _town_demand_per_day(obs, product)
    crowding = 1.0 + 0.18 * opponent_count + 0.08 * own_count
    return normalized_price * demand_support / crowding

def _herd_targets(obs, config, farm, private, capacity):
    day = int(obs.get("day", 0) or 0)
    left = TOTAL_DAYS - day
    placed = _farm_animal_counts(farm)
    owned = {
        animal: placed.get(animal, 0) + _private_item_total(private, animal)
        for animal in ("COW", "SHEEP")
    }
    if day < _cfg(config, "HERD_EXPANSION_DAY", HERD_EXPANSION_DAY):
        stage_target = _cfg(config, "CORE_HERD", CORE_HERD)
    elif day < _cfg(config, "HERD_FINAL_DAY", HERD_FINAL_DAY):
        stage_target = _cfg(config, "MID_HERD", MID_HERD)
    else:
        stage_target = _cfg(config, "TARGET_HERD", TARGET_HERD)
    if day > _cfg(config, "ANIMAL_PURCHASE_LAST_DAY", ANIMAL_PURCHASE_LAST_DAY) or left < 8:
        stage_target = sum(owned.values())
    target_total = min(capacity, max(sum(owned.values()), stage_target))

    targets = {
        animal: max(
            CORE_HERD_SEQUENCE.count(animal)
            if target_total >= _cfg(config, "CORE_HERD", CORE_HERD)
            else 0,
            owned[animal],
        )
        for animal in ("COW", "SHEEP")
    }
    opponents = _opponent_animal_counts(obs)
    while sum(targets.values()) < target_total:
        animal = max(
            ("COW", "SHEEP"),
            key=lambda name: (
                _livestock_score(
                    obs,
                    name,
                    targets[name],
                    opponents.get(name, 0),
                ),
                -targets[name],
                name == "COW",
            ),
        )
        targets[animal] += 1
    return targets

def _base_target_hands(obs, config, farm, private, roles):
    day = int(obs.get("day", 0) or 0)
    summary = _survey(farm, private, roles, day)
    due_jobs = len(_field_jobs(obs, config, farm, private, roles, liquidation=False))
    active_roles = (
        summary["plants"]
        + summary["animals"]
        + summary["plantable"]
        + summary["structures_todo"]
    )
    floor = 10 if day <= 27 and active_roles > 0 else 4
    risk_load = 2 * (
        summary["at_risk_animals"] + summary["at_risk_crops"]
    )
    demand_target = int(math.ceil((due_jobs + risk_load) / 7.0))
    return max(4, min(MAX_HANDS, max(floor, demand_target)))

_LABOR_GATE_BASE_TARGET_HANDS = _base_target_hands
_LABOR_GATE_CONDITIONS = ({'field': 'obs.day', 'operator': '>=', 'value': 20},)
_LABOR_GATE_INSIDE_CAP = 13
_LABOR_GATE_OUTSIDE_CAP = 12

def _labor_gate_active(obs, farm):
    for condition in _LABOR_GATE_CONDITIONS:
        field = condition["field"]
        operator = condition["operator"]
        threshold = condition["value"]
        if field == "obs.day":
            value = int(obs.get("day", 0) or 0)
        elif field == "farm.money_pre_action":
            value = float(farm.get("money", 0) or 0)
        else:
            return False
        if operator == "<=" and not value <= threshold:
            return False
        if operator == ">=" and not value >= threshold:
            return False
    return True

def _target_hands(obs, config, farm, private, roles):
    requested = int(_LABOR_GATE_BASE_TARGET_HANDS(obs, config, farm, private, roles))
    cap = (
        _LABOR_GATE_INSIDE_CAP
        if _labor_gate_active(obs, farm)
        else _LABOR_GATE_OUTSIDE_CAP
    )
    return min(requested, cap)

