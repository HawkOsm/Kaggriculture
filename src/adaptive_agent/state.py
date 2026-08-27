from .constants import CROPS, ANIMALS, MARKET, MELON_TILES_MAX, MELON_TILES_MIN, MELON_TILES_BASE, ANIMAL_SLOTS, CORE_HERD_SEQUENCE, CROP_MIX
from .config import get_config, LIQUIDATION_TURNS
# farm_plan depends on state, state depends on farm_plan: break cycle by importing here inside the function or just move _herd_targets to state?
# Wait, _role_plan calls _herd_targets. We should import _herd_targets.
def _herd_targets(*args, **kwargs):
    from .farm_plan import _herd_targets
    return _herd_targets(*args, **kwargs)


def _distance(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

def _shed_tiles(board_size, tiles=None):
    half = board_size // 2
    candidates = (
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    )
    if tiles is None:
        return candidates
    accessible = tuple(
        position
        for position in candidates
        if tiles[position[1]][position[0]] != "LOCKED"
    )
    return accessible or candidates[:1]

def _melon_target(obs):
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    price = int(prices.get("MELON", MARKET["MELON"][0]) or MARKET["MELON"][0])
    opponent_tiles = 0
    player = int(obs.get("player", 0) or 0)
    for index, other in enumerate(obs.get("farms", []) or []):
        if index == player:
            continue
        for row in other.get("tiles", []) or []:
            opponent_tiles += sum(
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == "MELON"
                for tile in row
            )
    if price >= 300 and opponent_tiles <= 5:
        return MELON_TILES_MAX
    if price <= 170 or opponent_tiles >= 12:
        return MELON_TILES_MIN
    if opponent_tiles >= 9:
        return MELON_TILES_BASE - 1
    return MELON_TILES_BASE

def _private_item_total(private, item):
    total = int((private.get("shed", {}) or {}).get(item, 0) or 0)
    for inventory in private.get("inventories", []) or []:
        total += int((inventory or {}).get(item, 0) or 0)
    return total

def _quadrant_of(position, board_size):
    x, y = position
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")

def _reserved_animal_slots(farm):
    tiles = farm.get("tiles", []) or []
    board_size = len(tiles)
    unlocked = set(farm.get("unlocked_quadrants", []) or ["NW"])
    sheds = _shed_tiles(board_size, tiles)
    slots = []
    by_quadrant = {}
    for quadrant in ("NW", "NE", "SW", "SE"):
        if quadrant not in unlocked:
            continue
        cells = []
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if tile == "LOCKED" or _quadrant_of((x, y), board_size) != quadrant:
                    continue
                distance = min(_distance((x, y), shed) for shed in sheds)
                cells.append((distance, y, x))
        cells.sort()
        count = min(ANIMAL_SLOTS[quadrant], len(cells))
        selected = [(x, y) for _, y, x in cells[:count]]
        by_quadrant[quadrant] = {
            "reserved": selected,
            "crops": [(x, y) for _, y, x in cells[count:]],
        }
        slots.extend(selected)
    return slots, by_quadrant

def _role_plan(obs, config, farm):
    private = obs.get("private", {}) or {}
    tiles = farm.get("tiles", []) or []
    animal_slots, zones = _reserved_animal_slots(farm)
    targets = _herd_targets(obs, config, farm, private, len(animal_slots))
    desired_animals = min(len(animal_slots), sum(targets.values()))
    active_slots = list(animal_slots[:desired_animals])
    for position in animal_slots:
        x, y = position
        tile = tiles[y][x]
        if isinstance(tile, dict) and "animal" in tile and position not in active_slots:
            active_slots.append(position)

    assigned = {"COW": 0, "SHEEP": 0}
    roles = {}
    core_sequence = CORE_HERD_SEQUENCE
    for index, position in enumerate(active_slots):
        x, y = position
        tile = tiles[y][x]
        actual = tile.get("animal") if isinstance(tile, dict) else None
        if actual in assigned:
            animal = actual
        elif index < len(core_sequence) and assigned[core_sequence[index]] < targets[
            core_sequence[index]
        ]:
            animal = core_sequence[index]
        else:
            animal = max(
                ("COW", "SHEEP"),
                key=lambda name: (
                    targets[name] - assigned[name],
                    _private_item_total(private, name),
                    name == "COW",
                ),
            )
        roles[position] = ("ANIMAL", animal)
        assigned[animal] += 1

    melon_target = _melon_target(obs)
    for quadrant in ("NW", "NE", "SW", "SE"):
        zone = zones.get(quadrant)
        if not zone:
            continue
        cells = zone["crops"]
        fixed = dict(CROP_MIX[quadrant])
        if quadrant == "NW":
            fixed["MELON"] = min(melon_target, len(cells))
        strawberry_count = max(0, len(cells) - sum(fixed.values()))
        sequence = []
        if quadrant == "NW":
            sequence.extend(["MELON"] * fixed.get("MELON", 0))
        sequence.extend(["STRAWBERRY"] * strawberry_count)
        sequence.extend(["WHEAT"] * fixed.get("WHEAT", 0))
        sequence.extend(["CARROT"] * fixed.get("CARROT", 0))
        if quadrant != "NW":
            sequence.extend(["MELON"] * fixed.get("MELON", 0))
        for position, crop in zip(cells, sequence):
            roles[position] = ("CROP", crop)
    return roles

def _inventory_total(inventory, excluded=()):
    return sum(
        int(value)
        for item, value in (inventory or {}).items()
        if item not in excluded and isinstance(value, (int, float)) and value > 0
    )

def _survey(farm, private, roles=None, day=0):
    summary = {
        "animals": 0,
        "unfed": 0,
        "at_risk_animals": 0,
        "at_risk_crops": 0,
        "open_structures": 0,
        "structures_todo": 0,
        "plants": 0,
        "plantable": 0,
        "weeds": 0,
    }
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if "animal" in tile:
                summary["animals"] += 1
                if not tile.get("fed_today", False):
                    summary["unfed"] += 1
                if int(tile.get("consecutive_unfed", 0) or 0) >= 1:
                    summary["at_risk_animals"] += 1
            elif tile.get("kind") in {"COOP", "PASTURE"}:
                summary["open_structures"] += 1
            elif tile.get("kind") == "PLANT":
                summary["plants"] += 1
                if int(tile.get("consecutive_unwatered", 0) or 0) >= 1:
                    summary["at_risk_crops"] += 1
            elif tile.get("kind") == "WEED":
                summary["weeds"] += 1

    if roles:
        for (x, y), (kind, item) in roles.items():
            tile = farm["tiles"][y][x]
            if kind == "ANIMAL" and tile is None:
                summary["structures_todo"] += 1
            elif (
                kind == "CROP"
                and tile is None
                and day <= CROPS[item]["last_plant"]
            ):
                summary["plantable"] += 1

    summary["wheat_stock"] = int((private.get("shed", {}) or {}).get("WHEAT", 0))
    summary["animal_stock"] = {
        animal: int((private.get("shed", {}) or {}).get(animal, 0) or 0)
        for animal in ANIMALS
    }
    summary["shed_load"] = sum(
        max(0, int(value or 0))
        for value in (private.get("shed", {}) or {}).values()
    )
    summary["carried_load"] = 0
    for inventory in private.get("inventories", []) or []:
        summary["wheat_stock"] += int((inventory or {}).get("WHEAT", 0))
        for animal in ANIMALS:
            summary["animal_stock"][animal] += int(
                (inventory or {}).get(animal, 0) or 0
            )
        summary["carried_load"] += _inventory_total(inventory)
    return summary

def _policy_phase(obs, config, farm, private, summary):
    day = int(obs.get("day", 0) or 0)
    step = int(obs.get("step", day * 24 + int(obs.get("hour", 0) or 0)) or 0)
    actions_left = max(0, 719 - step)
    # Must read the live (searched/tuned) value here, not the bare default --
    # dispatch.py's own liquidation check already does (`_cfg(config,
    # "LIQUIDATION_TURNS", ...)`); this function silently used the hardcoded
    # default instead, causing a real desync between the two phase checks
    # whenever a tuned config sets LIQUIDATION_TURNS != the default (22).
    if actions_left <= get_config(config, "LIQUIDATION_TURNS", LIQUIDATION_TURNS):
        return "LIQUIDATE"
    workers = 1 + len(farm.get("hands", []) or [])
    if (
        summary["at_risk_animals"] + summary["at_risk_crops"] > workers
        or summary["shed_load"] + summary["carried_load"] >= 95
    ):
        return "CRISIS"
    if day <= 4:
        return "BOOTSTRAP"
    if day <= 21:
        return "COMPOUND"
    return "REALIZE"
