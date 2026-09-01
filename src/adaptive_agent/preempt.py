"""Clone-aware premium front-run wrapper.

Reimplemented (not copied) from the CONCEPT in prvsiyan_frontier's decoded
`_preempt_shift` and Roman Tamrazov's "Hamburger" notebook: in a near-mirror
match, the opponent (a clone) will dump the same premium product we hold, so
selling one turn ahead captures the higher price before the shared-market crash.

The original is tape-based -- it pulls a SCHEDULED future sale forward. Our agent
is reactive with no future schedule, so instead: when the opponent's public farm
is a near-clone of ours, sell a bounded batch of any PREMIUM product we are
sitting on but did NOT already queue to sell this turn. Elo is win/loss only and
the top ladder is dominated by mirror matches, so this targets exactly the games
that decide rank. Config-gated so Optuna can tune/toggle it.
"""
from .config import get_config as _cfg
from .constants import MARKET

PREMIUM = ("STRAWBERRY", "MELON", "MILK", "WOOL")

_SIG_KEYS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "COW", "SHEEP", "GOOSE", "PASTURE", "COOP", "WEED",
)


def _public_signature(farm):
    counts = {k: 0 for k in _SIG_KEYS}
    for row in (farm.get("tiles", []) or []):
        for tile in (row if isinstance(row, list) else [row]):
            if not isinstance(tile, dict):
                continue
            for field in ("crop", "animal", "kind"):
                v = str(tile.get(field, "")).upper()
                if v in counts:
                    counts[v] += 1
                    break
    return (
        len(farm.get("hands", []) or []),
        len(farm.get("unlocked_quadrants", []) or []),
        tuple(counts[k] for k in sorted(counts)),
    )


def _clone_distance(obs):
    farms = list(obs.get("farms", []) or [])
    if len(farms) < 2:
        return 10 ** 9
    a, b = _public_signature(farms[0]), _public_signature(farms[1])
    return (
        abs(a[0] - b[0])
        + 3 * abs(a[1] - b[1])
        + sum(abs(x - y) for x, y in zip(a[2], b[2]))
    )


def clone_front_run(obs, config, farm, private, market):
    """Return `market` possibly augmented with front-run SELLs. Pure/no side effects."""
    if not _cfg(config, "FRONT_RUN_ENABLED", True):
        return market
    step = int(obs.get("step", 0) or 0)
    if not (_cfg(config, "FRONT_RUN_START", 120) <= step < _cfg(config, "FRONT_RUN_STOP", 680)):
        return market
    if _clone_distance(obs) > _cfg(config, "FRONT_RUN_MAX_CLONE_DIST", 6):
        return market

    orders = [list(o) for o in (market or [])]
    cap = int(_cfg(config, "maxMarketOrdersPerTurn", 10))
    if len(orders) >= cap:
        return market

    shed = (private.get("shed", {}) or {})
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    already = {o[1] for o in orders if len(o) >= 2 and o[0] == "SELL"}
    batch = int(_cfg(config, "FRONT_RUN_MAX_BATCH", 12))
    min_qty = int(_cfg(config, "FRONT_RUN_MIN_QTY", 4))
    price_ratio = float(_cfg(config, "FRONT_RUN_MIN_PRICE_RATIO", 0.5))

    for item in PREMIUM:
        if len(orders) >= cap:
            break
        if item in already:
            continue
        held = int(shed.get(item, 0) or 0)
        if held < min_qty:
            continue
        base = float(MARKET[item][0])
        price = float(prices.get(item, base) or base)
        if price < base * price_ratio:
            continue
        qty = min(held, batch)
        orders.append(["SELL", item, qty])

    return orders
