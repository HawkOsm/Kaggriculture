"""Build an improved agent ON TOP OF a strong public base, as a final wrapper layer.

The base (hanifnoerrofiq / "A Wonderful Life", public Code-tab notebook, OSI-licensed per
Foundational Rules 3.6 -- credited in CREDITS.md) is itself a chain of ~60 wrapper layers,
each of the form:

    _X_PARENT = agent
    def agent(obs, config=None): action = _X_PARENT(obs, config); ...adjust...; return action
    agent = globals().pop('agent')

We append ONE more layer in exactly that convention, so the base is untouched and our change
is isolated and reversible. The layer applies mechanisms observed in three other top agents:

  1. Roxy (yamakawanin, LB #61) -- premium front-running: its MUNIB expert runs with
     _FR_ITEMS = (WOOL, MILK, MELON, STRAWBERRY), selling premium goods ahead of an
     anticipated shared-market dump.
  2. kawa/boatlee route agent -- the shared-market dump signal itself: act when opponent
     visible supply exceeds town demand for the item.
  3. rayk (C95 "impact") -- price-impact discipline: never dump into an already-crashed
     price; gate on price relative to the item's base.

It also encodes our own measurement: perishables sold after the mid-game crash lose most of
their value (measured MILK ~157/unit early vs ~30 late, MELON 237 -> 66), which is the
failure mode all three mechanisms above are guarding against.

Guards: bounded quantity, respects maxMarketOrdersPerTurn, never sells below a price floor,
never touches field actions, and fails closed (any exception returns the base action).

Usage: python improve_hanif.py <base.py> <out.py>
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
base_p = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / ".claude/scratch/decoded_agents/hanif_current.py"
out_p = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "submission/main_hybrid.py"

LAYER = '''

# ============================================================================
# IMPROVEMENT LAYER (ours) -- premium front-run under a shared-market dump.
# Mechanisms borrowed from three other top agents (see src/distill/improve_hanif.py
# and CREDITS.md): Roxy's _FR_ITEMS premium front-running, kawa's opponent-supply
# dump signal, rayk's price-impact floor. Fails closed; field actions untouched.
# ============================================================================
_IMP_PREMIUM_BASE = {"WOOL": 200.0, "MELON": 250.0, "MILK": 160.0, "STRAWBERRY": 120.0}
_IMP_PRICE_FLOOR = 0.55      # never sell below this fraction of the item's base price
_IMP_MAX_EXTRA = 6           # bounded quantity per item per turn
_IMP_MAX_ORDERS = 10         # engine cap: maxMarketOrdersPerTurn
_IMP_MIN_STEP = 96           # only once production is running
_IMP_TELEMETRY = {"fired": 0, "units": 0, "errors": 0}
_IMP_PARENT = agent


def _imp_visible_opponent_supply(observation, seat, item):
    """Opponent stock of `item` visible on their farm tiles -- the incoming dump."""
    try:
        opp = observation["farms"][1 - seat]
    except Exception:
        return 0
    n = 0
    for row in (opp.get("tiles") or []):
        for tile in (row or []):
            if not isinstance(tile, dict):
                continue
            if item in ("MILK", "WOOL"):
                animal = tile.get("animal")
                if (item == "MILK" and animal == "COW") or (item == "WOOL" and animal == "SHEEP"):
                    n += 1
            elif tile.get("crop") == item:
                n += int(tile.get("yield_units", 0) or 0)
    return n


def agent(observation, configuration=None):
    action = _IMP_PARENT(observation, configuration)
    try:
        step = int(observation["step"])
        if step < _IMP_MIN_STEP:
            return action
        seat = int(observation["player"])
        market = observation.get("market") or {}
        prices = market.get("prices") or {}
        shed = (observation.get("private") or {}).get("shed") or {}
        orders = list(action.get("market") or [])
        if len(orders) >= _IMP_MAX_ORDERS:
            return action
        already = {o[1] for o in orders if len(o) >= 2 and o[0] == "SELL"}
        changed = False
        for item, base in _IMP_PREMIUM_BASE.items():
            if len(orders) >= _IMP_MAX_ORDERS:
                break
            if item in already:
                continue                                  # parent already selling it
            stock = int(shed.get(item, 0) or 0)
            if stock <= 0:
                continue
            price = float(prices.get(item, 0) or 0)
            if price < _IMP_PRICE_FLOOR * base:
                continue                                  # already crashed: don't dump
            if _imp_visible_opponent_supply(observation, seat, item) <= 0:
                continue                                  # no incoming dump: no urgency
            qty = min(stock, _IMP_MAX_EXTRA)
            orders.append(["SELL", item, qty])
            _IMP_TELEMETRY["fired"] += 1
            _IMP_TELEMETRY["units"] += qty
            changed = True
        if changed:
            action = dict(action, market=orders)
    except Exception:
        _IMP_TELEMETRY["errors"] += 1
        return action
    return action


agent.telemetry = _IMP_TELEMETRY
agent = globals().pop("agent")
'''

src = base_p.read_text()
if "globals().pop" not in src:
    raise SystemExit("base does not use the expected layering convention")
out_p.write_text(src + LAYER)
import ast
ast.parse(out_p.read_text())
print(f"wrote {out_p} ({out_p.stat().st_size} bytes) = base + 1 improvement layer; syntax OK")
