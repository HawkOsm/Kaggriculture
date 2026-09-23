"""Wrap a distilled tape with a THIN REPAIR layer (tape-based; the plan is unchanged).

Measured problem: replaying a tape against a different opponent makes ~12.8% of its work
actions no-ops -- the recording assumed a pasture/crop state that did not materialise here
(FEED 105, CARE 107, COLLECT_FERTILIZER 108, WATER 8 wasted per game out of 2569 work
actions). Those are dead worker-turns.

Repair rule: if the taped action is provably ineffective on the worker's ACTUAL tile,
substitute a useful action ON THE SAME TILE. Positions, movement and the market schedule
are never touched, so the tape's plan and its state trajectory are preserved -- we only
reclaim turns the tape was going to waste. This is the "thin repair wrapper" pattern used
by the top tape agents, not a reactive policy.

Usage: python tape_repair.py <tape_agent.py> <out.py>
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
src_p = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "submission/main_tape470.py"
out_p = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "submission/main_tape470_rep.py"

LAYER = '''

# ---------------------------------------------------------------------------
# THIN REPAIR LAYER (ours): reclaim taped work-actions that are no-ops here.
# Same tile, same position -- only a provably dead action is replaced.
# ---------------------------------------------------------------------------
_REP_ANIMAL = ("FEED", "CARE", "COLLECT_FERTILIZER")
_REP_STATS = {"repaired": 0, "to_water": 0, "to_harvest": 0, "errors": 0}
_REP_PARENT = agent


def _rep_tile(farm, pos):
    try:
        t = farm["tiles"][int(pos[1])][int(pos[0])]
        return t if isinstance(t, dict) else None
    except Exception:
        return None


def _rep_fix(act, tile):
    """Return a better action for this tile, or None to keep the taped one."""
    k = act[0] if act else "PASS"
    if tile is None:
        return None
    kind = tile.get("kind")
    ripe = int(tile.get("yield_units", 0) or 0) > 0
    dry = (kind == "PLANT") and not tile.get("watered_today", False)
    if k in _REP_ANIMAL and kind != "PASTURE":
        if ripe:
            return ["HARVEST"]
        if dry:
            return ["WATER"]
        return None
    if k == "WATER" and not dry:
        if ripe:
            return ["HARVEST"]
        return None
    if k == "HARVEST" and not ripe:
        if dry:
            return ["WATER"]
        return None
    return None


def agent(observation, configuration=None):
    action = _REP_PARENT(observation, configuration)
    try:
        seat = observation["player"] if "player" in observation else 0
        farm = observation["farms"][seat]
        positions = [farm.get("farmer")] + list(farm.get("hands") or [])
        units = [list(action.get("farmer") or ["PASS"])] + [list(h) for h in (action.get("hands") or [])]
        changed = False
        for i, pos in enumerate(positions):
            if i >= len(units) or not pos:
                continue
            fix = _rep_fix(units[i], _rep_tile(farm, pos))
            if fix is not None:
                _REP_STATS["repaired"] += 1
                _REP_STATS["to_water" if fix[0] == "WATER" else "to_harvest"] += 1
                units[i] = fix
                changed = True
        if changed:
            action = dict(action, farmer=units[0], hands=units[1:])
    except Exception:
        _REP_STATS["errors"] += 1
        return action
    return action


agent.repair_stats = _REP_STATS
# Re-insert `agent` LAST in module insertion order: the Kaggle loader picks the last
# callable defined, and our helpers above would otherwise win (this is why the base
# agents all end their layers with exactly this idiom).
agent = globals().pop("agent")
'''

src = src_p.read_text()
out_p.write_text(src + LAYER)
import ast
ast.parse(out_p.read_text())
print(f"wrote {out_p} ({out_p.stat().st_size} bytes); syntax OK")
