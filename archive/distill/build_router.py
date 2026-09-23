"""Build a SHOP-CONDITIONED ROUTER agent from one team's ladder replays.

Why this works: THIRD FARM CLUB's 90 extracted winning tapes all share an IDENTICAL
opening (steps 0-71), and the game always reveals its first town shop at exactly step 72.
So we can play their common opening blind, then at step 72 switch to the continuation
recorded in a game whose first shop matched -- with no incoherence, because every
continuation grew from the same opening.

That is the adaptivity a single frozen tape lacks (frozen tapes lose ~0/8 to current
adaptive agents). Routing rule mirrors the community's approach (kawa routes on the
yarn-unlock position); here we route on the identity of the first shop, which determines
which products get the 1.5-2.0x demand multiplier.

Data: public Kaggle episode replays (staff-confirmed allowed/encouraged). Tapes are
packed base85+zlib and baked in -- no runtime data access.
Usage: python build_router.py [team] [out_path]
"""
import sys, json, base64, zlib
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
KD = REPO / ".claude/scratch/kaggle_data"
TEAM = sys.argv[1] if len(sys.argv) > 1 else "THIRD FARM CLUB"
OUT = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "submission" / "main_router.py")
SPLIT = 72   # step at which the first shop becomes visible

T = json.load(open(KD / "top_tapes.json"))
S = json.load(open(KD / "shop_signatures.json"))


def sig(ep):
    return S.get(str(ep)) or S.get(ep)


ids = [i for i, t in enumerate(T) if t["team"] == TEAM]
# keep only the dominant opening so every continuation is compatible
openings = defaultdict(list)
for i in ids:
    openings[json.dumps(T[i]["tape"][:SPLIT])].append(i)
dom_open, dom_ids = max(openings.items(), key=lambda kv: len(kv[1]))
print(f"{TEAM}: {len(ids)} tapes, dominant opening covers {len(dom_ids)}")

# group by first shop; pick the tape with the best own-game margin in each group
groups = defaultdict(list)
for i in dom_ids:
    s = sig(T[i]["episode"])
    if s and s.get("first_shop"):
        groups[s["first_shop"]].append(i)

chosen = {}
for shop, gids in groups.items():
    best = max(gids, key=lambda i: T[i]["reward"] - T[i]["opp_reward"])
    chosen[shop] = best
    print(f"  {shop:16s} {len(gids):2d} tapes -> idx {best} "
          f"(margin {T[best]['reward']-T[best]['opp_reward']:+.0f})")

opening = json.loads(dom_open)
conts = {shop: T[i]["tape"][SPLIT:] for shop, i in chosen.items()}
default_shop = max(groups, key=lambda s: len(groups[s]))


def pack(obj):
    return base64.b85encode(zlib.compress(json.dumps(obj).encode(), 9)).decode()


payload = pack({"opening": opening, "conts": conts, "default": default_shop})

CODE = '''"""Kaggriculture pregame router agent.

Distilled from public Kaggle episode replays of team {team!r} (leaderboard rank {rank},
score {score}). All {n} source games share one identical opening, and the game reveals its
first town shop at exactly step {split}; so this agent plays that shared opening blind, then
switches to the continuation from a game whose first shop matched. Shop identity sets which
products get the 1.5-2.0x demand multiplier, so it is the decision that matters most.

Public replays are staff-confirmed allowed/encouraged for building a submission; the tapes
are packed inline (base85+zlib), so there is no runtime data access.
"""
import base64, json, zlib, copy

_P = json.loads(zlib.decompress(base64.b85decode(
    '{payload}'
)))
_OPENING = _P["opening"]
_CONTS = _P["conts"]
_DEFAULT = _P["default"]
_SPLIT = {split}
_state = {{"cont": None}}


def _shape(a, nh):
    hands = (list(a.get("hands") or []) + [["PASS"]] * nh)[:nh]
    return {{"farmer": a.get("farmer", ["PASS"]), "hands": hands,
            "market": a.get("market", [])}}


def agent(obs, configuration=None):
    step = int(obs["step"]) if "step" in obs else 0
    seat = obs["player"] if "player" in obs else 0
    nh = len(obs["farms"][seat].get("hands") or [])

    if step < _SPLIT:
        if step < len(_OPENING):
            return _shape(copy.deepcopy(_OPENING[step]), nh)
        return {{"farmer": ["PASS"], "hands": [["PASS"]] * nh, "market": []}}

    if _state["cont"] is None or step == _SPLIT:
        shops = ((obs.get("town") or {{}}).get("unlocked_shops") or []) if hasattr(obs, "get") \\
            else (getattr(obs, "town", {{}}) or {{}}).get("unlocked_shops") or []
        first = shops[0] if shops else None
        _state["cont"] = _CONTS.get(first) or _CONTS.get(_DEFAULT)

    cont = _state["cont"]
    j = step - _SPLIT
    if cont and 0 <= j < len(cont):
        return _shape(copy.deepcopy(cont[j]), nh)
    return {{"farmer": ["PASS"], "hands": [["PASS"]] * nh, "market": []}}
'''

code = CODE.format(payload=payload, split=SPLIT, team=TEAM,
                   rank=T[ids[0]]["rank"], score=T[ids[0]]["lb_score"], n=len(dom_ids))
Path(OUT).write_text(code)
print(f"\nwrote {OUT} ({len(code)} bytes) shops={len(conts)} default={default_shop}")
