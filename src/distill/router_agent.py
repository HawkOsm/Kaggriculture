"""Stage 3: the distillation ROUTER agent. Loads the tape library, and at each step
routes the live game-state profile to the nearest recorded state, then executes that
tape's action -- with a repair layer for legality. Two modes (env-selectable):

  DISTILL_MODE=full   : execute the routed tape's farmer+hands+market (position-risky)
  DISTILL_MODE=market : execute only the routed tape's MARKET action; field comes from
                        the champion dispatcher (position-independent; targets the
                        measured 36k market-suppression gap)

Routing: pick the tape whose recorded profile at the SAME step index is closest
(euclidean) to the live profile, re-selected every ROUTE_LOCK steps (default 8) to
allow adaptation while keeping coherent blocks (the top agents' 8-turn lock).
"""
import os, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "opponents"))
from build_library import profile

_LIB = None
_CHAMP = None
_state = {"tape_id": None, "lock_until": -1}
MODE = os.environ.get("DISTILL_MODE", "market")
ROUTE_LOCK = int(os.environ.get("ROUTE_LOCK", "8"))
LIB_PATH = os.environ.get("LIB_PATH", str(REPO / ".claude/scratch/distill/library.json"))


def _load():
    global _LIB, _CHAMP
    if _LIB is not None:
        return
    lib = json.load(open(LIB_PATH))
    # precompute per-tape profile matrices
    _LIB = []
    for entry in lib:
        profs = np.array([s[0] for s in entry["tape"]], dtype=np.float32)
        acts = [s[1] for s in entry["tape"]]
        _LIB.append({"profs": profs, "acts": acts, "source": entry["source"], "n": len(acts)})
    if MODE == "market":
        import importlib.util
        spec = importlib.util.spec_from_file_location("champmod", REPO / "submission" / "main.py")
        cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
        _CHAMP = cm.agent


def _route(prof, step):
    """Return (tape_id) of nearest recorded profile at this step index."""
    best_id, best_d = None, 1e18
    for i, t in enumerate(_LIB):
        if step >= t["n"]:
            continue
        d = float(np.sum((t["profs"][step] - prof) ** 2))
        if d < best_d:
            best_d, best_id = d, i
    return best_id


def agent(obs, config=None):
    _load()
    seat = obs.get("player", 0)
    step = int(obs.get("step", 0))
    prof = profile(obs, seat)

    if step >= _state["lock_until"] or _state["tape_id"] is None:
        tid = _route(prof, step)
        if tid is not None:
            _state["tape_id"] = tid
            _state["lock_until"] = step + ROUTE_LOCK
    tid = _state["tape_id"]

    tape_act = None
    if tid is not None and step < _LIB[tid]["n"]:
        tape_act = _LIB[tid]["acts"][step]

    if MODE == "market":
        base = _CHAMP(obs)
        if tape_act is not None and "market" in tape_act:
            base = dict(base); base["market"] = tape_act["market"]
        return base

    # full mode: execute the tape's action wholesale (repair = legality fallback)
    if tape_act is None:
        f = obs["farms"][seat]
        return {"farmer": ["PASS"], "hands": [["PASS"] for _ in (f.get("hands") or [])], "market": []}
    f = obs["farms"][seat]
    nh = len(f.get("hands") or [])
    hands = list(tape_act.get("hands") or [])
    hands = (hands + [["PASS"]] * nh)[:nh]
    return {"farmer": tape_act.get("farmer", ["PASS"]), "hands": hands, "market": tape_act.get("market", [])}
