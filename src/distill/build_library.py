"""Stage 2 of the distillation router: generate a TAPE LIBRARY by recording the
strong local agents (prvsiyan/boatlee/kaito/cropdusta -- top public Code-tab agents,
legal to run) across many seeds/opponents, keeping every game the source WON. Each
kept game -> a tape: a list of per-step (state_profile, action). The router (next
stage) k-NN matches the live state_profile against this library.

State profile = a compact fixed-length GAME-STATE vector (not per-unit): day/hour,
our money/hands/quadrants, per-crop tile counts (empty/growing/ripe), animal counts,
shed inventory per product, market prices per item, and the same for the opponent's
visible farm. This is the ~visible profile the top agents match on.
"""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "opponents"))
from kaggle_environments import make
from run_match import load_agent

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "MILK", "EGG", "WOOL", "FERTILIZER"]
ANIMALS = ["COW", "SHEEP", "GOOSE"]


def profile(obs, seat):
    """Fixed-length game-state feature vector for k-NN routing."""
    f = obs["farms"][seat]; opp = obs["farms"][1 - seat]
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    v = [obs.get("day", 0) / 30.0, obs.get("hour", 0) / 24.0]

    def farm_feats(fm):
        out = [float(fm.get("money", 0) or 0) / 50000.0,
               len(fm.get("hands", []) or []) / 13.0,
               len(fm.get("unlocked_quadrants", []) or []) / 4.0]
        empty = growing = ripe = 0
        acount = {a: 0 for a in ANIMALS}
        for row in (fm.get("tiles") or []):
            for tl in row:
                if tl is None:
                    empty += 1
                elif isinstance(tl, dict):
                    k = tl.get("kind")
                    if k in (None, "empty"): empty += 1
                    elif k == "PLANT":
                        if int(tl.get("yield_units", 0) or 0) > 0: ripe += 1
                        else: growing += 1
                    a = tl.get("animal")
                    if a in acount: acount[a] += 1
        out += [empty / 50.0, growing / 50.0, ripe / 50.0]
        out += [acount[a] / 12.0 for a in ANIMALS]
        return out

    v += farm_feats(f) + farm_feats(opp)
    v += [float(prices.get(p, 0) or 0) / 200.0 for p in PRODUCTS]
    return np.array(v, dtype=np.float32)


def _load_callable(path):
    """load_agent returns a PATH STRING (env.run resolves it), not a callable.
    To SPY on the source we need its real agent function -- import the module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("src_" + Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "agent")


def record_winner_tape(source_path, opp_path, seed):
    """Run source vs opp at seed; if source (seat 0) wins, return its tape
    [(profile, action), ...]; else None."""
    src = _load_callable(source_path); opp = load_agent(opp_path)
    tape = []
    def spy(obs, config=None):
        a = src(obs)
        tape.append((profile(obs, 0).tolist(), a))
        return a
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run([spy, opp])
    r0, r1 = env.steps[-1][0].reward, env.steps[-1][1].reward
    return (tape, r0, r1) if r0 > r1 else (None, r0, r1)


SOURCES = {
    "prvsiyan": str(REPO / "src/opponents/prvsiyan_frontier_agent.py"),
    "boatlee": str(REPO / "src/opponents/boatlee_v16_agent.py"),
    "kaito": str(REPO / "src/opponents/kaito_agent.py"),
    "cropdusta": str(REPO / "src/opponents/cropdusta_97601003_agent.py"),
    "rayk": str(REPO / "src/opponents/rayk_c95_agent.py"),
    "stevenleehans": str(REPO / "src/opponents/stevenleehans_agent.py"),
}
# include the TOP-2 as opponents: any seed where a source beats them harvests a
# prvsiyan/boatlee-beating tape (the only route to cracking the 0/24 wall).
OPPS = [str(REPO / "src/opponents/pilkwang_agent.py"),
        str(REPO / "src/opponents/cropdusta_97601003_agent.py"),
        str(REPO / "src/opponents/prvsiyan_frontier_agent.py"),
        str(REPO / "src/opponents/boatlee_v16_agent.py"),
        str(REPO / "src/opponents/kaito_agent.py"),
        str(REPO / "submission/main.py")]
SEEDS = 6


def _run_job(job):
    sname, spath, opp, seed = job
    tape, r0, r1 = record_winner_tape(spath, opp, seed)
    return (sname, opp, seed, tape, r0, r1)


if __name__ == "__main__":
    import multiprocessing as mp, os
    # HELDOUT: exclude this opponent (by substring) from BOTH sources and opponents,
    # to measure true generalization to an unseen opponent.
    heldout = os.environ.get("HELDOUT", "")
    out = REPO / os.environ.get("OUT", ".claude/scratch/distill/library.json")
    sources = {k: v for k, v in SOURCES.items() if not (heldout and heldout in v)}
    opps = [o for o in OPPS if not (heldout and heldout in o)]
    jobs = []
    for sname, spath in sources.items():
        for opp in opps:
            for seed in range(SEEDS):
                jobs.append((sname, spath, opp, seed))
    with mp.Pool(12) as pool:
        res = pool.map(_run_job, jobs)
    lib = []
    won = {s: 0 for s in sources}
    for sname, opp, seed, tape, r0, r1 in res:
        if tape is not None:
            lib.append({"source": sname, "opp": Path(opp).stem, "seed": seed, "tape": tape})
            won[sname] += 1
    json.dump(lib, open(out, "w"))
    print(f"HELDOUT={heldout!r}  wins per source:", won)
    print(f"library: {len(lib)} tapes, {sum(len(t['tape']) for t in lib)} states -> {out}")
