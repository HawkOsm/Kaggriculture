"""The deliverable core: replay a single distilled tape whole-game.

A tape is a list of per-step action dicts {"farmer","hands","market"}. We replay it
by step index; the engine no-ops any position-mismatched field action, while the
position-independent economic schedule (hire/buy/sell, herd, crop plan) carries it.
Verified ~0.78 vs the local top-agent pool (champion ~0.12). See README.md.
"""
import copy


def make_tape_agent(tape_actions):
    """tape_actions: list of per-step action dicts. Returns an agent(obs)."""
    def agent(obs, configuration=None):
        step = int(obs["step"]) if "step" in obs else 0
        seat = obs["player"] if "player" in obs else 0
        nh = len(obs["farms"][seat].get("hands") or [])
        if 0 <= step < len(tape_actions):
            a = copy.deepcopy(tape_actions[step])
            hands = (list(a.get("hands") or []) + [["PASS"]] * nh)[:nh]
            return {"farmer": a.get("farmer", ["PASS"]), "hands": hands,
                    "market": a.get("market", [])}
        return {"farmer": ["PASS"], "hands": [["PASS"]] * nh, "market": []}
    return agent


def load_tape(library_path, tape_id):
    """Load tape_id's action sequence (drops the routing profiles) from a library."""
    import json
    entry = json.load(open(library_path))[tape_id]
    return [step[1] for step in entry["tape"]], entry
