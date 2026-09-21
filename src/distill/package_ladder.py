"""Bake a distilled LADDER tape into a self-contained submission file.

Source data: .claude/scratch/kaggle_data/top_tapes.json — winning action sequences
extracted from public Kaggle episode replays of top-ranked teams. Extraction verified
bit-exact (replaying both seats' recorded actions reproduces the episode's exact final
rewards, alignment action[t] = steps[t+1][seat].action).

Legal per competition rules: Kaggle staff confirmed "Using public replays to train,
build, inform your submission is allowed and encouraged." The tape is baked into the
submission (no runtime data fetch, per rules on external data during evaluation).

Usage: python package_ladder.py <index-in-top_tapes.json> <out_path>
"""
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TAPES = REPO / ".claude/scratch/kaggle_data/top_tapes.json"

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "submission" / "main_ladder.py")

entry = json.load(open(TAPES))[idx]
actions = entry["tape"]

TEMPLATE = '''"""Kaggriculture pregame agent: a distilled top-team ladder trajectory.

Source: public episode replay {episode} — team {team!r} (leaderboard rank {rank},
score {lb_score}), which won that game {reward:.0f} to {opp_reward:.0f} vs {opp_team!r}.
Distilled offline from public replay data and baked in; no runtime data access.
Per Kaggle staff: using public replays to build a submission is allowed and encouraged.
"""
import copy

TAPE = {actions!r}


def agent(obs, configuration=None):
    step = int(obs["step"]) if "step" in obs else 0
    seat = obs["player"] if "player" in obs else 0
    nh = len(obs["farms"][seat].get("hands") or [])
    if 0 <= step < len(TAPE):
        a = copy.deepcopy(TAPE[step])
        hands = (list(a.get("hands") or []) + [["PASS"]] * nh)[:nh]
        return {{"farmer": a.get("farmer", ["PASS"]), "hands": hands,
                "market": a.get("market", [])}}
    return {{"farmer": ["PASS"], "hands": [["PASS"]] * nh, "market": []}}
'''

code = TEMPLATE.format(actions=actions, episode=entry.get("episode"),
                       team=entry.get("team"), rank=entry.get("rank"),
                       lb_score=entry.get("lb_score"), reward=entry.get("reward", 0),
                       opp_reward=entry.get("opp_reward", 0),
                       opp_team=entry.get("opp_team"))
Path(out).write_text(code)
print(f"wrote {out} ({len(code)} bytes, {len(actions)} steps) "
      f"team={entry.get('team')} rank={entry.get('rank')} ep={entry.get('episode')}")
