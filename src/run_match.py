#!/usr/bin/env python
"""Run a local Kaggriculture match between two agents and print the result.

Usage:
    python run_match.py <agent0> <agent1> [--episode-steps 720] [--no-save]

Agent specs are either a built-in name ("random", "pass", "starter"), a path
containing "/" to a self-contained .py file (loaded the same way Kaggle loads
a real submission -- exec'd by file path, last callable in the module
namespace wins, no import machinery), or "<module>:<function>" to import an
agent function from a .py file in this directory, e.g.:

    python run_match.py multi_crop:multi_crop melon_maxxer:melon_maxxer
    python run_match.py quick_start_agent:agent random
    python run_match.py submission/main.py melon_maxxer:melon_maxxer

Benchmark opponents (melon_maxxer, multi_crop, quick_start_agent, kawa_route_agent)
live in opponents/, kept separate from this repo's own agent code (agent.py
etc.) directly in src/.

Every run saves a result.json + replay.html + a best_config.json snapshot +
actions.csv into output/games/<timestamp>_<slug>/ by default (pass --no-save
to skip) -- open replay.html in a browser to watch the match, or actions.csv
for a continuous per-turn, per-unit log of every position and action both
players took (one row per unit per step; farmer + hands; both players, not
just ours) -- this is what every ad hoc instrumentation script this project
built during development (tier/movement audits, production audits, distance
probes) reconstructed by hand each time; now it's just there for any run.
This is the working-folder artifact backing a LOG.md entry: the log has the
scannable summary, this directory has the actual thing that was run.
output/games/ is for one-off test/verification games only -- RL training
checkpoints and logs live in src/rl_checkpoints/, untouched by this script.
"""
import argparse
import csv
import importlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "opponents"))

REPO_ROOT = HERE.parent
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "games"

BUILTIN = {"random", "pass", "starter"}


def load_agent(spec):
    if spec in BUILTIN:
        return spec
    if "/" in spec:
        # A real file path (e.g. submission/main.py) -- pass it straight
        # through so kaggle_environments loads it exactly like a real
        # submission does (exec by path, last callable wins), not via a
        # Python import. Drop any trailing ":name" -- file loading doesn't
        # take a function-name selector.
        return spec.split(":", 1)[0]
    if spec.endswith(".py"):
        spec = spec[:-3]
    module_name, _, func_name = spec.partition(":")
    func_name = func_name or module_name
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _format_action(action):
    if not action:
        return ""
    return " ".join(str(x) for x in action)


def _write_action_log(env, agent_names, out_path):
    """One row per unit per step per player: position + action taken.
    Reconstructed entirely from env.steps (each step's .action is what that
    player actually submitted, .observation.farms[player] has positions) --
    no agent instrumentation needed, works for any opponent including
    black-box ones we don't control the source of."""
    rows = []
    for step_idx, step_states in enumerate(env.steps):
        for player, state in enumerate(step_states):
            obs = state.observation
            action = state.action or {}
            day = obs.get("day", "") if hasattr(obs, "get") else ""
            money = ""
            farms = obs.get("farms", []) if hasattr(obs, "get") else []
            if 0 <= player < len(farms):
                money = farms[player].get("money", "")
                farmer_pos = tuple(farms[player].get("farmer", []) or [])
                hand_positions = [tuple(p) for p in (farms[player].get("hands", []) or [])]
            else:
                farmer_pos = ()
                hand_positions = []
            rows.append({
                "step": step_idx, "day": day, "player": player,
                "agent": agent_names[player], "money": money,
                "unit": "farmer", "pos": farmer_pos,
                "action": _format_action(action.get("farmer")),
            })
            hand_actions = action.get("hands") or []
            for i, pos in enumerate(hand_positions):
                hand_action = hand_actions[i] if i < len(hand_actions) else None
                rows.append({
                    "step": step_idx, "day": day, "player": player,
                    "agent": agent_names[player], "money": money,
                    "unit": f"hand{i}", "pos": pos,
                    "action": _format_action(hand_action),
                })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "day", "player", "agent", "money", "unit", "pos", "action"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("agent0")
    parser.add_argument("agent1")
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                         help="directory to save a timestamped run folder into (default: output/games)")
    parser.add_argument("--no-save", action="store_true", help="skip saving result.json/replay.html, just print")
    args = parser.parse_args()

    from kaggle_environments import make

    env = make("kaggriculture", configuration={"episodeSteps": args.episode_steps}, debug=True)
    env.run([load_agent(args.agent0), load_agent(args.agent1)])

    final = env.steps[-1]
    print(f"{args.agent0} vs {args.agent1}:")
    results = []
    for i, s in enumerate(final):
        print(f"  Player {i}: reward={s.reward}, status={s.status}")
        results.append({"player": i, "reward": s.reward, "status": s.status})

    if not args.no_save:
        slug = f"{args.agent0}_vs_{args.agent1}".replace(":", "-").replace("/", "-")
        run_dir = Path(args.out_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps({
            "agent0": args.agent0, "agent1": args.agent1,
            "episode_steps": args.episode_steps, "results": results,
        }, indent=2))
        (run_dir / "replay.html").write_text(env.render(mode="html"))
        _write_action_log(env, [args.agent0, args.agent1], run_dir / "actions.csv")
        best_config_path = HERE / "best_config.json"
        if best_config_path.exists():
            shutil.copy(best_config_path, run_dir / "best_config_snapshot.json")
        print(f"Saved -> {run_dir}")


if __name__ == "__main__":
    main()
