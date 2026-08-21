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
live in opponents/, kept separate from this repo's own agent code (robust_agent.py
etc.) directly in src/.

Every run saves a result.json + replay.html + a best_config.json snapshot into
output/games/<timestamp>_<slug>/ by default (pass --no-save to skip) -- open
replay.html in a browser to watch the match. This is the working-folder artifact
backing a LOG.md entry: the log has the scannable summary, this directory has the
actual thing that was run. output/games/ is for one-off test/verification games
only -- RL training checkpoints and logs live in src/rl_checkpoints/, untouched
by this script.
"""
import argparse
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
        best_config_path = HERE / "best_config.json"
        if best_config_path.exists():
            shutil.copy(best_config_path, run_dir / "best_config_snapshot.json")
        print(f"Saved -> {run_dir}")


if __name__ == "__main__":
    main()
