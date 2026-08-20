#!/usr/bin/env python
"""Run a local Kaggriculture match between two agents and print the result.

Usage:
    python run_match.py <agent0> <agent1> [--episode-steps 720] [--render out.html]

Agent specs are either a built-in name ("random", "pass", "starter"), or
"<module>:<function>" to load an agent function from a .py file in this
directory, e.g.:

    python run_match.py multi_crop:multi_crop melon_maxxer:melon_maxxer
    python run_match.py quick_start_agent:agent random
    python run_match.py multi_crop:multi_crop random --render replay.html
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BUILTIN = {"random", "pass", "starter"}


def load_agent(spec):
    if spec in BUILTIN:
        return spec
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
    parser.add_argument("--render", metavar="FILE.html", help="write an HTML replay viewer to this path")
    args = parser.parse_args()

    from kaggle_environments import make

    env = make("kaggriculture", configuration={"episodeSteps": args.episode_steps}, debug=True)
    env.run([load_agent(args.agent0), load_agent(args.agent1)])

    final = env.steps[-1]
    print(f"{args.agent0} vs {args.agent1}:")
    for i, s in enumerate(final):
        print(f"  Player {i}: reward={s.reward}, status={s.status}")

    if args.render:
        html = env.render(mode="html")
        Path(args.render).write_text(html)
        print(f"Replay written to {args.render} -- open it in a browser to watch.")


if __name__ == "__main__":
    main()
