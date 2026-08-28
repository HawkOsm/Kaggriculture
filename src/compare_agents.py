#!/usr/bin/env python
"""Compare two agent-source variants (e.g. current submission/main.py vs a
`.claude/scratch/main_before_*.py` snapshot) across multiple opponents with
alternating seats, and report both margin AND each side's own absolute
money separately.

Why this exists (see docs/tests/LOG.md, 2026-08-28 entries): testing a
candidate change against a single opponent's margin repeatedly gave
misleading signal this session. Two confounds, found by direct trace
inspection, not by assumption:

1. Fixed-tape opponents (prvsiyan_frontier_agent, kawa_route_agent) sell a
   fixed volume regardless of price. If a candidate change reduces OUR OWN
   wasteful overproduction, the shared market stops crashing -- and the
   fixed-tape opponent's un-adapting sales suddenly land at a much better
   price than before, since it can't and won't cut back to make room. That
   inflates the OPPONENT's score more than ours even when our own absolute
   money genuinely improved, making the margin look like a regression for a
   change that was actually a real improvement. Confirmed directly at least
   three separate times this session by tracing market prices before/after
   (see .claude/scratch/main_before_labor_floor.py's investigation,
   .claude/scratch/main_before_liquidation_window.py's, and the strawberry-
   cap experiment) -- not a one-off, a structural property of testing
   against a price-blind fixed seller.
2. All prior testing this session ran our agent as player 0 every time,
   never checking whether board-seat position itself biases the result --
   src/tuning/verification.py already avoids this (alternating seats via
   `i % 2`) for config-vs-config comparisons; this script carries that
   practice over to comparing two different agent *source files*.

Mitigation here is NOT a fix for #1 (there isn't a clean one against a
price-blind opponent) but exposure: this script always includes at least
one REACTIVE opponent (pilkwang_agent.py -- a live, price-aware agent, not
a recorded tape) alongside the fixed-tape ones. If a candidate's margin
improves against the reactive opponent but regresses against the fixed-tape
ones, that is itself the diagnostic -- it means the fixed-tape numbers are
confounded and the reactive-opponent number is the one to trust. If a
candidate regresses against BOTH kinds, that is a real regression, not a
measurement artifact.

Usage:
    python src/compare_agents.py <baseline_path> <candidate_path> \\
        [--opponents prvsiyan_frontier,kawa,pilkwang] [--seeds 1,2,3,4,5]

Agent paths are loaded exactly like run_match.py: a "/"-containing path is
passed straight to kaggle_environments (real submission-style loading), a
bare "module:function" is imported.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "opponents"))

DEFAULT_OPPONENTS = {
    # fixed-tape (price-blind, see module docstring)
    "prvsiyan_frontier": "src/opponents/prvsiyan_frontier_agent.py",
    "kawa": "src/opponents/kawa_route_agent.py",
    # reactive (live, price-aware) -- the confound-detector
    "pilkwang": "src/opponents/pilkwang_agent.py",
}
DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def _load_agent(spec):
    if "/" in spec:
        return spec.split(":", 1)[0]
    module_name, _, func_name = spec.partition(":")
    func_name = func_name or module_name
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _run_pair(agent_path_a, agent_path_b, seed):
    from kaggle_environments import make as make_env

    env = make_env(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": seed},
        debug=False,
    )
    env.run([str(agent_path_a), str(agent_path_b)])
    final = env.steps[-1]
    return final[0].reward, final[1].reward


def compare(baseline_path, candidate_path, opponents, seeds):
    """For each opponent, run `seeds` episodes with ALTERNATING seats for
    both baseline and candidate. Returns a dict:
    {opponent_name: {"baseline": {"own_money": [...], "margin": [...]},
                      "candidate": {...}}}
    """
    results = {}
    for opp_name, opp_path in opponents.items():
        opp_full = str(REPO_ROOT / opp_path) if not opp_path.startswith("/") else opp_path
        row = {
            "baseline": {"own_money": [], "margin": []},
            "candidate": {"own_money": [], "margin": []},
        }
        for label, agent_path in (("baseline", baseline_path), ("candidate", candidate_path)):
            agent_full = str(agent_path)
            for i, seed in enumerate(seeds):
                if i % 2 == 0:
                    r_us, r_opp = _run_pair(agent_full, opp_full, seed)
                else:
                    r_opp, r_us = _run_pair(opp_full, agent_full, seed)
                row[label]["own_money"].append(r_us)
                row[label]["margin"].append(r_us - r_opp)
        results[opp_name] = row
    return results


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def report(results):
    print(f"{'opponent':>18} {'base_money':>11} {'cand_money':>11} {'base_margin':>12} {'cand_margin':>12}")
    fixed_deltas = []
    reactive_deltas = []
    for opp_name, row in results.items():
        base_money = _avg(row["baseline"]["own_money"])
        cand_money = _avg(row["candidate"]["own_money"])
        base_margin = _avg(row["baseline"]["margin"])
        cand_margin = _avg(row["candidate"]["margin"])
        print(
            f"{opp_name:>18} {base_money:>11.0f} {cand_money:>11.0f} "
            f"{base_margin:>12.0f} {cand_margin:>12.0f}"
        )
        delta = cand_margin - base_margin
        if opp_name in DEFAULT_OPPONENTS and opp_name != "pilkwang":
            fixed_deltas.append(delta)
        elif opp_name == "pilkwang":
            reactive_deltas.append(delta)

    print()
    if fixed_deltas and reactive_deltas:
        fixed_avg = _avg(fixed_deltas)
        reactive_avg = _avg(reactive_deltas)
        print(f"avg margin delta vs fixed-tape opponents: {fixed_avg:+.0f}")
        print(f"avg margin delta vs reactive opponent:    {reactive_avg:+.0f}")
        if (fixed_avg < 0) != (reactive_avg < 0):
            print(
                "\n*** DIVERGENT SIGNAL: fixed-tape and reactive opponents disagree on "
                "direction -- this is the confound described in this file's docstring. "
                "Trust the reactive-opponent number and the own_money columns over the "
                "fixed-tape margin. ***"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--opponents", default=None, help="comma-separated subset of: " + ",".join(DEFAULT_OPPONENTS))
    parser.add_argument("--seeds", default=None, help="comma-separated seed list, default 1,2,3,4,5")
    args = parser.parse_args()

    opponents = DEFAULT_OPPONENTS
    if args.opponents:
        names = [n.strip() for n in args.opponents.split(",")]
        opponents = {n: DEFAULT_OPPONENTS[n] for n in names}
    seeds = DEFAULT_SEEDS
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    results = compare(args.baseline, args.candidate, opponents, seeds)
    report(results)


if __name__ == "__main__":
    main()
