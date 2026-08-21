"""Local Optuna search over agent.DEFAULT_CONFIG.

Frames tuning as beating a reigning champion instead of chasing a raw reward
number: each trial's candidate config plays a fixed opponent pool --
default is the five pulled-from-Kaggle 2500+ opponents (kawa, boatlee_v16,
rayk_c95, saiteja, kaito -- see CREDITS.md) plus the current champion config
(loaded from best_config.json, or DEFAULT_CONFIG on the very first run) --
and is scored by average reward *margin* (candidate - opponent), not raw
reward. That's what "steady progression" means here: every run of this
script tries to beat whatever the previous run's winner was, so
best_config.json only moves forward.

Because episodes aren't seeded, a single episode is noisy -- each trial
plays --episodes-per-opponent episodes against every opponent (both seat
orders by default) and reports the running average after each one, so
Optuna's pruner (MedianPruner) can kill a trial that's clearly worse than
the pack after just 1-2 episodes instead of burning the full budget on it.

Usage:
    python optimize.py --n-trials 200 --n-jobs 8
    python optimize.py --n-trials 50 --n-jobs 1 --opponents kawa,random

Resumable: reruns with the same --study-name/--storage continue the same
Optuna study instead of starting over.
"""

import argparse
import datetime
import json
import multiprocessing
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "opponents"))

import optuna

from kaggle_environments import make as make_env

from agent import DEFAULT_CONFIG, PRIORITY_TIER_NAMES, make_agent
from opponents.kawa_route_agent import kawa_route_agent
from opponents.boatlee_v16_agent import boatlee_v16_agent
from opponents.rayk_c95_agent import rayk_c95_agent
from opponents.saiteja_agent import saiteja_agent
from opponents.kaito_agent import kaito_agent
from opponents.tran_hh_agent import tran_hh_agent
from opponents.pilkwang_agent import pilkwang_agent
from opponents.romanrozen_agent import romanrozen_agent
from opponents.prvsiyan_frontier_agent import prvsiyan_frontier_agent

# Real opponent agents, keyed by the name used in --opponents. "champion" is
# handled separately (built fresh from champion_config each call, not fixed).
OPPONENT_REGISTRY = {
    "kawa": kawa_route_agent,
    "boatlee_v16": boatlee_v16_agent,
    "rayk_c95": rayk_c95_agent,
    "saiteja": saiteja_agent,
    "kaito": kaito_agent,
    "tran_hh": tran_hh_agent,
    "pilkwang": pilkwang_agent,
    "romanrozen": romanrozen_agent,
    "prvsiyan_frontier": prvsiyan_frontier_agent,
    "random": "random",
    "pass": "pass",
    "starter": "starter",
}


def _resolve_agent(name):
    if name not in OPPONENT_REGISTRY:
        raise ValueError(f"unknown opponent {name!r}; choose from {sorted(OPPONENT_REGISTRY)} or 'champion'")
    return OPPONENT_REGISTRY[name]


REPO_ROOT = HERE.parent
BEST_CONFIG_PATH = HERE / "best_config.json"
LOG_PATH = REPO_ROOT / "docs" / "tests" / "LOG.md"
OPTUNA_DIR = REPO_ROOT / "output" / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_STORAGE = f"sqlite:///{OPTUNA_DIR / 'optuna_study.db'}"
# Renamed when the search space changed for the scaling-strategy redesign
# (see docs/tests/LOG.md) -- old trials used a different, now-incompatible
# set of config keys (structure_money_threshold, no hire_money_floor, etc.),
# so this points at a fresh study rather than silently mixing histories.
DEFAULT_STUDY_NAME = "robust_agent_config_v2_scaling"


CROP_PROFILES = {
    # Broad exposure to avoid single-commodity collapse.
    "diversified": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"],
    # Faster cash cycling, lower long-horizon concentration.
    "fast_cash": ["WHEAT", "CARROT", "TOMATO"],
    # Keep high-upside melon while preserving short-cycle income.
    "melon_plus_short": ["WHEAT", "CARROT", "MELON"],
}


def load_champion_config():
    if BEST_CONFIG_PATH.exists():
        return json.loads(BEST_CONFIG_PATH.read_text())
    return dict(DEFAULT_CONFIG)


def sample_config(trial):
    cfg = {
        # Linear state-dependent sell-threshold policy (see robust_agent's
        # _dynamic_sell_fraction) -- Optuna searches the weights of a small
        # function instead of one constant. This is the "hybrid RL" layer:
        # same search mechanics as everything else here, but the thing
        # being searched is now a tiny policy, not a fixed number.
        "sell_fraction_base": trial.suggest_float("sell_fraction_base", 0.2, 0.8),
        "sell_fraction_day_weight": trial.suggest_float("sell_fraction_day_weight", -0.3, 0.3),
        "sell_fraction_cash_weight": trial.suggest_float("sell_fraction_cash_weight", -0.3, 0.3),
        "cash_scale": trial.suggest_int("cash_scale", 200, 4000, step=100),
        "max_sell_chunk": trial.suggest_int("max_sell_chunk", 3, 20),
        "sell_backlog_multiple": trial.suggest_float("sell_backlog_multiple", 1.0, 6.0),
        "money_reserve": trial.suggest_int("money_reserve", 20, 500, step=20),
        "seed_money_floor": trial.suggest_int("seed_money_floor", 5, 50),
        "hire_money_floor": trial.suggest_int("hire_money_floor", 0, 100, step=10),
        "hire_reserve_multiple": trial.suggest_float("hire_reserve_multiple", 0.5, 6.0),
        "max_hires_per_day": trial.suggest_int("max_hires_per_day", 0, 8),
        "land_utilization_threshold": trial.suggest_float("land_utilization_threshold", 0.4, 0.95),
        "land_startup_days": trial.suggest_int("land_startup_days", 0, 10),
        "animal_enabled": trial.suggest_categorical("animal_enabled", [True, False]),
        "enable_coop": trial.suggest_categorical("enable_coop", [True, False]),
        "max_structures": trial.suggest_int("max_structures", 0, 18),
        "pasture_target_ratio": trial.suggest_float("pasture_target_ratio", 0.0, 1.2),
        "animal_reserve_multiple": trial.suggest_float("animal_reserve_multiple", 1.0, 5.0),
        "startup_days": trial.suggest_int("startup_days", 0, 8),
        "buy_fertilizer": trial.suggest_categorical("buy_fertilizer", [True, False]),
        "hire_backlog_ratio": trial.suggest_float("hire_backlog_ratio", 0.3, 4.0),
        "diversification_weight": trial.suggest_float("diversification_weight", 0.0, 1.0),
        # Always on, not searched: selling into a still-healthy price before
        # the opponent's incoming supply craters it is a real signal we can
        # see (their public tiles), not a guess -- there's no game-theoretic
        # downside to using it, only a threshold/discount to tune. Left as
        # a togglable bool, a noisy 2-episode trial can (and did -- the
        # search that produced the current best_config.json turned it off)
        # mistake variance for a real signal and disable a strictly-useful
        # feature. See docs/tests/LOG.md.
        "opponent_awareness_enabled": True,
        "opponent_incoming_threshold": trial.suggest_int("opponent_incoming_threshold", 1, 10),
        "opponent_race_discount": trial.suggest_float("opponent_race_discount", 0.3, 1.0),
        "opponent_concentration_sensitivity": trial.suggest_float("opponent_concentration_sensitivity", 0.0, 1.0),
        "opponent_lookahead_days": trial.suggest_int("opponent_lookahead_days", 0, 5),
        "crop_profile": trial.suggest_categorical("crop_profile", sorted(CROP_PROFILES.keys())),
    }
    # One float knob per _plan_units dispatch tier -- lets the search find a
    # better task-priority order instead of it only changing when someone
    # reads actions.csv and hand-edits the function order (see
    # docs/tests/LOG.md, and DEFAULT_CONFIG's priority_weight_* comment in
    # agent.py for why this is float-per-tier rather than a searched
    # permutation).
    for name in PRIORITY_TIER_NAMES:
        cfg[f"priority_weight_{name}"] = trial.suggest_float(f"priority_weight_{name}", 0.0, 100.0)
    return cfg


def _resolve_search_config(params):
    """Convert Optuna trial params into a robust_agent-compatible config.

    Keeps search-only knobs (like crop_profile) out of runtime config while
    allowing older studies (without crop_profile) to remain readable.
    """
    cfg = dict(params)
    profile = cfg.pop("crop_profile", "diversified")
    cfg["crops"] = list(CROP_PROFILES.get(profile, CROP_PROFILES["diversified"]))
    if not cfg.get("animal_enabled", True):
        cfg["max_structures"] = 0
    return cfg


def play_episode(agent_a, agent_b, episode_steps=720):
    env = make_env("kaggriculture", configuration={"episodeSteps": episode_steps}, debug=True)
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    return final[0].reward, final[1].reward


def _risk_adjusted_score(margins, risk_aversion=0.2, tail_quantile=0.25, tail_weight=0.3):
    mean_margin = statistics.fmean(margins)
    std_margin = statistics.pstdev(margins) if len(margins) > 1 else 0.0
    tail_n = max(1, int(len(margins) * tail_quantile))
    tail_avg = statistics.fmean(sorted(margins)[:tail_n])
    return mean_margin - risk_aversion * std_margin + tail_weight * tail_avg


def evaluate_config(config, opponent_names, episodes_per_opponent, champion_config, trial=None,
                    risk_aversion=0.2, tail_quantile=0.25, tail_weight=0.3):
    """Average reward margin (candidate - opponent) across the opponent pool.
    Reports the running average to `trial` after each episode for pruning."""
    candidate = make_agent(config)
    opponents = []
    for name in opponent_names:
        opponents.append((name, make_agent(champion_config) if name == "champion" else _resolve_agent(name)))

    margins = []
    step = 0
    for name, opponent_agent in opponents:
        for seat in range(episodes_per_opponent):
            if seat % 2 == 0:
                r_candidate, r_opponent = play_episode(candidate, opponent_agent)
            else:
                r_opponent, r_candidate = play_episode(opponent_agent, candidate)
            margins.append(r_candidate - r_opponent)

            if trial is not None:
                running_score = _risk_adjusted_score(
                    margins,
                    risk_aversion=risk_aversion,
                    tail_quantile=tail_quantile,
                    tail_weight=tail_weight,
                )
                trial.report(running_score, step)
                step += 1
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return _risk_adjusted_score(
        margins,
        risk_aversion=risk_aversion,
        tail_quantile=tail_quantile,
        tail_weight=tail_weight,
    )


def make_objective(opponent_names, episodes_per_opponent, champion_config,
                   risk_aversion=0.2, tail_quantile=0.25, tail_weight=0.3):
    def objective(trial):
        sampled = sample_config(trial)
        config = _resolve_search_config(sampled)
        return evaluate_config(
            config,
            opponent_names,
            episodes_per_opponent,
            champion_config,
            trial=trial,
            risk_aversion=risk_aversion,
            tail_quantile=tail_quantile,
            tail_weight=tail_weight,
        )
    return objective


def verify_candidate(candidate_config, champion_config, episodes):
    """Dedicated candidate-vs-champion verification, independent of the
    search's own (noisy, mixed-opponent-pool) trial value. Promotion should
    never be decided by the same 6-episode number that picked the trial --
    that's exactly the kind of thin sample the search's pruner is built to
    distrust mid-search, so the final promotion gate shouldn't trust it
    either. Returns (wins, losses, ties, avg_margin)."""
    candidate = make_agent(candidate_config)
    champion = make_agent(champion_config)
    wins = losses = ties = 0
    margins = []
    for i in range(episodes):
        if i % 2 == 0:
            r_candidate, r_champion = play_episode(candidate, champion)
        else:
            r_champion, r_candidate = play_episode(champion, candidate)
        margin = r_candidate - r_champion
        margins.append(margin)
        if margin > 0:
            wins += 1
        elif margin < 0:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties, sum(margins) / len(margins)


def _worker(study_name, storage, n_trials, opponent_names, episodes_per_opponent, champion_config,
            risk_aversion, tail_quantile, tail_weight):
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(
        make_objective(
            opponent_names,
            episodes_per_opponent,
            champion_config,
            risk_aversion=risk_aversion,
            tail_quantile=tail_quantile,
            tail_weight=tail_weight,
        ),
        n_trials=n_trials,
    )


def _log_outcome(study, opponent_names, episodes_per_opponent, n_trials, n_jobs, had_champion_before,
                  verification_episodes, wins, losses, ties, verify_margin, promoted):
    date = datetime.date.today().isoformat()
    best = study.best_trial
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    entry = f"""## {date} — optimize.py: Optuna search over robust_agent config
- Command: `python src/optimize.py --n-trials {n_trials} --n-jobs {n_jobs} --opponents {','.join(opponent_names)}`
- Result: {n_complete} complete trials, {n_pruned} pruned. Search's own best trial margin (noisy, {episodes_per_opponent} episodes x {len(opponent_names)}-opponent pool): {best.value:.1f}. **Dedicated candidate-vs-champion verification ({verification_episodes} episodes): {wins}W-{losses}L-{ties}T, avg margin {verify_margin:.1f}.**
- Notes: opponent pool was {opponent_names}, champion going in was `{'best_config.json' if had_champion_before else 'DEFAULT_CONFIG (first run)'}`. Promotion gate is the dedicated verification (wins > losses), not the search's own trial value. {"Promoted to new champion (best_config.json updated)." if promoted else "Did NOT clear the verification bar -- best_config.json left unchanged."} Best trial params: `{json.dumps(best.params)}`.

"""
    # LOG.md convention is newest-first: insert right after the header/divider
    # instead of appending, which would silently bury new entries at the end.
    text = LOG_PATH.read_text()
    marker = "---\n"
    idx = text.index(marker) + len(marker)
    LOG_PATH.write_text(text[:idx] + "\n" + entry + text[idx:].lstrip("\n"))
    print(f"Logged outcome to {LOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=100, help="total trials across all workers")
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--episodes-per-opponent", type=int, default=2, help="must be even for balanced seating")
    parser.add_argument(
        "--opponents",
        default="kawa,boatlee_v16,rayk_c95,saiteja,kaito,tran_hh,pilkwang,romanrozen,prvsiyan_frontier,champion",
        help=f"comma-separated: {sorted(OPPONENT_REGISTRY)} or champion",
    )
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--timeout", type=int, default=None, help="wall-clock budget in seconds, across all workers")
    parser.add_argument("--verification-episodes", type=int, default=10,
                         help="dedicated candidate-vs-champion episodes deciding promotion (independent of the search's own trial value)")
    parser.add_argument("--risk-aversion", type=float, default=0.2,
                        help="penalty weight on per-episode margin variance (higher = safer configs)")
    parser.add_argument("--tail-quantile", type=float, default=0.25,
                        help="fraction of worst margins treated as downside tail")
    parser.add_argument("--tail-weight", type=float, default=0.3,
                        help="weight on worst-tail average margin (higher = more crash-averse)")
    args = parser.parse_args()

    opponent_names = [o.strip() for o in args.opponents.split(",") if o.strip()]
    had_champion_before = BEST_CONFIG_PATH.exists()
    champion_config = load_champion_config()

    optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )

    n_jobs = max(1, args.n_jobs)
    per_worker = [args.n_trials // n_jobs] * n_jobs
    for i in range(args.n_trials % n_jobs):
        per_worker[i] += 1

    print(f"Running {args.n_trials} trials across {n_jobs} worker process(es), "
          f"opponents={opponent_names}, episodes_per_opponent={args.episodes_per_opponent}")

    if n_jobs == 1:
        _worker(
            args.study_name,
            args.storage,
            args.n_trials,
            opponent_names,
            args.episodes_per_opponent,
            champion_config,
            args.risk_aversion,
            args.tail_quantile,
            args.tail_weight,
        )
    else:
        procs = []
        for n in per_worker:
            if n <= 0:
                continue
            p = multiprocessing.Process(
                target=_worker,
                args=(
                    args.study_name,
                    args.storage,
                    n,
                    opponent_names,
                    args.episodes_per_opponent,
                    champion_config,
                    args.risk_aversion,
                    args.tail_quantile,
                    args.tail_weight,
                ),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    best = study.best_trial
    print(f"\nSearch's own best trial margin (noisy, {args.episodes_per_opponent} episodes x {len(opponent_names)}-opponent pool): {best.value:.1f}")
    print(f"Best params: {json.dumps(best.params, indent=2)}")

    merged = dict(DEFAULT_CONFIG)
    merged.update(_resolve_search_config(best.params))

    print(f"\nRunning dedicated verification: {args.verification_episodes} episodes, candidate vs champion only...")
    wins, losses, ties, verify_margin = verify_candidate(merged, champion_config, args.verification_episodes)
    print(f"Verification: {wins}W-{losses}L-{ties}T, avg margin {verify_margin:.1f}")

    # Promotion gate is this dedicated head-to-head, not the search's own
    # trial value -- that value came from a 6-episode mixed-opponent-pool
    # average (the full opponent pool + champion combined), which can
    # look positive even if the candidate barely edges or loses to the
    # champion specifically. Require strictly more wins than losses.
    promoted = wins > losses
    if promoted:
        BEST_CONFIG_PATH.write_text(json.dumps(merged, indent=2))
        print(f"New champion written to {BEST_CONFIG_PATH}")
    else:
        print("Candidate did not beat the champion head-to-head (wins <= losses) -- best_config.json left unchanged.")

    _log_outcome(study, opponent_names, args.episodes_per_opponent, args.n_trials, n_jobs, had_champion_before,
                 args.verification_episodes, wins, losses, ties, verify_margin, promoted)


if __name__ == "__main__":
    main()
