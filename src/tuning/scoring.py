import math
import statistics

import optuna

from .search_space import sample_config
from .simulation import play_episode
from .config import SEARCH_SEEDS, OPPONENT_REGISTRY

from adaptive_agent.agent import make_agent


def _resolve_agent(name):
    if name not in OPPONENT_REGISTRY:
        raise ValueError(f"unknown opponent {name!r}; choose from {sorted(OPPONENT_REGISTRY)} or 'champion'")
    return OPPONENT_REGISTRY[name]


DEFAULT_WIN_SCALE = 15000.0


def _squash(margin, win_scale):
    """Map a raw coin margin onto (-1, +1), saturating for blowouts -- see
    the original optimize.py's docstring (docs/tests/LOG.md history) for why:
    the competition's rating is win/loss/tie only, so raw-margin scoring lets
    a handful of huge-magnitude unwinnable matchups (kawa, pilkwang) dominate
    the objective over the genuinely contestable ones. tanh keeps a usable
    gradient near the win/loss boundary while treating "-140k" and "-120k" as
    equally-just-a-loss."""
    return math.tanh(margin / win_scale)


def _risk_adjusted_score(margins, risk_aversion=0.2, tail_quantile=0.25, tail_weight=0.3,
                          win_scale=None):
    if win_scale is not None:
        margins = [_squash(m, win_scale) for m in margins]
    mean_margin = statistics.fmean(margins)
    std_margin = statistics.pstdev(margins) if len(margins) > 1 else 0.0
    tail_n = max(1, int(len(margins) * tail_quantile))
    tail_avg = statistics.fmean(sorted(margins)[:tail_n])
    return mean_margin - risk_aversion * std_margin + tail_weight * tail_avg


def evaluate_config(config, opponent_names, episodes_per_opponent, champion_config, trial=None,
                     risk_aversion=0.2, tail_quantile=0.25, tail_weight=0.3,
                     win_scale=DEFAULT_WIN_SCALE, seeded=True):
    """Average risk-adjusted margin (candidate - opponent) across the
    opponent pool. Reports the running score to `trial` after each episode
    for pruning."""
    margins = []
    step = 0
    for name in opponent_names:
        for seat in range(episodes_per_opponent):
            candidate = make_agent(config)
            opponent_agent = make_agent(champion_config) if name == "champion" else _resolve_agent(name)
            seed = SEARCH_SEEDS[(step // 2) % len(SEARCH_SEEDS)] if seeded else None
            if seat % 2 == 0:
                r_candidate, r_opponent = play_episode(candidate, opponent_agent, seed=seed)
            else:
                r_opponent, r_candidate = play_episode(opponent_agent, candidate, seed=seed)
            margins.append(r_candidate - r_opponent)

            if trial is not None:
                running_score = _risk_adjusted_score(
                    margins, risk_aversion=risk_aversion, tail_quantile=tail_quantile,
                    tail_weight=tail_weight, win_scale=win_scale,
                )
                trial.report(running_score, step)
                step += 1
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return _risk_adjusted_score(
        margins, risk_aversion=risk_aversion, tail_quantile=tail_quantile,
        tail_weight=tail_weight, win_scale=win_scale,
    )


def make_objective(opponent_names, episodes_per_opponent, champion_config,
                    risk_aversion=0.2, tail_quantile=0.25, tail_weight=0.3,
                    win_scale=DEFAULT_WIN_SCALE, seeded=True):
    def objective(trial):
        config = sample_config(trial)
        return evaluate_config(
            config, opponent_names, episodes_per_opponent, champion_config, trial=trial,
            risk_aversion=risk_aversion, tail_quantile=tail_quantile, tail_weight=tail_weight,
            win_scale=win_scale, seeded=seeded,
        )
    return objective
