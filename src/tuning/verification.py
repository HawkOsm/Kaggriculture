from adaptive_agent.agent import make_agent

from .simulation import play_episode
from .scoring import _squash, DEFAULT_WIN_SCALE
from .config import SEARCH_SEEDS, HOLDOUT_SEEDS, HOLDOUT_ROTATIONS, OPPONENT_REGISTRY


def _resolve_agent(name):
    if name not in OPPONENT_REGISTRY:
        raise ValueError(f"unknown opponent {name!r}; choose from {sorted(OPPONENT_REGISTRY)} or 'champion'")
    return OPPONENT_REGISTRY[name]


class HoldoutGate:
    def __init__(self, rotation_idx):
        self.rotation_idx = rotation_idx
        self.active_holdout = HOLDOUT_ROTATIONS[rotation_idx]

    def verify_holdout(self, candidate_config, champion_config, episodes, tolerance, seeded=True):
        """Out-of-sample regression check -- does the candidate do materially
        WORSE than the champion against opponents excluded from every search
        pool? Returns (passed, per_opponent) where per_opponent maps name ->
        (cand_wins, cand_margin, champ_wins, champ_margin, regressed)."""
        per_opponent = {}
        passed = True
        for name in self.active_holdout:
            opponent = _resolve_agent(name)
            results = {}
            for label, cfg in (("cand", candidate_config), ("champ", champion_config)):
                wins = 0
                margins = []
                for i in range(episodes):
                    agent = make_agent(cfg)
                    seed = HOLDOUT_SEEDS[(i // 2) % len(HOLDOUT_SEEDS)] if seeded else None
                    if i % 2 == 0:
                        r_us, r_them = play_episode(agent, opponent, seed=seed)
                    else:
                        r_them, r_us = play_episode(opponent, agent, seed=seed)
                    margins.append(r_us - r_them)
                    wins += int(r_us > r_them)
                results[label] = (wins, sum(margins) / len(margins))
            (cand_wins, cand_margin), (champ_wins, champ_margin) = results["cand"], results["champ"]

            if cand_wins < champ_wins:
                regressed = True
            else:
                cand_s, champ_s = _squash(cand_margin, DEFAULT_WIN_SCALE), _squash(champ_margin, DEFAULT_WIN_SCALE)
                regressed = cand_s < champ_s - tolerance * abs(champ_s)
            per_opponent[name] = (cand_wins, cand_margin, champ_wins, champ_margin, regressed)
            if regressed:
                passed = False
        return passed, per_opponent


def verify_candidate(candidate_config, champion_config, episodes, seeded=True):
    """Dedicated candidate-vs-champion verification, independent of the
    search's own (noisy, mixed-opponent-pool) trial value -- the only valid
    promotion signal (see AGENT.md). Returns (wins, losses, ties, avg_margin)."""
    wins = losses = ties = 0
    margins = []
    for i in range(episodes):
        candidate = make_agent(candidate_config)
        champion = make_agent(champion_config)
        seed = SEARCH_SEEDS[(i // 2) % len(SEARCH_SEEDS)] if seeded else None
        if i % 2 == 0:
            r_candidate, r_champion = play_episode(candidate, champion, seed=seed)
        else:
            r_champion, r_candidate = play_episode(champion, candidate, seed=seed)
        margin = r_candidate - r_champion
        margins.append(margin)
        if margin > 0:
            wins += 1
        elif margin < 0:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties, sum(margins) / len(margins)
