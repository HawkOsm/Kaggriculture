"""Opponent registry, seed panels, and holdout rotations for tuning
`adaptive_agent`. Ported from the pre-restructuring `optimize.py` (last
committed version, `git show HEAD:src/optimize.py` -- the opponent pool
itself doesn't depend on which agent architecture is being tuned) with paths
adjusted for this package's location.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "opponents"))

from opponents.kawa_route_agent import kawa_route_agent
from opponents.boatlee_v16_agent import boatlee_v16_agent
from opponents.rayk_c95_agent import rayk_c95_agent
from opponents.saiteja_agent import saiteja_agent
from opponents.kaito_agent import kaito_agent
from opponents.tran_hh_agent import tran_hh_agent
from opponents.pilkwang_agent import pilkwang_agent
from opponents.romanrozen_agent import romanrozen_agent
from opponents.prvsiyan_frontier_agent import prvsiyan_frontier_agent
from opponents.premaananda108_agent import agent as premaananda108_agent
from opponents.sakhawathossen_agent import agent as sakhawathossen_agent
from opponents.stevenleehans_agent import stevenleehans_agent
from opponents.selfvar_no_expand_agent import selfvar_no_expand_agent
from opponents.selfvar_slow_expand_agent import selfvar_slow_expand_agent
from opponents.selfvar_panic_seller_agent import selfvar_panic_seller_agent
from opponents.selfvar_no_animals_agent import selfvar_no_animals_agent
from opponents.selfvar_labour_starved_agent import selfvar_labour_starved_agent
from opponents.selfvar_bad_dispatch_agent import selfvar_bad_dispatch_agent

# "champion" is handled separately by the caller (built fresh from
# champion_config, not fixed) -- not listed here.
# NOTE: pilkwang is INCLUDED (adaptive_agent is modeled on it, not a copy of
# it -- see CREDITS.md -- beating the source it was adapted from is a
# meaningful, not circular, signal).
#
# Pool refreshed 2026-08-27 (see docs/tests/LOG.md): the former near/mid-tier
# opponents (rajan1673, chaitanyajamble, ektarr, nagatakengo, umutdorukztrk,
# iamsdt, kiykhoi, mansiaggarwal88, moncefelm, daisy023) were all beaten
# decisively (4W-0L each, avg margin +58k to +128k) by the current
# `adaptive_agent` champion -- zero signal left, so they were deleted rather
# than kept as dead weight. Every opponent remaining below is one the current
# champion still LOSES to (0W across kawa/boatlee_v16/rayk_c95/saiteja/kaito/
# tran_hh/pilkwang/romanrozen/prvsiyan_frontier/sakhawathossen/stevenleehans)
# or is genuinely close (premaananda108, 2W-2L) -- real signal for tracking
# further progress, sourced by CURRENT public leaderboard score, not stale
# pull-time numbers.
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
    "sakhawathossen": sakhawathossen_agent,
    "stevenleehans": stevenleehans_agent,
    "premaananda108": premaananda108_agent,
    "selfvar_no_expand": selfvar_no_expand_agent,
    "selfvar_slow_expand": selfvar_slow_expand_agent,
    "selfvar_panic_seller": selfvar_panic_seller_agent,
    "selfvar_no_animals": selfvar_no_animals_agent,
    "selfvar_labour_starved": selfvar_labour_starved_agent,
    "selfvar_bad_dispatch": selfvar_bad_dispatch_agent,
    "random": "random",
    "pass": "pass",
    "starter": "starter",
}

REPO_ROOT = SRC.parent
BEST_CONFIG_PATH = SRC / "adaptive_best_config.json"
LOG_PATH = REPO_ROOT / "docs" / "tests" / "LOG.md"
OPTUNA_DIR = REPO_ROOT / "output" / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_STORAGE = f"sqlite:///{OPTUNA_DIR / 'adaptive_agent_study.db'}"
DEFAULT_STUDY_NAME = "adaptive_agent_v1"

# Disjointness is the point: if the holdout gate reused search seeds, a
# candidate could be overfit to those specific worlds and the gate would not
# be out-of-sample on worlds, only on opponents. They must never overlap.
SEARCH_SEEDS = [1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042, 10042, 11042, 12042]
HOLDOUT_SEEDS = [1337, 2337, 3337, 4337, 5337, 6337, 7337, 8337, 9337, 10337, 11337, 12337]
assert set(SEARCH_SEEDS).isdisjoint(set(HOLDOUT_SEEDS)), "SEARCH_SEEDS and HOLDOUT_SEEDS must be disjoint"

# kawa is ALWAYS included in every rotation as a structural-collapse guard --
# unwinnable, so never a promotion target, but a candidate that collapses
# further against it broke something structural, worth catching.
# Rotations rebuilt 2026-08-27 from the refreshed pool (see note above
# OPPONENT_REGISTRY) -- default search pool below draws from whichever
# opponents are NOT in the active rotation, same disjointness rule as before.
HOLDOUT_ROTATIONS = [
    ["saiteja", "sakhawathossen", "kawa"],
    ["boatlee_v16", "romanrozen", "kawa"],
    ["prvsiyan_frontier", "premaananda108", "kawa"],
]
