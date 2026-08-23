"""Local Optuna search over agent.DEFAULT_CONFIG.

Frames tuning as beating a reigning champion instead of chasing a raw reward
number: each trial's candidate config plays a fixed opponent pool --
default is nine pulled-from-Kaggle 2500+ opponents (kawa, boatlee_v16,
rayk_c95, saiteja, kaito, tran_hh, pilkwang, romanrozen, prvsiyan_frontier
-- see CREDITS.md) plus two near our own live rating (rajan1673,
chaitanyajamble, weighted 2x each -- see OPPONENT_REGISTRY's comment)
plus the current champion config (loaded from best_config.json, or
DEFAULT_CONFIG on the very first run) -- and is scored by average reward
*margin* (candidate - opponent), not raw reward. That's what "steady
progression" means here: every run of this script tries to beat whatever
the previous run's winner was, so best_config.json only moves forward.

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
import math
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
from opponents.rajan1673_agent import agent as rajan1673_agent
from opponents.chaitanyajamble_agent import agent as chaitanyajamble_agent
from opponents.ektarr_agent import agent as ektarr_agent
from opponents.nagatakengo_agent import agent as nagatakengo_agent
from opponents.premaananda108_agent import agent as premaananda108_agent
from opponents.sakhawathossen_agent import agent as sakhawathossen_agent
# Near-tier opponents sourced 2026-08-22 to fix a search pool with almost no
# winnable targets (see HOLDOUT_OPPONENTS' comment). Leaderboard scores 380-496,
# i.e. genuinely around our own ~500. moncefelm is the exception: its author
# scores 442 but the agent itself beats our champion by ~89k (saturating the
# objective), so it is registry-only and deliberately NOT in the default pool.
from opponents.umutdorukztrk_agent import umutdorukztrk_agent
from opponents.iamsdt_agent import iamsdt_agent
from opponents.kiykhoi_agent import kiykhoi_agent
from opponents.mansiaggarwal88_agent import mansiaggarwal88_agent
from opponents.moncefelm_agent import moncefelm_agent
from opponents.daisy023_agent import daisy023_agent
# Calibrated self-variants (2026-08-22): degraded snapshots of the champion,
# built because two exhaustive passes over public Kaggle notebooks found no
# more real opponents in the gradient band (see docs/tests/LOG.md). Each
# embeds a FROZEN hardcoded config -- deliberately never reads
# best_config.json, so promoting a new champion cannot silently change their
# strength and invalidate past measurements (the champion-drift bug, hit once
# already today). Measured champion margins: no_expand -3,560 [-0.23],
# slow_expand -2,708 [-0.18], panic_seller +333 [+0.02], no_animals +3,516
# [+0.23], labour_starved +4,670 [+0.30], bad_dispatch +9,237 [+0.55].
# NOTE these are sparring partners, not real competitors -- tuning primarily
# against variants of oneself is a documented echo-chamber failure mode, so
# only the two filling the emptiest region are in the default pool and the
# holdout gate stays anchored on REAL agents.
from opponents.selfvar_no_expand_agent import selfvar_no_expand_agent
from opponents.selfvar_slow_expand_agent import selfvar_slow_expand_agent
from opponents.selfvar_panic_seller_agent import selfvar_panic_seller_agent
from opponents.selfvar_no_animals_agent import selfvar_no_animals_agent
from opponents.selfvar_labour_starved_agent import selfvar_labour_starved_agent
from opponents.selfvar_bad_dispatch_agent import selfvar_bad_dispatch_agent

# Real opponent agents, keyed by the name used in --opponents. "champion" is
# handled separately (built fresh from champion_config each call, not fixed).
# rajan1673/chaitanyajamble/ektarr/nagatakengo are near our own live rating
# (~467, found via the public leaderboard -- see docs/tests/LOG.md, "pulled
# two near-tier opponents" and "sourced 4 more real opponents"); everything
# else here is a stronger Elo stretch-goal opponent, several confirmed this
# session to be pre-solved fixed-route scripts (kawa/prvsiyan specifically,
# see LOG.md's "MAJOR REFRAME" entry) rather than live adaptive strategies
# -- so near-tier wins are the more realistic near-term signal despite the
# pool being dominated by the stronger set.
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
    "rajan1673": rajan1673_agent,
    "chaitanyajamble": chaitanyajamble_agent,
    "ektarr": ektarr_agent,
    "nagatakengo": nagatakengo_agent,
    "premaananda108": premaananda108_agent,
    "sakhawathossen": sakhawathossen_agent,
    "umutdorukztrk": umutdorukztrk_agent,
    "iamsdt": iamsdt_agent,
    "kiykhoi": kiykhoi_agent,
    "mansiaggarwal88": mansiaggarwal88_agent,
    "moncefelm": moncefelm_agent,
    "daisy023": daisy023_agent,
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
# Renamed a sixth time after two more search-space additions on top of v8:
# shadow pricing (lambda_labor, lambda_land -- opportunity-cost penalties on
# crop/animal scoring) and cash-flow lookahead / MPC-lite (income_discount,
# income_lookahead_days -- lets projected near-term guaranteed income offset
# the reserve cushions). Both were individually verified (see docs/tests/
# LOG.md: shadow pricing kept land term / dropped labor term via v10 solo
# search; cash-flow lookahead mechanically verified but mixed 6-episode
# opponent-dependent result) but never searched *together* against real
# opponents -- this is the first study to combine all four v8-era mechanics
# (reserve rework, priority-weight fix already hand-applied to
# best_config.json, shadow pricing, cash-flow lookahead) in one search space
# scored against the actual champion-tier opponent pool. v8 had zero trials
# logged under this shape (confirmed via optuna.study.get_all_study_summaries
# before reusing the number), so no separate placeholder rename was needed
# for it on its own -- same as the v8 pass note below.
# Renamed an eighth time. Two changes, either of which alone would force it:
# (1) SEEDED PAIRED EVALUATION is now on by default -- trials no longer see
# random worlds, so a trial value means something different (and far less
# noisy) than any v12 value; (2) the --opponents default pool was rebuilt
# from 12 saturating 1700-2700 opponents to 6 near-tier ones chosen for
# measured gradient. v12's own history is unusable regardless: its landscape
# was flat (0.0212 spread across all 21 completed trials, 95% pruned) because
# _squash saturated every opponent in that pool -- see docs/tests/LOG.md.
# Renamed a ninth time: the opponent pool changed again (two calibrated
# self-variants added to fill the near-zero/negative gradient band, and
# holdout rotation now varies the pool per rotation index). Different pool
# means a different objective, so v13 trial values are not comparable.
# Rotation index is auto-appended to this name at runtime (_rot<N>).
#
# Renamed a tenth time (2026-08-22): search-space shape changed again --
# removed the 4 relative-wealth knobs (wealth_margin_scale/risk_sensitivity/
# risk_scale_min/risk_scale_max; the mechanism itself, a real promoted
# effect in v14's champion, not a no-op, was deleted from agent.py) and
# added 5 new ones (crash_sell_enabled/crash_sell_margin/
# crash_sell_multiplier/crash_sell_chunk/cluster_shed_weight -- see
# docs/tests/LOG.md). Same MedianPruner-contamination reasoning as every
# rename above: v14's trial history was never scored on this shape.
# Renamed an eleventh time (2026-08-22): the relative-wealth mechanism was
# restored (it had been deleted, despite wealth_margin_scale being the single
# highest-importance parameter of the v14 search at 0.204) and is now searched
# as an explicit toggle plus 4 shape knobs. Search-space shape changed, so v15
# trial values are not comparable -- same MedianPruner reasoning as every
# rename above. Restored code is a verified no-op at its defaults (identical
# rewards on fixed seeds vs the pre-restore agent).
# Renamed a twelfth time (2026-08-22), same day: fixed a MedianPruner bug
# discovered by auditing v16's own result (see git history / LOG.md) --
# n_warmup_steps=1 let a trial be pruned after its first of 16 episodes,
# starving any delayed-payoff strategy. animal_enabled=True was sampled in
# only 6% of all 2000 v16 trials (not 50%) because early animal trials got
# killed before their investment paid off, and TPE learned from that to stop
# sampling the arm. v16's "animals are useless" conclusion is void -- the
# search never fairly evaluated the option. n_warmup_steps now covers a full
# pass over the opponent pool before any pruning decision. This changes which
# trials survive to be compared, so v16's pruning history is not reusable.
DEFAULT_STUDY_NAME = "robust_agent_config_v17_pruner_fix"
#
# --- superseded reasoning, kept for context ---
# DEFAULT_STUDY_NAME = "robust_agent_config_v16_relwealth_restored"  # superseded, see above
#
# --- superseded reasoning, kept for context ---
# DEFAULT_STUDY_NAME = "robust_agent_config_v15_crash_sell_shed_cluster"  # superseded, see above
#
# --- superseded reasoning, kept for context ---
# DEFAULT_STUDY_NAME = "robust_agent_config_v14_variants_rotation"  # superseded, see above
# DEFAULT_STUDY_NAME = "robust_agent_config_v13_seeded_neartier"  # superseded, see above
#
# --- superseded reasoning, kept for context ---
# DEFAULT_STUDY_NAME = "robust_agent_config_v12_winrate_objective"  # superseded, see above
#
# --- superseded reasoning, kept for context ---
# Renamed a seventh time for the most consequential change yet: the OBJECTIVE
# itself. Trials were scored on risk-adjusted raw coin margin, but this
# competition's rating is win/loss/tie ONLY -- coin difference never affects
# the rating change. Since kawa (~-142k) and prvsiyan (~-97k) dominate the
# pool average by an order of magnitude over the contestable near-tier
# matchups (+-6k..25k), the old objective spent most of its optimization
# pressure making unwinnable losses marginally less lopsided, which earns
# nothing. Scores now pass through _squash (tanh) so blowouts saturate and
# near-boundary matchups keep the gradient. Every prior study's trial values
# are in different units and are NOT comparable to these -- a harder break
# than any previous rename, which only changed the search space or opponent
# pool. v11's own history stays valid on its own terms (and its promotion was
# reverted anyway, see docs/tests/LOG.md).
# DEFAULT_STUDY_NAME = "robust_agent_config_v11_combined_mechanics"  # superseded, see above
# v8 rename covered the reserve/risk-scale architecture rewrite: new
# land_reserve_multiple key added, and money_reserve/hire_money_floor now
# mean something different (small fixed survival floors, not spending
# buffers -- bounds narrowed accordingly). Both a search-space shape change
# and a semantic change to existing keys, so v7's history wasn't a valid
# baseline for pruning comparisons. The default --opponents pool also
# reverted to the strong-only set (near-tier weighting removed, see that
# flag's own comment) in the same pass -- no separate rename needed for that
# on its own, since that study had zero trials logged under either change
# yet at the time.
# DEFAULT_STUDY_NAME = "robust_agent_config_v8_reserve_rework"  # superseded, see above
# Renamed a fourth time after adding ektarr/nagatakengo (near-tier, weighted
# 2x) and premaananda108/sakhawathossen (stronger, 1x) to the default
# --opponents pool -- same reasoning as the v5->v6 rename just below: a
# different opponent pool means a different reward distribution at every
# step, which is exactly the kind of change MedianPruner contamination
# cares about even though the search-space shape itself didn't move. v6's
# own history (real promotion, 9W-1L verification -- see docs/tests/LOG.md)
# stays valid on its own terms, just not comparable to trials scored
# against this new, larger 20-entry pool.
# DEFAULT_STUDY_NAME = "robust_agent_config_v7_more_opponents"  # superseded, see above
#
# --- superseded reasoning, kept for context ---
# Renamed a third time after adding rajan1673/chaitanyajamble (weighted 2x)
# to the default --opponents pool. Search-space shape didn't change, but
# MedianPruner contamination isn't only about param shape -- it's about the
# objective's meaning at each step, and a different opponent pool (14
# entries now vs 10, different reward scale/distribution at every episode
# count) is exactly that kind of change. v5's own history (real promotion,
# 6W-4L verification, see docs/tests/LOG.md) stays valid on its own terms;
# it's just not comparable to trials scored against this new pool.
# DEFAULT_STUDY_NAME = "robust_agent_config_v6_near_tier_weighted"  # superseded, see above
#
# --- superseded reasoning, kept for context ---
# Renamed again after adding 4 relative-wealth knobs (wealth_margin_scale,
# risk_sensitivity, risk_scale_min, risk_scale_max -- 29-dim search space,
# up from 25). Same MedianPruner-contamination reasoning as the v3->v4
# rename below: a persistent study's pruning decisions are judged against
# the median of *all prior trials at that step*, so reusing v4's history
# (whose trials never had these 4 keys) would silently handicap every new
# trial regardless of its actual params. v4 itself is not abandoned/wasted
# -- see its own history for the 87/400 completion-rate confirmation that
# a truly fresh study works as intended. (Note: "v3_dynamic_sell" is a
# separate, already-abandoned 100-trial study from earlier in the session
# -- not reused either.) See docs/tests/LOG.md.
#
# --- superseded reasoning, kept for context ---
# Renamed after adding the 11 priority_weight_* dispatch-tier knobs +
# opponent_concentration_sensitivity (25-dim search space, up from 13/22).
# Two searches (450 then 1000 trials, 2081 pruned total) against
# robust_agent_config_v2_scaling never produced a single completed trial
# with the new keys -- MedianPruner compares each new trial's intermediate
# progress against the median of *all prior trials at that step*, and v2's
# history (8+ searches, an early margin-36325.9 outlier never beaten since)
# makes every new trial look bad almost immediately regardless of its actual
# params. A fresh study starts pruning decisions from a clean baseline
# instead of being judged against a different search space's history.
# DEFAULT_STUDY_NAME = "robust_agent_config_v5_relative_wealth"  # superseded, see above


# Two DISJOINT frozen seed panels for paired evaluation.
# Disjointness is the point: if the holdout gate reused search seeds, a candidate could be
# overfit to those specific worlds and the gate would not be out-of-sample on worlds, only on
# opponents. They must never overlap.
SEARCH_SEEDS = [1042, 2042, 3042, 4042, 5042, 6042, 7042, 8042, 9042, 10042, 11042, 12042]
HOLDOUT_SEEDS = [1337, 2337, 3337, 4337, 5337, 6337, 7337, 8337, 9337, 10337, 11337, 12337]
assert set(SEARCH_SEEDS).isdisjoint(set(HOLDOUT_SEEDS)), "SEARCH_SEEDS and HOLDOUT_SEEDS must be disjoint"


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
        # sell_backlog_multiple deliberately NOT searched -- see the matching
        # comment on agent.py's KNOB_SPECS. It's genuinely unused by
        # _market_orders (the force-sell logic that read it was reverted),
        # so searching it burns a real dimension for zero behavioral effect.
        # This removal was decided and documented earlier in the session but
        # the edit to this file didn't actually land then -- every search
        # since (through v7) searched it anyway; harmless (just wasted
        # search budget), not a correctness bug.
        # Narrowed from (20, 500) -- money_reserve is now a small fixed
        # survival floor, not a spending-policy buffer (see agent.py
        # DEFAULT_CONFIG's comment) -- same scale as the floors below.
        "money_reserve": trial.suggest_int("money_reserve", 10, 150, step=10),
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
        # New: land's own cost-scaled reserve, same shape as
        # animal_reserve_multiple -- previously land only used the flat
        # money_reserve buffer directly (see agent.py DEFAULT_CONFIG).
        "land_reserve_multiple": trial.suggest_float("land_reserve_multiple", 0.2, 4.0),
        # ROI payback gate, replacing animal_reserve_multiple/
        # land_reserve_multiple's flat cushion with a projected-payback
        # check when enabled (see agent.py DEFAULT_CONFIG's
        # roi_gate_enabled comment). New, unverified.
        "roi_gate_enabled": trial.suggest_categorical("roi_gate_enabled", [True, False]),
        "roi_margin": trial.suggest_float("roi_margin", 1.0, 3.0),
        # Phase-window gates, modeled on pilkwang_agent.py -- see agent.py
        # DEFAULT_CONFIG's phase_gate_enabled comment. New, unverified.
        "phase_gate_enabled": trial.suggest_categorical("phase_gate_enabled", [True, False]),
        "land_min_days_left": trial.suggest_int("land_min_days_left", 0, 20),
        "animal_purchase_last_day": trial.suggest_int("animal_purchase_last_day", 5, 27),
        "crisis_backlog_ratio": trial.suggest_float("crisis_backlog_ratio", 0.5, 6.0),
        "startup_days": trial.suggest_int("startup_days", 0, 8),
        "buy_fertilizer": trial.suggest_categorical("buy_fertilizer", [True, False]),
        "hire_backlog_ratio": trial.suggest_float("hire_backlog_ratio", 0.3, 4.0),
        "diversification_weight": trial.suggest_float("diversification_weight", 0.0, 1.0),
        # Shadow prices -- see agent.py DEFAULT_CONFIG's lambda_labor/
        # lambda_land comment. New, unverified.
        "lambda_labor": trial.suggest_float("lambda_labor", 0.0, 15.0),
        "lambda_land": trial.suggest_float("lambda_land", 0.0, 80.0),
        # Cash-flow lookahead -- see agent.py DEFAULT_CONFIG's comment. New,
        # unverified.
        "income_lookahead_days": trial.suggest_int("income_lookahead_days", 0, 5),
        "income_discount": trial.suggest_float("income_discount", 0.0, 1.0),
        # Scripted opening window -- see agent.py DEFAULT_CONFIG's
        # opening_enabled comment. New, unverified; defaults to a no-op
        # (opening_enabled=False) so existing champion history stays valid
        # until this is actually searched.
        "opening_enabled": trial.suggest_categorical("opening_enabled", [True, False]),
        "opening_days": trial.suggest_int("opening_days", 0, 5),
        "opening_reserve_scale": trial.suggest_float("opening_reserve_scale", 0.0, 1.0),
        "opening_hires_day0": trial.suggest_int("opening_hires_day0", 0, 8),
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
        # Item-holding crash-sell mechanic, replacing the removed
        # relative-wealth CPPI risk scale (see agent.py DEFAULT_CONFIG's
        # crash_sell_enabled comment) -- searched as a togglable bool, same
        # as opening_enabled, since it's a genuinely new/unverified
        # mechanism rather than one already known to help.
        "crash_sell_enabled": trial.suggest_categorical("crash_sell_enabled", [True, False]),
        # Relative-wealth mechanism, restored 2026-08-22 after a prior session
        # deleted it despite it being the highest-importance parameter of the
        # v14 search (0.204). Searched as a toggle + 4 shape knobs so the
        # search decides whether it earns its place, rather than either of us
        # assuming from one run.
        "relative_wealth_enabled": trial.suggest_categorical("relative_wealth_enabled", [True, False]),
        "wealth_margin_scale": trial.suggest_float("wealth_margin_scale", 500.0, 10000.0),
        "risk_sensitivity": trial.suggest_float("risk_sensitivity", 0.0, 2.0),
        "risk_scale_min": trial.suggest_float("risk_scale_min", 0.7, 1.5),
        "risk_scale_max": trial.suggest_float("risk_scale_max", 1.0, 4.0),
        "crash_sell_margin": trial.suggest_int("crash_sell_margin", 1, 20),
        "crash_sell_multiplier": trial.suggest_float("crash_sell_multiplier", 1.0, 3.0),
        "crash_sell_chunk": trial.suggest_int("crash_sell_chunk", 10, 100),
        # Shed-distance tile-placement penalty -- see agent.py
        # DEFAULT_CONFIG's cluster_shed_weight comment. New, unverified.
        "cluster_shed_weight": trial.suggest_float("cluster_shed_weight", 0.0, 2.0),
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


def play_episode(agent_a, agent_b, episode_steps=720, seed=None):
    config = {"episodeSteps": episode_steps}
    if seed is not None:
        config["seed"] = seed
    env = make_env("kaggriculture", configuration=config, debug=True)
    env.run([agent_a, agent_b])
    final = env.steps[-1]
    return final[0].reward, final[1].reward


DEFAULT_WIN_SCALE = 15000.0


def _squash(margin, win_scale):
    """Map a raw coin margin onto (-1, +1), saturating for blowouts.

    The competition's rating system is win/loss/tie ONLY -- the coin
    difference in a match does not affect the rating change at all. Scoring
    trials on raw margin therefore pointed most of the search's optimization
    pressure at the wrong thing: kawa (~-142k) and prvsiyan (~-97k) dominate
    the pool average by an order of magnitude over the genuinely contestable
    near-tier matchups (+-6k..25k), so "improve the average margin" largely
    meant "lose slightly less lopsidedly to opponents we cannot beat", which
    earns exactly zero rating. Squashing fixes the incentive: -140k and -120k
    both saturate to ~-1 (correctly worth the same -- both are just a loss),
    while margins near the win/loss boundary keep a usable gradient.

    tanh rather than a hard win/loss indicator because at 2 episodes per
    opponent a 3-valued signal is far too noisy for TPE to learn from, and
    it would give zero signal on a matchup where every episode loses (a
    candidate 500 coins from flipping kawa would score identically to one
    500k away). win_scale sets the width of the contestable band.
    """
    return math.tanh(margin / win_scale)


def _risk_adjusted_score(margins, risk_aversion=0.2, tail_quantile=0.25, tail_weight=0.3,
                         win_scale=None):
    """Risk-adjusted trial score. With `win_scale` set (the default path, see
    _squash), margins are squashed to win/loss-like units first; pass
    win_scale=None for the legacy raw-margin behavior."""
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
    """Average reward margin (candidate - opponent) across the opponent pool.
    Reports the running average to `trial` after each episode for pruning."""
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
                    margins,
                    risk_aversion=risk_aversion,
                    tail_quantile=tail_quantile,
                    tail_weight=tail_weight,
                    win_scale=win_scale,
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
        win_scale=win_scale,
    )


def make_objective(opponent_names, episodes_per_opponent, champion_config,
                   risk_aversion=0.2, tail_quantile=0.25, tail_weight=0.3,
                   win_scale=DEFAULT_WIN_SCALE, seeded=True):
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
            win_scale=win_scale,
            seeded=seeded,
        )
    return objective


def verify_candidate(candidate_config, champion_config, episodes, seeded=True):
    """Dedicated candidate-vs-champion verification, independent of the
    search's own (noisy, mixed-opponent-pool) trial value. Promotion should
    never be decided by the same 6-episode number that picked the trial --
    that's exactly the kind of thin sample the search's pruner is built to
    distrust mid-search, so the final promotion gate shouldn't trust it
    either. Returns (wins, losses, ties, avg_margin)."""
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


# Opponents held out of every search's own --opponents pool, used ONLY by the
# promotion gate below. Rationale (see docs/tests/LOG.md, the v11 revert):
# verify_candidate alone tests candidate-vs-champion in self-play and nothing
# else, so a config that beats the specific current champion while genuinely
# regressing against the wider field clears the gate anyway -- which is
# exactly what happened to v11 (6W-4L vs champion, +1942.7 margin, promoted,
# then found to have doubled its deficit vs chaitanyajamble and lost cleanly
# to ektarr, and was reverted). Holding a fixed set out of the search entirely
# is what makes this an out-of-sample check rather than more of the same
# signal the search already optimized against. Independently corroborated by
# the Halite IV 4th-place solution's evolutionary setup (input/HaliteIV_0Zeta,
# evolutionary_optimization.py), which scores every genome against a standing
# pool of baseline bots plus retained older genomes rather than against the
# current best alone.
# ROTATIONS: We rotate which near-tier opponents are held out across runs.
# A fixed holdout allowed the search to quietly specialize against the 
# opponents in its pool while drifting on the ones held out. Rotating makes
# specialization much harder to sustain.
#
# Coverage verification: All 7 near-tier opponents (chaitanyajamble, daisy023,
# kiykhoi, rajan1673, umutdorukztrk, ektarr, nagatakengo) are held out at 
# least once across these 3 rotations. Each rotation holds out 2-3 near-tier 
# opponents, leaving at least 4 available for the search pool to provide 
# sufficient gradient.
#
# kawa is ALWAYS included in every rotation as a structural-collapse guard. 
# It is unwinnable, so it can never be a promotion target, but a candidate 
# that collapses further against it has broken something structural, which is
# worth catching.
HOLDOUT_ROTATIONS = [
    ["ektarr", "nagatakengo", "kawa"],
    ["chaitanyajamble", "daisy023", "kiykhoi", "kawa"],
    ["rajan1673", "umutdorukztrk", "kawa"],
]

# Measured squashed values vs the current champion (see docs/tests/LOG.md):
# chaitanyajamble -0.42, ektarr +0.20, daisy023 +0.52, nagatakengo +0.66,
# kiykhoi +0.85, rajan1673 +0.87, umutdorukztrk +0.95. Only opponents whose
# value sits well inside the band carry real gradient -- a pool of matchups we
# already win lopsidedly is nearly as uninformative as a pool of ones we lose
# lopsidedly, which is what flattened the v12 landscape.
# chaitanyajamble (the only near-tier opponent we LOSE to) and ektarr (the only
# roughly EVEN one) are therefore the two most valuable opponents to have in a
# pool. The first draft of these rotations held out BOTH in rotation 0 -- the
# default -- leaving an all-winning pool. Invariant below prevents that
# recurring: every rotation must leave at least one of the two in the pool.
_GRADIENT_ANCHORS = {"chaitanyajamble", "ektarr"}
for _i, _rot in enumerate(HOLDOUT_ROTATIONS):
    assert _GRADIENT_ANCHORS - set(_rot), (
        f"holdout rotation {_i} holds out every gradient anchor {_GRADIENT_ANCHORS}; "
        "the resulting search pool would be all-winning and nearly flat -- keep at least one"
    )


def verify_holdout(candidate_config, champion_config, episodes, tolerance, active_holdout, seeded=True):
    """Out-of-sample regression check against the active holdout rotation.

    The question this answers is NOT "does the candidate beat these opponents"
    -- some of them (chaitanyajamble, ektarr) are losing matchups for the
    champion too, and demanding wins there would reject everything. It's
    "does the candidate do materially WORSE against them than the champion
    does", i.e. is the head-to-head gain a real improvement or just a
    reallocation that quietly gives up ground elsewhere.

    Both configs play the same opponents over the same episode count with the
    same seat alternation. A candidate fails if, for any holdout opponent, its
    average margin is worse than the champion's by more than `tolerance` times
    the size of the champion's own margin -- relative rather than absolute
    because margins differ by orders of magnitude across opponents (a ~-6k
    matchup vs a ~-100k one), so one absolute threshold can't serve both.

    Returns (passed, per_opponent) where per_opponent maps name ->
    (cand_wins, cand_margin, champ_wins, champ_margin, regressed)."""
    per_opponent = {}
    passed = True
    for name in active_holdout:
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
        # Wins are the real currency (rating is win/loss/tie only -- coin
        # difference never affects it), so losing episodes the champion won
        # is a regression outright, no tolerance. Below that, compare on
        # SQUASHED margin rather than raw: on a matchup both configs lose 0-4,
        # win counts give no signal at all, and raw margin would let a
        # meaningless -140k -> -120k shift on an unwinnable opponent mask a
        # real slip on a contestable one. Squashing puts every opponent on the
        # same win/loss-shaped scale first (see _squash).
        if cand_wins < champ_wins:
            regressed = True
        else:
            cand_s, champ_s = _squash(cand_margin, DEFAULT_WIN_SCALE), _squash(champ_margin, DEFAULT_WIN_SCALE)
            regressed = cand_s < champ_s - tolerance * abs(champ_s)
        per_opponent[name] = (cand_wins, cand_margin, champ_wins, champ_margin, regressed)
        if regressed:
            passed = False
    return passed, per_opponent


def _worker(study_name, storage, n_trials, opponent_names, episodes_per_opponent, champion_config,
            risk_aversion, tail_quantile, tail_weight, win_scale, seeded):
    study = optuna.load_study(study_name=study_name, storage=storage)
    study.optimize(
        make_objective(
            opponent_names,
            episodes_per_opponent,
            champion_config,
            risk_aversion=risk_aversion,
            tail_quantile=tail_quantile,
            tail_weight=tail_weight,
            win_scale=win_scale,
            seeded=seeded,
        ),
        n_trials=n_trials,
    )


def _log_outcome(study, opponent_names, episodes_per_opponent, n_trials, n_jobs, had_champion_before,
                  verification_episodes, wins, losses, ties, verify_margin, promoted,
                  active_holdout, rotation_idx, holdout_detail=None, holdout_passed=True):
    date = datetime.date.today().isoformat()
    best = study.best_trial
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    if holdout_detail:
        rows = "; ".join(
            f"{n}: cand {cw}W {cm:+.1f} vs champ {hw}W {hm:+.1f}{' **REGRESSED**' if reg else ''}"
            for n, (cw, cm, hw, hm, reg) in holdout_detail.items()
        )
        holdout_txt = f" **Out-of-sample holdout check ({'PASSED' if holdout_passed else 'FAILED'} on rotation {rotation_idx}): {rows}.**"
    else:
        holdout_txt = " Holdout check not run (candidate already rejected head-to-head, or --holdout-episodes 0)."
    entry = f"""## {date} — optimize.py: Optuna search over robust_agent config
- Command: `python src/optimize.py --n-trials {n_trials} --n-jobs {n_jobs} --opponents {','.join(opponent_names)} --holdout-rotation {rotation_idx}`
- Result: {n_complete} complete trials, {n_pruned} pruned. Search's own best trial margin (noisy, {episodes_per_opponent} episodes x {len(opponent_names)}-opponent pool): {best.value:.1f}. **Dedicated candidate-vs-champion verification ({verification_episodes} episodes): {wins}W-{losses}L-{ties}T, avg margin {verify_margin:.1f}.**{holdout_txt}
- Notes: opponent pool was {opponent_names}, champion going in was `{'best_config.json' if had_champion_before else 'DEFAULT_CONFIG (first run)'}`. Promotion now requires BOTH gates: the dedicated candidate-vs-champion verification (wins > losses) AND no material regression against the active held-out opponents {active_holdout} (rotation {rotation_idx}), which are excluded from every search pool (see HOLDOUT_ROTATIONS' comment in optimize.py and the v11 revert entry). {"Promoted to new champion (best_config.json updated)." if promoted else "Did NOT clear the bar -- best_config.json left unchanged."} Best trial params: `{json.dumps(best.params)}`.

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
    parser.add_argument("--holdout-rotation", type=int, default=0,
                        help="index of the holdout rotation to use")
    parser.add_argument(
        "--opponents",
        # Default is computed at runtime based on the active holdout rotation:
        # "all near-tier opponents NOT in the active holdout, plus champion".
        default="DEFAULT",
        help=f"comma-separated: {sorted(OPPONENT_REGISTRY)} or champion",
    )
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME,
                        help="Optuna study name. If using default, it will be suffixed with the rotation index.")
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--timeout", type=int, default=None, help="wall-clock budget in seconds, across all workers")
    parser.add_argument("--verification-episodes", type=int, default=10,
                         help="dedicated candidate-vs-champion episodes deciding promotion (independent of the search's own trial value)")
    parser.add_argument("--holdout-episodes", type=int, default=4,
                        help="episodes per held-out opponent for the out-of-sample regression check; 0 disables it")
    parser.add_argument("--holdout-tolerance", type=float, default=0.5,
                        help="how much worse than the champion a candidate may be on a held-out opponent, as a fraction of the champion's own margin, before promotion is blocked")
    parser.add_argument("--risk-aversion", type=float, default=0.2,
                        help="penalty weight on per-episode margin variance (higher = safer configs)")
    parser.add_argument("--tail-quantile", type=float, default=0.25,
                        help="fraction of worst margins treated as downside tail")
    parser.add_argument("--tail-weight", type=float, default=0.3,
                        help="weight on worst-tail average margin (higher = more crash-averse)")
    parser.add_argument("--win-scale", type=float, default=DEFAULT_WIN_SCALE,
                        help="coin-margin width of the contestable band before the trial score saturates "
                             "toward a win/loss-like +-1 (see _squash). Lower = closer to pure win/loss. "
                             "Pass 0 to disable squashing and score raw margins (legacy behavior).")
    parser.add_argument("--seeded", action=argparse.BooleanOptionalAction, default=True,
                        help="use paired seeded evaluation to cancel world variance (default: ON)")
    args = parser.parse_args()

    if args.win_scale <= 0:
        args.win_scale = None  # legacy raw-margin scoring
        
    rotation_idx = args.holdout_rotation % len(HOLDOUT_ROTATIONS)
    active_holdout = HOLDOUT_ROTATIONS[rotation_idx]

    if args.opponents == "DEFAULT":
        # Real near-tier opponents, plus the two self-variants that fill the
        # emptiest part of the gradient band (near zero / negative, where the
        # real pool is thinnest: only chaitanyajamble -0.42 and ektarr +0.20
        # sit below +0.5). Variants stay a minority of the pool on purpose --
        # see their import comment for the echo-chamber caveat. They are NOT
        # eligible for the holdout gate, which stays anchored on real agents.
        all_near_tier = ["chaitanyajamble", "daisy023", "kiykhoi", "rajan1673", "umutdorukztrk", "ektarr", "nagatakengo",
                         "selfvar_no_expand", "selfvar_panic_seller"]
        default_opponents = [op for op in all_near_tier if op not in active_holdout] + ["champion"]
        opponent_names = default_opponents
    else:
        opponent_names = [o.strip() for o in args.opponents.split(",") if o.strip()]

    # A holdout opponent inside the search pool silently destroys the whole
    # point of the second gate -- the candidate would have been optimized
    # against it, so "no regression there" stops being out-of-sample
    # evidence. Refuse rather than warn: this is easy to do by accident (the
    # v6/v7 searches deliberately had these four IN the pool) and the failure
    # is invisible in the output if it's allowed through.
    leaked = sorted(set(opponent_names) & set(active_holdout))
    if leaked and args.holdout_episodes > 0:
        parser.error(
            f"opponents {leaked} are held out for the promotion gate (active holdout) and cannot "
            f"also be in the search pool. Either drop them from --opponents, or pass "
            f"--holdout-episodes 0 to deliberately disable the out-of-sample gate."
        )
        
    # Study separation: mixing rotations in one study would corrupt MedianPruner 
    # comparisons since each rotation is a DIFFERENT objective (different opponents 
    # in the pool).
    if args.study_name == DEFAULT_STUDY_NAME:
        args.study_name = f"{args.study_name}_rot{rotation_idx}"
    had_champion_before = BEST_CONFIG_PATH.exists()
    champion_config = load_champion_config()

    optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        # n_warmup_steps was 1 -- a trial could be pruned after its FIRST of 16
        # episodes (len(opponents)*episodes_per_opponent), a single opponent
        # single episode compared against the running median. Confirmed to
        # starve any delayed-payoff strategy: animal_enabled=True scored worse
        # on average at step 0 (upfront cash cost, no payoff yet), got pruned
        # disproportionately early, and TPE -- which learns from pruned trials
        # too -- all but stopped sampling it: 6% of all 2000 samples in the
        # v16 run, cascading to 1.5% of completions. Not "the search rejected
        # animals" -- the search never fairly evaluated them. Raised to give
        # every trial a full pass over the opponent pool before any pruning
        # decision, so an investment that pays off by opponent 3 isn't killed
        # for looking weak against opponent 1.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=len(opponent_names) * args.episodes_per_opponent,
        ),
        load_if_exists=True,
    )

    n_jobs = max(1, args.n_jobs)
    per_worker = [args.n_trials // n_jobs] * n_jobs
    for i in range(args.n_trials % n_jobs):
        per_worker[i] += 1

    print(f"Running {args.n_trials} trials across {n_jobs} worker process(es), "
          f"opponents={opponent_names}, episodes_per_opponent={args.episodes_per_opponent}")
    print(f"Active holdout rotation: {rotation_idx} (held out: {active_holdout})")
    print(f"Study name: {args.study_name}")

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
            args.win_scale,
            args.seeded,
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
                    args.win_scale,
                    args.seeded,
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
    wins, losses, ties, verify_margin = verify_candidate(merged, champion_config, args.verification_episodes, seeded=args.seeded)
    print(f"Verification: {wins}W-{losses}L-{ties}T, avg margin {verify_margin:.1f}")

    # Promotion gate is this dedicated head-to-head, not the search's own
    # trial value -- that value came from a 6-episode mixed-opponent-pool
    # average (the full opponent pool + champion combined), which can
    # look positive even if the candidate barely edges or loses to the
    # champion specifically. Require strictly more wins than losses.
    beat_champion = wins > losses

    # Second, independent gate: even a genuine head-to-head win doesn't
    # promote if it came at the cost of the wider field (see
    # HOLDOUT_OPPONENTS' comment and the v11 revert in docs/tests/LOG.md).
    # Only run it when the first gate passed -- it costs 2 * holdout_episodes
    # * len(HOLDOUT_OPPONENTS) episodes, pure waste on a candidate already
    # rejected.
    holdout_passed, holdout_detail = True, {}
    if beat_champion and args.holdout_episodes > 0:
        print(f"\nRunning out-of-sample holdout check: {args.holdout_episodes} episodes each vs {active_holdout} (candidate and champion)...")
        holdout_passed, holdout_detail = verify_holdout(
            merged, champion_config, args.holdout_episodes, args.holdout_tolerance, active_holdout, seeded=args.seeded
        )
        for name, (c_w, c_m, h_w, h_m, regressed) in holdout_detail.items():
            flag = "REGRESSED" if regressed else "ok"
            print(f"  {name:18s} candidate {c_w}W avg {c_m:+10.1f} | champion {h_w}W avg {h_m:+10.1f}  [{flag}]")

    promoted = beat_champion and holdout_passed
    if promoted:
        BEST_CONFIG_PATH.write_text(json.dumps(merged, indent=2))
        print(f"New champion written to {BEST_CONFIG_PATH}")
    elif not beat_champion:
        print("Candidate did not beat the champion head-to-head (wins <= losses) -- best_config.json left unchanged.")
    else:
        regressed_on = [n for n, v in holdout_detail.items() if v[4]]
        print(f"Candidate beat the champion head-to-head but REGRESSED against held-out opponent(s) {regressed_on} "
              f"-- best_config.json left unchanged.")

    _log_outcome(study, opponent_names, args.episodes_per_opponent, args.n_trials, n_jobs, had_champion_before,
                 args.verification_episodes, wins, losses, ties, verify_margin, promoted,
                 active_holdout, rotation_idx, holdout_detail, holdout_passed)


if __name__ == "__main__":
    main()
