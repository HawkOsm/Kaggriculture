"""Optuna search space for `adaptive_agent`'s tunable surface (see PLAN.md's
structural-vs-tunable split, `.claude/scratch/pilkwang_rebuild/plan/PLAN.md`)
-- phase thresholds, herd schedule, land expansion timing, dispatch priority
scaling, and cash-reserve fractions. The phase machine, spatial zoning,
dispatch algorithm, and starvation-escape check themselves stay structural,
not searched. Validated during the original 150-trial pilkwang-rebuild search
(see docs/tests/LOG.md, 2026-08-27) -- ported here unchanged.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEST_CONFIG_PATH = HERE.parent / "adaptive_best_config.json"


def load_champion_config():
    if BEST_CONFIG_PATH.exists():
        return json.loads(BEST_CONFIG_PATH.read_text())
    from adaptive_agent import config as _default_config
    return {
        "LIQUIDATION_TURNS": _default_config.LIQUIDATION_TURNS,
        "HERD_EXPANSION_DAY": _default_config.HERD_EXPANSION_DAY,
        "HERD_FINAL_DAY": _default_config.HERD_FINAL_DAY,
        "ANIMAL_PURCHASE_LAST_DAY": _default_config.ANIMAL_PURCHASE_LAST_DAY,
        "CORE_HERD": _default_config.CORE_HERD,
        "MID_HERD": _default_config.MID_HERD,
        "TARGET_HERD": _default_config.TARGET_HERD,
        "LAND_OPEN_DAYS": _default_config.LAND_OPEN_DAYS,
        "MAX_EXTRA_LAND": _default_config.MAX_EXTRA_LAND,
        "TRAVEL_COST": _default_config.TRAVEL_COST,
        "PRIORITY_BONUS": dict(_default_config.PRIORITY_BONUS),
        "CASH_RESERVE": _default_config.CASH_RESERVE,
        "RESERVE_FRACTION": dict(_default_config.RESERVE_FRACTION),
    }


def sample_config(trial):
    cfg = {}

    # Phase Thresholds
    cfg["LIQUIDATION_TURNS"] = trial.suggest_int("LIQUIDATION_TURNS", 15, 30)

    # Herd Schedule Cutoffs
    cfg["HERD_EXPANSION_DAY"] = trial.suggest_int("HERD_EXPANSION_DAY", 3, 10)
    cfg["HERD_FINAL_DAY"] = trial.suggest_int("HERD_FINAL_DAY", cfg["HERD_EXPANSION_DAY"] + 1, 15)
    cfg["ANIMAL_PURCHASE_LAST_DAY"] = trial.suggest_int(
        "ANIMAL_PURCHASE_LAST_DAY", cfg["HERD_FINAL_DAY"] + 1, 25
    )

    # Herd Sizes
    cfg["CORE_HERD"] = trial.suggest_int("CORE_HERD", 2, 8)
    cfg["MID_HERD"] = trial.suggest_int("MID_HERD", cfg["CORE_HERD"] + 1, 15)
    cfg["TARGET_HERD"] = trial.suggest_int("TARGET_HERD", cfg["MID_HERD"] + 1, 20)

    # Land Expansion -- represented as two ints, reassembled into the tuple
    # the agent actually reads.
    land_day_1 = trial.suggest_int("LAND_OPEN_DAY_1", 2, 8)
    land_day_2 = trial.suggest_int("LAND_OPEN_DAY_2", land_day_1 + 1, 15)
    cfg["LAND_OPEN_DAYS"] = (land_day_1, land_day_2)
    cfg["MAX_EXTRA_LAND"] = trial.suggest_int("MAX_EXTRA_LAND", 0, 2)

    # Dispatch priority scaling -- score = PRIORITY_BONUS[priority] + value -
    # TRAVEL_COST * distance (see adaptive_agent/dispatch.py's _unit_actions).
    cfg["TRAVEL_COST"] = trial.suggest_float("TRAVEL_COST", 2.0, 15.0)
    cfg["PRIORITY_BONUS"] = {
        -1: trial.suggest_float("PRIORITY_BONUS_MINUS_1", 50000.0, 200000.0),
        0: trial.suggest_float("PRIORITY_BONUS_0", 50000.0, 150000.0),
        1: trial.suggest_float("PRIORITY_BONUS_1", 500.0, 5000.0),
        2: trial.suggest_float("PRIORITY_BONUS_2", 200.0, 2000.0),
        3: trial.suggest_float("PRIORITY_BONUS_3", 50.0, 500.0),
        4: trial.suggest_float("PRIORITY_BONUS_4", -100.0, 100.0),
        5: trial.suggest_float("PRIORITY_BONUS_5", -500.0, 0.0),
    }

    # Cash reserves
    cfg["CASH_RESERVE"] = trial.suggest_int("CASH_RESERVE", 50, 500)
    cfg["RESERVE_FRACTION"] = {
        "WHEAT": trial.suggest_float("RESERVE_WHEAT", 0.4, 0.9),
        "CARROT": trial.suggest_float("RESERVE_CARROT", 0.3, 0.8),
        "TOMATO": trial.suggest_float("RESERVE_TOMATO", 0.3, 0.8),
        "STRAWBERRY": trial.suggest_float("RESERVE_STRAWBERRY", 0.3, 0.8),
        "MELON": trial.suggest_float("RESERVE_MELON", 0.3, 0.8),
        "EGG": trial.suggest_float("RESERVE_EGG", 0.4, 0.9),
        "MILK": trial.suggest_float("RESERVE_MILK", 0.2, 0.7),
        "WOOL": trial.suggest_float("RESERVE_WOOL", 0.2, 0.7),
        "FERTILIZER": trial.suggest_float("RESERVE_FERTILIZER", 0.0, 0.5),
    }

    return cfg
