"""Self-variant opponent: NO ANIMAL ECONOMY.

Degradation: animal_enabled=False, max_structures=0
  -> Agent skips all pasture/coop building and animal purchases entirely,
     relying purely on crop revenue.

Measured against champion (2026-08-22 best_config.json):
  avg_margin = +3516  (champion has a mild edge)
  squashed   = +0.2302
  champion_wins = 1/2 episodes

Surprisingly competitive -- the saved cash from not buying animals gets
reinvested into crops and land, and on some seeds the pure-crop economy
actually outperforms the champion's mixed strategy.

Frozen snapshot: config hardcoded below, does NOT read best_config.json.
Source champion: best_config.json as of 2026-08-22T16:54+03:00.
"""

import sys
sys.path.insert(0, "/home/osm/Projects/Kaggriculture/src")
from agent import make_agent

# Full champion config (DEFAULT_CONFIG merged with best_config.json),
# then degradation applied. Hardcoded -- never reads best_config.json.
_CONFIG = {
    # --- DEGRADATION: no animal economy ---
    'animal_enabled': False,
    # --- end degradation ---
    'animal_reserve_multiple': 2.595799954753388,
    'broke_phase_days_frac': 0.0,
    'broke_phase_reserve_scale': 1.0,
    'buy_fertilizer': False,
    'cash_scale': 3900,
    'crops': ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON'],
    'diversification_weight': 0.46489068096515895,
    'enable_coop': False,
    'hire_backlog_ratio': 2.0586600581339973,
    'hire_money_floor': 60,
    'hire_reserve_multiple': 1.2868527815789883,
    'income_discount': 0.0,
    'income_lookahead_days': 0,
    'lambda_labor': 0.0,
    'lambda_land': 0.0,
    'land_reserve_multiple': 1.5,
    'land_startup_days': 8,
    'land_utilization_threshold': 0.7844950137160052,
    'max_hires_per_day': 6,
    'max_sell_chunk': 8,
    # --- DEGRADATION: no structures ---
    'max_structures': 0,
    # --- end degradation ---
    'money_reserve': 420,
    'opening_days': 3,
    'opening_enabled': False,
    'opening_hires_day0': 5,
    'opening_reserve_scale': 0.15,
    'opponent_awareness_enabled': True,
    'opponent_concentration_sensitivity': 0.6634759244356456,
    'opponent_incoming_threshold': 9,
    'opponent_lookahead_days': 3,
    'opponent_race_discount': 0.9655340065792685,
    'pasture_target_ratio': 0.6139913922913309,
    'priority_weight_care': 34.998862122942356,
    'priority_weight_collect_fertilizer': 68.0900009691404,
    'priority_weight_empty_build': 15.226715136687174,
    'priority_weight_empty_coop_place': 6.420598313711167,
    'priority_weight_empty_pasture_place': 14.14611078775056,
    'priority_weight_empty_plant': 65.01973167073385,
    'priority_weight_feed': 28.43172059455725,
    'priority_weight_fertilize': 20.728901657756897,
    'priority_weight_harvest': 93.12309658795056,
    'priority_weight_water': 12.008227925293447,
    'priority_weight_weeds': 38.60857710296399,
    'relative_wealth_enabled': True,
    'risk_scale_max': 2.3531408542357126,
    'risk_scale_min': 0.8862056218250035,
    'risk_sensitivity': 0.8922000048258688,
    'season_days': 30,
    'seed_money_floor': 25,
    'sell_backlog_multiple': 5.073588763331714,
    'sell_fraction_base': 0.46288312129404735,
    'sell_fraction_cash_weight': -0.06388422078868353,
    'sell_fraction_day_weight': 0.052944245483376506,
    'sell_fraction_max': 0.9,
    'sell_fraction_min': 0.15,
    'startup_days': 2,
    'wealth_margin_scale': 2925.410674583168,
    'wind_down_days': 3,
}

selfvar_no_animals_agent = make_agent(_CONFIG)
