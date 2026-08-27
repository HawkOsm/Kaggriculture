"""Configuration defaults for the pilkwang_style_agent."""

# Phase Thresholds
LIQUIDATION_TURNS = 22 # L171

# Herd Schedule Cutoffs
HERD_EXPANSION_DAY = 7 # L156
HERD_FINAL_DAY = 11 # L157
ANIMAL_PURCHASE_LAST_DAY = 18 # L158

# Herd Sizes
CORE_HERD = 4 # L153
MID_HERD = 11 # L154
TARGET_HERD = 15 # L155

# Land Expansion
LAND_OPEN_DAYS = (5, 9) # L176
MAX_EXTRA_LAND = 2 # L169

# Priority Scaling
TRAVEL_COST = 8.0 # L173
PRIORITY_BONUS = { # L177-185
    -1: 120_000.0,
    0: 100_000.0,
    1: 1_500.0,
    2: 750.0,
    3: 250.0,
    4: 0.0,
    5: -100.0,
}

# Reserve Cash/Fractions
CASH_RESERVE = 250 # L170
RESERVE_FRACTION = { # L128-138
    "WHEAT": 0.68,
    "CARROT": 0.55,
    "TOMATO": 0.50,
    "STRAWBERRY": 0.48,
    "MELON": 0.58,
    "EGG": 0.65,
    "MILK": 0.42,
    "WOOL": 0.40,
    "FERTILIZER": 0.18,
}

def get_config(config, key, default):
    """Retrieve config, similar to original _cfg."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
