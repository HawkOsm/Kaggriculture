# Domain constants extracted from pilkwang_agent.py


CROPS = {
    "WHEAT": {
        "seed": 10,
        "first": 2,
        "max_day": 4,
        "max_yield": 6,
        "ongoing": False,
        "ripe": 4,
        "last_plant": 24,
    },
    "CARROT": {
        "seed": 20,
        "first": 2,
        "max_day": 3,
        "max_yield": 4,
        "ongoing": False,
        "ripe": 3,
        "last_plant": 25,
    },
    "TOMATO": {
        "seed": 50,
        "first": 8,
        "max_day": 8,
        "max_yield": 4,
        "ongoing": True,
        "interval": 1,
        "ripe": 8,
        "last_plant": 19,
    },
    "STRAWBERRY": {
        "seed": 100,
        "first": 10,
        "max_day": 10,
        "max_yield": 4,
        "ongoing": True,
        "interval": 2,
        "ripe": 10,
        "last_plant": 18,
    },
    "MELON": {
        "seed": 80,
        "first": 10,
        "max_day": 12,
        "max_yield": 6,
        "ongoing": False,
        "ripe": 10,
        "last_plant": 18,
    },
}

ANIMALS = {
    "GOOSE": {
        "cost": 300,
        "structure": "COOP",
        "product": "EGG",
        "first": 4,
        "interval": 1,
        "max_held": 4,
    },
    "COW": {
        "cost": 400,
        "structure": "PASTURE",
        "product": "MILK",
        "first": 8,
        "interval": 2,
        "max_held": 6,
    },
    "SHEEP": {
        "cost": 500,
        "structure": "PASTURE",
        "product": "WOOL",
        "first": 6,
        "interval": 3,
        "max_held": 6,
    },
}

PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)

MARKET = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}

SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

MOVES = (
    ("NORTH", 0, -1),
    ("WEST", -1, 0),
    ("SOUTH", 0, 1),
    ("EAST", 1, 0),
)

LAND_PRICES = (1000, 2000, 4000)

MARKET_I0 = 10000

TOTAL_DAYS = 30

MAX_MARKET_ORDERS = 10

MAX_HANDS = 12

CORE_HERD_SEQUENCE = ("COW", "COW", "COW", "SHEEP")

CORE_HERD = len(CORE_HERD_SEQUENCE)

MID_HERD = 11

TARGET_HERD = 15

HERD_EXPANSION_DAY = 7

HERD_FINAL_DAY = 11

ANIMAL_PURCHASE_LAST_DAY = 18

ANIMAL_SLOTS = {"NW": 4, "NE": 7, "SW": 4, "SE": 0}

CROP_MIX = {
    "NW": {"MELON": 10, "WHEAT": 4, "CARROT": 2},
    "NE": {"WHEAT": 4, "CARROT": 1},
    "SW": {"WHEAT": 4, "CARROT": 1},
    "SE": {"WHEAT": 5, "CARROT": 2},
}

MELON_TILES_MIN = 8

MELON_TILES_BASE = 10

MELON_TILES_MAX = 12

MAX_EXTRA_LAND = 2

CASH_RESERVE = 250

LIQUIDATION_TURNS = 22

SHED_CAPACITY = 100

TRAVEL_COST = 8.0

FEED_STOCK_DAYS = 3

SCHEDULE_WHEAT_CAP = 100

LAND_OPEN_DAYS = (5, 9)

PRIORITY_BONUS = {
    -1: 120_000.0,
    0: 100_000.0,
    1: 1_500.0,
    2: 750.0,
    3: 250.0,
    4: 0.0,
    5: -100.0,
}

HINGE_GAIN = 8.0

MAX_HANDS = 13
