"""nagatakengo_agent: local benchmark opponent, NOT original work.

Pulled from a public Kaggle notebook (nk, "Kaggriculture"):
https://www.kaggle.com/code/nagatakengo/kaggriculture
Extracted from its `%%writefile main.py` cell on 2026-08-22. Near our own
current live rating tier (rank ~4567/5684, score 428.5 at pull time) --
deliberately NOT top-tier, added to give a locally testable opponent
representative of what the live ladder actually matches us against right
now. Comments are verbatim Japanese from the source notebook. See
CREDITS.md for full attribution.
"""
import random


def get_market_price(obs, product):
    """現在価格を取得する。"""
    return obs["market"]["prices"][product]


def score(input):
    """max_depth=3 のDecisionTreeモデル。"""

    if input[3] <= 10096.5:
        if input[0] <= 257.5:
            if input[2] <= 265.0:
                var0 = 262.30210213139543
            else:
                var0 = 269.14311998948614
        else:
            if input[2] <= 180.0:
                var0 = 169.52286374133948
            else:
                var0 = 192.71375464684016
    else:
        if input[0] <= 497.5:
            if input[3] <= 10127.5:
                var0 = 95.84500378501136
            else:
                var0 = 82.26066931619805
        else:
            if input[3] <= 10154.5:
                var0 = 19.437829958238122
            else:
                var0 = 6.73007806147341

    return var0


def should_sell_melon(
    melon_price,
    melon_stock,
    melon_in_shed,
    day,
    step,
):
    """MELONをSELLするかHOLDするか判断する。"""

    # 終盤は価格に関係なく売却
    if step >= 710:
        return True

    # 12ターン以内の最高価格を予測
    pred_price = score([
        step,
        melon_in_shed,
        melon_price,
        melon_stock,
        0,
        0,
        0,
    ])

    # 現在価格の方が予測最高価格以上なら売却
    if melon_price >= pred_price:
        return True

    return False


def get_harvest_age(crop_name):
    """現在の戦略で使う収穫開始日を返す。"""
    if crop_name == "MELON":
        return 10

    return 2


def get_tile_action(tile, day):
    """足元のタイルで今すぐ行う作業を返す。"""
    if not isinstance(tile, dict):
        return None

    if tile.get("kind") == "WEED":
        return ["DIG"]

    if tile.get("kind") != "PLANT":
        return None

    crop_name = tile.get("crop", "WHEAT")
    crop_age = day - tile.get("planted_day", day)
    harvest_age = get_harvest_age(crop_name)

    if crop_age >= harvest_age:
        return ["HARVEST"]

    if not tile.get("watered_today", True):
        return ["WATER"]

    return None


def step_toward(fx, fy, tx, ty, tiles):
    """
    目的地に近づく方向へ1マス移動する。
    目的地へ直接進めない場合は、移動可能な方向を選ぶ。
    """
    max_y = len(tiles) - 1
    max_x = len(tiles[0]) - 1

    candidates = []

    if fx < tx and fx < max_x and tiles[fy][fx + 1] != "LOCKED":
        candidates.append("EAST")
    elif fx > tx and fx > 0 and tiles[fy][fx - 1] != "LOCKED":
        candidates.append("WEST")

    if fy < ty and fy < max_y and tiles[fy + 1][fx] != "LOCKED":
        candidates.append("SOUTH")
    elif fy > ty and fy > 0 and tiles[fy - 1][fx] != "LOCKED":
        candidates.append("NORTH")

    if candidates:
        return random.choice(candidates)

    valid_dirs = []

    if fy > 0 and tiles[fy - 1][fx] != "LOCKED":
        valid_dirs.append("NORTH")

    if fy < max_y and tiles[fy + 1][fx] != "LOCKED":
        valid_dirs.append("SOUTH")

    if fx < max_x and tiles[fy][fx + 1] != "LOCKED":
        valid_dirs.append("EAST")

    if fx > 0 and tiles[fy][fx - 1] != "LOCKED":
        valid_dirs.append("WEST")

    if valid_dirs:
        return random.choice(valid_dirs)

    return "PASS"


def find_target_tile(
    tiles,
    fx,
    fy,
    has_seeds,
    day,
    excluded_coords=None,
):
    """
    現在位置から各作業候補を評価し、
    最もスコアの高いタイルの座標を返す。
    """
    if excluded_coords is None:
        excluded_coords = set()

    best_target = None
    best_score = -9999

    for y in range(len(tiles)):
        for x in range(len(tiles[0])):

            if (x, y) in excluded_coords:
                continue

            tile = tiles[y][x]

            if tile == "LOCKED":
                continue

            base_score = None

            # 雑草
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                if day >= 27:
                    continue

                base_score = 0

            # 植物
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop_name = tile.get("crop", "WHEAT")
                crop_age = day - tile.get("planted_day", day)
                harvest_age = get_harvest_age(crop_name)

                if crop_age >= harvest_age:
                    if crop_name == "MELON":
                        base_score = 250
                    else:
                        base_score = 25

                elif not tile.get("watered_today", True):
                    base_score = 50

            # 空き地
            elif tile is None and has_seeds:
                if day >= 27:
                    continue

                base_score = 50

            if base_score is None:
                continue

            movement_cost = abs(x - fx) + abs(y - fy)
            task_score = base_score - movement_cost

            if task_score > best_score:
                best_score = task_score
                best_target = (x, y)

    return best_target


def build_market_actions(
    me,
    seeds,
    shed,
    day,
    step,
    melon_price,
    melon_stock,
):
    """現在の市場売買・雇用・土地購入ルールから注文一覧を作る。"""

    market = []

    money = me.get("money", 0)
    current_hands = me.get("hands", [])
    unlocked_quads = me.get(
        "unlocked_quadrants",
        ["NW"],
    )

    wheat_seeds = seeds.get("WHEAT", 0)
    melon_seeds = seeds.get("MELON", 0)

    wheat_in_shed = shed.get("WHEAT", 0)
    melon_in_shed = shed.get("MELON", 0)

    # -------------------------
    # 種を購入
    # -------------------------

    if money >= 500:
        if melon_seeds == 0:
            market.append(
                ["BUY_SEED", "MELON", 10]
            )


    if wheat_seeds == 0 and money >= 10:
        market.append(
            ["BUY_SEED", "WHEAT", 3]
        )

    # -------------------------
    # WHEAT売却
    # -------------------------

    if wheat_in_shed > 0:
        market.append(
            ["SELL", "WHEAT", wheat_in_shed]
        )

    # -------------------------
    # MELON売却
    # -------------------------

    if melon_in_shed > 0:
        if should_sell_melon(
            melon_price,
            melon_stock,
            melon_in_shed,
            day,
            step,
        ):
            market.append(
                ["SELL", "MELON", melon_in_shed]
            )

    # -------------------------
    # 雇用
    # -------------------------

    total_people = 1 + len(current_hands)

    if total_people < 5 and money >= 200:
        market.append(
            ["HIRE"]
        )

    # -------------------------
    # 土地購入
    # -------------------------

    if len(unlocked_quads) < 2 and money >= 1000:
        market.insert(
            0,
            ["BUY_LAND"],
        )

    return market


def agent(obs, config=None):

    # ========================================
    # 状態取得
    # ========================================

    player = obs["player"]

    me = obs["farms"][player]
    private = obs["private"]

    tiles = me["tiles"]

    fx, fy = me["farmer"]
    farmer_tile = tiles[fy][fx]

    seeds = private.get("seeds", {})
    shed = private.get("shed", {})

    day = obs.get("day", 0)
    step = obs.get("step", 0)

    current_hands = me.get("hands", [])

    # 現在のMELON価格
    melon_price = get_market_price(
        obs,
        "MELON",
    )

    # 現在のMELON市場在庫
    melon_stock = obs["market"]["inventory"]["MELON"]

    wheat_seeds = seeds.get("WHEAT", 0)
    melon_seeds = seeds.get("MELON", 0)

    has_seeds = (
        wheat_seeds > 0
        or melon_seeds > 0
    )

    # ========================================
    # 市場
    # ========================================

    market = build_market_actions(
        me,
        seeds,
        shed,
        day,
        step,
        melon_price,
        melon_stock,
    )

    # ========================================
    # メイン農家
    # ========================================

    farmer_action = None

    if farmer_tile is None:

        if melon_seeds > 0:
            farmer_action = [
                "PLANT",
                "MELON",
            ]

        elif wheat_seeds > 0:
            farmer_action = [
                "PLANT",
                "WHEAT",
            ]

    else:
        farmer_action = get_tile_action(
            farmer_tile,
            day,
        )

    # ========================================
    # メイン農家の移動
    # ========================================

    claimed_targets = set()

    if farmer_action is None:

        target = find_target_tile(
            tiles,
            fx,
            fy,
            has_seeds,
            day,
            claimed_targets,
        )

        if target is not None:

            claimed_targets.add(target)

            move_dir = step_toward(
                fx,
                fy,
                target[0],
                target[1],
                tiles,
            )

            farmer_action = [move_dir]

        else:
            farmer_action = ["PASS"]

    else:
        claimed_targets.add(
            (fx, fy)
        )

    # ========================================
    # 作業員
    # ========================================

    hands_actions = []

    for hand in current_hands:

        hx, hy = hand
        hand_tile = tiles[hy][hx]

        hand_action = get_tile_action(
            hand_tile,
            day,
        )

        if hand_action is None:

            target = find_target_tile(
                tiles,
                hx,
                hy,
                has_seeds,
                day,
                claimed_targets,
            )

            if target is not None:

                claimed_targets.add(target)

                move_dir = step_toward(
                    hx,
                    hy,
                    target[0],
                    target[1],
                    tiles,
                )

                hand_action = [move_dir]

            else:
                hand_action = ["PASS"]

        else:
            claimed_targets.add(
                (hx, hy)
            )

        hands_actions.append(
            hand_action
        )

    # ========================================
    # 出力
    # ========================================

    return {
        "farmer": farmer_action,
        "hands": hands_actions,
        "market": market,
    }


nagatakengo_agent = agent
