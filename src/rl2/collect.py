import sys
import os
import json
import numpy as np
from pathlib import Path
from collections import Counter
from kaggle_environments import make

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "opponents"))
from run_match import load_agent
from farm_utils import shed_tiles

DEMOS = [
    "boatlee_v16_agent",
    "kaito_agent",
    "prvsiyan_frontier_agent",
    "cropdusta_97601003_agent",
    "rayk_c95_agent",
]

OPPONENTS = DEMOS + ["pilkwang_agent"]

VOCAB = [
    "PASS", "NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "PLANT", 
    "FEED", "CARE", "COLLECT_FERTILIZER", "DROP", "PICKUP", "BUILD_PASTURE", "BUILD_COOP"
]
VOCAB_MAP = {k: i for i, k in enumerate(VOCAB)}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
CROPS = ["MELON", "WHEAT", "CARROT", "STRAWBERRY", "TOMATO"]

def extract_features(obs, player_idx, unit_pos, is_farmer, inv):
    farm = obs.farms[player_idx]
    money = farm.money
    opp_money = obs.farms[1 - player_idx].money
    hands = farm.hands or []
    
    # 1. unit (x,y) normalized
    ux, uy = unit_pos
    ux_n, uy_n = ux / 9.0, uy / 9.0
    
    # 2. is_farmer
    isf = 1.0 if is_farmer else 0.0
    
    # 3. unit inventory
    carried = sum(inv.get(p, 0) for p in PRODUCTS)
    wheat = inv.get("WHEAT", 0)
    fert = inv.get("FERTILIZER", 0)
    an_carried = sum(inv.get(a, 0) for a in ANIMALS)
    # normalize by 10
    inv_feats = [carried/10.0, wheat/10.0, fert/10.0, an_carried/10.0]
    
    # 4. 5x5 window
    window = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            tx, ty = ux + dx, uy + dy
            if 0 <= tx < 10 and 0 <= ty < 10:
                tile = (farm.tiles or [])[ty][tx]
                if tile == "LOCKED":
                    # not own land
                    cell_feat = [0.0]*16 # is_own_land=0
                elif isinstance(tile, dict):
                    # own land
                    kind = tile.get("kind", "empty")
                    kind_onehot = [
                        kind == "empty", kind == "PLANT", kind == "WEED", 
                        kind == "COOP", kind == "PASTURE"
                    ]
                    has_animal = ("animal" in tile)
                    crop = tile.get("crop", "")
                    crop_onehot = [crop == c for c in CROPS]
                    watered = tile.get("watered_today", False)
                    consec_unwatered = tile.get("consecutive_unwatered", 0)
                    yield_units = tile.get("yield_units", 0)
                    ripe = (yield_units > 0)
                    
                    cell_feat = [1.0] + [float(k) for k in kind_onehot] + [float(has_animal)] + \
                                [float(c) for c in crop_onehot] + [float(watered), float(consec_unwatered), float(yield_units), float(ripe)]
                elif tile is None:
                    # empty own land
                    cell_feat = [1.0] + [1.0,0.0,0.0,0.0,0.0] + [0.0] + [0.0]*len(CROPS) + [0.0,0.0,0.0,0.0]
                else:
                    cell_feat = [0.0]*16
            else:
                cell_feat = [0.0]*16
            window.extend(cell_feat)
            
    # 5. globals
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    d_norm = day / 30.0
    h_norm = hour / 24.0
    phase_boot = 1.0 if hour < 8 else 0.0
    phase_mid = 1.0 if 8 <= hour <= 24 else 0.0 # hour goes up to 24? day is 24 hours. Wait, "mid 8-24 / end>24". Wait, 24 is max for hour? 
    phase_end = 1.0 if hour > 24 else 0.0
    own_m = money / 50000.0
    opp_m = opp_money / 50000.0
    hands_norm = len(hands) / 13.0
    
    stiles = shed_tiles(10)
    dist = min(abs(ux - sx) + abs(uy - sy) for sx, sy in stiles) / 20.0
    
    glob_feats = [d_norm, h_norm, phase_boot, phase_mid, phase_end, own_m, opp_m, hands_norm, dist]
    
    return [ux_n, uy_n, isf] + inv_feats + window + glob_feats

def run():
    out_dir = REPO_ROOT / ".claude/scratch/rl2"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    N = 40
    X = []
    y = []
    
    # Preload agents to avoid reloading
    loaded_agents = {name: load_agent(name) for name in OPPONENTS}
    
    for demo in DEMOS:
        demo_agent = loaded_agents[demo]
        opps = [o for o in OPPONENTS if o != demo] # demo might play against itself if we don't exclude, but it's 5 others. Wait, 5 others if we exclude demo, 5 + 1 - 1 = 5.
        for seed in range(N):
            opp_name = opps[seed % len(opps)]
            opp_agent = loaded_agents[opp_name]
            
            seat = seed % 2
            agents = [demo_agent, opp_agent] if seat == 0 else [opp_agent, demo_agent]
            
            env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
            env.run(agents)
            
            # extract
            # env.steps[t+1][player].action is the action taken by player reacting to env.steps[t]
            for t in range(len(env.steps) - 1):
                state = env.steps[t][0].observation
                action_state = env.steps[t+1][seat].action or {}
                
                farm = state.farms[seat]
                farmer_pos = farm.farmer or []
                hands_pos = farm.hands or []
                
                invs = state.private.inventories if state.private else []
                # private can be missing?
                if not hasattr(state, "private") or state.private is None:
                    continue
                
                farmer_action = (action_state.get("farmer") or ["PASS"])[0]
                if farmer_action == "PLANT": farmer_action = "PLANT" # ignore crop
                if farmer_pos:
                    f_feat = extract_features(state, seat, farmer_pos, True, invs[0] if len(invs) > 0 else {})
                    if farmer_action in VOCAB_MAP:
                        X.append(f_feat)
                        y.append(VOCAB_MAP[farmer_action])
                        
                hands_actions = action_state.get("hands") or []
                for i, hpos in enumerate(hands_pos):
                    h_action = (hands_actions[i] if i < len(hands_actions) else ["PASS"])[0]
                    if h_action == "PLANT": h_action = "PLANT"
                    f_feat = extract_features(state, seat, hpos, False, invs[i+1] if i+1 < len(invs) else {})
                    if h_action in VOCAB_MAP:
                        X.append(f_feat)
                        y.append(VOCAB_MAP[h_action])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    np.save(out_dir / "demo_X.npy", X)
    np.save(out_dir / "demo_y.npy", y)
    
    with open(out_dir / "feature_spec.json", "w") as f:
        json.dump({"dim": X.shape[1], "vocab": VOCAB}, f)
        
    print(f"Total transitions M: {len(X)}")
    counts = Counter(y)
    for i, c in counts.most_common():
        print(f"{VOCAB[i]}: {c}")
    print(f"Feature dim D: {X.shape[1]}")

if __name__ == "__main__":
    run()
