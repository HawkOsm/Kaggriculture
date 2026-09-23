import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

HERE = Path("src/rl2").resolve()
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from adaptive_agent.agent import Dispatcher
from adaptive_agent.state import _role_plan
from rl2.collect import extract_features, VOCAB, VOCAB_MAP
from rl2.policy import BCPolicy

_model = None
_scaler_mean = None
_scaler_std = None
_dispatcher = None
_champion_config = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _init():
    global _model, _scaler_mean, _scaler_std, _dispatcher, _champion_config
    if _model is not None:
        return
        
    out_dir = REPO_ROOT / ".claude/scratch/rl2"
    
    with open(out_dir / "feature_spec.json", "r") as f:
        spec = json.load(f)
        
    _model = BCPolicy(in_dim=spec["dim"], out_dim=len(VOCAB)).to(_device)
    _model.load_state_dict(torch.load(out_dir / "bc_model.pt", map_location=_device))
    _model.eval()
    
    scaler = np.load(out_dir / "scaler.npz")
    _scaler_mean = scaler["mean"].astype(np.float32)
    _scaler_std = scaler["std"].astype(np.float32)
    
    config_path = REPO_ROOT / "src" / "adaptive_best_config.json"
    with open(config_path, "r") as f:
        _champion_config = json.load(f)
        
    _dispatcher = Dispatcher(_champion_config)

def check_legality(op, x, y, farm, inv):
    if op == "PASS": return True
    if op in ["NORTH", "SOUTH", "EAST", "WEST"]:
        # All directions are technically legal if they don't fall off board
        # But if they fall off board, the engine handles it?
        nx, ny = x, y
        if op == "NORTH": ny -= 1
        elif op == "SOUTH": ny += 1
        elif op == "EAST": nx += 1
        elif op == "WEST": nx -= 1
        if not (0 <= nx < 10 and 0 <= ny < 10): return False
        return True
        
    tile = farm.tiles[y][x] if (0 <= x < 10 and 0 <= y < 10) else None
    
    if op == "WATER":
        if tile is None or tile == "LOCKED": return False
        if not isinstance(tile, dict): return False
        if tile.get("kind") != "PLANT": return False
        return True
        
    if op == "HARVEST":
        if tile is None or tile == "LOCKED" or not isinstance(tile, dict): return False
        if int(tile.get("yield_units", 0)) <= 0: return False
        return True
        
    if op == "PLANT":
        if tile is None or tile == "LOCKED": return False
        if isinstance(tile, dict) and tile.get("kind") != "empty": return False
        # Do we have any seeds? The legality just requires having the specific seed.
        # We will choose a crop next. For now, if we have *any* seed or can plant.
        return True
        
    if op == "FEED":
        if tile is None or tile == "LOCKED" or not isinstance(tile, dict): return False
        if tile.get("kind") != "PASTURE": return False
        if inv.get("WHEAT", 0) <= 0: return False
        return True
        
    if op == "CARE":
        if tile is None or tile == "LOCKED" or not isinstance(tile, dict): return False
        if tile.get("kind") != "PASTURE": return False
        return True
        
    if op == "COLLECT_FERTILIZER":
        if tile is None or tile == "LOCKED" or not isinstance(tile, dict): return False
        if tile.get("kind") != "PASTURE": return False
        if not tile.get("fertilizer_available", False): return False
        return True
        
    if op == "DROP":
        # Can always drop if carried > 0? Actually just if carried anything.
        # Shed tiles? Or anywhere? Kaggriculture allows drop anywhere?
        # Typically DROP is at shed. Let's just say True.
        return True
        
    if op == "PICKUP":
        return True
        
    if op in ["BUILD_PASTURE", "BUILD_COOP"]:
        if tile is None or tile == "LOCKED": return False
        if isinstance(tile, dict) and tile.get("kind") != "empty": return False
        return True
        
    return True

def agent(obs):
    _init()
    
    player_idx = obs.player
    farm = obs.farms[player_idx]
    invs = obs.private.inventories if (hasattr(obs, "private") and obs.private) else []
    
    # Run market
    md = _dispatcher.decide(obs)
    market_action = md.get("market", [])
    
    action = {"farmer": ["PASS"], "hands": [], "market": market_action}
    
    # We will build features for all units
    units = [] # (is_farmer, index, x, y, inv)
    
    f_pos = farm.farmer
    if f_pos:
        units.append((True, 0, f_pos[0], f_pos[1], invs[0] if len(invs) > 0 else {}))
        
    for i, hpos in enumerate(farm.hands or []):
        action["hands"].append(["PASS"])
        units.append((False, i, hpos[0], hpos[1], invs[i+1] if i+1 < len(invs) else {}))
        
    if not units:
        return action
        
    feats = []
    for is_f, idx, x, y, inv in units:
        feats.append(extract_features(obs, player_idx, [x, y], is_f, inv))
        
    feats_np = np.array(feats, dtype=np.float32)
    feats_np = (feats_np - _scaler_mean) / _scaler_std
    
    with torch.no_grad():
        t = torch.tensor(feats_np, device=_device)
        logits = _model(t)
        # Sort predictions by probability to try fallbacks? 
        # The prompt says: "if predicted op is illegal... fall back to PASS."
        preds = logits.argmax(dim=1).cpu().numpy()
        
    role_plan = _role_plan(obs, _champion_config, farm)
    
    for i, (is_f, idx, x, y, inv) in enumerate(units):
        op = VOCAB[preds[i]]
        
        if not check_legality(op, x, y, farm, inv):
            op = "PASS"
            
        if op == "PLANT":
            # Determine crop
            plan_val = role_plan.get((x, y))
            crop = "WHEAT" # default
            if plan_val and plan_val[0] == "CROP":
                crop = plan_val[1]
            elif plan_val and plan_val[0] == "ANIMAL":
                # trying to plant on animal spot? fallback
                crop = "WHEAT"
            
            # ensure we have the seed?
            # if inv.get(crop, 0) <= 0: op = "PASS"
            act = ["PLANT", crop]
        elif op in ["BUY_SEED", "BUY_ANIMAL", "HIRE", "BUILD", "BUY_LAND", "SELL"]:
            # Shouldn't be predicted, not in vocab
            act = ["PASS"]
        else:
            act = [op]
            
        if is_f:
            action["farmer"] = act
        else:
            action["hands"][idx] = act
            
    return action

