"""Deploy the champion-guided RL best.pt as a normal agent(obs) for env.run.
Mirrors train_rl.run_episode's eval path (eps=0, no forcing): per-unit features +
champion-op one-hot -> ActorCritic -> argmax op, legality via the same oracles,
champion market. Used for the clean head-to-head verification vs the champion."""
import sys, json
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "opponents"))

from adaptive_agent.agent import Dispatcher
from adaptive_agent.state import _role_plan
from rl2.collect import extract_features, VOCAB, VOCAB_MAP
from rl2.policy import ActorCritic
from rl2.bc_agent import check_legality
from rl2.train_rl import _champ_op_index, _can_plant, CHAMP_DIM

_M = None
_mean = _std = _cfg = _disp = None

def _init():
    global _M, _mean, _std, _cfg, _disp
    if _M is not None:
        return
    out = REPO / ".claude/scratch/rl2"
    sc = np.load(out / "scaler.npz")
    _mean = sc["mean"].astype(np.float32); _std = sc["std"].astype(np.float32)
    _cfg = json.load(open(REPO / "src" / "adaptive_best_config.json"))
    _disp = Dispatcher(_cfg)
    dim = json.load(open(out / "feature_spec.json"))["dim"] + CHAMP_DIM
    _M = ActorCritic(in_dim=dim, out_dim=len(VOCAB))
    _M.load_state_dict(torch.load(out / "checkpoints" / "best.pt", map_location="cpu"))
    _M.eval()

def agent(obs):
    _init()
    pi = obs.player
    farm = obs.farms[pi]
    invs = obs.private.inventories if (hasattr(obs, "private") and obs.private) else []
    md = _disp.decide(obs)
    role_plan = _role_plan(obs, _cfg, farm)
    action = {"farmer": ["PASS"], "hands": [], "market": md.get("market", [])}

    units = []
    if farm.farmer:
        units.append((True, 0, farm.farmer[0], farm.farmer[1], invs[0] if invs else {}))
    for i, h in enumerate(farm.hands or []):
        action["hands"].append(["PASS"])
        units.append((False, i, h[0], h[1], invs[i + 1] if i + 1 < len(invs) else {}))
    if not units:
        return action

    feats, champ_idx = [], []
    for is_f, idx, x, y, inv in units:
        feats.append(extract_features(obs, pi, [x, y], is_f, inv))
        champ_idx.append(_champ_op_index(md, is_f, idx))
    f = (np.array(feats, dtype=np.float32) - _mean) / _std
    oh = np.zeros((len(units), CHAMP_DIM), dtype=np.float32)
    oh[np.arange(len(units)), champ_idx] = 1.0
    f = np.concatenate([f, oh], axis=1)
    with torch.no_grad():
        logits, _ = _M(torch.tensor(f))
        preds = logits.argmax(dim=1).numpy()

    for i, (is_f, idx, x, y, inv) in enumerate(units):
        op = VOCAB[preds[i]]
        if op == "PLANT":
            if not _can_plant(x, y, farm, role_plan):
                op = "PASS"
        elif not check_legality(op, x, y, farm, inv):
            op = "PASS"
        if op == "PLANT":
            pv = role_plan.get((x, y)); crop = "WHEAT"
            if pv and pv[0] == "CROP": crop = pv[1]
            act = ["PLANT", crop]
        elif op in ["BUY_SEED", "BUY_ANIMAL", "HIRE", "BUILD", "BUY_LAND", "SELL"]:
            act = ["PASS"]
        else:
            act = [op]
        if is_f: action["farmer"] = act
        else: action["hands"][idx] = act
    return action
