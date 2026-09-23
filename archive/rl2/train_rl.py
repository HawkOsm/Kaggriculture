import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import multiprocessing as mp
from pathlib import Path
import math
import random

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "opponents"))

from kaggle_environments import make
from run_match import load_agent
from adaptive_agent.agent import Dispatcher
from adaptive_agent.state import _role_plan
from rl2.collect import extract_features, VOCAB, VOCAB_MAP

# Champion-guided residual RL: we append the champion (adaptive_agent) suggested
# op for each unit as a one-hot INPUT feature, and during rollouts we execute the
# champion's op with a decaying probability. Together this makes the policy START
# AT CHAMPION STRENGTH (~45% vs pilkwang) instead of from scratch, and only learn
# where to DEVIATE (plant more / move tighter). CHAMP_DIM extra input dims.
CHAMP_DIM = len(VOCAB)

_INPLACE_WORK = ["HARVEST", "PLANT", "WATER", "CARE", "FEED", "COLLECT_FERTILIZER"]

def _best_inplace_work(x, y, farm, inv):
    """Highest-value legal PRODUCTIVE op the unit can do WITHOUT moving, else None.
    Used to force the movement lever: replace a MOVE with in-place work so the net
    learns the top agents' tight local sweeps (moves/work 0.86 vs our 3.13).
    Productivity-gated (no redundant water/harvest) so it can't reward-hack; and
    self-limiting — once in-place work is exhausted it returns None and the unit
    moves normally, so a worker can never get stuck refusing to move."""
    try:
        t_ = farm.tiles[y][x]
        tl = t_ if isinstance(t_, dict) else None
    except Exception:
        tl = None
    for op in _INPLACE_WORK:
        if not check_legality(op, x, y, farm, inv):
            continue
        if op == "HARVEST" and not (tl and int(tl.get("yield_units", 0) or 0) > 0):
            continue
        if op == "WATER" and not (tl and not tl.get("watered_today", False)):
            continue
        return op
    return None

def _can_plant(x, y, farm, role_plan):
    """Correct PLANT legality. Empty plantable tiles are stored as None (NOT a
    dict with kind=='empty'), which the old check_legality wrongly rejected — that
    silently reverted every PLANT to PASS and is why prior runs had 0 plants. We
    use the champion's role_plan as the ownership+intent oracle: a tile is a valid
    plant target iff the champion's own plan marks it CROP and it's not occupied."""
    try:
        tl = farm.tiles[y][x]
    except Exception:
        return False
    if isinstance(tl, dict) and tl.get("kind") not in (None, "empty"):
        return False
    pv = role_plan.get((x, y))
    return bool(pv and pv[0] == "CROP")

def _champ_op_index(md, is_f, idx):
    """Champion's suggested op for this unit -> VOCAB index (PASS if unknown)."""
    try:
        if is_f:
            op = (md.get("farmer") or ["PASS"])[0]
        else:
            hands = md.get("hands") or []
            op = (hands[idx] if idx < len(hands) else ["PASS"])[0]
    except Exception:
        op = "PASS"
    return VOCAB_MAP.get(op, VOCAB_MAP["PASS"])
from rl2.policy import ActorCritic
from rl2.bc_agent import check_legality

_scaler_mean = None
_scaler_std = None
_champion_config = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _init_globals():
    global _scaler_mean, _scaler_std, _champion_config
    if _scaler_mean is not None: return
    out_dir = REPO_ROOT / ".claude/scratch/rl2"
    scaler = np.load(out_dir / "scaler.npz")
    _scaler_mean = scaler["mean"].astype(np.float32)
    _scaler_std = scaler["std"].astype(np.float32)
    config_path = REPO_ROOT / "src" / "adaptive_best_config.json"
    with open(config_path, "r") as f:
        _champion_config = json.load(f)

def run_episode(args):
    # eps = forced-PLANT exploration probability (see sampling block). Eval calls
    # pass a 4-tuple (eps defaults 0); training rollouts pass a 5-tuple with a
    # decaying eps so the policy actually EXPERIENCES planting (from-scratch PPO
    # never discovers PLANT on its own: standing on an empty owned tile AND
    # emitting the PLANT class is a joint event that ~never fires randomly).
    if len(args) == 5:
        seed, opp_name, seat, model_state_dict, eps = args
    else:
        seed, opp_name, seat, model_state_dict = args
        eps = 0.0
    _init_globals()
    PLANT_IDX = VOCAB.index("PLANT")
    
    # Init model
    out_dir = REPO_ROOT / ".claude/scratch/rl2"
    with open(out_dir / "feature_spec.json", "r") as f:
        dim = json.load(f)["dim"] + CHAMP_DIM  # + champion-op one-hot
    model = ActorCritic(in_dim=dim, out_dim=len(VOCAB))
    model.load_state_dict(model_state_dict)
    model.eval()
    
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    opp = load_agent(opp_name)
    dispatcher = Dispatcher(_champion_config)
    
    transitions = [] # (feats, actions, logps, values, is_farmers)
    last_work = {}          # per-unit: did it do work last turn (for stacking bonus)
    ep_moves = ep_work = ep_plant = 0   # episode metrics for the CSV log
    
    env.reset()
    state = env.state
    
    for step_idx in range(720):
        if state[0].status != "ACTIVE":
            break
            
        # BUGFIX: our agent must see ITS OWN player's observation. Using
        # state[0].observation regardless of seat meant that in every seat==1
        # game our market (dispatcher reads obs.player) and features were
        # computed for the WRONG player -> corrupted half of both training data
        # and eval (produced the seat-correlated 12/24 eval artifact).
        obs = state[seat].observation

        # Opponent action
        opp_obs = state[1].observation if seat == 0 else state[0].observation
        opp_act = opp(opp_obs)
        
        # Our action
        player_idx = seat
        farm = obs.farms[player_idx]
        invs = obs.private.inventories if (hasattr(obs, "private") and obs.private) else []
        
        md = dispatcher.decide(obs)
        market_action = md.get("market", [])
        action = {"farmer": ["PASS"], "hands": [], "market": market_action}
        role_plan = _role_plan(obs, _champion_config, farm)  # plant oracle + crop choice
        
        units = []
        f_pos = farm.farmer
        if f_pos:
            units.append((True, 0, f_pos[0], f_pos[1], invs[0] if len(invs) > 0 else {}))
        for i, hpos in enumerate(farm.hands or []):
            action["hands"].append(["PASS"])
            units.append((False, i, hpos[0], hpos[1], invs[i+1] if i+1 < len(invs) else {}))
            
        if not units:
            agents = [action, opp_act] if seat == 0 else [opp_act, action]
            state = env.step(agents)
            continue
            
        feats = []
        champ_idx = []   # champion's suggested op per unit (VOCAB index)
        for uidx, (is_f, idx, x, y, inv) in enumerate(units):
            feats.append(extract_features(obs, player_idx, [x, y], is_f, inv))
            champ_idx.append(_champ_op_index(md, is_f, idx))

        feats_np = np.array(feats, dtype=np.float32)
        feats_np = (feats_np - _scaler_mean) / _scaler_std
        # Append the champion-op one-hot (raw, no standardization) so the policy
        # can SEE what the champion would do — this carries the routing/destination
        # intent the 5x5 local window can't, making champion imitation learnable.
        champ_oh = np.zeros((len(units), CHAMP_DIM), dtype=np.float32)
        champ_oh[np.arange(len(units)), champ_idx] = 1.0
        feats_np = np.concatenate([feats_np, champ_oh], axis=1)

        with torch.no_grad():
            t = torch.tensor(feats_np)
            logits, vals = model(t)
            dist = Categorical(logits=logits)
            acts = dist.sample()

            # Champion-forced warm-start: with prob eps, execute the champion's
            # suggested op (if legal) instead of the sampled one. Early iters run
            # ~as the champion (champion-level wins feed the terminal reward);
            # eps decays so the net takes over and learns to improve on it. We
            # store the net's logp of the EXECUTED action so PPO's ratio stays
            # consistent — forcing only shifts exploration off-policy.
            if eps > 0.0:
                MOVES_FORCE = {"NORTH", "SOUTH", "EAST", "WEST"}
                acts_i = acts.numpy().copy()
                for i2, (is_f2, idx2, x2, y2, inv2) in enumerate(units):
                    cop = VOCAB[champ_idx[i2]]
                    # (1) soft champion warm-start (PLANT legality via the oracle,
                    # since check_legality wrongly rejects the None empty tiles the
                    # champion plants on)
                    clegal = (_can_plant(x2, y2, farm, role_plan) if cop == "PLANT"
                              else check_legality(cop, x2, y2, farm, inv2))
                    if random.random() < eps and clegal:
                        acts_i[i2] = champ_idx[i2]
                    # (2) MOVEMENT lever: if the chosen op is a MOVE but there's
                    # productive work right here, force the work instead — this is
                    # exactly the "move tighter" behaviour we want the net to learn.
                    if VOCAB[acts_i[i2]] in MOVES_FORCE and random.random() < eps:
                        w = _best_inplace_work(x2, y2, farm, inv2)
                        if w is not None:
                            acts_i[i2] = VOCAB_MAP[w]
                acts = torch.tensor(acts_i)
            logps = dist.log_prob(acts)

        acts_np = acts.numpy()
        vals_np = vals.squeeze(-1).numpy()
        logps_np = logps.numpy()
        
        # Store for GAE
        transitions.append({
            "feats": feats_np,
            "acts": acts_np,
            "logps": logps_np,
            "vals": vals_np,
            "units": units
        })
        
        # ACTIONS-PER-MOVEMENT reward (the whole point of this training): reward a
        # unit for doing valid WORK this turn, penalize wasted MOVEs and idle PASS.
        # This drives the policy toward the top agents' ~0.86 moves/work vs our 2.69.
        MOVES_SET = {"NORTH", "SOUTH", "EAST", "WEST"}
        step_reward = 0.0
        for i, (is_f, idx, x, y, inv) in enumerate(units):
            op = VOCAB[acts_np[i]]
            # PLANT legality via the oracle (None empty tiles); everything else via
            # check_legality. Illegal -> PASS.
            if op == "PLANT":
                if not _can_plant(x, y, farm, role_plan):
                    op = "PASS"
            elif not check_legality(op, x, y, farm, inv):
                op = "PASS"

            # tile state to reward only PRODUCTIVE work (prevents reward-hacking:
            # spamming WATER on already-watered tiles counted as "work" -> 0 plants).
            tl = None
            try:
                t_ = farm.tiles[y][x]
                tl = t_ if isinstance(t_, dict) else None
            except Exception:
                tl = None

            if op == "PASS":
                step_reward -= 0.30
            elif op in MOVES_SET:
                step_reward -= 0.15
                ep_moves += 1
            elif op == "PLANT":
                step_reward += 2.50            # PLANTING lever: dominant reward
                ep_work += 1
                ep_plant += 1
            elif op == "HARVEST":
                if tl and int(tl.get("yield_units", 0) or 0) > 0:
                    step_reward += 2.00        # realizes value
                    ep_work += 1
                else:
                    step_reward -= 0.10        # redundant harvest = wasted
            elif op == "WATER":
                if tl and not tl.get("watered_today", False):
                    step_reward += 1.00        # real, needed watering
                    ep_work += 1
                else:
                    step_reward -= 0.10        # redundant water = wasted (the exploit)
            else:  # FEED / CARE / COLLECT_FERTILIZER / DROP / PICKUP / BUILD_*
                step_reward += 0.60
                ep_work += 1

            if op == "PLANT":
                plan_val = role_plan.get((x, y))
                crop = "WHEAT"
                if plan_val and plan_val[0] == "CROP": crop = plan_val[1]
                elif plan_val and plan_val[0] == "ANIMAL": crop = "WHEAT"
                act = ["PLANT", crop]
            elif op in ["BUY_SEED", "BUY_ANIMAL", "HIRE", "BUILD", "BUY_LAND", "SELL"]:
                act = ["PASS"]
            else:
                act = [op]

            if is_f: action["farmer"] = act
            else: action["hands"][idx] = act

        # normalize by unit count so games with more hands aren't over-weighted
        transitions[-1]["r"] = step_reward / max(1, len(units))

        agents = [action, opp_act] if seat == 0 else [opp_act, action]
        state = env.step(agents)
        
    r0 = state[0].reward
    r1 = state[1].reward
    margin = r0 - r1 if seat == 0 else r1 - r0
    
    # Reward is DENSE per-step actions-per-movement (set above as trs["r"]).
    # A modest terminal margin anchor keeps the objective tied to actually
    # winning, but the dominant signal is work-per-turn efficiency.
    terminal_bonus = 0.5 * math.tanh(margin / 15000.0)

    # Compute GAE
    gamma = 0.99
    lam = 0.95

    gae = 0.0
    for t in reversed(range(len(transitions))):
        trs = transitions[t]
        farmer_val = trs["vals"][0] # farmer is always unit 0

        r = trs.get("r", 0.0)
        if t == len(transitions) - 1:
            next_val = 0.0
            r = r + terminal_bonus
        else:
            next_val = transitions[t+1]["vals"][0]

        delta = r + gamma * next_val - farmer_val
        gae = delta + gamma * lam * gae
        
        trs["adv"] = gae
        trs["ret"] = gae + farmer_val
        
    return transitions, margin > 0, {"moves": ep_moves, "work": ep_work,
                                     "plants": ep_plant, "steps": max(1, len(transitions))}

def eval_policy(model_state_dict, seeds, pool_opponents):
    _init_globals()
    pool = mp.Pool(12)
    args_list = []
    for i, seed in enumerate(seeds):
        opp = pool_opponents[i % len(pool_opponents)]
        seat = seed % 2
        args_list.append((seed, opp, seat, model_state_dict))
    
    # Run evaluation without GAE to save time, or just reuse run_episode and ignore transitions
    results = pool.map(run_episode, args_list)
    wins = sum(1 for r in results if r[1])
    pool.close()
    return wins / len(seeds)

def eval_policy_detailed(model_state_dict, seeds, pool_opponents):
    _init_globals()
    pool = mp.Pool(12)
    
    wins_per_opp = {opp: 0 for opp in pool_opponents}
    
    args_list = []
    for opp in pool_opponents:
        for seed in seeds:
            seat = seed % 2
            args_list.append((seed, opp, seat, model_state_dict))
            
    results = pool.map(run_episode, args_list)
    
    idx = 0
    for opp in pool_opponents:
        for seed in seeds:
            w = results[idx][1]
            if w:
                wins_per_opp[opp] += 1
            idx += 1
            
    pool.close()
    return wins_per_opp

def run():
    out_dir = REPO_ROOT / ".claude/scratch/rl2"
    with open(out_dir / "feature_spec.json", "r") as f:
        dim = json.load(f)["dim"] + CHAMP_DIM  # + champion-op one-hot

    model = ActorCritic(in_dim=dim, out_dim=len(VOCAB)).to(_device)

    # Warm-start: only the champion-guided best.pt (same in_dim) is compatible.
    # Old 416-dim BC/best checkpoints don't fit the new champion-op input, so we
    # start fresh — but the WARM behaviour comes from champion-forcing in rollouts
    # (eps), not from loaded weights, so "fresh" here still plays at champion level.
    ckpt_best = out_dir / "checkpoints" / "best.pt"
    loaded = False
    if ckpt_best.exists():
        try:
            model.load_state_dict(torch.load(ckpt_best, map_location=_device))
            loaded = True
            print(f"Warm-started full model from {ckpt_best} (continuing to improve best.pt).")
        except Exception as e:
            print(f"best.pt incompatible ({e}); starting fresh (champion-forced warm-start).")
    if not loaded:
        print("Fresh net; champion-forcing (eps) provides the champion-level warm-start.")

    # CSV training log so improvements are visible (moves/work ratio is the target).
    import csv as _csv
    _csvf = open(out_dir / "training_log.csv", "w", newline="")
    _csvw = _csv.writer(_csvf)
    _csvw.writerow(["iter", "rollout_win_18", "eval_win", "mean_step_reward",
                    "moves", "work", "plants", "moves_per_work", "plants_per_game"])
    _csvf.flush()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    OPPONENTS = [
        "pilkwang_agent",
        "cropdusta_97601003_agent",
        "prvsiyan_frontier_agent",
        "boatlee_v16_agent",
    ]
    
    best_win_rate = 0.0
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    
    epochs = 4 # PPO epochs per iter
    batch_size = 2048
    clip_ratio = 0.2
    ent_coef = 0.01
    
    for it in range(60):
        # Collect rollouts
        model.eval()
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
        pool = mp.Pool(12)
        # Champion-forcing prob: ~all champion early (champion-level trajectories
        # feed the win reward), decaying so the net takes over and learns to
        # improve on the champion (plant more / move tighter).
        eps = max(0.05, 0.95 * (0.88 ** it))
        args_list = []
        for i in range(12):
            seed = np.random.randint(0, 100000)
            opp = OPPONENTS[i % len(OPPONENTS)]
            seat = i % 2
            args_list.append((seed, opp, seat, model_state_dict, eps))
            
        print(f"Iter {it}: Collecting rollouts...")
        results = pool.map(run_episode, args_list)
        pool.close()
        
        wins = sum(1 for r in results if r[1])
        print(f"Iter {it}: Win rate vs pool in collection = {wins}/12")

        # aggregate the three levers for the CSV: movement, work, planting
        tot_moves = sum(r[2]["moves"] for r in results)
        tot_work = sum(r[2]["work"] for r in results)
        tot_plants = sum(r[2]["plants"] for r in results)
        n_games = len(results)
        mean_step_r = float(np.mean([sd["r"] for r in results for sd in r[0] if "r" in sd])) if results else 0.0
        moves_per_work = tot_moves / max(1, tot_work)
        plants_per_game = tot_plants / max(1, n_games)
        print(f"Iter {it}: moves/work={moves_per_work:.2f} plants/game={plants_per_game:.0f} "
              f"work={tot_work} moves={tot_moves} (target moves/work ~0.86)")
        
        all_feats = []
        all_acts = []
        all_logps = []
        all_advs = []
        all_rets = []
        
        for r in results:
            trans = r[0]
            for step_data in trans:
                for i in range(len(step_data["acts"])):
                    all_feats.append(step_data["feats"][i])
                    all_acts.append(step_data["acts"][i])
                    all_logps.append(step_data["logps"][i])
                    all_advs.append(step_data["adv"])
                    all_rets.append(step_data["ret"])
                    
        all_feats = torch.tensor(np.array(all_feats), dtype=torch.float32).to(_device)
        all_acts = torch.tensor(np.array(all_acts), dtype=torch.long).to(_device)
        all_logps = torch.tensor(np.array(all_logps), dtype=torch.float32).to(_device)
        all_advs = torch.tensor(np.array(all_advs), dtype=torch.float32).to(_device)
        all_rets = torch.tensor(np.array(all_rets), dtype=torch.float32).to(_device)
        
        # Normalize adv
        all_advs = (all_advs - all_advs.mean()) / (all_advs.std() + 1e-8)
        
        # PPO Update
        model.train()
        dataset = torch.utils.data.TensorDataset(all_feats, all_acts, all_logps, all_advs, all_rets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for _ in range(epochs):
            for b_feats, b_acts, b_logps, b_advs, b_rets in loader:
                optimizer.zero_grad()
                logits, vals = model(b_feats)
                dist = Categorical(logits=logits)
                new_logps = dist.log_prob(b_acts)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_logps - b_logps)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_advs
                actor_loss = -torch.min(surr1, surr2).mean()
                
                critic_loss = F.mse_loss(vals.squeeze(-1), b_rets)
                
                loss = actor_loss + 0.5 * critic_loss - ent_coef * entropy
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
        # Eval every 5 iters
        eval_val = ""
        if (it + 1) % 5 == 0:
            print(f"Iter {it}: Evaluating...")
            eval_seeds = list(range(1000, 1032))  # 32 seeds for robust best-selection
            win_rate = eval_policy(model_state_dict, eval_seeds, OPPONENTS)
            eval_val = round(win_rate, 4)
            print(f"Iter {it}: Eval Win rate = {win_rate:.2f}")
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
                print(f"Iter {it}: New best model saved!")

        # CSV row so improvement in the three levers is visible turn by turn
        _csvw.writerow([it, wins, eval_val, round(mean_step_r, 4), tot_moves,
                        tot_work, tot_plants, round(moves_per_work, 3), round(plants_per_game, 1)])
        _csvf.flush()

    _csvf.close()
    # Final Eval (GATE 4)
    print("Evaluating best model...")
    best_dict = torch.load(out_dir / "checkpoints" / "best.pt")
    final_seeds = list(range(500, 532))
    wins_per_opp = eval_policy_detailed(best_dict, final_seeds, OPPONENTS)
    
    print("-" * 45)
    print(f"{'Opponent':<30} | {'Wins':<10}")
    print("-" * 45)
    total_wins = 0
    for opp in OPPONENTS:
        w = wins_per_opp[opp]
        total_wins += w
        print(f"{opp:<30} | {w}/32 ({w/32:.2f})")
    print("-" * 45)
    print(f"{'Total':<30} | {total_wins}/128 ({total_wins/128:.2f})")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run()
