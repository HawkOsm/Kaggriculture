"""RL training for the agent.py policy hook (see agent.KNOB_SPECS
/ make_agent(policy_fn=...)).

Architecture (see docs/tests/LOG.md for why this shape specifically):
  - Rollout collection is CPU-bound (the kaggriculture engine itself is
    plain-Python simulation) and runs across many parallel worker
    processes, each playing full games against a mix of the current
    champion config, the current policy itself (self-play), and the two
    builtin benchmark bots -- using its own CPU copy of the current policy
    snapshot for inference, so there's no live GPU round-trip mid-episode.
  - Every generation, all of that (hundreds of games' worth of per-day
    decisions) gets flattened into one big batch and run through PPO
    updates on the GPU together, in large minibatches over several epochs --
    this is where "hundreds of games batched through the GPU" actually
    happens, at the update step rather than turn-by-turn during rollout.
    That's a deliberate reliability tradeoff: true per-turn synchronized
    GPU-batched rollout would need bidirectional IPC between dozens of
    worker processes and a central GPU inference server, which is real
    distributed-systems complexity with real deadlock/hang risk for an
    unattended 8-hour run. This gets most of the same GPU throughput with
    far less that can go wrong overnight.
  - A background thread polls nvidia-smi and throttles the PPO update loop
    (a short sleep after each optimizer step) to keep GPU utilization in a
    target band instead of pegging at 100%.

Usage:
    python train_rl.py --hours 8
    python train_rl.py --hours 0.05 --n-workers 4 --episodes-per-worker 2  # smoke test
"""

import argparse
import collections
import csv
import json
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "opponents"))

from agent import (
    DEFAULT_CONFIG,
    SAFE_FALLBACK,
    _market_orders,
    _opponent_incoming_supply,
    _plan_units,
    _scan_farm,
    decode_knobs,
    extract_features,
    make_agent,
)
from rl_policy import ActorCritic

REPO_ROOT = HERE.parent
LOG_PATH = REPO_ROOT / "docs" / "tests" / "LOG.md"

REWARD_SCALE = 1000.0  # day-over-day money delta, scaled down for PPO stability
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MAX_GRAD_NORM = 0.5
# Lower than PPO's usual 3e-4 default, deliberately: smaller steps keep the
# per-epoch KL divergence low enough that more epochs stay safe under
# target_kl below, which is what actually lets a generation's update phase
# run long enough to matter against CPU-bound rollout collection (see
# ppo_update's docstring) instead of early-stopping after a single epoch.
# Dropped further (1e-4 -> 5e-5) after live data showed even 1e-4 wasn't
# enough: with a large replay buffer, a single epoch alone has hundreds of
# minibatch steps, and their *cumulative* drift was tripping target_kl
# before epoch 2 ever started, regardless of buffer size. Smaller steps
# reduce that cumulative drift per epoch.
LR = 5e-5

# "champion"/"self" keep training anchored to our own best config and
# self-play; the other nine are the pulled-from-Kaggle 2500+ opponents (see
# CREDITS.md) -- added after best.pt (trained only against champion/self/
# melon_maxxer/multi_crop) lost 0W-6L to kawa_route_agent alone, confirming
# the old pool never exposed training to anything close to real leaderboard
# difficulty (see docs/tests/LOG.md).
OPPONENT_POOL = [
    "champion", "self", "kawa", "boatlee_v16", "rayk_c95", "saiteja", "kaito",
    "tran_hh", "pilkwang", "romanrozen", "prvsiyan_frontier",
]
OPPONENT_WEIGHTS = [0.15, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08, 0.09, 0.08, 0.08, 0.10]


# --------------------------------------------------------------------------
# Training-time agent: same mechanical layer as agent.make_agent,
# but records one transition per in-game day for PPO instead of just
# applying config overrides.
# --------------------------------------------------------------------------

def make_training_agent(net, base_cfg, transitions, deterministic=False):
    """transitions: list appended to in place, one dict per in-game day:
    {features, action, log_prob, value, reward}. `reward` for day D is
    filled in when day D+1 starts (or via finalize() at episode end)."""
    state = {"last_day": None, "day_money": 0.0, "opp_day_money": 0.0, "cfg": dict(base_cfg), "unit_targets": {}}

    def agent(obs):
        try:
            farms = obs.get("farms", [])
            player = obs.get("player", 0)
            if not farms or player >= len(farms):
                return dict(SAFE_FALLBACK)
            farm = farms[player]
            opponent_farm = next((f for i, f in enumerate(farms) if i != player), None)
            private = obs.get("private", {}) or {}
            day = obs.get("day", 0)
            board_size = len(farm["tiles"])
            prices = (obs.get("market", {}) or {}).get("prices", {}) or {}
            money = farm.get("money", 0.0)
            opp_money = opponent_farm.get("money", 0.0) if opponent_farm is not None else state["opp_day_money"]

            if day != state["last_day"]:
                if state["last_day"] is not None and transitions and transitions[-1]["reward"] is None:
                    # Margin-delta, not absolute own-money delta: the
                    # competition's Elo-like ranking only scores win/loss
                    # (see docs/GAME_GUIDE.md -- coin margin at game end
                    # doesn't even matter there, only who has more), so a
                    # day where we grow $500 while the opponent grows $2000
                    # should read as a bad day, not a good one. Still dense
                    # (every day, not just game-end) for credit assignment.
                    our_delta = money - state["day_money"]
                    opp_delta = opp_money - state["opp_day_money"]
                    transitions[-1]["reward"] = (our_delta - opp_delta) / REWARD_SCALE
                state["last_day"] = day
                state["day_money"] = money
                state["opp_day_money"] = opp_money
                features = extract_features(obs, base_cfg)
                action, log_prob, value = net.act(
                    torch.tensor(features, dtype=torch.float32), deterministic=deterministic
                )
                action_list = action.tolist()
                turn_cfg = dict(base_cfg)
                turn_cfg.update(decode_knobs(action_list))
                state["cfg"] = turn_cfg
                transitions.append({
                    "features": features,
                    "action": action_list,
                    "log_prob": float(log_prob.item()),
                    "value": float(value.item()),
                    "reward": None,
                })
            turn_cfg = state["cfg"]

            info = _scan_farm(farm, board_size, day)
            opponent_supply = _opponent_incoming_supply(
                opponent_farm, board_size, day, turn_cfg["opponent_lookahead_days"]
            )
            market_orders = _market_orders(farm, private, info, turn_cfg, prices, day, opponent_supply)
            farmer_action, hands_actions = _plan_units(
                farm, private, board_size, day, info, turn_cfg, prices, state["unit_targets"]
            )
            return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}
        except Exception as exc:  # noqa: BLE001
            print(f"train_rl agent: swallowed exception: {exc!r}", file=sys.stderr)
            return dict(SAFE_FALLBACK)

    def finalize(final_money, opp_final_money):
        if transitions and transitions[-1]["reward"] is None:
            our_delta = final_money - state["day_money"]
            opp_delta = opp_final_money - state["opp_day_money"]
            transitions[-1]["reward"] = (our_delta - opp_delta) / REWARD_SCALE

    return agent, finalize


def _build_opponent(spec, champion_config, net, base_cfg):
    if spec == "champion":
        return make_agent(champion_config)
    if spec == "self":
        agent, _ = make_training_agent(net, base_cfg, [], deterministic=False)
        return agent
    if spec == "kawa":
        from kawa_route_agent import kawa_route_agent
        return kawa_route_agent
    if spec == "boatlee_v16":
        from boatlee_v16_agent import boatlee_v16_agent
        return boatlee_v16_agent
    if spec == "rayk_c95":
        from rayk_c95_agent import rayk_c95_agent
        return rayk_c95_agent
    if spec == "saiteja":
        from saiteja_agent import saiteja_agent
        return saiteja_agent
    if spec == "kaito":
        from kaito_agent import kaito_agent
        return kaito_agent
    if spec == "tran_hh":
        from tran_hh_agent import tran_hh_agent
        return tran_hh_agent
    if spec == "pilkwang":
        from pilkwang_agent import pilkwang_agent
        return pilkwang_agent
    if spec == "romanrozen":
        from romanrozen_agent import romanrozen_agent
        return romanrozen_agent
    if spec == "prvsiyan_frontier":
        from prvsiyan_frontier_agent import prvsiyan_frontier_agent
        return prvsiyan_frontier_agent
    raise ValueError(f"unknown opponent spec {spec!r}")


# --------------------------------------------------------------------------
# Rollout worker (runs in a separate process; CPU only)
# --------------------------------------------------------------------------

def rollout_worker(payload):
    (state_dict_cpu, opponent_spec, champion_config, n_episodes, episode_steps, seed, deterministic) = payload
    torch.set_num_threads(1)  # many worker processes -- avoid BLAS thread oversubscription
    random.seed(seed)
    torch.manual_seed(seed)

    from kaggle_environments import make as make_env

    net = ActorCritic()
    net.load_state_dict(state_dict_cpu)
    net.eval()

    base_cfg = dict(DEFAULT_CONFIG)
    episodes = []
    for i in range(n_episodes):
        transitions = []
        agent, finalize = make_training_agent(net, base_cfg, transitions, deterministic=deterministic)
        opponent_agent = _build_opponent(opponent_spec, champion_config, net, base_cfg)

        env = make_env("kaggriculture", configuration={"episodeSteps": episode_steps}, debug=True)
        if i % 2 == 0:
            env.run([agent, opponent_agent])
            idx = 0
        else:
            env.run([opponent_agent, agent])
            idx = 1
        final = env.steps[-1]
        final_money = final[idx].reward or 0.0
        opp_final_money = final[1 - idx].reward or 0.0
        margin = final_money - opp_final_money
        finalize(final_money, opp_final_money)

        if transitions and all(t["reward"] is not None for t in transitions):
            episodes.append({"transitions": transitions, "margin": margin, "opponent": opponent_spec})
    return episodes


# --------------------------------------------------------------------------
# GAE
# --------------------------------------------------------------------------

def compute_gae(transitions):
    values = [t["value"] for t in transitions]
    rewards = [t["reward"] for t in transitions]
    advantages = [0.0] * len(transitions)
    gae = 0.0
    for t in reversed(range(len(transitions))):
        next_value = values[t + 1] if t + 1 < len(transitions) else 0.0
        delta = rewards[t] + GAMMA * next_value - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * gae
        advantages[t] = gae
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


# --------------------------------------------------------------------------
# GPU utilization throttle -- background thread polling nvidia-smi, a
# simple proportional controller keeping the rolling-average utilization in
# [target_low, target_high] by adjusting a sleep applied after each PPO
# optimizer step.
# --------------------------------------------------------------------------

class GPUThrottle:
    def __init__(self, target_low=65.0, target_high=75.0, poll_interval=1.0, window=10):
        self.target_low = target_low
        self.target_high = target_high
        self.poll_interval = poll_interval
        self.sleep_s = 0.0
        self._lock = threading.Lock()
        self._readings = collections.deque(maxlen=window)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _read_util(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return float(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            u = self._read_util()
            if u is not None:
                self._readings.append(u)
                avg = sum(self._readings) / len(self._readings)
                with self._lock:
                    if avg > self.target_high:
                        self.sleep_s = min(0.25, self.sleep_s + 0.006)
                    elif avg < self.target_low:
                        self.sleep_s = max(0.0, self.sleep_s - 0.006)
            self._stop.wait(self.poll_interval)

    def tick(self):
        with self._lock:
            s = self.sleep_s
        if s > 0:
            time.sleep(s)

    def current_avg(self):
        return sum(self._readings) / len(self._readings) if self._readings else 0.0


# --------------------------------------------------------------------------
# Main training loop
# --------------------------------------------------------------------------

@torch.no_grad()
def _refresh_values(net, device, episodes, minibatch_size=4096):
    """Recompute value estimates for every transition in `episodes` with the
    current network, in place. Needed because episodes now live in a replay
    buffer spanning many generations (see ppo_update) instead of being used
    once -- without this, GAE for older episodes would use value estimates
    from a long-stale network."""
    flat = [(ep, i) for ep in episodes for i in range(len(ep["transitions"]))]
    for start in range(0, len(flat), minibatch_size):
        chunk = flat[start:start + minibatch_size]
        states = torch.tensor(
            [ep["transitions"][i]["features"] for ep, i in chunk],
            dtype=torch.float32, device=device,
        )
        _, _, values = net(states)
        for (ep, i), v in zip(chunk, values.cpu().tolist()):
            ep["transitions"][i]["value"] = v


def ppo_update(net, optimizer, device, replay_episodes, minibatch_size, throttle,
               max_epochs, target_kl):
    """`replay_episodes` is the full replay buffer (recent generations'
    episodes retained, not just the one just collected) -- see main()'s
    REPLAY_EPISODES sizing note for why: rollout collection is CPU-bound and
    inherently much slower than a GPU update over one generation's worth of
    data, so processing a much larger retained batch each time is what
    actually gives the GPU sustained, genuine (not padded) work across an
    8-hour session, instead of brief spikes followed by long idle stretches.

    Epoch count is capped, not fixed -- and stops early once the average KL
    divergence between the old and updated policy crosses `target_kl` (the
    standard PPO safeguard, e.g. OpenAI Spinning Up's default range 0.01-
    0.05). This protects training quality regardless of how large the
    replay buffer or epoch cap gets: more data extends real GPU time, it
    never buys extra epochs at the cost of policy collapse."""
    _refresh_values(net, device, replay_episodes)

    states, actions, old_log_probs, old_values, advantages, returns = [], [], [], [], [], []
    for ep in replay_episodes:
        adv, ret = compute_gae(ep["transitions"])
        for t, a, r in zip(ep["transitions"], adv, ret):
            states.append(t["features"])
            actions.append(t["action"])
            old_log_probs.append(t["log_prob"])
            old_values.append(t["value"])
            advantages.append(a)
            returns.append(r)

    if not states:
        return None

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
    old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
    old_values_t = torch.tensor(old_values, dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    n = states_t.shape[0]
    last_policy_loss = last_value_loss = last_entropy = 0.0
    epochs_run = 0
    for epoch in range(max_epochs):
        perm = torch.randperm(n, device=device)
        kls = []
        for start in range(0, n, minibatch_size):
            idx = perm[start:start + minibatch_size]
            log_probs, entropy, values = net.evaluate(states_t[idx], actions_t[idx])
            ratio = torch.exp(log_probs - old_log_probs_t[idx])
            surr1 = ratio * advantages_t[idx]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages_t[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            # Value clipping (PPO2/Baselines): cap how far the value estimate
            # can move from its pre-update snapshot in a single update, same
            # idea as the policy-ratio clip above. Without this the critic
            # can take unbounded steps against a replay buffer whose returns
            # keep shifting as _refresh_values recomputes them each
            # generation -- exactly the failure mode a live run showed
            # (value_loss climbing every generation for 10 generations
            # straight post-reset, entropy collapsing alongside it; see
            # docs/tests/LOG.md).
            values_clipped = old_values_t[idx] + (values - old_values_t[idx]).clamp(-CLIP_EPS, CLIP_EPS)
            value_loss = torch.max(
                F.mse_loss(values, returns_t[idx], reduction="none"),
                F.mse_loss(values_clipped, returns_t[idx], reduction="none"),
            ).mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            throttle.tick()

            with torch.no_grad():
                kls.append((old_log_probs_t[idx] - log_probs).mean().item())
            last_policy_loss, last_value_loss, last_entropy = (
                policy_loss.item(), value_loss.item(), entropy_mean.item()
            )
        epochs_run = epoch + 1
        if sum(kls) / len(kls) > target_kl:
            break

    return {
        "n_transitions": n,
        "epochs_run": epochs_run,
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--n-workers", type=int, default=max(1, min(20, (os.cpu_count() or 8) - 2)))
    parser.add_argument("--episodes-per-worker", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--max-ppo-epochs", type=int, default=8)
    parser.add_argument("--target-kl", type=float, default=0.04)
    parser.add_argument("--minibatch-size", type=int, default=1536)
    # Dropped from 12000 -> 2000 after a live 8h run showed a real bug: at
    # 12000 with ~200 new episodes/generation, episodes sat in the buffer
    # for ~60 generations (2.5h+) before aging out, so most of every update
    # was importance-sampled against old_log_probs from a policy dozens of
    # generations stale. PPO's clipped surrogate objective assumes
    # near-on-policy data; this broke that assumption and produced a real,
    # sustained eval_margin regression (confirmed via training_log.csv:
    # steady degradation from -4604.8 best down to -9229.9 over ~1h once
    # the buffer filled), even though mean_margin on the stale-mixed batch
    # looked flat/noisy. 2000 episodes / 200 new per gen = ~10 generations
    # turnover (~25-30min lag), which keeps data close enough to on-policy.
    # This trades away some GPU-utilization headroom (less data per update
    # = less GPU work) for actual training correctness -- see docs/tests/LOG.md.
    parser.add_argument("--replay-episodes", type=int, default=2000,
                         help="retained episode history the GPU update trains over each generation "
                              "(see ppo_update docstring -- this is what makes the update phase long "
                              "enough to matter against CPU-bound rollout collection)")
    parser.add_argument("--checkpoint-dir", default=str(REPO_ROOT / "output" / "training"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--gpu-target-low", type=float, default=65.0)
    parser.add_argument("--gpu-target-high", type=float, default=75.0)
    # Elite-retention safeguard: if eval keeps coming back meaningfully
    # worse than the best-known policy, the live net has likely drifted off
    # a good trajectory (the failure mode above) -- revert to best.pt
    # rather than let an unattended run keep training away from its own
    # best result for hours.
    parser.add_argument("--regression-margin", type=float, default=1500.0)
    parser.add_argument("--regression-patience", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    net = ActorCritic().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    log_path = ckpt_dir / "training_log.csv"

    generation = 0
    best_margin = float("-inf")
    regression_streak = 0
    resumed_replay = None
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        generation = ckpt["generation"]
        best_margin = ckpt.get("best_margin", float("-inf"))
        resumed_replay = ckpt.get("replay_buffer")
        print(f"Resumed from generation {generation}, best_margin={best_margin:.1f}, "
              f"replay_buffer={len(resumed_replay) if resumed_replay else 0} episodes")

    champion_config = json.loads((HERE / "best_config.json").read_text())

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "generation", "elapsed_s", "new_episodes", "replay_size", "n_transitions",
                "epochs_run", "mean_reward", "mean_margin", "eval_margin",
                "policy_loss", "value_loss", "entropy", "gpu_util_avg",
            ])

    throttle = GPUThrottle(args.gpu_target_low, args.gpu_target_high)
    throttle.start()

    mp_ctx = multiprocessing.get_context("spawn")  # CUDA lives in the main process; fork would corrupt it
    # +1 worker reserved for the periodic eval job -- without it, eval was
    # competing with the regular rollout batch for the same fixed pool and
    # stalling an entire generation (confirmed live: every eval_every'th
    # generation showed a ~110s stall at 0% GPU).
    executor = ProcessPoolExecutor(max_workers=args.n_workers + 1, mp_context=mp_ctx)
    replay_buffer = collections.deque(resumed_replay or [], maxlen=args.replay_episodes)

    def submit_generation():
        state_dict_cpu = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        jobs = []
        for _ in range(args.n_workers):
            opp = random.choices(OPPONENT_POOL, weights=OPPONENT_WEIGHTS, k=1)[0]
            seed = random.randint(0, 2**31 - 1)
            jobs.append(executor.submit(
                rollout_worker,
                (state_dict_cpu, opp, champion_config, args.episodes_per_worker,
                 args.episode_steps, seed, False),
            ))
        return jobs

    start_time = time.time()
    deadline = start_time + args.hours * 3600
    print(f"Training for {args.hours}h ({args.n_workers} workers, "
          f"{args.episodes_per_worker} episodes/worker/generation), GPU target "
          f"[{args.gpu_target_low},{args.gpu_target_high}]%, replay cap {args.replay_episodes} episodes")

    # Pipelining: the next generation's rollout jobs are submitted (using
    # whatever weights are current at that moment) immediately after
    # collecting the previous generation's results and BEFORE running the
    # GPU update -- so CPU workers are already collecting generation N+1
    # while the GPU is still updating on generation N's (replay-buffer-
    # extended) batch, instead of the two phases stacking sequentially.
    pending_jobs = submit_generation()
    pending_eval = None  # (future, generation_it_was_submitted_at) or None

    try:
        while time.time() < deadline:
            gen_start = time.time()
            new_episodes = []
            for j in pending_jobs:
                try:
                    new_episodes.extend(j.result())
                except Exception as exc:  # noqa: BLE001 -- one bad worker shouldn't kill an 8h run
                    print(f"worker failed, skipping: {exc!r}", file=sys.stderr)

            pending_jobs = submit_generation()

            if not new_episodes:
                print("generation produced no usable episodes, retrying")
                continue

            replay_buffer.extend(new_episodes)
            stats = ppo_update(net, optimizer, device, list(replay_buffer), args.minibatch_size,
                                throttle, max_epochs=args.max_ppo_epochs, target_kl=args.target_kl)
            generation += 1
            # Captured right after the GPU-bound update, before the eval
            # block's blocking (CPU-only) rollout collection -- otherwise
            # eval's ~90-110s of zero-GPU time pushes the update's real
            # readings out of the throttle's short rolling window before we
            # ever log them, making every eval_every'th generation falsely
            # read ~0% regardless of how the update itself actually went.
            gpu_avg = throttle.current_avg()

            mean_reward = sum(sum(t["reward"] for t in ep["transitions"]) for ep in new_episodes) / len(new_episodes)
            mean_margin = sum(ep["margin"] for ep in new_episodes) / len(new_episodes)

            # Non-blocking eval: check a previously-submitted eval future
            # without waiting on it (never stalls the main loop), and start
            # a new one if it's due and none is currently in flight. Uses
            # the reserved +1 worker (see executor construction above), so
            # it runs fully overlapped with regular rollout collection
            # instead of adding ~90-110s of dead time to whichever
            # generation happens to trigger it.
            eval_margin = ""
            if pending_eval is not None and pending_eval[0].done():
                eval_future, eval_gen = pending_eval
                try:
                    eval_episodes = eval_future.result()
                    eval_margin_val = sum(ep["margin"] for ep in eval_episodes) / max(1, len(eval_episodes))
                    eval_margin = f"{eval_margin_val:.1f}"
                    if eval_margin_val > best_margin:
                        best_margin = eval_margin_val
                        regression_streak = 0
                        torch.save({"model": net.state_dict(), "generation": eval_gen,
                                    "best_margin": best_margin}, best_path)
                        print(f"  new best (deterministic vs champion, gen {eval_gen}): margin={eval_margin_val:.1f}")
                    elif eval_margin_val < best_margin - args.regression_margin:
                        regression_streak += 1
                        print(f"  eval regressed ({eval_margin_val:.1f} vs best {best_margin:.1f}), "
                              f"streak={regression_streak}/{args.regression_patience}")
                        if regression_streak >= args.regression_patience and best_path.exists():
                            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
                            net.load_state_dict(best_ckpt["model"])
                            optimizer = torch.optim.Adam(net.parameters(), lr=LR)
                            replay_buffer.clear()
                            regression_streak = 0
                            print(f"  reverted to best.pt (gen {best_ckpt['generation']}, "
                                  f"margin={best_margin:.1f}) after {args.regression_patience} "
                                  f"consecutive regressed evals; replay buffer cleared")
                    else:
                        regression_streak = 0
                except Exception as exc:  # noqa: BLE001
                    print(f"eval failed: {exc!r}", file=sys.stderr)
                pending_eval = None

            if pending_eval is None and generation % args.eval_every == 0:
                eval_state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                eval_future = executor.submit(
                    rollout_worker,
                    (eval_state_dict, "champion", champion_config, args.eval_episodes,
                     args.episode_steps, random.randint(0, 2**31 - 1), True),
                )
                pending_eval = (eval_future, generation)

            torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "generation": generation, "best_margin": best_margin,
                        "replay_buffer": list(replay_buffer)}, latest_path)

            elapsed_total = time.time() - start_time
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    generation, f"{elapsed_total:.0f}", len(new_episodes), len(replay_buffer),
                    stats["n_transitions"] if stats else 0,
                    stats["epochs_run"] if stats else 0,
                    f"{mean_reward:.3f}", f"{mean_margin:.1f}", eval_margin,
                    f"{stats['policy_loss']:.4f}" if stats else "",
                    f"{stats['value_loss']:.4f}" if stats else "",
                    f"{stats['entropy']:.4f}" if stats else "",
                    f"{gpu_avg:.1f}",
                ])
            print(f"gen {generation:4d} | {elapsed_total/3600:5.2f}h elapsed | "
                  f"{len(new_episodes):3d} new / {len(replay_buffer):5d} replay eps | "
                  f"epochs {stats['epochs_run'] if stats else 0:2d} | mean_margin {mean_margin:9.1f} | "
                  f"gen_time {time.time()-gen_start:5.1f}s | gpu_avg {gpu_avg:5.1f}%", flush=True)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        throttle.stop()
        torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(),
                    "generation": generation, "best_margin": best_margin,
                    "replay_buffer": list(replay_buffer)}, latest_path)
        print(f"Stopped at generation {generation}, best_margin={best_margin:.1f}, "
              f"replay_buffer={len(replay_buffer)} episodes. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
