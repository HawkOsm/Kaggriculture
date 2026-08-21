"""Actor-critic network for the RL policy layer (see train_rl.py).

The policy only ever outputs KNOB_SPECS-shaped values (see agent.py)
once per in-game day -- everything mechanical (movement, task priority,
harvesting) stays the proven rule-based code in agent.py untouched.
This network is deliberately larger than the ~12-dimensional control problem
strictly needs; see docs/tests/LOG.md for why (GPU utilization target on an
8-hour unattended training run).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent import FEATURE_NAMES, KNOB_SPECS

STATE_DIM = len(FEATURE_NAMES)
ACTION_DIM = len(KNOB_SPECS)


class ActorCritic(nn.Module):
    def __init__(self, hidden=2048, depth=4):
        super().__init__()
        layers = []
        d = STATE_DIM
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.alpha_head = nn.Linear(hidden, ACTION_DIM)
        self.beta_head = nn.Linear(hidden, ACTION_DIM)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        # softplus + 1 keeps alpha/beta > 1 -- a unimodal Beta, not the
        # degenerate U-shape that piles mass at 0/1 when either param < 1.
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        value = self.value_head(h).squeeze(-1)
        return alpha, beta, value

    @torch.no_grad()
    def act(self, state, deterministic=False):
        """state: 1D tensor [STATE_DIM]. Returns (action[ACTION_DIM] in
        [0,1], log_prob scalar, value scalar) -- all detached CPU tensors,
        for use during rollout collection (no grad needed there)."""
        alpha, beta, value = self.forward(state.unsqueeze(0))
        alpha, beta, value = alpha.squeeze(0), beta.squeeze(0), value.squeeze(0)
        dist = torch.distributions.Beta(alpha, beta)
        action = (alpha / (alpha + beta)) if deterministic else dist.sample()
        action = action.clamp(1e-4, 1 - 1e-4)
        log_prob = dist.log_prob(action).sum()
        return action, log_prob, value

    def evaluate(self, states, actions):
        """states: [N, STATE_DIM], actions: [N, ACTION_DIM] in [0,1] --
        used during the PPO update, with grad. Returns (log_probs[N],
        entropy[N], values[N])."""
        alpha, beta, values = self.forward(states)
        dist = torch.distributions.Beta(alpha, beta)
        actions = actions.clamp(1e-4, 1 - 1e-4)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, values
