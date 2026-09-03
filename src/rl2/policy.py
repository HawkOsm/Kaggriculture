import torch
import torch.nn as nn

class BCPolicy(nn.Module):
    def __init__(self, in_dim, out_dim=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, in_dim, out_dim=15):
        super().__init__()
        self.actor = BCPolicy(in_dim, out_dim)
        self.critic = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        logits = self.actor(x)
        val = self.critic(x)
        return logits, val
