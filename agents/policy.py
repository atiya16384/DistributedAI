import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=128, output_dim=1):
        super(PolicyNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Outputs between -1 and 1
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Value function output
        )

    def forward(self, x):
        action = self.actor(x)
        value = self.critic(x)
        return action, value
