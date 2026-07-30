"""Actor/critic networks plus the auxiliary MLP and backward LSTM used by Train.

MLP and BackwardsLSTM were referenced by the training loop in the original
script but never defined, so it could not run. Both are reconstructed here
to match the interfaces the training loop expects:
  - MLP:           state -> (mean, std), a plain feed-forward stand-in for
                   the "forward" action distribution used in the auxiliary loss.
  - BackwardsLSTM: single state -> (mean, std), an LSTM-based model used
                   during the reverse (backwards) pass over a rollout.
"""
import torch
from torch import nn
from torch.distributions import normal


def _init_linear_weights(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight)
            layer.bias.data.zero_()


class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.fc1 = nn.Linear(in_features=n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.mu = nn.Linear(in_features=64, out_features=n_actions)
        self.log_std = nn.Parameter(torch.zeros(1, n_actions))

        _init_linear_weights(self)

    def forward(self, inputs):
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)
        std = self.log_std.exp()
        dist = normal.Normal(mu, std)
        return dist, mu, std


class Critic(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states

        self.fc1 = nn.Linear(in_features=n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.value = nn.Linear(in_features=64, out_features=1)

        _init_linear_weights(self)

    def forward(self, inputs):
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        return self.value(x)


class MLP(nn.Module):
    """Feed-forward state -> (mean, std) network used for the auxiliary loss in Train."""

    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_features=n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.mean = nn.Linear(in_features=64, out_features=n_actions)
        self.log_std = nn.Parameter(torch.zeros(1, n_actions))

        _init_linear_weights(self)

    def forward(self, inputs):
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        std = self.log_std.exp()
        return mean, std


class BackwardsLSTM(nn.Module):
    """Single-state -> (mean, std) LSTM used during Train.backwardsPass."""

    def __init__(self, n_states, n_actions, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_states, hidden_size=hidden_size, batch_first=True)
        self.mean = nn.Linear(hidden_size, n_actions)
        self.log_std = nn.Parameter(torch.zeros(1, n_actions))

    def forward(self, state):
        x = state.view(1, 1, -1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        mean = self.mean(out)
        std = self.log_std.exp()
        return mean.squeeze(0), std
