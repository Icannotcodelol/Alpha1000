"""PPO policy/value network with optional LSTM core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PolicyOutputs:
    logits: torch.Tensor
    value: torch.Tensor
    hidden: Tuple[torch.Tensor, torch.Tensor]


class PolicyValueNet(nn.Module):
    """Simple MLP + single-layer LSTM head producing policy logits and value."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 512,
        lstm_size: int = 256,
    ) -> None:
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=0.1),
        )
        self.lstm = nn.LSTM(hidden_size, lstm_size, batch_first=True)
        core_hidden = max(lstm_size // 2, 128)
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_size, core_hidden),
            nn.Tanh(),
            nn.Linear(core_hidden, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Linear(lstm_size, core_hidden),
            nn.Tanh(),
            nn.Linear(core_hidden, 1),
        )
        self.lstm_size = lstm_size

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> PolicyOutputs:
        """
        Args:
            obs: Tensor of shape (batch, obs_size) or (batch, seq_len, obs_size).
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        batch, seq_len, obs_size = obs.shape
        x = obs.view(batch * seq_len, obs_size)
        encoded = self.obs_encoder(x)
        encoded = encoded.view(batch, seq_len, -1)
        if hidden is None:
            hidden = self.initial_state(batch=batch, device=obs.device)

        lstm_out, next_hidden = self.lstm(encoded, hidden)
        core = lstm_out[:, -1, :]
        logits = self.policy_head(core)
        value = self.value_head(core).squeeze(-1)
        return PolicyOutputs(logits=logits, value=value, hidden=next_hidden)

    def initial_state(
        self,
        *,
        batch: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, self.lstm_size, device=device)
        c = torch.zeros(1, batch, self.lstm_size, device=device)
        return h, c


class LegacyPolicyValueNet(nn.Module):
    """Legacy network architecture (smaller MLP + LSTM)."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 256,
        lstm_size: int = 128,
    ) -> None:
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(hidden_size, lstm_size, batch_first=True)
        self.policy_head = nn.Linear(lstm_size, action_size)
        self.value_head = nn.Linear(lstm_size, 1)
        self.lstm_size = lstm_size

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> PolicyOutputs:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        batch, seq_len, obs_size = obs.shape
        x = obs.view(batch * seq_len, obs_size)
        encoded = self.obs_encoder(x)
        encoded = encoded.view(batch, seq_len, -1)
        if hidden is None:
            hidden = self.initial_state(batch=batch, device=obs.device)

        lstm_out, next_hidden = self.lstm(encoded, hidden)
        core = lstm_out[:, -1, :]
        logits = self.policy_head(core)
        value = self.value_head(core).squeeze(-1)
        return PolicyOutputs(logits=logits, value=value, hidden=next_hidden)

    def initial_state(
        self,
        *,
        batch: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch, self.lstm_size, device=device)
        c = torch.zeros(1, batch, self.lstm_size, device=device)
        return h, c


def build_policy_value_net(
    obs_size: int,
    action_size: int,
    *,
    hidden_size: int,
    lstm_size: int,
    legacy: bool = False,
) -> nn.Module:
    if legacy:
        return LegacyPolicyValueNet(obs_size=obs_size, action_size=action_size, hidden_size=hidden_size, lstm_size=lstm_size)
    return PolicyValueNet(obs_size=obs_size, action_size=action_size, hidden_size=hidden_size, lstm_size=lstm_size)
