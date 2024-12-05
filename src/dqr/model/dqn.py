from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torchcontrib.optim import SWA

from dqr.model.mdp import BasicBuffer, State
from dqr.util.preprocess import get_model_inputs, get_multiple_model_inputs

if TYPE_CHECKING:
    import pandas as pd

    from dqr.model.mdp import Batch


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 32), nn.ReLU(), nn.Linear(32, self.output_dim)
        )

    def forward(self, state):
        return self.fc(state)


class DQNAgent:
    def __init__(
        self,
        input_dim: int,
        dataset: pd.DataFrame,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        buffer: BasicBuffer | None = None,
        buffer_size: int = 10000,
        tau: float = 0.999,
        swa: bool = False,
        pre_trained_model: DQN | None = None,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.model = DQN(input_dim, 1)
        if pre_trained_model:
            self.model = pre_trained_model
        base_opt = torch.optim.Adam(self.model.parameters())
        self.dataset = dataset
        self.MSE_loss = nn.MSELoss()
        self.replay_buffer = buffer or BasicBuffer(30000)
        if swa:
            self.optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
        else:
            self.optimizer = base_opt

    def get_action(self, state: State, dataset: pd.DataFrame | None = None) -> str:
        if dataset is None:
            dataset = self.dataset
        inputs = get_multiple_model_inputs(state, state.remaining, dataset)
        model_inputs = autograd.Variable(torch.from_numpy(inputs).float().unsqueeze(0))
        expected_returns = self.model.forward(model_inputs)
        value, index = expected_returns.max(1)
        return state.remaining[index[0]]

    def compute_loss(
        self,
        batch: Batch,
        dataset,
        verbose=False,
    ) -> torch.Tensor:
        states, actions, rewards, next_states, dones = batch
        model_inputs = np.array(
            [
                get_model_inputs(states[i], actions[i], dataset)
                for i in range(len(states))
            ]
        )
        model_inputs = torch.FloatTensor(model_inputs)

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(model_inputs)
        model_inputs = np.array(
            [
                get_model_inputs(next_states[i], actions[i], dataset)
                for i in range(len(next_states))
            ]
        )
        model_inputs = torch.FloatTensor(model_inputs)
        next_Q = self.model.forward(model_inputs)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_Q

        if verbose:
            print(curr_Q, expected_Q)
        loss = self.MSE_loss(curr_Q.squeeze(0), expected_Q.detach())
        return loss

    def update(self, batch_size, verbose=False):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch, self.dataset, verbose)
        train_loss = loss.float()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if isinstance(self.optimizer, SWA):
            self.optimizer.swap_swa_sgd()
        return train_loss
