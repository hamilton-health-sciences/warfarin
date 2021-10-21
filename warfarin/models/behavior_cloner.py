"""Wrapper for the behavior cloning model."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from .nets import build_mlp


class BehaviorCloner(nn.Module):
    def __init__(self,
                 state_dim,
                 num_actions,
                 num_layers,
                 hidden_dim,
                 lr,
                 device):
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device

        self.backbone = build_mlp(state_dim,
                                  hidden_dim,
                                  num_actions,
                                  num_layers).to(device)
        self.optim = Adam(self.backbone.parameters(), lr=lr)

    def forward(self, state):
        logit = self.backbone(state)
        
        return F.softmax(logit, dim=1)

    def train(self, batch):
        self.optim.zero_grad()

        _, state, option, _, _, _ = batch
        logprob = F.log_softmax(self.backbone(state), dim=1)
        loss = F.cross_entropy(logprob, option.squeeze())
        loss.backward()

        self.optim.step()

        return loss.item()

    def save(self, filename):
        weights = self.backbone.state_dict()
        torch.save(weights, filename)
