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
                 likelihood="ordered",
                 cutpoints=None,
                 device="cuda"):
        super().__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lr = lr
        if likelihood in ["discrete", "ordered"]:
            self.likelihood = likelihood
        else:
            raise ValueError("Likelihood must be one of 'discrete', 'ordered'")

        self.device = device

        if likelihood == "discrete":
            output_dim = num_actions
        else:
            output_dim = 2
            if cutpoints is None:
                cutpoints = torch.arange(num_actions - 1).float()
                cutpoints = cutpoints - cutpoints.mean()
            self.cutpoints = nn.Parameter(cutpoints.to(self.device),
                                          requires_grad=False)
        self.backbone = build_mlp(state_dim,
                                  hidden_dim,
                                  output_dim,
                                  num_layers).to(device)

        self.optim = Adam(self.backbone.parameters(), lr=lr)

    def log_prob(self, state):
        if self.likelihood == "discrete":
            logit = self.backbone(state)
            logprobs = F.log_softmax(logit, dim=1)
        elif self.likelihood == "ordered":
            backbone_output = self.backbone(state)
            loc = backbone_output[:, 0]
            scale = torch.exp(backbone_output[:, 1])
            cutpoint_shifted_scaled = (
                self.cutpoints.repeat(loc.shape[0], 1) - loc.unsqueeze(1)
            ) / scale.unsqueeze(1)
            cdf = torch.cat(
                (torch.zeros(loc.shape[0], 1).to(loc.device),
                 torch.special.expit(cutpoint_shifted_scaled),
                 torch.ones(loc.shape[0], 1).to(loc.device)),
                dim=1
            )
            # Add small offset to each bin to prevent zeros in loss
            probs = (cdf[:, 1:] - cdf[:, :-1] + 1e-8)
            probs /= probs.sum(dim=1).unsqueeze(1)
            logprobs = torch.log(probs)

        return logprobs

    def forward(self, state):
        if self.likelihood == "discrete":
            logit = self.backbone(state)
            probs = F.softmax(logit, dim=1)
        elif self.likelihood == "ordered":
            probs = torch.exp(self.log_prob(state))

        return probs

    def train(self, batch):
        self.optim.zero_grad()

        _, state, option, _, _, _ = batch
        logprob = self.log_prob(state)
        loss = F.cross_entropy(logprob, option.squeeze())
        loss.backward()

        self.optim.step()

        return loss.item()

    def save(self, filename):
        weights = self.state_dict()
        torch.save(weights, filename)

    @staticmethod
    def load(filename):
        params = torch.load(filename)

        # Get dimensionality of model from savefile
        keys = list(params.keys())
        input_key, output_key = "backbone.0.weight", keys[-1]
        state_dim = params[input_key].shape[1]
        hidden_dim = params[input_key].shape[0]
        num_layers = len([k for k in keys if "backbone" in k]) // 2
        if "cutpoints" in keys:
            likelihood = "ordered"
            num_actions = params["cutpoints"].shape[0] + 1
        else:
            likelihood = "discrete"
            num_actions = params[output_key].shape[0]

        model = BehaviorCloner(state_dim=state_dim,
                               num_actions=num_actions,
                               num_layers=num_layers,
                               hidden_dim=hidden_dim,
                               likelihood=likelihood,
                               lr=1e-3,
                               device="cuda")
        model.load_state_dict(params)

        return model
