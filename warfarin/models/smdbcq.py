"""Implementation of the SMDP-formulated dBCQ model."""

import copy

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class FCQ(nn.Module):
    """
    A feed-forward MLP implementation of the Q network.
    """

    def __init__(self, state_dim, num_actions, hidden_states=10, num_layers=2):
        """
        Args:
            state_dim: The dimensionality of the state space.
            num_actions: The number of actions in the discrete action space.
            hidden_states: The number of hidden units in each hidden layer.
            num_layers: The number of layers of the Q-network.
        """
        super().__init__()

        if num_layers == 3:
            self.q1 = nn.Linear(state_dim, hidden_states)
            self.q2 = nn.Linear(hidden_states, hidden_states)
            self.q3 = nn.Linear(hidden_states, num_actions)

            self.i1 = nn.Linear(state_dim, hidden_states)
            self.i2 = nn.Linear(hidden_states, hidden_states)
            self.i3 = nn.Linear(hidden_states, num_actions)

        if num_layers == 2:
            self.q1 = nn.Linear(state_dim, hidden_states)
            self.q3 = nn.Linear(hidden_states, num_actions)

            self.i1 = nn.Linear(state_dim, hidden_states)
            self.i3 = nn.Linear(hidden_states, num_actions)

        self.num_layers = num_layers

    def forward(self, state):
        """
        Given a state, compute the estimated Q-value of each action and the log-
        probability of the generative model of the observed data.

        Args:
            state: The input state.

        Returns:
            qval: The estimated Q-value.
            lp: The log-probability of the generative model representing the
                observed policy.
            i: The raw output of the generative model.
        """
        if self.num_layers == 2:
            q = F.relu(self.q1(state))
            i = F.relu(self.i1(state))
            i = F.relu(self.i3(i))

        elif self.num_layers == 3:
            q = F.leaky_relu(self.q1(state))
            q = F.leaky_relu(self.q2(q))
            i = F.leaky_relu(self.i1(state))
            i = F.leaky_relu(self.i2(i))
            i = F.leaky_relu(self.i3(i))

        qval = self.q3(q)
        lp = F.log_softmax(i, dim=1)

        return qval, lp, i


class SMDBCQ(object):
    """
    Trainer class for the semi-Markov decision process formulation of the
    discrete batch-constrained Q-learning model.
    """

    def __init__(self,
                 num_actions,
                 state_dim,
                 device,
                 BCQ_threshold=0.3,
                 discount=0.99,
                 optimizer="Adam",
                 optimizer_parameters=None,
                 polyak_target_update=False,
                 target_update_frequency=8e3,
                 tau=0.005,
                 hidden_states=25,
                 num_layers=3):
        # Dimensionality of action space
        self.num_actions = num_actions

        # Observed probability ratio threshold
        self.tau = tau

        # Device to store the model on
        self.device = device

        # Build the network and optimizer
        self.q = FCQ(state_dim,
                     num_actions,
                     hidden_states=hidden_states,
                     num_layers=num_layers).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        if optimizer_parameters is None:
            optimizer_parameters = {}
        self.q_optimizer = getattr(torch.optim, optimizer)(
            self.q.parameters(), **optimizer_parameters
        )

        # Discount factor
        self.discount = discount

        # Target update rule
        self.maybe_update_target = (self.polyak_target_update
                                    if polyak_target_update
                                    else self.copy_target_update)
        self.target_update_frequency = target_update_frequency

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)

        # Threshold for "unlikely" actions
        self.threshold = BCQ_threshold

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state: torch.Tensor):
        """
        Select the action with the maximum predicted Q-value.

        Args:
            state: The input state.

        Returns:
            actions: The actions selected by the model.
        """
        with torch.no_grad():
            q, imt, _ = self.q(state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()
            actions = np.array(
                (imt * q + (1. - imt) * -1e8).argmax(1).to("cpu")
            )
            actions = actions.reshape(-1, 1)

            return actions

    def train(self, replay_buffer):
        """
        Train the DBCQ model for with one batch of transitions.

        Args:
            replay_buffer: The replay buffer from which transitions should be
                           sampled.

        Returns:
            q_loss: The Q-network loss.
        """
        # Sample replay buffer
        k, state, action, next_state, reward, not_done = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            q, imt, i = self.q(next_state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[0] >= self.threshold).float()

            # Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

            q, imt, i = self.q_target(next_state)
            target_q = (
                reward +
                not_done * (self.discount**k) * q.gather(
                    1, next_action
                ).reshape(-1, 1)
            )

        # Get current Q estimate
        current_q, imt, i = self.q(state)
        current_q = current_q.gather(1, action)

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_q, target_q)
        i_loss = F.nll_loss(imt, action.reshape(-1))

        total_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        # Optimize the Q
        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        return total_loss

    def polyak_target_update(self):
        """
        Perform a moving average update of the target Q-network.
        """
        for param, target_param in zip(self.q.parameters(),
                                       self.q_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def copy_target_update(self):
        """
        Perform a copy update of the target Q-network.
        """
        if self.iterations % self.target_update_frequency == 0:
            print("Updating target Q Network")
            self.q_target.load_state_dict(self.q.state_dict())
