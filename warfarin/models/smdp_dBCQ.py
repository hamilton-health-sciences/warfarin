import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_states=10, num_layers=2):
        super(FC_Q, self).__init__()

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

        return self.q3(q), F.log_softmax(i, dim=1), i



class discrete_BCQ(object):
    def __init__(
            self,
            num_actions,
            state_dim,
            device,
            BCQ_threshold=0.3,
            discount=0.99,
            optimizer="Adam",
            optimizer_parameters={},
            polyak_target_update=False,
            target_update_frequency=8e3,
            tau=0.005,
            initial_eps=1,
            end_eps=0.001,
            eps_decay_period=25e4,
            eval_eps=0.00,
            hidden_states=25,
            num_layers=3
    ):
        self.device = device

        # Determine network type
        self.Q = FC_Q(state_dim, num_actions, hidden_states=hidden_states, num_layers=num_layers).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        self.num_actions = num_actions

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Threshold for "unlikely" actions
        self.threshold = BCQ_threshold

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        with torch.no_grad():
            # NOTE: Had to convert state to cpu because it was expecting cpu instead of cuda
            state = torch.FloatTensor(state).reshape(self.state_shape).to(self.device)
            q, imt, i = self.Q(state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[0] > self.threshold).float()
            # Use large negative number to mask actions from argmax
            # return ((imt * q + (1. - imt) * -1e8).argmax(1)).astype(int)
            actions = np.array([[x] for x in np.array((imt * q + (1. - imt) * -1e8).argmax(1).to("cpu"))])
            return actions

    def train(self, replay_buffer, events_buffer, is_behav=False):

        # Sample replay buffer
        k, state, action, next_state, reward, done = replay_buffer.sample()
        events_k, events_state, events_action, events_next_state, events_reward, events_done = events_buffer.sample()

        state = torch.cat((state, events_state), 0)
        k = torch.cat((k, events_k), 0)
        action = torch.cat((action, events_action), 0)
        next_state = torch.cat((next_state, events_next_state), 0)
        reward = torch.cat((reward, events_reward), 0)
        done = torch.cat((done, events_done), 0)

        # Compute the target Q value
        with torch.no_grad():
            q, imt, i = self.Q(next_state)
            imt = imt.exp()
            imt = (imt / imt.max(1, keepdim=True)[0] >= self.threshold).float()

            # Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)

            q, imt, i = self.Q_target(next_state)
            target_Q = reward + done * (self.discount ** k) * q.gather(1, next_action).reshape(-1, 1)

        # Get current Q estimate
        current_Q, imt, i = self.Q(state)
        current_Q = current_Q.gather(1, action)

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        i_loss = F.nll_loss(imt, action.reshape(-1))

        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        return Q_loss


    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            print(f"Updating target Q Network")
            self.Q_target.load_state_dict(self.Q.state_dict())
