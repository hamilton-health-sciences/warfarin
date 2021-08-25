import torch.nn as nn
import torch.nn.functional as F



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

#         print(f"Q: {self.q3(q)}")
#         print(f"Imt: {F.log_softmax(i, dim=1)}")
        return self.q3(q), F.log_softmax(i, dim=1), i

