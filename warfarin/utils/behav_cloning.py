# IMPORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Simple full-connected supervised network for Behavior Cloning of batch data
class FC_BC(nn.Module):
    def __init__(self, state_dim=33, num_actions=25, num_nodes=64):
        super(FC_BC, self).__init__()
        self.l1 = nn.Linear(state_dim, num_nodes)
        self.bn1 = nn.BatchNorm1d(num_nodes)
        self.l2 = nn.Linear(num_nodes, num_nodes)
        self.bn2 = nn.BatchNorm1d(num_nodes)
        self.l3 = nn.Linear(num_nodes, num_actions)

    def forward(self, state):
        out = F.relu(self.l1(state))
        out = self.bn1(out)
        out = F.relu(self.l2(out))
        out = self.bn2(out)
        return self.l3(out)



class BehaviorCloning(object):
    def __init__(self, input_dim, num_actions, num_nodes=256, learning_rate=1e-3, weight_decay=0.1, optimizer_type='adam', device='cpu'):
        '''Implement a fully-connected network that produces a supervised prediction of the actions
        preserved in the collected batch of data following observations of patient health.
        INPUTS:
        input_dim: int, the dimension of an input array. Default: 33
        num_actions: int, the number of actions available to choose from. Default: 25
        num_nodes: int, the number of nodes
        '''

        self.device = device
        self.state_shape = input_dim
        self.num_actions = num_actions
        self.lr = learning_rate

        # Initialize the network
        self.model = FC_BC(input_dim, num_actions, num_nodes).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=weight_decay)

        self.iterations = 0

    def train_epoch(self, train_dataloader):
        '''Sample batches of data from training dataloader, predict actions using the network,
        Update the parameters of the network using CrossEntropyLoss.'''

        losses = []

        # Loop through the training data
        for state, action in train_dataloader:
            t2 = time.time()
            state = state.to(self.device)
            action = action.to(self.device)
            t3 = time.time()
                
            # Predict the action with the network
            pred_actions = self.model(state)
            t4 = time.time()
            
            # Compute loss
            try:
                t0 = time.time()
                loss = self.loss_func(pred_actions, action.flatten())
                t1 = time.time()
            except:
                print("LOL ERRORS")

            # Optimize the network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            t5 = time.time()

            losses.append(loss.item())
#             print(f"Move device to {self.device}: {t3 - t2}, pred actions: {t4 - t3}, Loss function: {t1 - t0}, Backward step: {t5 - t1}")
            
        self.iterations += 1

        return np.mean(losses)
