import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_DECAY = 0.97
EPSILON_MIN = 0.02
LEARNING_RATE = 1e-5
TARGET_UPDATE = 100
DISCOUNT_FACTOR = 0.99
MEMORY_SIZE = 30000

ob_size = 17
ac_size = 9

class NN(nn.Module):
    def __init__(self, in_features=ob_size, num_actions=ac_size):
        super(NN, self).__init__()
        self.hidden_layer = 256
        self.fc1 = nn.Linear(in_features, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.hidden_layer*2)
        self.fc3 = nn.Linear(self.hidden_layer*2, self.hidden_layer*4)
        self.fc4 = nn.Linear(self.hidden_layer*4, self.hidden_layer*4)
        self.fc5 = nn.Linear(self.hidden_layer*4, self.hidden_layer*4)
        self.fc6 = nn.Linear(self.hidden_layer*4, self.hidden_layer*2)        
        self.fc7 = nn.Linear(self.hidden_layer*2, self.hidden_layer)
        self.fc8 = nn.Linear(self.hidden_layer, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

class DQNAgent:
    def __init__(self, n_features=ob_size, n_actions=ac_size):
        self.n_features = n_features
        self.n_actions = n_actions

        self.policy_net = NN().to(device)

    def choose_action(self, state):
        with torch.no_grad():
            return self.policy_net(torch.from_numpy(state).float().to(device).unsqueeze(0)).max(1)[1].view(1, 1)