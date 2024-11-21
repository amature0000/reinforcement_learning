import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import numpy as np
np.set_printoptions(precision=3, suppress=True)

WIDTH = 34
HEIGHT = 34
CHANNELS = 8
SHOW_SCREEN = True
def logging(q_values):
    #print(q_values.detach().cpu().numpy(), f"{self.epsilon:.2f}")
    q_value_print = q_values.tolist()
    max_index = q_value_print[0].index(max(q_value_print[0]))
    for i in range(3):
        if i == max_index: print("\033[95m" + f"{q_value_print[0][i]:.3f}" + "\033[0m", end=' ')
        else: print(f"{q_value_print[0][i]:.3f}", end=' ')
    print()
    
class DeepQNetwork:
    def __init__(self, device='cpu'):
        self.device = device
        self.policy_net = DQN().to(device)

    def choose_action(self, features):
        state = torch.tensor(features, dtype=torch.float32).to(self.device).unsqueeze(0)
        q_values = self.policy_net(state)
        if SHOW_SCREEN: logging(q_values)
        with torch.no_grad(): return q_values.max(1)[1].item()

class DQN(nn.Module):
    def __init__(self, input_dim=35, hidden_dim=128, num_actions=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x