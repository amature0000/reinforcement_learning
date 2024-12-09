import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple
from colorama import Fore

SC = True
def print_probs(probs):
    prob_list = probs.cpu().tolist()
    max_index = prob_list.index(max(prob_list))
    for i in range(9):
        if i == max_index: print(Fore.RED + f"{prob_list[i]:.3f}" + Fore.RESET, end=' ')
        else: print(f"{prob_list[i]:.3f}", end=' ')
    print()

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
        #x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x
    
class PPOAgent:
    def __init__(self, device, features, actions=9, epsilon=0.2, epsilon_decay=0.8, epsilon_min=0.05, gamma=0.9999, lr=0.0001, batch=128, memsize=1000):
        self.device = device
        self.features = features
        print(f"device : {self.device}")
        self.actions = actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.actor_loss = torch.tensor(0)
        self.critic_loss = torch.tensor(0)
        self.lr = lr
        self.batch_size = batch

        self.actor = NN(features, 1024, actions).to(self.device)
        self.critic = NN(features, 1024, 1).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(memsize)

        self.actor.apply(self.weights_init)

    def choose_action(self, state, deterministic=False):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            logits = self.actor(state)
            if deterministic:
                action = torch.argmax(logits)
            else: 
                probs = F.softmax(logits, dim=-1)
                print_probs(probs)
                action = torch.multinomial(probs, num_samples=1).item()
                self.log_prob = torch.log(probs[action])
        return action
    
    def store(self, state, action, prob, reward, next_state, done):
        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        self.memory.push(state, action, prob, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size: return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        batch_state = torch.cat(batch.state).view(self.batch_size, self.features).to(self.device)
        batch_action = torch.cat(batch.action).long().to(self.device)
        batch_reward = torch.cat(batch.reward).float().to(self.device)
        batch_next_state = torch.cat(batch.next_state).view(self.batch_size, self.features).to(self.device)
        batch_done = torch.tensor(batch.done, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            next_values = self.critic(batch_next_state).squeeze(1) * (1 - batch_done)
            values = self.critic(batch_state).squeeze(1)  # [batch_size]
            targets = batch_reward + self.gamma * next_values  # [batch_size]
            advantages = (targets - values).detach()  # [batch_size]

        old_log_probs = torch.stack(batch.log_prob).to(self.device)

        logits = self.actor(batch_state)
        probs = F.softmax(logits, dim=-1)
        new_log_probs = torch.log(probs.gather(1, batch_action).squeeze(1))
        ratio = torch.exp(new_log_probs - old_log_probs)

        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)        
        self.actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        entropy_weight = max(0.01, self.epsilon / 10)
        self.actor_loss = self.actor_loss - (entropy_weight * entropy)

        self.actor_optimizer.zero_grad()
        self.actor_loss.backward()
        self.actor_optimizer.step()

        values = self.critic(batch_state).squeeze(1)
        self.critic_loss = F.mse_loss(values, targets)

        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        self.memory.clear()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def prob(self): return self.log_prob

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)


Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'next_state', 'done'))
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()
    
    def __len__(self):
        return len(self.memory)