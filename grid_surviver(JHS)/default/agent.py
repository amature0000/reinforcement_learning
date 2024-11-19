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


class DeepQNetwork:
    def __init__(self, actions=3, epsilon=0.99, decay = 0.99, min_epsilon=0.05, gamma=0.9, lr=0.001, batch_size = 64, update_freq=1000, buffer_size=200000, device='cpu'):
        #self.n_features = n_features
        self.n_actions = actions
        self.epsilon = epsilon
        self.epsilon_decay = decay
        self.epsilon_min = min_epsilon
        self.gamma = gamma
        self.loss = torch.tensor(0)
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.lr)
        self.memory = ReplayBuffer(buffer_size)

        self.learn_step_counter = 0
        self.target_update = update_freq

        self.policy_net.apply(self.weights_init)

    # DQN layer의 weight을 적절한 값으로 초기화
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    # epsilon
    def choose_action_while_train(self, features):
        #temp = self.choose_action(features)
        if random.random() < self.epsilon: return random.randint(0, 2)
        return self.choose_action(features) # replace temp to track all Q-values
    # greedy
    def choose_action(self, features):
        state = torch.tensor(features, dtype=torch.float32).to(self.device).unsqueeze(0)
        q_values = self.policy_net(state)
        """
        #print(q_values.detach().cpu().numpy(), f"{self.epsilon:.2f}")
        q_value_print = q_values.tolist()
        max_index = q_value_print[0].index(max(q_value_print[0]))
        for i in range(3):
            if i == max_index: print("\033[95m" + f"{q_value_print[0][i]:.3f}" + "\033[0m", end=' ')
            else: print(f"{q_value_print[0][i]:.3f}", end=' ')
        print()
        """
        with torch.no_grad(): return q_values.max(1)[1].item()

    def learn(self):
        if len(self.memory) < 10000: return # 충분한 버퍼 체크
        if len(self.memory) < self.batch_size: return # 최소한의 버퍼 체크
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state, dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device).unsqueeze(1)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device).unsqueeze(1)

        # policy net에서 예측한 Q value
        Q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # target net에서 예측한 next Q value
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_Q_values = self.target_net(next_state_batch).gather(1, next_actions)
            expected_Q_values = reward_batch + (self.gamma * next_Q_values * (1 - done_batch))

        # Loss 계산
        criterion = nn.SmoothL1Loss() #nn.MSELoss()
        self.loss = criterion(Q_values, expected_Q_values)

        # 역전파 및 optimizer 스텝
        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # 타겟 네트워크 업데이트
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # epsilon 감소
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def store_transition(self, state, a, r, next_state, done):
        self.memory.push(state, a, r, next_state, done)


class DQN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, num_actions=3):
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
    
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)