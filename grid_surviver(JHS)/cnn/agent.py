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

grid_size  = WIDTH * HEIGHT
ob_size = grid_size + 1

class DeepQNetwork:
    def __init__(self, n_features=ob_size, actions=3, epsilon=0.99, decay = 0.9999, min_epsilon=0.2, gamma=0.9, lr=1.0, batch_size = 64, update_freq=1000, buffer_size=200000, device='cpu'):
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
    def choose_action_while_train(self, grid):
        if random.random() < self.epsilon: return random.randint(0, 2)
        else: return self.choose_action(grid)
    # greedy
    def choose_action(self, grid):
        grid = torch.from_numpy(grid).float().to(self.device).view(1, CHANNELS, WIDTH, HEIGHT)
        temp = self.policy_net(grid)
        print(temp.detach().cpu().numpy()[0])
        with torch.no_grad(): return temp.max(1)[1].view(1, 1).item()

    def learn(self):
        if len(self.memory) < 10000: return # 충분한 버퍼 체크
        if len(self.memory) < self.batch_size: return # 최소한의 버퍼 체크
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # buffer에서 값 읽어오기
        grid_batch = torch.cat([s.view(1, CHANNELS, WIDTH, HEIGHT) for s in batch.state]).to(self.device)
        next_grid_batch = torch.cat([s.view(1, CHANNELS, WIDTH, HEIGHT) for s in batch.next_state]).to(self.device)
        batch_a = torch.cat(batch.action).long().to(self.device)
        batch_r = torch.cat(batch.reward).float().to(self.device)
        batch_done = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        # policy net에서 예측한 Q value
        Q_values = self.policy_net(grid_batch).gather(1, batch_a)
        
        # target net에서 예측한 next Q value
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_grid_batch).max(1)[0].detach()

        # target value 계산
        expected_Q_values = batch_r + (next_state_values * self.gamma) * (1 - batch_done)

        # Loss 계산
        criterion = nn.SmoothL1Loss() #nn.MSELoss()
        self.loss = criterion(Q_values, expected_Q_values.unsqueeze(1))

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


    def store_transition(self, grid, a, r, next_grid, done):
        self.memory.push(torch.from_numpy(grid).float(), a, r, torch.from_numpy(next_grid).float(), done)


class DQN(nn.Module):
    def __init__(self, num_actions=3):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=CHANNELS, out_channels=64, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, grid):
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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