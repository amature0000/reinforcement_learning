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

SHOW_SCREEN = False

def logging(q_values):
    
    q_value_print = q_values.tolist()

    max_index = q_value_print[0].index(max(q_value_print[0]))

    for i in range(3):

        if i == max_index: print("\033[95m" + f"{q_value_print[0][i]:.3f}" + "\033[0m", end=' ')

        else: print(f"{q_value_print[0][i]:.3f}", end=' ')

    print()

    

class DeepQNetwork:

    def __init__(self, actions=3, epsilon=0.99, decay = 0.999, min_epsilon=0.05, gamma=0.9, lr=0.001, batch_size = 64, update_freq=1000, buffer_size=200000, device='cpu'):

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

        state_grid = torch.tensor(features[0], dtype=torch.float32).to(self.device).unsqueeze(0)  # 배치 차원을 추가

        state_dir = torch.tensor(features[1], dtype=torch.float32).to(self.device).unsqueeze(0)

        q_values = self.policy_net(state_grid, state_dir)

        if SHOW_SCREEN: logging(q_values)

        with torch.no_grad(): return q_values.max(1)[1].item()



    def learn(self):

        if len(self.memory) < 10000: return # 충분한 버퍼 체크

        if len(self.memory) < self.batch_size: return # 최소한의 버퍼 체크

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        

        state_grid_batch = torch.cat([s.view(1, 4, 11, 11) for s in batch.state_grid]).to(self.device)

        state_dir_batch = torch.cat([s.unsqueeze(0) for s in batch.state_dir]).to(self.device)

        

        # state_grid_batch = torch.tensor(batch.state_grid, dtype=torch.float32).to(self.device)

        # state_dir_batch = torch.tensor(batch.state_dir, dtype=torch.float32).to(self.device)





        # next_grid_state_batch = torch.cat([torch.tensor(s.clone().detach(), dtype=torch.float32).view(1, 4, 11, 11) for s in batch.next_state_grid]).to(self.device)

        next_grid_state_batch = torch.cat([s.view(1, 4, 11, 11) for s in batch.next_state_grid]).to(self.device)

        next_dir_state_batch = torch.cat([s.unsqueeze(0) for s in batch.next_state_dir]).to(self.device)



        # next_grid_state_batch = torch.tensor(batch.next_state_grid, dtype=torch.float32).to(self.device)

        # next_dir_state_batch = torch.tensor(batch.next_state_dir, dtype=torch.float32).to(self.device)



        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device).unsqueeze(1)

        # print(action_batch, 'action')

        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device).unsqueeze(1)

        # print(reward_batch, 'reward')

        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device).unsqueeze(1)

        # print(done_batch, 'done')



        # policy net에서 예측한 Q value

        Q_values = self.policy_net(state_grid_batch, state_dir_batch).gather(1, action_batch)

        

        # target net에서 예측한 next Q value

        with torch.no_grad():

            next_actions = self.policy_net(next_grid_state_batch, next_dir_state_batch).max(1)[1].unsqueeze(1)

            next_Q_values = self.target_net(next_grid_state_batch, next_dir_state_batch).gather(1, next_actions)

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



    def store_transition(self, state_grid, state_dir, action, reward, next_state_grid, next_state_dir, done):

        self.memory.push((torch.from_numpy(state_grid).float(), torch.from_numpy(state_dir).float(), action, reward, torch.from_numpy(next_state_grid).float(), torch.from_numpy(next_state_dir).float(), done))





class DQN(nn.Module):

    def __init__(self, input_channels=4, hidden_dim=128, num_actions=3):

        super(DQN, self).__init__()
        
        
        
        

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)


        

        # FC 계층: CNN 출력 크기와 숫자 데이터 크기를 합친 값으로 정의

        self.fc1 = nn.Linear(128 * 11 * 11 + 9, 512)  # 추가된 숫자 데이터의 크기 포함

        self.fc2 = nn.Linear(512, num_actions)



    def forward(self, x, additional_features):

        # CNN 처리

        # print(x.shape, additional_features.shape)

        # self.conv1(x)

        x = torch.relu(self.conv1(x))

        x = torch.relu(self.conv2(x))

        x = torch.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # CNN 출력을 평탄화



        # 숫자 데이터와 CNN 출력을 결합

        x = torch.cat((x, additional_features), dim=1)  # 추가된 숫자 데이터를 결합



        # FC 계층 처리

        x = torch.relu(self.fc1(x))

        x = self.fc2(x)

        return x



    

Transition = namedtuple('Transition', ('state_grid', 'state_dir', 'action', 'reward', 'next_state_grid', 'next_state_dir', 'done'))

class ReplayBuffer(object):

    def __init__(self, capacity):

        self.memory = deque([], maxlen=capacity)

    

    def push(self, args):

        self.memory.append(Transition(*args))



    def sample(self, batch_size):

        return random.sample(self.memory, batch_size)

    

    def __len__(self):

        return len(self.memory)

"""

10by10

꿀벌

말벌

킬러



플레이어

상하좌우 벌 개수



"""