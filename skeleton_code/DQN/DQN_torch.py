import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def normalize(state):
    # implement
    return state


BATCH_SIZE = 128 # batch size : 너무 크면 overfitting+오래걸림, 너무 작으면 학습이 불안정해짐
EPSILON_START = 0.9
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.0001
LEARNING_RATE = 0.0005
TARGET_UPDATE = 50 # n번 학습이 진행될때마다 target net을 policy net으로 update (크면 클 수록 학습이 안정적)
DISCOUNT_FACTOR = 0.9 
MEMORY_SIZE = 20000 # Replay Buffer size


class DQN(nn.Module):
    # DQN layer : input layer - FC256 - ReLU - FC128 - ReLU - FC64 - ReLU - FC32 - ReLU - output layer
    def __init__(self, in_features=ob_size, num_actions=ac_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
    
# (s, a, r, s_) transition 을 저장하기 위한 Replay Buffer
# Replayer Buffer에서 batch size만큼 sampling해서 학습함. 여긴 딱히 건드릴거 없음
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DeepQNetwork:
    def __init__(self, n_features=ob_size, n_actions=ac_size):
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.gamma = DISCOUNT_FACTOR
        self.loss = 0
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE

        self.policy_net = DQN().to(device) # CUDA 세팅안되어있으면 그냥 CPU로 돌아감
        self.target_net = DQN().to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr = self.lr) # learning rate을 적절히 조정해줌
        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.learn_step_counter = 0
        self.target_update = TARGET_UPDATE

        self.policy_net.apply(self.weights_init)

    # DQN layer의 weight을 적절한 값으로 초기화
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # epsilon-greedy
    def choose_action(self, state):
        if random.random() > self.epsilon: # follow policy
            with torch.no_grad():
                action = self.policy_net(torch.from_numpy(state).type(dtype).unsqueeze(0)).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

        return action
    

    def learn(self):
        if len(self.memory) < self.batch_size: return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # episode의 맨 마지막 (next state가 없는)을 필터링하기 위한 mask
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).view(self.batch_size, self.n_features)

        batch_s = torch.cat(batch.state).view(-1, self.n_features).to(device)
        batch_a = torch.cat(batch.action).long().to(device)
        batch_r = torch.cat(batch.reward).float().to(device)

        Q_values = self.policy_net(batch_s).gather(1, batch_a) # policy net에서 예측한 Q value
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_Q_values = (batch_r + next_state_values * self.gamma) # target net에서 예측한 Q value

        criterion = nn.SmoothL1Loss()
        self.loss = criterion(Q_values, expected_Q_values.unsqueeze(1)) # policy net과 target net에서 예측한 Q value 차

        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    def store_transition(self, s, a, r, s_):
        self.memory.push(torch.from_numpy(s).float(), a, r, torch.from_numpy(s_).float())




# Main
if __name__ == "__main__":
    DQNAgent = DeepQNetwork()

    cur_ep = 0
    cur_step = 0

    try:
        while True:
            print("Start episode: ", cur_ep)
            obs = env.reset()
            obs = normalize(obs)

            while True:
                cur_step += 1

                if obs is None:
                    obs = env.reset()
                    obs = normalize(obs)

                action = DQNAgent.choose_action(obs)

                # print("---action: ", action)
                
                obs_, reward, done, info = env.step(np.array([action.item()], dtype=object))
                reward = torch.tensor([reward], device=device)

                try:
                    obs_ = normalize(obs_)
                except:
                    continue


                DQNAgent.store_transition(obs, action, reward, obs_)


                if cur_step > 500 and cur_step % 50 == 0:
                    DQNAgent.learn()


                obs = obs_

    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
    finally:
        env.render()
        env.close()
        print("Done")
