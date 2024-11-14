import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pickle
from state import State
from collections import deque
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Agent:
    def __init__(self, 
                 alpha=0.001, 
                 gamma=0.99, 
                 epsilon=0.8, 
                 decay=0.999, 
                 min_epsilon=0.2, 
                 num_features=19, 
                 possible_actions=3, 
                 buffer_capacity=10000, 
                 batch_size=64, 
                 target_update_freq=1000,
                 device=DEVICE):
        """
        alpha = 학습률 (optimizer의 lr과 동일)
        gamma = 할인율
        epsilon = 초기 탐험 비율
        decay = epsilon 감쇠율
        min_epsilon = 최소 epsilon 값
        num_features = 상태의 특성 수
        possible_actions = 가능한 행동의 수
        buffer_capacity = 경험 재생 버퍼의 용량
        batch_size = 학습 배치 크기
        target_update_freq = 타겟 네트워크 업데이트 주기
        device = 'cpu' 또는 'cuda'
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.num_features = num_features
        self.possible_actions = possible_actions
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        print(self.device)

        # 메인 네트워크와 타겟 네트워크
        self.policy_net = DQNNetwork(num_features, possible_actions).to(self.device)
        self.target_net = DQNNetwork(num_features, possible_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 학습하지 않음

        # 옵티마이저와 손실 함수
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

        # 경험 재생 버퍼
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # 업데이트 카운터
        self.update_count = 0

    def save(self, filename='dqn_save.pkl'):
        save_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
    def load(self, filename='dqn_save.pkl'):
        with open(filename, 'rb') as f:
            save_dict = pickle.load(f)
            self.policy_net.load_state_dict(save_dict['policy_net_state_dict'])
            self.target_net.load_state_dict(save_dict['target_net_state_dict'])
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
            self.epsilon = save_dict['epsilon']

    def compute_q_values(self, state):
        """
        주어진 상태에 대한 모든 행동의 Q-값을 계산합니다.
        state: numpy array of shape (num_features,)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, num_features)
        with torch.no_grad():
            q_values = self.policy_net(state)  # (1, possible_actions)
        return q_values.cpu().numpy()[0]
    def compute_q_values(self, state):
        """
        주어진 상태에 대한 모든 행동의 Q-값을 계산합니다.
        state: numpy array of shape (num_features,)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, num_features)
        with torch.no_grad():
            q_values = self.policy_net(state)  # (1, possible_actions)
        return q_values.cpu().numpy()[0]
    
    def choose_action_while_train(self, state: State):
        """
        학습 중 행동을 선택합니다. epsilon-탐험 전략을 사용.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.possible_actions - 1)
        else: return self.choose_action(state)
    
    def choose_action(self, state: State):
        """
        학습 후 행동을 선택합니다. 탐험 없이 최대 Q-값을 가진 행동 선택.
        """
        features = state.process_features()
        q_values = self.compute_q_values(features)
        return np.argmax(q_values)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        경험을 저장합니다.
        state: numpy array (features)
        action: int
        reward: float
        next_state: numpy array (features)
        done: bool
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_model(self):
        """
        경험 재생 버퍼에서 샘플을 추출하여 신경망을 업데이트합니다.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # 충분한 경험이 쌓이지 않음
        
        # 경험 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 텐서 변환
        states = torch.FloatTensor(states).to(self.device)          # (batch_size, num_features)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # (batch_size, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # (batch_size, 1)
        next_states = torch.FloatTensor(next_states).to(self.device)  # (batch_size, num_features)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)      # (batch_size, 1)
        
        # 현재 Q-값
        current_q = self.policy_net(states).gather(1, actions)  # (batch_size, 1)
        
        # 타겟 Q-값
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)  # (batch_size, 1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))  # (batch_size, 1)
        
        # 손실 계산
        loss = self.criterion(current_q, target_q)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 타겟 네트워크 업데이트
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # epsilon 감쇠
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        
        # theta 변화량을 출력하여 학습 정도를 확인
        delta_theta_norm = loss.item()
        print(f"Update {self.update_count}: Loss={delta_theta_norm:.6f}, Epsilon={self.epsilon:.4f}")

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        input_dim = amount of input features
        output_dim = actions
        """
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  
        self.fc2 = nn.Linear(128, 128)        
        self.fc3 = nn.Linear(128, output_dim) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
"""
class Agent:
    def __init__(self, alpha = 0.01, gamma = 0.99, epsilon = 0.8, decay = 0.999, min_epsilon = 0.2, num_features = 17, possible_actions = 3):
        # alpha = learning rate
        # gamma = discount rate
        # epsilon = epsilon init value
        # decay = epsilon decay rate
        # min_epsilon = min epsilon value
        # theta = FA parameter
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.theta = np.random.uniform(-0.01, 0.01, num_features)
        self.possible_actions = list(range(possible_actions))

    def save(self):
        with open('save.pkl', 'wb') as file:
            pickle.dump({
                'theta': self.theta,
                'epsilon': self.epsilon
            }, file)
    def load(self):
        with open('save.pkl', 'rb') as file:
            data = pickle.load(file)
            self.theta = data['theta']
            self.epsilon = data['epsilon']

    def compute_q(self, features):
        return np.dot(self.theta, features)
    def compute_future_q(self, state:State):
        return np.array([self.compute_q(state.process_features(a)) for a in self.possible_actions])

    def choose_action_while_train(self, state):
        if random.random() < self.epsilon:
            return np.random.choice(self.possible_actions)
        else:
            return self.choose_action(state)
    def choose_action(self, state):
        q_values = self.compute_future_q(state)
        return np.argmax(q_values)

    def update(self, state:State, action, reward, next_state, done):
        features = state.process_features(action)
        q_next = 0 if done else max(self.compute_future_q(next_state))

        td_error = reward + self.gamma * q_next - self.compute_q(features)
        #print(td_error)
        self.theta += self.alpha * td_error * features
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        print(self.epsilon)
"""