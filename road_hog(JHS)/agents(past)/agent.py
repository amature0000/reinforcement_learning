import torch
import torch.nn as nn
import torch.optim as optim

# Policy Network 정의
class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.softmax(x, dim=-1)

# Agent 정의
class PolicyGradientAgent:
    def __init__(self, features, actions=9, gamma = 0.9999, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using : {self.device}")
        self.network = Network(features, 512, actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma  # Discount factor
        self.log_probs = []  # Store log probabilities of actions
        self.rewards = []  # Store rewards for each step

    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action_probs = self.network(obs_tensor)
        if deterministic:
            action = torch.argmax(action_probs)  # 확률이 가장 높은 행동 선택
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()  # 확률적으로 행동 선택 (학습용)
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # Normalize rewards for stability
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        # Calculate policy loss
        policy_loss = sum(-log_prob * reward for log_prob, reward in zip(self.log_probs, discounted_rewards))
        print(f"{policy_loss=:.3f}", end=" ")

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear memory
        self.log_probs = []
        self.rewards = []
    def save(self, filename="save"):
        torch.save(self.network.state_dict(), filename)
        print("save")
    def load(self, filename="save"):
        self.network.load_state_dict(torch.load(filename, map_location=self.device))
        self.network.eval()  # 평가 모드로 전환
        print("load")

