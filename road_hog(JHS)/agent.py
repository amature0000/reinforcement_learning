import torch
import torch.nn as nn
import torch.optim as optim

# Policy Network 정의
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Hidden layer
        self.fc2 = nn.Linear(128, action_dim)  # Output layer (action logits)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = self.fc2(x)  # Output logits (will convert to probabilities later)
        return torch.softmax(x, dim=-1)  # Action probabilities

# Agent 정의
class PolicyGradientAgent:
    def __init__(self, features, actions=9, gamma = 0.05, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"cuda = {self.device}")
        self.policy = PolicyNetwork(features, actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma  # Discount factor
        self.log_probs = []  # Store log probabilities of actions
        self.rewards = []  # Store rewards for each step

    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action_probs = self.policy(obs_tensor)
        if deterministic:
            action = torch.argmax(action_probs).item()  # 확률이 가장 높은 행동 선택
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()  # 확률적으로 행동 선택 (학습용)
        return action

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
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)  # Gradient ascent

        policy_loss = torch.cat(policy_loss).sum()

        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear memory
        self.log_probs = []
        self.rewards = []
    def save(self, filename="save"):
        torch.save(self.policy.state_dict(), filename)
        print("save")
    def load(self, filename="save"):
        self.policy.load_state_dict(torch.load(filename, map_location=self.device))
        self.policy.eval()  # 평가 모드로 전환
        print("load")

# 사용 예시
# 환경에서 관찰값 obs를 가져와 에이전트가 행동을 선택하는 루프
input_dim = 4  # Example input dimension (e.g., obs shape)
action_dim = 9  # Example action space size
agent = PolicyGradientAgent(input_dim, action_dim)

for episode in range(1000):  # Number of episodes
    obs = env.reset()  # Reset the environment (가정된 코드)
    total_reward = 0

    while True:
        # 에이전트가 행동 선택
        action = agent.select_action(obs)

        # 환경으로부터 새로운 상태, 보상, 종료 여부 받기 (가정된 코드)
        next_obs, reward, done, _ = env.step(action)
        agent.store_reward(reward)  # 보상 저장

        total_reward += reward
        obs = next_obs

        if done:
            break

    # 에피소드 종료 후 정책 업데이트
    agent.update_policy()
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")
