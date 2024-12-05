import torch
import torch.nn as nn
import torch.optim as optim

# Actor Network 정의
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action_probs = torch.softmax(self.fc4(x), dim=-1)
        return action_probs

# Critic Network 정의
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        state_value = self.fc4(x)
        return state_value

# Actor-Critic Agent
class PolicyGradientAgent:
    def __init__(self, features, actions=9, gamma=0.9999, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using : {self.device}")
        
        self.actor = ActorNetwork(features, 512, actions).to(self.device)
        self.critic = CriticNetwork(features, 512).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma  # Discount factor
        
        self.log_probs = []   # 행동의 로그 확률 저장
        self.values = []      # 상태 가치 저장
        self.rewards = []     # 보상 저장
        self.dones = []       # 에피소드 종료 여부 저장

    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action_probs = self.actor(obs_tensor)
        state_value = self.critic(obs_tensor)
        
        if deterministic:
            action = torch.argmax(action_probs)
            log_prob = torch.log(action_probs[action] + 1e-10)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()  # 확률적으로 행동 선택 (학습용)
            log_prob = dist.log_prob(action)
        
        self.log_probs.append(log_prob)
        self.values.append(state_value)
        
        return action.item()

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def update_policy(self, next_obs=None):
        # 다음 상태의 가치 예측
        next_value = torch.zeros(1).to(self.device)
        if next_obs is not None:
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
            next_value = self.critic(next_obs_tensor).detach()
        
        # 할인된 보상 계산
        discounted_rewards = []
        R = next_value
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            R = reward + self.gamma * R * (1 - done)
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.stack(discounted_rewards).squeeze().to(self.device)
        # 보상 정규화
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Critic의 가치 예측
        values = torch.stack(self.values).squeeze().to(self.device)
        
        # Advantage 계산
        advantages = discounted_rewards - values
        
        # Actor 손실 계산: -log_prob * Advantage
        actor_loss = - (torch.stack(self.log_probs) * advantages.detach()).mean()
        
        # Critic 손실 계산: MSE between predicted values and discounted rewards
        critic_loss = nn.functional.mse_loss(values, discounted_rewards)
        
        # Actor 업데이트
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Critic 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 메모리 초기화
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def save(self, filename="save.pth"):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
        print("Model saved")

    def load(self, filename="save.pth"):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor.eval()
        self.critic.eval()
        print("Model loaded")