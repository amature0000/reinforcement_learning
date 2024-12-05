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
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        action_probs = torch.softmax(self.fc8(x), dim=-1)
        return action_probs

# Critic Network 정의
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        state_value = self.fc8(x)
        return state_value

# Actor-Critic TD Agent
class PolicyGradientAgent:
    def __init__(self, features, actions=9, gamma=0.99, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device : {self.device}")
        
        self.actor = ActorNetwork(features, 1024, actions).to(self.device)
        self.critic = CriticNetwork(features, 1024).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr*0.1)
        
        self.gamma = gamma
        
        self.log_prob = None  # 현재 스텝의 로그 확률
        self.value = None     # 현재 스텝의 가치
        self.reward = None    # 현재 스텝의 보상
        self.done = False     # 현재 스텝의 종료 여부
    
    def select_action(self, obs, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action_probs = self.actor(obs_tensor)
        state_value = self.critic(obs_tensor)
        formatted_probs = ['{:.3f}'.format(x) for x in action_probs.tolist()]
        print(formatted_probs)
        
        if deterministic:
            action = torch.argmax(action_probs)
            log_prob = torch.log(action_probs[action] + 1e-10)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        # 현재 스텝의 로그 확률과 가치 저장
        self.log_prob = log_prob
        self.value = state_value
        
        return action.item()
    
    def store_reward(self, reward, done):
        self.reward = reward
        self.done = done
    
    def update_policy(self, next_state=None):
        if next_state is not None:
            next_obs_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                next_value = self.critic(next_obs_tensor)
        else:
            next_value = torch.zeros(1).to(self.device)
        
        # TD 오차 계산
        td_target = self.reward + self.gamma * next_value * (1 - int(self.done))
        td_error = td_target - self.value
        
        # loss
        critic_loss = td_error.pow(2)
        actor_loss = -self.log_prob * td_error.detach()
        
        # Critic 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient Clipping (옵션)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        # Actor 업데이트
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient Clipping (옵션)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        self.log_prob = None
        self.value = None
        self.reward = None
        self.done = False
    
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