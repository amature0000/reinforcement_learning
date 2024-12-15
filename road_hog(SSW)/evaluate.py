from knu_rl_env.road_hog import evaluate
from agent import DQNAgent
from process import process_obs, ACTION_SPACE
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.DQN = DQNAgent()

    def act(self, obs):
        state = process_obs(obs)
        return ACTION_SPACE[self.DQN.evaluate(state)]
    
    def save_model(self, episode):
        model_filename = f"model_{episode}.pth"
        torch.save(self.DQN.policy_net.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    def load_model(self, model):
        self.DQN.epsilon = 0.0
        self.DQN.policy_net.load_state_dict(torch.load(model))

# Main
if __name__ == "__main__":
    print(device)
    agent = Agent()
    agent.load_model("model_1.pth")
    evaluate(agent)