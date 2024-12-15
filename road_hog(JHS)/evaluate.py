from knu_rl_env.road_hog import evaluate
from agent import DQNAgent
from process import process_obs, ACTION_SPACE
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UPDATE_INTERVAL = 10
SC = False

class Agent:
    def __init__(self):
        self.DQN_a = DQNAgent()
        self.DQN_b = DQNAgent()

    def act(self, obs):
        state = process_obs(obs)
        if state[-1] > 50: return ACTION_SPACE[self.DQN_a.evaluate(state)]
        return ACTION_SPACE[self.DQN_b.evaluate(state)]
    
    def load_model(self):
        self.DQN_a.policy_net.load_state_dict(torch.load("model_a.pth"))
        self.DQN_b.policy_net.load_state_dict(torch.load("model_b.pth"))
        print("load")
    
# Main
if __name__ == "__main__":
    print(device)
    agent = Agent()
    agent.load_model()
    evaluate(agent)