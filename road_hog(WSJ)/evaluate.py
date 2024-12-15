from knu_rl_env.road_hog import make_road_hog, evaluate
from dqn_agent import DQNAgent, device
from process import get_speed, car_in_view, normalize_obs, process_obs

import torch
import numpy as np
import math

UPDATE_INTERVAL = 10
SC = True

class Agent:
    def __init__(self):
        self.DQN = DQNAgent()
        self.DQN2 = DQNAgent()
        self.done_cnt = 0

    def act(self, obs):
        state = normalize_obs(process_obs(obs))
        dist = ((state[0] ** 2) + (state[1] ** 2)) ** 0.5
        if dist > 50:
            return self.DQN.evaluate(state)
        return self.DQN2.evaluate(state)

    def load_model(self):
        self.DQN.epsilon = 0.1
        self.DQN.policy_net.load_state_dict(torch.load("model_0.pth"))
        self.DQN2.epsilon = 0.1
        self.DQN2.policy_net.load_state_dict(torch.load("model2_1.pth"))

# Main
if __name__ == "__main__":
    print(device)
    agent = Agent()
    agent.load_model()
    evaluate(agent)