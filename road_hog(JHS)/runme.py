import numpy as np
from knu_rl_env.road_hog import RoadHogAgent, make_road_hog, evaluate, run_manual
from agent import PolicyGradientAgent

class RoadHogRLAgent(RoadHogAgent):
    def __init__(self):
        self.env = make_road_hog(
            show_screen=False
        )
        self.agent = PolicyGradientAgent(features=...)

    def act(self, obs):
        return self.agent.select_action(obs, True)
    def fit(self, obs):
        return self.agent.select_action(obs)
    
    def save(self):
        self.agent.save()
    def load(self):
        self.agent.load()
    def store_reward(self, reward):
        self.agent.store_reward(reward)
    def update_policy(self):
        self.agent.update_policy()

    def train(self):
        curIt = 1
        while True:
            self.save()
            print(f"{curIt=}")
            obs, _ = self.env.reset()
            while True:
                action = self.fit(obs)
                obs, reward, done, _ = self.env.step(action)
                self.store_reward(reward)
                if done: break
            self.update_policy()
            curIt += 1

if __name__ == '__main__':
    #run_manual()
    agent = RoadHogRLAgent()
    agent.train()
    evaluate()