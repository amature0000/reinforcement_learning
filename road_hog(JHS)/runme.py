import numpy as np
from knu_rl_env.road_hog import RoadHogAgent, make_road_hog, evaluate, run_manual
from actorcritic import PolicyGradientAgent
from state import process_reward, process_obs

SC = True
class RoadHogRLAgent(RoadHogAgent):
    def __init__(self):
        self.env = make_road_hog(show_screen=SC)
        max_near = 4
        self.agent = PolicyGradientAgent(features=4 + 4 * max_near + 2)

    def act(self, obs):
        return self.agent.select_action(process_obs(obs), True)
    def test(self, obs):
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
            print(f"{curIt=}", end=" ")
            obs, _, _, _, _ = self.env.reset()
            state = process_obs(obs)
            rewards = 0
            while True:
                action = self.test(state)

                next_obs, _, terminated, truncated, _ = self.env.step(action)
                next_state = process_obs(next_obs)
                reward, done = process_reward(obs, next_obs)
                done = done or terminated or truncated

                self.store_reward(reward, done)
                
                rewards += reward
                if done: break
                state = next_state
            self.update_policy()
            print(f"{rewards=}")
            curIt += 1

if __name__ == '__main__':
    #run_manual()
    agent = RoadHogRLAgent()
    agent.train()
    evaluate()