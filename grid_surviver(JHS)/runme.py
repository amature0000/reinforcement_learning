from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor, evaluate, run_manual
from agent import Agent
from utils import process_reward
from state import State
import numpy as np

SC = True

class GridSurvivorRLAgent(GridSurvivorAgent):
    def __init__(self):
        self.agent = Agent()

    def act(self, obs):
        return self.agent.choose_action(obs)
    def fit(self, obs):
        return self.agent.choose_action_while_train(obs)

    def save(self):
        self.agent.save()
        print("save")
    def load(self):
        self.agent.load()
        print("load")

    def train(self):
        cur_ep = 0
        cur_step = 0
        state = State()
        next_state = State()
        env = make_grid_survivor(show_screen=SC)
        # obs: 2-dim map, hit_points(hp)
        while True:
            cur_ep += 1
            print(f"{cur_ep=}")
            state.reset()
            next_state.reset()
            obs, _ = env.reset()
            while True:
                cur_step += 1

                state.process_state(obs)
                action = self.fit(obs)

                next_obs, _, terminated, truncated, _ = env.step(action)
                next_state.process_state(next_obs)
                reward = process_reward(next_state, terminated)
                done = terminated or truncated

                self.agent.update(state, action, reward, next_state, done)

                if done: break
                obs = next_obs
            self.agent.save()

if __name__ == '__main__':
    #run_manual()
    agent = GridSurvivorRLAgent()
    #agent.load()
    agent.train()
    evaluate(agent)