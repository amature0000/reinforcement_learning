# TODO: 코드수정해야함
import numpy as np
from knu_rl_env.road_hog import RoadHogAgent, make_road_hog, evaluate, run_manual

class RoadHogRLAgent(RoadHogAgent):
    def __init__(self):
        self.env = make_road_hog(
            show_screen=False
        )
        self.agent = ...

    def act(self, state):
        return self.agent.select_action(state)
    def fit(self, state):
        return self.agent.select_action_while_train(state)
    
    def save(self):
        self.agent.save()
    def load(self):
        self.agent.load()

    def train(self):
        curIt = 1
        while True:
            self.save()
            print("\nstart iteration : " +  str(curIt))
            obs, _ = self.env.reset()
            while True:
                pass

            curIt += 1


if __name__ == '__main__':
    run_manual()