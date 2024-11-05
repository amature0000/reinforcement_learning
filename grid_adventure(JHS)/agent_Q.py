import numpy as np
from Q import QAgent
from knu_rl_env.grid_adventure import GridAdventureAgent, make_grid_adventure, evaluate, run_manual
from utils import get_reward, reset_stats

class GridAdventureRLAgent(GridAdventureAgent):
    def __init__(self):
        self.env = make_grid_adventure(
            show_screen=False
        )
        self.agent = QAgent()

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
        stepIdx = 0
        while True:
            self.save()
            print("\nstart iteration : " +  str(curIt))
            obs, _ = self.env.reset()
            reset_stats()
            # obs type : obs, reward, terminated, truncated, {}
            while True:
                stepIdx += 1

                action = self.fit(obs)
                strange_tuple_whatisthis = self.env.step(np.array([action], dtype=object))
                next_obs = strange_tuple_whatisthis[0]
                origin_obs_second = strange_tuple_whatisthis[2]
                origin_obs_third = strange_tuple_whatisthis[3]

                reward, terminate, stupid = get_reward(obs, action, next_obs, origin_obs_second, origin_obs_third)
                self.agent.update(obs, action, reward, next_obs, terminate, stupid)
                #print(f'{1+reward:.2f}')
                obs = next_obs
                if terminate:
                    break

            curIt += 1


if __name__ == '__main__':
    #run_manual()
    agent = GridAdventureRLAgent()
    agent.load()
    #agent.train()

    evaluate(agent)