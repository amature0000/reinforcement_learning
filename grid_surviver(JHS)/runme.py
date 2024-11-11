from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor, evaluate
from agent import Agent

'''
Implement your agent by overriding knu_rl_env.grid_survivor.GridSurvivorAgent
'''
class GridSurvivorRLAgent(GridSurvivorAgent):
    def __init__(self):
        self.env = make_grid_survivor(
            show_screen=True # or, False
        )
        self.agent = Agent()
    def act(self, state):
        pass
    def fit(self, state):
        pass

    def save(self):
        self.agent.save()
    def load(self):
        self.agent.load()

    def train():
        pass


if __name__ == '__main__':
    #run_manual()
    agent = GridSurvivorRLAgent()
    agent.load()
    evaluate(agent)