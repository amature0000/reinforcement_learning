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
        '''
        Return value is one of actions following:
        - GridSurvivorAgent.ACTION_LEFT
        - GridSurvivorAgent.ACTION_RIGHT
        - GridSurvivorAgent.ACTION_FORWARD
        '''
        pass
    def save(self):
        self.agent.save()
    def load(self):
        self.agent.load()

    def train():
        '''
        Below is to create the grid adventure environment.
        '''
        env = make_grid_survivor(
            show_screen=True # or, False
        )

if __name__ == '__main__':
    #run_manual()
    agent = GridSurvivorRLAgent()
    agent.load()
    evaluate(agent)