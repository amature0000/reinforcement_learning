from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor, evaluate, run_manual
import torch
from qagent import QLearningAgent
from state import State, process_reward

SC = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GridSurvivorRLAgent(GridSurvivorAgent):
    def __init__(self):
        self.agent = QLearningAgent(3, 2 ** 12)
        self.state = State()
        self.device = DEVICE

    def act(self, obs=None):
        if obs != None: self.state.process_state(obs)
        return self.agent.choose_action(self.state.features(), True)
    
    def load(self):
        self.agent.load_model("q_save")

# Main
if __name__ == "__main__":
    agent = GridSurvivorRLAgent()
    agent.load()
    try: evaluate(agent)
    except: exit()
    finally: exit()
