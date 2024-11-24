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
    
    def iter_eval(self):
        total_bee = 0
        total_step = 0
        best = 0
        episode = 0
        try:
            while True:
                env = make_grid_survivor(show_screen=SC)
                current_step = 0
                obs, _ = env.reset()
                self.state.process_state(obs)
                episode += 1
                while True:
                    current_step += 1
                    action = self.act()
                    
                    obs, _, terminated, truncated, _ = env.step(action)
                    self.state.process_state(obs)
                    if terminated or truncated: break
                total_step += current_step
                bee = 50 - self.state.b
                total_bee += bee
                best = max(best, bee)
                print(",", bee)
                #print("bee : ", bee, " / actions : ", current_step, " / total step : ", total_step, " / total bee : ", total_bee, " / eval step : ", total_step / episode, " / eval bee : ", total_bee / episode, " / best : ", best)
        except KeyboardInterrupt:
            print("Ctrl-C -> Exit")
            env.render()
            env.close()

# Main
if __name__ == "__main__":
    agent = GridSurvivorRLAgent()
    agent.load()
    agent.iter_eval()
    #evaluate(agent)
