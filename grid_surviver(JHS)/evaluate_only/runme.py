from knu_rl_env.grid_survivor import GridSurvivorAgent, evaluate
#from lightgrid import GridSurvivorAgent, make_light_grid_survivor
import torch
from agent import DeepQNetwork
from state import State, process_reward

SHOW_SCREEN = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridSurvivorRLAgent(GridSurvivorAgent):
    def __init__(self):
        self.agent = DeepQNetwork(device=DEVICE)
        self.state = State()
        self.device = DEVICE

    def act(self, obs):
        self.state.process_state(obs)
        return self.agent.choose_action(self.state.features())
    
    def load(self):
        self.agent.policy_net.load_state_dict(torch.load("save.pth"))
        print("load")

# Main
if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(f"사용 중인 장치: {DEVICE}")
    # ===========================
    agent = GridSurvivorRLAgent()
    agent.load()
    evaluate(agent)