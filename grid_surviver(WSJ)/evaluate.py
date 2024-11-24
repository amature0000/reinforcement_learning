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
        self.ev = False

    def act(self, obs):
        self.state.process_state(obs)
        # print(self.state.features(), end = '  ')
        # self.agent.print_q_value(self.state.features())
        return self.agent.choose_action(self.state.features(), self.ev)
    
    def load(self):
        self.agent.load_model("q_save")
        self.ev = True
    
    def train(self):
        current_episode = 0
        env = make_grid_survivor(show_screen=SC)
        next_state = State()
        total_step = 0
        ema = 0.0
        best = 0
        try:
            while True:
                current_step = 0
                obs, _ = env.reset()

                while True:
                    current_step += 1
                    action = self.act(obs)
                    
                    next_obs, _, terminated, truncated, _ = env.step(action)
                    next_state.process_state(next_obs)
                    reward = process_reward(self.state, next_state)
                    done = terminated or truncated
                    # if _reward == -10.0: done = True
                    # self.agent.store_transition(self.state.features(), action, reward, next_state.features(), done)

                    self.agent.learn(self.state.features(), action, reward, next_state.features())
                    if done: break
                    obs = next_obs
                current_episode += 1
                total_step += current_step
                best = max(best, 50-next_state.b)
                ema = (ema * 0.9) + ((50-next_state.b) * 0.1)
                print("bee : ", 50 - next_state.b, " / actions : ", current_step, " / epsilon : ", self.agent.epsilon, "lr : ", self.agent.lr, " / total step : ", total_step, " / ema : ", ema, " / best : ", best)
                self.agent.save_model("q_save")
        except KeyboardInterrupt:
            print("Ctrl-C -> Exit")
        finally:
            env.render()
            env.close()

# Main
if __name__ == "__main__":
    agent = GridSurvivorRLAgent()
    agent.load()
    agent.train()
    #evaluate(agent)
