from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor, evaluate, run_manual
import torch
from agent import DeepQNetwork
from state import State, process_reward
import numpy as np

SC = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GridSurvivorRLAgent(GridSurvivorAgent):
    def __init__(self):
        self.agent = DeepQNetwork(device=DEVICE)
        self.state = State()
        self.device = DEVICE

    def act(self, obs):
        self.state.process_state(obs)
        return self.agent.choose_action(self.state.input_data)
    def test(self, obs):
        self.state.process_state(obs)
        return self.agent.choose_action_while_train(self.state.input_data)
    
    def save(self):
        torch.save(self.agent.policy_net.state_dict(), "save.pth")
        print(f"save")

    def load(self):
        self.agent.epsilon_min = 0.0
        self.agent.epsilon = 0.0
        self.agent.policy_net.load_state_dict(torch.load("save.pth"))
    
    def train(self):
        current_episode = 0
        env = make_grid_survivor(show_screen=SC)
        next_state = State()
        try:
            while True:
                current_step = 0
                current_episode += 1
                print(f"{current_episode=}")
                obs, _ = env.reset()

                while True:
                    current_step += 1
                    action = self.test(obs)
                    
                    next_obs, _, terminated, truncated, _ = env.step(action)
                    next_state.process_state(next_obs)
                    reward = torch.tensor([process_reward(self.state, next_state)], device=self.device)
                    done = terminated or truncated
                    
                    self.agent.store_transition(self.state.input_data, action, reward, next_state.input_data)

                    if current_step % 10 == 0: self.agent.learn()
                    if done: break
                    obs = next_obs
                self.save()
        except KeyboardInterrupt:
            print("Ctrl-C -> Exit")
        finally:
            env.render()
            env.close()
            self.save()

# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(f"사용 중인 장치: {device}")
    agent = GridSurvivorRLAgent()
    #agent.load()
    agent.train()
    #evaluate(agent)
