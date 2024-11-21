from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor, evaluate, run_manual
import torch
from agent import DeepQNetwork, logging
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
    def test(self, obs):
        self.state.process_state(obs)
        return self.agent.choose_action_while_train(self.state.features())
    
    def save(self):
        torch.save(self.agent.policy_net.state_dict(), "save.pth")
        #print(f"save")

    def load(self):
        self.agent.policy_net.load_state_dict(torch.load("save.pth"))
        self.agent.policy_net.eval()
        print("load")
    
    def train(self):
        episode = 0
        env = make_grid_survivor(show_screen=SHOW_SCREEN)
        next_state = State()
        total_step = 0
        max_bee = 50
        try:
            while True:
                current_step = 0
                obs, _ = env.reset()
                stupid = 0
                while True:
                    current_step += 1
                    action = self.test(obs)
                    
                    next_obs, _, terminated, truncated, _ = env.step(action)
                    next_state.process_state(next_obs)
                    reward = process_reward(self.state, next_state)
                    done = terminated or truncated
                    temp = done
                    if reward == -10.0:
                        temp = True # 벽에 박으면 future reward를 0으로 설정한다.
                        stupid += 1
                    self.agent.store_transition(self.state.features(), action, reward, next_state.features(), temp)
                    if done: break
                    obs = next_obs
                max_bee = min(max_bee, next_state.b)
                print(f"{episode=}, {total_step=}, {current_step=}, {next_state.b=}, {self.agent.epsilon=:.2f}, {stupid=}, {max_bee=}")
                logging(self.agent.q_values)
                episode += 1
                total_step += current_step
                for _ in range(100): self.agent.learn()
                self.save()
        except KeyboardInterrupt:
            print("Ctrl-C -> Exit")
            env.render()
            env.close()
            self.save()
            exit()

# Main
if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(f"사용 중인 장치: {DEVICE}")
    # ===========================
    agent = GridSurvivorRLAgent()
    #agent.load()
    agent.train()
    #evaluate(agent)
