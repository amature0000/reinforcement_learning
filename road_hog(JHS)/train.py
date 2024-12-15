from knu_rl_env.road_hog import make_road_hog
from agent import DQNAgent
from process import get_reward, process_obs, ACTION_SPACE, ACTION_SPACE_REVERSE
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UPDATE_INTERVAL = 10
SC = False

class Agent:
    def __init__(self):
        self.DQN_a = DQNAgent()
        self.DQN_b = DQNAgent()

    def act(self, obs):
        state = process_obs(obs)
        if state[-1] > 50: return ACTION_SPACE[self.DQN_a.choose_action(state)]
        return ACTION_SPACE[self.DQN_b.choose_action(state)]
    
    def save_model(self):
        torch.save(self.DQN_a.policy_net.state_dict(), "model_a.pth")
        torch.save(self.DQN_b.policy_net.state_dict(), "model_b.pth")
        print("save")

    def load_model(self):
        self.DQN_a.policy_net.load_state_dict(torch.load("model_a.pth"))
        self.DQN_b.policy_net.load_state_dict(torch.load("model_b.pth"))
        print("load")
    
    def train(self):
        cur_ep = 0
        cur_step = 0
        best = -9999
        best_done = False
        env = make_road_hog(show_screen=SC)
        
        while True:
            print("Start episode: ", cur_ep)
            obs, _ = env.reset() # _, done, truncated,

            out, crash = 0, 0
            total_reward = 0.0
            
            while True:
                cur_step += 1

                action = self.act(obs)
                # print("---action: ", action)
                
                obs_, _, done, truncated, _ = env.step(action)
                if obs_["time"] >= 40.0: truncated=True
                if not obs_["is_on_load"]: out += 1
                if obs_["is_crashed"]: crash += 1
                if done: best_done+=1

                reward = get_reward(obs, obs_, done)
                reward = torch.tensor([reward], device=device)
                total_reward += reward.item()

                state = process_obs(obs)
                state_ = process_obs(obs_)
                if state[-1] > 50:
                    self.DQN_a.store_transition(state, torch.tensor([[ACTION_SPACE_REVERSE[action]]], device=device, dtype=torch.long), reward, state_)
                else:
                    self.DQN_b.store_transition(state, torch.tensor([[ACTION_SPACE_REVERSE[action]]], device=device, dtype=torch.long), reward, state_)

                if len(self.DQN_a.memory) >= 2000 and cur_step % UPDATE_INTERVAL == 0:
                    self.DQN_a.learn()
                    print("DQN-A")
                    print("loss : ", self.DQN_a.loss.item())
                    print("epsilon : ", self.DQN_a.epsilon)
                    self.save_model()

                if len(self.DQN_b.memory) >= 2000 and cur_step % UPDATE_INTERVAL == 0:
                    self.DQN_b.learn()
                    print("DQN-B")
                    print("loss : ", self.DQN_b.loss.item())
                    print("epsilon : ", self.DQN_b.epsilon)
                    self.save_model()

                if done or truncated:
                    best = max(best, total_reward)
                    print("########### Done : ", total_reward, "Best : ", best, " / out : ", out, " / crash : ", crash, "/ done: ", done, "/ best_done: ", best_done)
                    break
                obs = obs_
            cur_ep += 1

# Main
if __name__ == "__main__":
    print(device)
    agent = Agent()
    agent.train()