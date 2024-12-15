from knu_rl_env.road_hog import make_road_hog
from agent import DQNAgent
from process import get_reward, process_obs, ACTION_SPACE, ACTION_SPACE_REVERSE
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UPDATE_INTERVAL = 10
SC = False

class Agent:
    def __init__(self):
        self.DQN = DQNAgent()

    def act(self, obs):
        state = process_obs(obs)
        return ACTION_SPACE[self.DQN.choose_action(state)]
    
    def save_model(self, episode):
        model_filename = f"model_{episode}.pth"
        torch.save(self.DQN.policy_net.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    def load_model(self, model):
        self.DQN.epsilon = 0.0
        self.DQN.policy_net.load_state_dict(torch.load(model))
    
    def train(self):
        cur_ep = 0
        cur_step = 0
        best = -9999
        best_done = False
        env = make_road_hog(show_screen=SC)
        
        # try:
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

                reward = get_reward(obs, obs_)
                reward = torch.tensor([reward], device=device)
                total_reward += reward.item()

                ori_obs = obs
                obs = process_obs(obs)
                ori_obs_ = obs_
                obs_ = process_obs(obs_)

                if not ori_obs_["is_on_load"]: out += 1
                if ori_obs_["is_crashed"]: crash += 1

                if done:
                    reward = torch.tensor([10.0], device=device)
                    best_done+=1
                    # total_reward = 10

                # done = done or reward < 0.0
                if ori_obs_["time"] >= 40.0: truncated=True

                self.DQN.store_transition(obs, torch.tensor([[ACTION_SPACE_REVERSE[action]]], device=device, dtype=torch.long), reward, obs_)

                if cur_step >= 2000 and cur_step % UPDATE_INTERVAL == 0:
                    self.DQN.learn()
                    print("loss : ", self.DQN.loss.item())
                    print("epsilon : ", self.DQN.epsilon)
                    if cur_step % (UPDATE_INTERVAL * 10) == 0: self.save_model(1)

                if done or truncated:
                    best = max(best, total_reward)
                    print("########### Done : ", total_reward, "Best : ", best, " / out : ", out, " / crash : ", crash, "/ done: ", done, "/ best_done: ", best_done)
                    break
                obs = ori_obs_
            cur_ep += 1

# Main
if __name__ == "__main__":
    print(device)
    agent = Agent()
    agent.train()