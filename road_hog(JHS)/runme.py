import torch
from knu_rl_env.road_hog import RoadHogAgent, make_road_hog, evaluate, run_manual
from ppo_agent import PPOAgent
from state import process_reward, process_obs

UPDATE_INTERVAL = 128
SC = True
class RoadHogRLAgent(RoadHogAgent):
    def __init__(self):
        self.env = make_road_hog(show_screen=SC)
        max_near = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = PPOAgent(device = self.device, features=4 + 4 * max_near + 2)

    # save&load
    def save(self):
        torch.save(self.agent.actor.state_dict(), "save.pth")

    def load(self):
        self.agent.actor.load_state_dict(torch.load("save.pth"))
        print("load")
    # wrap
    def store(self, state, action, reward, next_state, done):
        self.agent.store(state, action, self.agent.prob(), reward, next_state, done)

    def learn(self):
        self.agent.learn()

    def act(self, obs):
        return self.agent.choose_action(process_obs(obs), True)
    
    def test(self, state):
        return self.agent.choose_action(state)

    def train(self):
        episode = 1
        while True:
            self.save()
            obs, _ = self.env.reset()
            state = process_obs(obs)
            rewards = 0
            while True:
                action = self.test(state)
                next_obs, _, terminated, truncated, _ = self.env.step(action)
                reward, done = process_reward(obs, next_obs)
                #print(reward)
                next_state = process_obs(next_obs)
                done = done or terminated or truncated

                self.store(state, torch.tensor([[action]], device=self.device, dtype=torch.long), torch.tensor([reward], device=self.device), next_state, done)
                rewards += reward

                if len(self.agent.memory) >= UPDATE_INTERVAL: 
                    print("learn")
                    self.learn()
                if done: break
                obs = next_obs
                state = next_state
            print(f"{episode=} {rewards=:.3f}")
            episode += 1

if __name__ == '__main__':
    #run_manual()
    agent = RoadHogRLAgent()
    agent.train()
    #evaluate()