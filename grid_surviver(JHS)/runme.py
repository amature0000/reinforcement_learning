from knu_rl_env.grid_survivor import GridSurvivorAgent, make_grid_survivor, evaluate, run_manual
from agent import Agent
from state import State

SC = True

class GridSurvivorRLAgent(GridSurvivorAgent):
    def __init__(self):
        self.agent = Agent()
        self.state = State()

    def act(self, obs):
        self.state.process_state(obs)
        return self.agent.choose_action(self.state)
    def test(self):
        return self.agent.choose_action_while_train(self.state)

    def save(self):
        self.agent.save()
        print("save")
    def load(self):
        self.agent.load()
        print("load")

    def train(self):
        episode = 0
        next_state = State()
        env = make_grid_survivor(show_screen=SC)
        # obs: 2-dim map, hit_points(hp)
        while True:
            episode += 1
            print(f"{episode=}")
            self.state.reset()
            next_state.reset()
            obs, _ = env.reset()
            while True:
                self.state.process_state(obs)
                action = self.test()

                obs, _, terminated, truncated, _ = env.step(action)
                next_state.process_state(obs)
                done = terminated or truncated

                self.agent.update(self.state, action, next_state, done)
                if done: break
            self.agent.save()

if __name__ == '__main__':
    #run_manual()
    agent = GridSurvivorRLAgent()
    #agent.load()
    agent.train()
    #evaluate(agent)