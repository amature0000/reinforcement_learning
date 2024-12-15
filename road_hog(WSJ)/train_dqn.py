from knu_rl_env.road_hog import make_road_hog, evaluate
from dqn_agent import DQNAgent, device

import torch
import numpy as np
import math

UPDATE_INTERVAL = 10
SC = True

def check_intersection(a, b, c, d, max_distance=60.0):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d == 0: return max_distance

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    if not (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)):
        return max_distance
    if not (min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
        return max_distance
    
    distance = ((px - x1) ** 2 + (py - y1) ** 2)**0.5
    return min(max_distance, distance)

def get_rotated_points(x, y, c, s, points):
    '''
        [[cos, sin] x
         [-sin, cos]] y
    '''

    rotated_points = []
    for px, py in points:
        rx = px - x
        ry = py - y

        new_rx = (c * rx) + (s * ry) + x
        new_ry = (-s * rx) + (c * ry) + y
        rotated_points.append((new_rx, new_ry))
    
    return rotated_points

def get_angles(x, y, c, s, dis=10.0):
    angles = [(dis * math.cos(math.radians(i)), dis * math.sin(math.radians(i))) for i in range(0, 360, 30)]

    ret = []
    for i in range(12):
        rx = angles[i][0]
        ry = angles[i][1]
        new_rx = (c * rx + s * ry) + x
        new_ry = (-s * rx + c * ry) + y
        ret.append((new_rx, new_ry))
    return ret


def get_corners(x, y):
    l,w=5,2
    return [(x - l/2, y - w/2), (x + l/2, y - w/2), (x + l/2, y + w/2), (x - l/2, y + w/2)]

def car_in_view(x, y, c, s, other_cars, max_distance=60.0, car_length=5.0, car_width=2.0):
    # corners = get_rotated_points(x, y, c, s, get_corners(x, y))
    others = []
    for car in other_cars:
        if car[0] == car[1] == car[2] == car[3] == 0: continue
        others.append(get_rotated_points(car[0], car[1], car[2], car[3], get_corners(car[0], car[1])))

    end_points = get_angles(x, y, c, s)
    ret = []
    for angles in range(12):
        min_dis = max_distance
        for car in others:
            for i in range(4):
                line_start = car[i]
                line_end = car[(i + 1) % 4]
                my_start = (x, y)
                my_end = end_points[angles]
                dis = check_intersection(my_start, my_end, line_start, line_end, max_distance=max_distance)
                min_dis = min(min_dis, dis)
        ret.append(min_dis)
    return ret

def normalize_obs(obs):
    return np.array(obs)
    obs[0] = ((obs[0] + 250) / 330) * 1.0
    obs[1] = ((obs[1] + 70) / 140) * 1.0
    obs[2] = obs[2] / 40
    obs[3] = obs[3] / 40
    for i in range(4, 16): obs[i] = obs[i] / 60

    return np.array(obs)

def get_speed(vx, vy, cos_value, sin_value):
    return (vx * cos_value) + (vy * sin_value)


class Agent:
    def __init__(self):
        self.DQN = DQNAgent()
        self.done_cnt = 0


    def process_obs(self, obs):
        player = obs["observation"][0]
        near = obs["observation"][1:]
        goal = obs["goal_spot"]
        is_on_load = obs["is_on_load"]
        is_crashed = obs["is_crashed"]

        near_xycs = [(i[0], i[1], i[4], i[5]) for i in near]
        speed = get_speed(player[4], player[5], player[2], player[3])
        check = car_in_view(player[0], player[1], player[4], player[5], near_xycs, max_distance=60.0) # min(6.0, abs(speed) * 2)

        state = [
            goal[0] - player[0], goal[1] - player[1], player[2], player[3], # px, py, vx, vy
            speed,
        ] + check

        return state


    def get_reward(self, obs, action, obs_):
        player = obs_["observation"][0]
        player_pre = obs["observation"][0]
        goal_spot = obs_["goal_spot"]

        if obs_["is_crashed"]: return -2.0
        if not obs_["is_on_load"]: return -0.5

        if abs(goal_spot[0] - player[0]) >= 300 or abs(goal_spot[1] - player[1]) >= 100:
            return -5.0

        dis = ((player[0] - goal_spot[0])**2 + (player[1] - goal_spot[1])**2)**0.5
        dis_pre = ((player_pre[0] - goal_spot[0])**2 + (player_pre[1] - goal_spot[1])**2)**0.5

        if dis < dis_pre: return ((dis_pre - dis) / 20) * (1 / (dis + 0.1)) / 5
        else: return -max(0.1, ((dis - dis_pre) / 20) * (1 / (dis + 0.1)) / 5)
        

    def act(self, obs):
        obs_ = normalize_obs(self.process_obs(obs))
        return self.DQN.choose_action(obs_)
    
    def save_model(self, episode):
        model_filename = f"model_{episode}.pth"
        torch.save(self.DQN.policy_net.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    def load_model(self, model):
        self.DQN.epsilon = 0.1
        self.DQN.policy_net.load_state_dict(torch.load(model))
    
    def train(self):
        cur_ep = 0
        cur_step = 0
        best = -9999

        env = make_road_hog(show_screen=SC)
        

        try:
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

                    reward = self.get_reward(obs, action, obs_)
                    reward = torch.tensor([reward], device=device)
                    total_reward += reward.item()

                    ori_obs = obs
                    obs = normalize_obs(self.process_obs(obs))
                    ori_obs_ = obs_
                    obs_ = normalize_obs(self.process_obs(obs_))
                    # print(obs_)

                    if not ori_obs_["is_on_load"]: out += 1
                    if ori_obs_["is_crashed"]: crash += 1
                    # time = ori_obs_["time"] # + (out * 0.6) + (crash * 10)

                    if done:
                        reward = torch.tensor([10.0], device=device)
                        self.done_cnt += 1

                    # done = done or reward < 0.0
                    if reward.item() < -3.0: truncated = True

                    self.DQN.store_transition(obs, torch.tensor([[action]], device=device, dtype=torch.long), reward, obs_)

                    if cur_step >= 1000 and cur_step % UPDATE_INTERVAL == 0:
                        self.DQN.learn()
                        print("loss : ", self.DQN.loss.item())
                        print("epsilon : ", self.DQN.epsilon)
                        if cur_step % (UPDATE_INTERVAL * 10) == 0: self.save_model(0)

                    if done or truncated:
                        best = max(best, total_reward)
                        print("########### Done : ", total_reward, "Best : ", best, " / out : ", out, " / crash : ", crash, " / done : ", self.done_cnt)
                        break

                    obs = ori_obs_

                cur_ep += 1

        except KeyboardInterrupt:
            print("Ctrl-C -> Exit")
        finally:
            env.render()
            env.close()
            print("Done")

# Main
if __name__ == "__main__":
    agent = Agent()
    # agent.train()
    agent.load_model("model_0.pth")
    evaluate(agent)


# -202, 0