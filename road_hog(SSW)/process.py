from knu_rl_env.road_hog import RoadHogAgent
from math  import degrees, acos
import numpy as np

ACTION_SPACE = [
    RoadHogAgent.FORWARD_ACCEL_RIGHT,
    RoadHogAgent.FORWARD_ACCEL_LEFT,
    RoadHogAgent.BACKWARD_ACCEL_LEFT,
    RoadHogAgent.BACKWARD_ACCEL_RIGHT
]

ACTION_SPACE_REVERSE = {
    RoadHogAgent.FORWARD_ACCEL_RIGHT:0,
    RoadHogAgent.FORWARD_ACCEL_LEFT:1,
    RoadHogAgent.BACKWARD_ACCEL_LEFT:2,
    RoadHogAgent.BACKWARD_ACCEL_RIGHT:3
}

def get_reward(obs, obs_):
    player = obs_["observation"][0]
    player_pre = obs["observation"][0]
    goal_spot = obs_["goal_spot"]
    reward = 0
    
    if obs_["is_crashed"]:
        reward += -0.09
    if not obs_["is_on_load"]:
        reward += -1
    if get_speed(player) < 4:
        reward -= 1
        
    dis = (player[0] - goal_spot[0])**2 + (player[1] - goal_spot[1])**2
    reward *= 0.1
    reward += -dis/62500
    return reward

def process_obs(obs):
    """
    state = [
        px, py, pdx, pdy, ps,
        진행방향 15도 단위로 일정거리내 장애물 유무(0, 1), => 8개
        on_load, is_crashed,
        goal_dist
    ]
    features : 16
    """
    player = obs["observation"][0]
    near = obs["observation"][1:]
    goal_spot = obs["goal_spot"]
    is_on_load = obs["is_on_load"]
    is_crashed = obs["is_crashed"]
    list_can_crash = [0] * 8
    for car in near:
        if goal_spot[0]==car[0] and goal_spot[1]==car[1]:
            continue
        direction = check_crash(player, car)
        if not direction:
            continue
        list_can_crash[direction-1] = 1

    state = [
        player[0], player[1], player[2], player[3], get_speed(player)
    ]
    state.extend(list_can_crash)
    state.extend([is_on_load, is_crashed, get_dis(player[0], player[1], goal_spot[0], goal_spot[1])])
    return np.array(state)

def get_degree(cos_value):
    if cos_value >= 0: return degrees(acos(cos_value))
    return 360 - degrees(acos(cos_value))

def get_dis(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def check_crash(car1, car2):

    x1, y1 = car1[0], car1[1]
    x2, y2 = car2[0], car2[1]
    # v1 = car1[3]/abs(car1[3])
    # v2 = car2[3]/abs(car2[3])
    deg1 = get_degree(car1[5]) * np.sign(car1[3]) + 180
    deg2 = get_degree(car2[5]) * np.sign(car2[3]) + 180
    
    deg_relative1 = 90
    deg_relative2 = deg2 - (90 - deg1)
    dis = get_dis(x1, y1, x2, y2)
    # print(deg1, deg2)
    if dis > 10:
        return 0
    
    if 30 <= deg_relative2 < 45:
        return 1
    elif 45 <= deg_relative2 < 60:
        return 2
    elif 60 <= deg_relative2 < 75:
        return 3
    elif 75 <= deg_relative2 < 90:
        return 4
    elif 90 <= deg_relative2 < 105:
        return 5
    elif 105 <= deg_relative2 < 120:
        return 6
    elif 120 <= deg_relative2 < 135:
        return 7
    elif 135 <= deg_relative2 < 150:
        return 8
    return 0

def get_speed(car):
    return car[2]*car[4]+car[3]*car[5]
