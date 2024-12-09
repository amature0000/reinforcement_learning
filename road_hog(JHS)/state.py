from math import degrees, acos
import numpy as np
import heapq

def process_reward(next_obs, terminated): # asserted "terminated" as "goal reached"
    reward = 0
    p = next_obs["observation"][0]
    speed = abs(get_speed(p[4], p[5], p[2], p[3]))
    # =====
    if speed == 0: reward = -0.1
    if next_obs["is_crashed"] == True: reward = -0.1
    if next_obs["is_on_load"] == False:  reward = -0.2
    if terminated: reward = 10.0
    return reward

def process_obs(obs, max_near=4):
    """
    state = [px, py, pd, ps,
            o1x, o1y, o1d, o1s,
            o2x, o2y, o2d, o2s,
            ...
            gx, gy, gd, gs
            is_on_load, is_crashed
            ]
    max_near is [0, 9]
    o is sorted, goalspot removed.
    4 + 4n + 4 + 2 features (max=46)
    """
    player = obs["observation"][0]
    near = obs["observation"][1:]
    goal_spot = obs["goal_spot"]
    is_on_load = obs["is_on_load"]
    is_crashed = obs["is_crashed"]

    def is_padding(obj): return obj[0] == 0 and obj[1] == 0
    def is_goal(obj, goal): return obj[0] == goal[0] and obj[1] == goal[1]
    def distance(obj): return (obj[0] - player[0])**2 + (obj[1] - player[1])**2 # squared l2 norm

    actual_near = [obj for obj in near if not is_padding(obj) and not is_goal(obj, goal_spot)]
    actual_near_sorted = heapq.nsmallest(max_near, actual_near, key=distance)
    padding_count = max_near - len(actual_near_sorted)

    state = normalize_features([
        player[0], player[1], get_degree(player[5]), # px, py, pd
        get_speed(player[2], player[3], player[4], player[5]), # ps
    ])
    for obj in actual_near_sorted:
        state.extend(normalize_features([
            obj[0], obj[1], get_degree(obj[5]), # ox, oy, od
            get_speed(obj[2], obj[3], obj[4], obj[5])  # os
        ]))
    for _ in range(padding_count): state.extend([0, 0, 0, 0]) # padding
    state.extend(normalize_features([
        goal_spot[0], goal_spot[1], get_degree(goal_spot[5]), #gx, gy, gd
        get_speed(goal_spot[2], goal_spot[3], goal_spot[4], goal_spot[5]) # gs
    ]))
    state.extend([is_on_load, is_crashed])
    return np.array(state)

def get_degree(cos_value):
    if cos_value >= 0: return degrees(acos(cos_value))
    return 360 - degrees(acos(cos_value))

def get_speed(cos_value, sin_value, vx, vy):
    return (vx * cos_value) + (vy * sin_value)

def normalize_features(list):
    list[0] = (list[0] + 250) / 330
    list[1] = (list[1] + 70) / 140
    list[2] = list[2] / 360
    list[3] = list[3] / 40
    return list