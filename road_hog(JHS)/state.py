from math import degrees, acos
import numpy as np
import heapq

def process_reward(obs, next_obs):
    done = False
    x = obs["observation"][0][0]
    p = next_obs["observation"][0]
    next_x = p[0]
    spd = abs(get_speed(p[4], p[5], p[2], p[3]))
    # if x no movement
    reward = -1.0
    # if x moved
    if x < next_x: reward = 0.02 * spd
    elif x > next_x: reward = -0.02 * (20 - spd)
    if next_obs["is_crashed"] == True:
        reward += -1.0
    if next_obs["is_on_load"] == False: 
        done = True
        reward = -3.0
    #print(x, next_x, reward)
    #print(spd, end=" ")
    return reward, done

def process_obs(obs, max_near=4):
    """
    state = [px, py, pd, ps,
            o1x, o1y, o1d, o1s,
            o2x, o2y, o2d, o2s,
            ...
            is_on_load, is_crashed
            ]
    max_near is [0, 9]
    o is sorted
    4 + 4n + 2 features (max=42)
    """
    player = obs["observation"][0]
    near = obs["observation"][1:]
    is_on_load = obs["is_on_load"]
    is_crashed = obs["is_crashed"]

    def is_padding(obj): return obj[0] == 0 and obj[1] == 0
    def distance(obj): return (obj[0] - player[0])**2 + (obj[1] - player[1])**2 # non squared l2 norm

    actual_near = [obj for obj in near if not is_padding(obj)] # remove padding
    near_sorted = heapq.nsmallest(max_near, actual_near, key=distance) # sort non-padding list
    padding_count = max_near - len(near_sorted)

    state = normalize_features([
        player[0], player[1], get_degree(player[5]), # px, py, pd
        get_speed(player[2], player[3], player[4], player[5]), # ps
    ])
    for obj in near_sorted:
        state.extend(normalize_features([
            obj[0], obj[1], get_degree(obj[5]), # ox, oy, od
            get_speed(obj[2], obj[3], obj[4], obj[5])  # os
        ]))
    for _ in range(padding_count): state.extend([0, 0, 0, 0]) # padding

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