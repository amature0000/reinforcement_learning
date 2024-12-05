from math import degrees, acos
import heapq

# TODO: x좌표 따라 보상 주기
def process_reward(obs, next_obs):
    done = False
    # if no movement x: -0.01 (or zero?)
    reward = -0.01
    # if x moved right: positive
    # if x moved left: negative
    if obs["is_on_load"] == False or obs["is_crashed"] == True: 
        done = True
        reward = -2.0
    return reward, done

# TODO: normalization
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

    state = [
        player[0], player[1], get_degree(player[5]), # px, py, pd
        get_speed(player[2], player[3], player[4], player[5]), # ps
    ]
    for obj in near_sorted:
        state.extend([
            obj[0], obj[1], get_degree(obj[5]), # ox, oy, od
            get_speed(obj[2], obj[3], obj[4], obj[5])  # os
        ])
    for _ in range(padding_count): state.extend([0, 0, 0, 0]) # padding

    state.extend([is_on_load, is_crashed])
    return state

def get_degree(cos_value):
    if cos_value >= 0: return degrees(acos(cos_value))
    return 360 - degrees(acos(cos_value))

def get_speed(cos_value, sin_value, vx, vy):
    return (vx * cos_value) + (vy * sin_value)