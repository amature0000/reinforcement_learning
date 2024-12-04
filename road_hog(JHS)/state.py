from math import degrees, acos

def process_reward(obs):
    done = False
    p = obs["observation"][0]
    reward = (abs(get_speed(p[4], p[5], p[2], p[3])) - 10) / 10
    if obs["is_on_load"] == False or obs["is_crashed"] == True: 
        done = True
        reward = -2.0
    return reward, done

def process_obs(obs):
    """
    state = [px, py, pd, ps,
            o1x, o1y, o1d, o1s,
            o2x, o2y, o2d, o2s,
            ...
            is_on_load, is_crashed
            ]
    o is sorted
    4 + 4n + 2 features (max=42)
    """
    player = obs["observation"][0]
    near = obs["observation"][1:]
    is_on_load = obs["is_on_load"]
    is_crashed = obs["is_crashed"]

    near = sorted(near, key=lambda obj: (obj[0] - player[0])**2 + (obj[1] - player[1])**2)

    state = [
        player[0], player[1], get_degree(player[5]), # px, py, pd
        get_speed(player[4], player[5], player[2], player[3]), # speed

        near[0][0], near[0][1], get_degree(near[0][5]),
        get_speed(near[0][4], near[0][5], near[0][2], near[0][3]),
        near[1][0], near[1][1], get_degree(near[1][5]),
        get_speed(near[1][4], near[1][5], near[1][2], near[1][3]),
        near[2][0], near[2][1], get_degree(near[2][5]),
        get_speed(near[1][4], near[1][5], near[1][2], near[1][3]),
        near[3][0], near[3][1], get_degree(near[3][5]),
        get_speed(near[1][4], near[1][5], near[1][2], near[1][3]),
        is_on_load, is_crashed
    ]
    return state

def get_degree(cos_value):
    if cos_value >= 0: return degrees(acos(cos_value))
    return 360 - degrees(acos(cos_value))

def get_speed(vx, vy, cos_value, sin_value):
    return (vx * cos_value) + (vy * sin_value)