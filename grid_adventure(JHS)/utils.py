import numpy as np
from collections import deque

def footstep(y, x, dir):
    if dir == 0: return y, x - 1 # L
    if dir == 1: return y, x + 1 # R
    if dir == 2: return y - 1, x # U
    if dir == 3: return y + 1, x # D
        
def process_map(obs):
    #obs = obs[0]
    map = np.zeros((26, 26))
    dx = 0; dy = 0
    for y in range(26):
        for x in range(26):
            if obs[y][x][0] == "A":
                dy = y
                dx = x
            elif obs[y][x][0] == "W": map[y][x] = 1
            elif obs[y][x][0] == "L": map[y][x] = 1
            elif len(obs[y][x]) == 3: 
                if obs[y][x][2] == "L": map[y][x] = 1
                if obs[y][x][2] == "C": map[y][x] = 1
            elif obs[y][x][0] == "K": map[y][x] = 1
    return map, dy, dx

def bfs(obs, sx, sy):
    map_grid, dx, dy = process_map(obs)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    visited = [[False for _ in range(26)] for _ in range(26)]
    visited[sx][sy] = True

    queue = deque()
    queue.append(((sx, sy), 0))

    while queue:
        current_pos, distance = queue.popleft()
        x, y = current_pos
        if (x, y) == (dx, dy):
            return distance

        for dir_x, dir_y in directions:
            new_x, new_y = x + dir_x, y + dir_y
            if 0 <= new_x < 26 and 0 <= new_y < 26:
                if not visited[new_x][new_y] and map_grid[new_x][new_y] == 0:
                    visited[new_x][new_y] = True
                    queue.append(((new_x, new_y), distance + 1))

    print("거리 정보를 계산할 수 없습니다.")
    return -1


D_POS = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
D_NAME = ['', '', '', '', '']
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

D_POS_search_end = False
py = 1
px = 1
pd = 1
k = 0
past_obs_p = np.array([py, px, pd, k, 0])
past_front = "E"
bfs_result = 0

def process_obs(obs):
    global D_POS, D_NAME, D_POS_search_end, py, px, pd, k, past_obs_p, past_front
    # character info(x,y,dir)
    for y, x in DIRECTIONS:
        if obs[py + y][px + x][0] == "A":
            px += x; py += y
            if obs[py][px][1] == "L": pd = 0 # L
            elif obs[py][px][1] == "R": pd = 1 # R
            elif obs[py][px][1] == "U": pd = 2 # U
            elif obs[py][px][1] == "D": pd = 3 # D
            break

    # initial door pos. search
    nth_door = 0
    if D_POS_search_end == False:
        for y in range(26):
            if nth_door == 5: break
            for x in range(26):
                if nth_door == 5: break
                if obs[y][x][0] == "D":
                    D_NAME[nth_door] = obs[y][x][1]
                    D_POS[nth_door] = [y, x]
                    nth_door += 1
    D_POS_search_end = True

    # door info(bitmap, detailed_info)
    rd_open = False; gd_open = False; bd_open = False
    rd_close = False; gd_close = False; bd_close = False
    nth_door = 0
    terminate = False
    door_state = 0
    for idx in range(5):        
        target = obs[D_POS[idx][0]][D_POS[idx][1]]
        temp = 1
        if target[0] == "D":
            # if door is closed
            if target[2] == "L" or target[2] == "C":
                temp = 0
                if target[1] == "R": rd_close = True
                elif target[1] == "G": gd_close = True
                elif target[1] == "B": bd_close = True
            # if door is open
            else:
                if target[1] == "R": rd_open = True
                elif target[1] == "G": gd_open = True
                elif target[1] == "B": bd_open = True
        # if door is open(case 2)
        else:
            if D_NAME[nth_door] == "R":  rd_open = True
            if D_NAME[nth_door] == "G": gd_open = True
            if D_NAME[nth_door] == "B": bd_open = True
        # key nearby chk
        for dy, dx in DIRECTIONS:
            new_target = obs[D_POS[idx][0] + dy][D_POS[idx][1] + dx]
            if new_target[0] == "K":
                print("way blocked")
                terminate = True
        door_state |= temp << nth_door
        nth_door += 1
    # key info (do I have key?)
    next_y, next_x = footstep(py, px, pd)
    front = obs[next_y][next_x]
    if past_front[0] == "K" and front[0] != "K" and py == past_obs_p[0] and px == past_obs_p[1] and pd == past_obs_p[2] :
        if past_front[1] == "R": k = 1
        elif past_front[1] == "G": k = 2
        elif past_front[1] == "B": k = 3
    elif front[0] == "K" and past_front[0] != "K" and py == past_obs_p[0] and px == past_obs_p[1] and pd == past_obs_p[2] :
        k = 0
    # make obs_p
    obs_p = np.array([py, px, pd, k, door_state]) 
    past_obs_p = obs_p
    past_front = front
    return obs_p, front, rd_open, gd_open, bd_open, rd_close, gd_close, bd_close, terminate, terminate

def reset_stats():
    global bfs_result, D_POS, D_NAME, DIRECTIONS, D_POS_search_end, py, px, pd, k, past_obs_p, past_front

    D_POS = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    D_NAME = ['', '', '', '', '']
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    D_POS_search_end = False
    py = 1
    px = 1
    pd = 1
    k = 0
    past_obs_p = np.array([py, px, pd, k, 0])
    past_front = "E"
    bfs_result = 0

def obs_equal(obs1, obs2):
    return np.array_equal(obs1[0], obs2[0]) and obs1[1:] == obs2[1:]

# ACTION_SPACE = [ACTION_LEFT, ACTION_RIGHT, ACTION_FORWARD, ACTION_PICKUP, ACTION_DROP, ACTION_UNLOCK]
def get_reward(obs, action, next_obs, origin_obs_second, origin_obs_third):
    global bfs_result
    terminate = False
    stupid = False
    next_obs_p, next_front, next_obs_rd_open, next_obs_gd_open, next_obs_bd_open, next_obs_rd_close, next_obs_gd_close, next_obs_bd_close, stupid, terminate = process_obs(next_obs)
    obs_p, front, obs_rd, obs_gd, obs_bd, obs_rd_close, obs_gd_close, obs_bd_close, _stupid, _terminate = process_obs(obs)

    
    # default value settings
    reward = 0
    bfs_result_current = bfs(next_obs, 1, 1)
    # no movement
    if bfs_result == bfs_result_current:
        reward = -5
    # no bfs
    if bfs_result_current == -1 or bfs(next_obs, 24, 24) == -1:
        print("road is blocked!")
        terminate = True
        stupid = True
    # move backward
    elif bfs_result > bfs_result_current:
        reward = -20 * (bfs_result - bfs_result_current)
    #else: reward = (bfs_result_current - bfs_result) * 2 
    bfs_result = bfs_result_current
    # heading to wall or door
    if action == 2 and front == "W":
        print("heading to wall")
        stupid = True
    elif action == 2 and len(front) == 3 and (front[2] == "L" or front[2] == "C"):
        print("heading to door")
        stupid = True
    # key pickup
    if action == 3:
        # already have key
        if obs_p[3] != 0:
            print("already have key!")
            stupid = True
        else:
            # empty key pickup
            if next_obs_p[3] == 0:
                print("there's no key")
                stupid = True
            # RK (in state)
            if next_obs_p[3] == 1:
                # is there any unlocked door? (in state)
                if next_obs_rd_open == False:
                    reward += 300
                    print("rk pick")
                else:
                    print("already open:rk")
                    stupid = True
            # GK (in state)
            if next_obs_p[3] == 2: 
                # is there any unlocked door? (in state)
                if next_obs_gd_open == False: 
                    reward += 300
                    print("gk pick")
                else:
                    print("already open:gk")
                    stupid = True
            # BK (in state)
            if next_obs_p[3] == 3:
                # is there any unlocked door? (in state)
                if next_obs_bd_open == False: 
                    reward += 300
                    print("bk pick")
                else:
                    print("already open:bk")
                    stupid = True
    # key drop
    elif action == 4:
        # drop key to obj   
        if front != "E" and obs_p[3] != 0:
            print("you can't build there")
            stupid = True
            terminate = True
        # I had RK, and any RD has been unlocked
        elif obs_p[3] == 1 and next_obs_rd_open == True: reward -= 200
        # I had GK, and any GD has been unlocked
        elif obs_p[3] == 2 and next_obs_gd_open == True: reward += 200
        # I had BK, and any BD has been unlocked
        elif obs_p[3] == 3 and next_obs_bd_open == True: reward += 200
        else : # useful key drop
            print("key drop")
            stupid = True
            # if I had any key, reset
            if obs_p[3] != 0: terminate = True
    # door interaction
    elif action == 5:
        # I locked door, reset
        if len(next_front) == 3 and next_front[2] == 'C':
            print("door locking action!")
            stupid = True
            terminate = True
        # door_state changed(better)
        if obs_p[4] < next_obs_p[4]: 
            reward += 400
            if next_obs_p[3] == 1: print("rd open")
            if next_obs_p[3] == 2: print("gd open")
            if next_obs_p[3] == 3: print("bd open")
        # useless open action
        else : stupid = True
    # I reached goal
    if next_obs_p[0] == 24 and next_obs_p[1] == 24:
        reward = 400
        terminate = True
        print("GOAL")
    # or LAVA
    elif origin_obs_second:
        terminate = True
        stupid = True
        print("LAVA!")
    # too many actions
    if origin_obs_third: 
        terminate = True
    #if obs_equal(obs, next_obs) and stupid == False: print(obs_p, action)
    return np.clip(float(reward)/500.0, -1.0, 1.0), terminate, stupid