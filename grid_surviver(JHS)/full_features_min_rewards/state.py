from collections import deque
import numpy as np

L = 0
U = 1
R = 2
D = 3
MAX_COR = 34
CHANNELS = 8

class State:
    def process_state(self, obs):
        char_map = obs['grid']
        # reset states
        self.saved = False
        self.damaged = False
        self.dead = False
        # get hp
        self.hp = obs['hit_points']
        # process map
        types = ['E', 'W', 'K', 'H', 'B']
        b_cnt = 0
        k_cnt = 0
        h_cnt = 0
        for y in range(MAX_COR):
            for x in range(MAX_COR):
                # 에이전트 위치 저장
                if char_map[y][x][0] == "A":
                    self.py, self.px = y, x
                    if char_map[y][x][1] == "L": self.pd = L
                    elif char_map[y][x][1] == "R": self.pd = R
                    elif char_map[y][x][1] == "U": self.pd = U
                    elif char_map[y][x][1] == "D": self.pd = D
                    self.upleft = types.index(char_map[y-1][x-1][0]) if y > 0 else -1
                    self.up = types.index(char_map[y-1][x][0]) if y > 0 else -1
                    self.upright = types.index(char_map[y-1][x+1][0]) if y > 0 else -1
                    self.left = types.index(char_map[y][x-1][0]) if x > 0 else -1
                    self.right = types.index(char_map[y][x+1][0]) if x < MAX_COR else -1
                    self.down = types.index(char_map[y+1][x][0]) if y < MAX_COR else -1
                    self.downleft = types.index(char_map[y+1][x-1][0]) if y < MAX_COR else -1
                    self.downright = types.index(char_map[y+1][x+1][0]) if y < MAX_COR else -1
                if char_map[y][x] == "B": b_cnt += 1
                elif char_map[y][x] == "H": h_cnt += 1
                elif char_map[y][x] == "K": k_cnt += 1
        self.b = b_cnt
        self.h = h_cnt
        self.k = k_cnt
        # bfs
        self.b_stat = beefs(self.py, self.px, self.pd, 'B', 50, char_map)
        self.h_stat = beefs(self.py, self.px, self.pd, 'H', 22, char_map)
        self.k_stat = beefs(self.py, self.px, self.pd, 'K', 8, char_map)
    def features(self):
        # xy 2 + hp 1 + b 50*2 + h 22*2 + k 8*2 + pd 4 + nearby 8*5 = 207 features
        normalized_py = float(self.py) / (MAX_COR - 1)
        normalized_px = float(self.px) / (MAX_COR - 1)

        normalized_b = (self.b_stat[:, :2] / (MAX_COR - 1)).flatten().tolist()
        normalized_h = (self.h_stat[:, :2] / (MAX_COR - 1)).flatten().tolist()
        normalized_k = (self.k_stat[:, :2] / (MAX_COR - 1)).flatten().tolist()

        normalized_hp = float(self.hp) / 100
        onehot_pd = onehot(self.pd, 4)

        onehot_nearby = []
        attrs = ['upleft', 'up', 'upright', 'left', 'right', 'downleft', 'down', 'downright']
        for attr in attrs:
            value = getattr(self, attr, -1)
            onehot_nearby.extend(onehot(value, 5))

        return [normalized_py, normalized_px, normalized_hp] + normalized_b + normalized_h + normalized_k + onehot_pd + onehot_nearby

def onehot(value, size):
    if 0 <= value < size:
        vec = [0.0] * size
        vec[value] = 1.0
        return vec
    else: return [0.0] * size

def process_reward(state:State, next_state:State):
    # touched honeybee
    if state.b > next_state.b: return 1.0
    # dead
    if state.k > next_state.k: return -1.0
    # dead
    if next_state.hp < 10 : return -1.0
    # damaged
    if state.hp > next_state.hp: return -0.1
    # no movement
    if state.px == next_state.px and state.py == next_state.py and state.pd == next_state.pd: return -1.0
    return 0

def beefs(start_y, start_x, dir, target, amount, map):
    if map[start_y][start_x][0] == target: return start_y, start_x, 0
    queue = deque()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue.append((start_y, start_x, dir, 0))
    visited = set()
    visited.add((start_y, start_x, dir))
    result = []
    while queue:
        current_y, current_x, current_dir, current_distance = queue.popleft()
        tempy, tempx = directions[current_dir]
        tempy += current_y
        tempx += current_x
        # boundary check
        if 0 < tempy < MAX_COR and 0 < tempx < MAX_COR:
            # return
            if map[tempy][tempx][0] == target: 
                result.append((tempy, tempx, current_distance + 1))
                if len(result) == amount: return np.array(result)
            # forward step
            if ((tempy, tempx, current_dir) not in visited) and (map[tempy][tempx][0] != 'W' or map[tempy][tempx][0] != 'K'):
                visited.add((tempy, tempx, current_dir))
                queue.append([tempy, tempx, current_dir, current_distance + 1])
        # rotate step
        for new_dir in [(current_dir + 1) % 4, (current_dir + 3) % 4]:
            if (current_y, current_x, new_dir) not in visited:
                visited.add((current_y, current_x, new_dir))
                queue.append((current_y, current_x, new_dir, current_distance + 1))
    result_array = np.array(result)
    if len(result_array) < amount:
        padding = np.array([(-1, -1, -1)] * (amount - len(result_array)))
        result_array = np.vstack([result_array, padding])
    return result_array