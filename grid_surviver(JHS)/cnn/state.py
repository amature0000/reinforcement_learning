from collections import deque
import numpy as np

L = 0
R = 1
U = 2
D = 3
MAX_COR = 34
CHANNELS = 8

class State:
    def __init__(self, py=16, px=17, pd=R, hp=0, b=0, h=0, k=0, bdist=0):
        self.px = px
        self.py = py
        self.pd = pd
        self.hp = hp
        self.b = b
        self.h = h
        self.k = k
        self.b_dist = bdist
        self.input_data = np.zeros((CHANNELS, MAX_COR, MAX_COR))
    def reset(self):
        self.__init__()
    
    def process_state(self, obs):
        char_map = obs['grid']
        # reset states
        self.saved = False
        self.damaged = False
        self.dead = False
        # get hp
        self.hp = obs['hit_points']
        # process map
        types = ['W', 'AL', 'AR', 'AU', 'AD', 'B', 'H', 'K'] # 'E'
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
                type = char_map[y][x]
                if type in types:
                    channel_idx = types.index(type)
                    self.input_data[channel_idx, y, x] = 1
                if type == "B": b_cnt += 1
                elif type == "H": h_cnt += 1
                elif type == "K": k_cnt += 1

        self.b = b_cnt
        self.h = h_cnt
        self.k = k_cnt
        # bfs
        _, self.b_dist = beefs(self.py, self.px, 'B', char_map)


def process_reward(state:State, next_state:State):
    # touched honeybee = 1.0
    if state.b > next_state.b: return 10.0
    # dead = -1.0
    if state.k > next_state.k: return -10.0
    # dead = -1.0
    if next_state.hp < 10 : return -10.0
    # damaged = -0.2
    if state.hp > next_state.hp: return -2.0
    # no movement = -1.0
    if state.px == next_state.px and state.py == next_state.py and state.pd == next_state.pd: return -10.0
    # TODO: bfs based reward(normal:-.02, backward:-.2, forward:+.2)
    if state.b_dist > next_state.b_dist: return 0.2
    if state.b_dist < next_state.b_dist: return -0.2
    return -0.01

def beefs(start_y, start_x, target,map):
    queue = deque()
    rows = int(MAX_COR)
    cols = int(MAX_COR)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue.append((start_y, start_x, 0))
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    visited[start_y][start_x] = True

    while queue:
        current_y, current_x, distance = queue.popleft()
        # current block check
        if map[current_y][current_x][0] == target:
            return (current_y, current_x), distance
        # search
        for dy, dx in directions:
            nx, ny = current_x + dx, current_y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if not visited[ny][nx] and map[ny][nx] != 'W':
                    visited[ny][nx] = True
                    queue.append((ny, nx, distance + 1))
    return (start_y, start_x), -1