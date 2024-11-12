import numpy as np
from collections import deque

MAX_COR = 33
MAX_DIR = 3
MAX_HONEYBEE = 50
MAX_HORNET = 22
MAX_KILLERBEE = 8
MAX_HEALTH = 150
MAX_FAR = 63.0
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

class State:
    def __init__(self, px=0, py=0, pd=0, hp=0, kx=0, ky=0, hx=0, hy=0, bx=0, by=0, kdir=0, hdir=0, bdir=0):
        self.px = px
        self.py = py
        self.pd = pd
        self.hp = hp
        self.kx = kx
        self.ky = ky
        self.hx = hx
        self.hy = hy
        self.bx = bx
        self.by = by
        self.kdir = kdir
        self.hdir = hdir
        self.bdir = bdir
    def reset(self):
        self.__init__()

state = State()

def reset_state():
    global state
    state.reset()

def _process_state(obs_before_process):
    global state
    types = ['E', 'W', 'AL', 'AR', 'AU', 'AD', 'B', 'H', 'K']
    obs = obs_before_process['grid']
    #get hp
    state.hp = obs_before_process['hit_points']
    #process map and get pos
    processed_map = _process_map(obs)
    #find nearest obj by bfs
    tpos, tdir = _bfs(processed_map)
    state.by, state.bx = tpos['B']
    state.ky, state.kx = tpos['K']
    state.hy, state.hx = tpos['H']
    state.bdir = tdir['B']
    state.kdir = tdir['K']
    state.hdir = tdir['H']
    
def process_features(obs, action):
    # 16 features(px, py, pd, kx, ky, hx, hy, bx, by, kdir, hdir, bdir, hp, a1, a2, a3)
    global state
    _process_state(obs)
    features = []
    # coord incode
    features.append(state.px / MAX_COR)
    features.append(state.py / MAX_COR)
    features.append(state.pd / MAX_DIR)
    
    # killerbee incode
    features.append(state.kx / MAX_COR)
    features.append(state.ky / MAX_COR)

    # hornet incode
    features.append(state.hx / MAX_COR)
    features.append(state.hy / MAX_COR)
    
    # honeybee incode
    features.append(state.bx / MAX_COR)
    features.append(state.by / MAX_COR)

    # dir incode
    features.append(state.kdir / MAX_FAR)
    features.append(state.hdir / MAX_FAR)
    features.append(state.bdir / MAX_FAR)
    
    # hp incode
    features.append(state.hp) # 1 if hp > 10, else 0

    # action incode
    for i in range(3):
        features.append(1.0 if i == action else 0.0)

    return np.array(features)

def process_reward(obs, terminated):
    global state
    _process_state(obs)
    # touched all honeybee
    if state.bdir == None: return 1
    # dead
    if terminated: return -1
    # touched honeybee
    # TODO: implementation
    # basic movement
    return -0.01

# process map and extract current location
def _process_map(obs):
    global state
    types = {
        'E': 0,
        'W': 1,
        'A': 0,
        'B': 2,
        'H': 3,
        'K': 4
    }    
    processed_map = []
    
    for y in range(len(obs)):
        row = []
        for x in range(len(obs[0])):
            char = obs[y][x][0]
            num = types[char]
            row.append(num)
            if char == 'A':
                state.py = y
                state.px = x
                if obs[state.py][state.px][1] == "L": state.pd = 0 # L
                elif obs[state.py][state.px][1] == "R": state.pd = 1 # R
                elif obs[state.py][state.px][1] == "U": state.pd = 2 # U
                elif obs[state.py][state.px][1] == "D": state.pd = 3 # D
        processed_map.append(row)
    return processed_map

# bfs algorithm to find nearest objects' information
def _bfs(processed_map):
    queue = deque()
    rows = len(processed_map)
    cols = len(processed_map[0]) if rows > 0 else 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    start_x, start_y = state.px, state.py
    queue.append((start_x, start_y, 0))
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    visited[start_y][start_x] = True

    targets = {'B': 2, 'H': 3, 'K': 4}
    target_position = {'B': (None, None), 'H': (None, None), 'K': (None, None)}
    target_distance = {'B': None, 'H': None, 'K': None}
    target_found = {'B': False, 'H': False, 'K': False}

    while queue:
        current_x, current_y, distance = queue.popleft()
        # current block check
        for key, value in targets.items():
            if processed_map[current_y][current_x] == value and not target_found[key]:
                target_position[key] = (current_y, current_x)
                target_distance[key] = distance
                target_found[key] = True
                #print(f"{key} : ({current_x}, {current_y}), {distance}")
                # early exit
                if all(target_found.values()):
                    return target_position, target_distance
        # search
        for dx, dy in directions:
            nx, ny = current_x + dx, current_y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if not visited[ny][nx] and processed_map[ny][nx] != 1:
                    visited[ny][nx] = True
                    queue.append((nx, ny, distance + 1))

    return target_position, target_distance