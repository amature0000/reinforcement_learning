from collections import deque
import numpy as np

L = 0
R = 1
U = 2
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
                    self.up = types.index(char_map[y-1][x][0]) if y > 0 else -1
                    self.down = types.index(char_map[y+1][x][0]) if y < MAX_COR else -1
                    self.left = types.index(char_map[y][x-1][0]) if x > 0 else -1
                    self.right = types.index(char_map[y][x+1][0]) if x < MAX_COR else -1
                if char_map[y][x] == "B": b_cnt += 1
                elif char_map[y][x] == "H": h_cnt += 1
                elif char_map[y][x] == "K": k_cnt += 1
        self.b = b_cnt
        self.h = h_cnt
        self.k = k_cnt
        # bfs
        location, distance = beefs(self.py, self.px, char_map)
        self.b_dist = distance['B']
        self.by, self.bx = location['B']
        self.hy, self.hx = location['H']
        self.ky, self.kx = location['K']
    def features(self):
        normalized_py = self.py / (MAX_COR - 1)
        normalized_px = self.px / (MAX_COR - 1)
        normalized_by = self.by / (MAX_COR - 1)
        normalized_bx = self.bx / (MAX_COR - 1)
        normalized_hy = self.hy / (MAX_COR - 1)
        normalized_hx = self.hx / (MAX_COR - 1)
        normalized_ky = self.ky / (MAX_COR - 1)
        normalized_kx = self.kx / (MAX_COR - 1)

        pd_one_hot = np.zeros(4)
        up_one_hot = np.zeros(5)
        down_one_hot = np.zeros(5)
        left_one_hot = np.zeros(5)
        right_one_hot = np.zeros(5)
        pd_one_hot[self.pd] = 1.0
        if self.up != -1: up_one_hot[self.up] = 1.0
        if self.down != -1: down_one_hot[self.down] = 1.0
        if self.left != -1: left_one_hot[self.left] = 1.0
        if self.right != -1: right_one_hot[self.right] = 1.0

        result = [
            normalized_py,
            normalized_px,
            normalized_by,
            normalized_bx,
            normalized_hy,
            normalized_hx,
            normalized_ky,
            normalized_kx
        ] + pd_one_hot.tolist() + up_one_hot.tolist() + down_one_hot.tolist() + left_one_hot.tolist() + right_one_hot.tolist()
        return result


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
    # bfs based reward(normal:-.02, backward:-.2, forward:+.2)
    if state.b_dist > next_state.b_dist: return 0.2
    if state.b_dist < next_state.b_dist: return -0.2
    return -0.01

def beefs(start_y, start_x, map):
    queue = deque()
    rows = int(MAX_COR)
    cols = int(MAX_COR)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue.append((start_y, start_x, 0))
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    visited[start_y][start_x] = True

    targets = {'B', 'H', 'K'}
    target_position = {'B': (start_y, start_x), 'H': (start_y, start_x), 'K': (start_y, start_x)}
    target_distance = {'B': 0, 'H': 0, 'K': 0}
    target_found = {'B': False, 'H': False, 'K': False}

    while queue:
        current_y, current_x, distance = queue.popleft()
        # current block check
        for key in targets:
            if map[current_y][current_x][0] == key and not target_found[key]:
                target_position[key] = (current_y, current_x)
                target_distance[key] = distance
                target_found[key] = True
                # early exit
                if all(target_found.values()):
                    return target_position, target_distance
        # search
        for dy, dx in directions:
            nx, ny = current_x + dx, current_y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if not visited[ny][nx] and map[ny][nx] != 'W':
                    visited[ny][nx] = True
                    queue.append((ny, nx, distance + 1))

    return target_position, target_distance