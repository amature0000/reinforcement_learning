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
        """
        location, distance = beefs(self.py, self.px, char_map)
        self.b_dist = distance['B']
        self.by, self.bx = location['B']
        self.hy, self.hx = location['H']
        self.ky, self.kx = location['K']
        """
        b1, b2 = beefs(self.py, self.px, self.pd, 'B', char_map)
        self.by, self.bx, self.b_dist = b1
        self.by2, self.bx2, self.b_dist2 = b2
        h1, h2 = beefs(self.py, self.px, self.pd, 'H', char_map)
        self.hy, self.hx, self.h_dist = h1
        self.hy2, self.hx2, self.h_dist2 = h2
        k1, k2 = beefs(self.py, self.px, self.pd, 'K', char_map)
        self.ky, self.kx, self.k_dist = k1
        self.ky2, self.kx2, self.k_dist2 = k2
    def features(self):
        # coordinate 14 + hp 1 + onehot_dir 4 + nearby blocks onehot 5*8 = 59
        normalized_py = float(self.py) / (MAX_COR - 1)
        normalized_px = float(self.px) / (MAX_COR - 1)

        normalized_by = float(self.by) / (MAX_COR - 1)
        normalized_bx = float(self.bx) / (MAX_COR - 1)
        normalized_by2 = float(self.by2) / (MAX_COR - 1)
        normalized_bx2 = float(self.bx2) / (MAX_COR - 1)

        normalized_hy = float(self.hy) / (MAX_COR - 1)
        normalized_hx = float(self.hx) / (MAX_COR - 1)
        normalized_hy2 = float(self.hy2) / (MAX_COR - 1)
        normalized_hx2 = float(self.hx2) / (MAX_COR - 1)

        normalized_ky = float(self.ky) / (MAX_COR - 1)
        normalized_kx = float(self.kx) / (MAX_COR - 1)
        normalized_ky2 = float(self.ky2) / (MAX_COR - 1)
        normalized_kx2 = float(self.kx2) / (MAX_COR - 1)

        normalized_hp = float(self.hp) / 100
        pd_one_hot = np.zeros(4)
        up_one_hot = np.zeros(5)
        upleft_one_hot = np.zeros(5)
        upright_one_hot = np.zeros(5)
        down_one_hot = np.zeros(5)
        downleft_one_hot = np.zeros(5)
        downright_one_hot = np.zeros(5)
        left_one_hot = np.zeros(5)
        right_one_hot = np.zeros(5)
        pd_one_hot[self.pd] = 1.0

        if self.upleft != -1: upleft_one_hot[self.upleft] = 1.0
        if self.up != -1: up_one_hot[self.up] = 1.0
        if self.upright != -1: upright_one_hot[self.upright] = 1.0
        if self.left != -1: left_one_hot[self.left] = 1.0
        if self.right != -1: right_one_hot[self.right] = 1.0
        if self.downleft != -1: downleft_one_hot[self.downleft] = 1.0
        if self.down != -1: down_one_hot[self.down] = 1.0
        if self.downright != -1: downright_one_hot[self.downright] = 1.0

        result = [
            normalized_py,
            normalized_px,
            normalized_by,
            normalized_bx,
            normalized_by2,
            normalized_bx2,
            normalized_hy,
            normalized_hx,
            normalized_hy2,
            normalized_hx2,
            normalized_ky,
            normalized_kx,
            normalized_ky2,
            normalized_kx2,
            normalized_hp
        ] + pd_one_hot.tolist() + upleft_one_hot.tolist() + up_one_hot.tolist() + upright_one_hot.tolist() + left_one_hot.tolist() + right_one_hot.tolist() + downleft_one_hot.tolist() + down_one_hot.tolist() + downright_one_hot.tolist()
        return result


def process_reward(state:State, next_state:State):
    # touched honeybee
    if state.b > next_state.b: return 10.0
    # dead
    if state.k > next_state.k: return -10.0
    # dead
    if next_state.hp < 10 : return -10.0
    # damaged
    if state.hp > next_state.hp: return -0.5
    # no movement
    if state.px == next_state.px and state.py == next_state.py and state.pd == next_state.pd: return -10.0
    # getting closer to B
    if state.b_dist > next_state.b_dist: 
        if state.b_dist2 > next_state.b_dist2: return 0.4
        else: return 0.2
    elif state.b_dist2 > next_state.b_dist2: return 0.2
    # or run away from K
    if state.k_dist < next_state.k_dist: return 0.25
    return 0
"""
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
                if not visited[ny][nx] and map[ny][nx] == 'E':
                    visited[ny][nx] = True
                    queue.append((ny, nx, distance + 1))

    return target_position, target_distance
"""
"""
def beefs(start_y, start_x, target, map):
    if map[start_y][start_x][0] == target: return start_y, start_x, 0
    queue = deque()
    rows = int(MAX_COR)
    cols = int(MAX_COR)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue.append((start_y, start_x, 0))
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    visited[start_y][start_x] = True

    target_distance = 0

    while queue:
        current_y, current_x, distance = queue.popleft()
        for dy, dx in directions:
            nx, ny = current_x + dx, current_y + dy
            # return
            if map[ny][nx][0] == target:
                return ny, nx, target_distance + 1
            # search
            if 0 <= nx < cols and 0 <= ny < rows:
                if not visited[ny][nx] and map[ny][nx] == 'E': # every object is considered as a "wall"
                    visited[ny][nx] = True
                    queue.append((ny, nx, distance + 1))

    return start_y, start_x, 0
    """
def beefs(start_y, start_x, dir, target, map):
    if map[start_y][start_x][0] == target: return start_y, start_x, 0
    queue = deque()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue.append((start_y, start_x, dir, 0))
    visited = {}
    result = []
    while queue:
        current_y, current_x, current_dir, current_distance = queue.popleft()
        tempy, tempx = directions[current_dir]
        tempy += current_y
        tempx += current_x
        # boundary check
        if tempy < MAX_COR and tempx < MAX_COR and tempy > 0 and tempx > 0:
            # return
            if map[tempy][tempx][0] == target: 
                result.append((tempy, tempx, current_distance + 1))
                if len(result) == 2: return result
            # forward step
            if ((tempy, tempx, current_dir) not in visited) and map[tempy][tempx][0] == 'E':
                visited[(tempy, tempx, current_dir)] = True
                queue.append([tempy, tempx, current_dir, current_distance + 1])
        # rotate step
        if (current_y, current_x, (current_dir + 1) % 4) not in visited:
            visited[(current_y, current_x, (current_dir + 1) % 4)] = True
            queue.append([current_y, current_x, (current_dir + 1) % 4, current_distance + 1])
        if (current_y, current_x, (current_dir + 3) % 4) not in visited:
            visited[(current_y, current_x, (current_dir + 3) % 4)] = True
            queue.append([current_y, current_x, (current_dir + 3) % 4, current_distance + 1])
    result.append((start_y, start_x, 0))
    if len(result) == 1: result.append((start_y, start_x, 0))
    return result