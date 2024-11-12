from collections import deque
import numpy as np

MAX_COR = 34.0
MAX_DIR = 3.0
MAX_FAR = 10.0
L = 0
R = 1
U = 2
D = 3

class State:
    def __init__(self, py=16, px=17, pd=R, hp=0, kx=0, ky=0, hx=0, hy=0, bx=0, by=0, kdir=0, hdir=0, bdir=0):
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

    def process_reward(self, terminated):
        # touched all honeybee
        if self.bdir == 0: return 1
        # dead
        if terminated: return -1
        # touched honeybee
        # TODO: implementation
        # no state changed
        # TODO: implementation
        # basic movement
        return -0.01

    def process_state(self, obs_before_process):
        obs = obs_before_process['grid']
        # get hp
        self.hp = obs_before_process['hit_points']
        # get p
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for y, x in directions:
            if obs[self.py + y][self.px + x][0] == "A":
                self.px += x; self.py += y
                if obs[self.py][self.px][1] == "L": self.pd = L
                elif obs[self.py][self.px][1] == "R": self.pd = R
                elif obs[self.py][self.px][1] == "U": self.pd = U
                elif obs[self.py][self.px][1] == "D": self.pd = D
                break
        # get the nearest objects
        tpos, tdir = self.bfs(obs)
        print(tpos)
        print(tdir)
        self.by, self.bx = tpos['B']
        self.ky, self.kx = tpos['K']
        self.hy, self.hx = tpos['H']
        self.bdir = tdir['B']
        self.kdir = tdir['K']
        self.hdir = tdir['H']
    
    def process_features(self, action):
        # 16 features(px, py, pd, kx, ky, hx, hy, bx, by, kdir, hdir, bdir, hp, a1, a2, a3)
        features = []
        # coord incode
        features.append(self.px / MAX_COR)
        features.append(self.py / MAX_COR)
        features.append(self.pd / MAX_DIR)
        
        # killerbee incode
        features.append(self.kx / MAX_COR)
        features.append(self.ky / MAX_COR)

        # hornet incode
        features.append(self.hx / MAX_COR)
        features.append(self.hy / MAX_COR)
        
        # honeybee incode
        features.append(self.bx / MAX_COR)
        features.append(self.by / MAX_COR)

        # dir incode
        features.append(1 - min(self.kdir / MAX_FAR, 1))
        features.append(1 - min(self.hdir / MAX_FAR, 1))
        features.append(1 - min(self.bdir / MAX_FAR, 1))
        
        # hp incode
        features.append(self.hp) # 1 if hp > 10, else 0

        # action incode
        for i in range(3):
            features.append(1.0 if i == action else 0.0)

        return np.array(features)

    # bfs algorithm to find the nearest objects' information
    def bfs(self, map):
        queue = deque()
        rows = int(MAX_COR)
        cols = int(MAX_COR)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        start_x, start_y = self.px, self.py
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
                    #print(f"{key} : ({current_x}, {current_y}), {distance}")
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