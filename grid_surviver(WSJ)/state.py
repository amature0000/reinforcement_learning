from collections import deque
import numpy as np
import copy

L = 0
R = 1
U = 2
D = 3
MAX_COR = 34
CHANNELS = 8

class State:
    def process_state(self, obs):
        self.char_map = obs['grid']
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
        self.near = [0 for _ in range(12)]

        for y in range(MAX_COR):
            for x in range(MAX_COR):
                if self.char_map[y][x][0] == "A":
                    self.py, self.px = y, x
                    if self.char_map[y][x][1] == "L": self.pd = L
                    elif self.char_map[y][x][1] == "R": self.pd = R
                    elif self.char_map[y][x][1] == "U": self.pd = U
                    elif self.char_map[y][x][1] == "D": self.pd = D            

                if self.char_map[y][x] == "B": b_cnt += 1
                elif self.char_map[y][x] == "H": h_cnt += 1
                elif self.char_map[y][x] == "K": k_cnt += 1

        tmp = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]]
        if self.pd == L: 
            mv = [[0, -2], [0, -1], [1, -1], [-1, -1], [2, 0], [1, 0], [-2, 0], [-1, 0], [0, 2], [0, 1], [1, 1], [-1, 1]]
            tmp = [[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]] #FBRL
            self.forward = "WEBHK".index(self.char_map[self.py][self.px - 1])
        elif self.pd == R: 
            mv = [[0, 2], [0, 1], [-1, 1], [1, 1], [-2, 0], [-1, 0], [2, 0], [1, 0], [0, -2], [0, -1], [-1, -1], [1, -1]] 
            tmp = [[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]] #FBRL
            self.forward = "WEBHK".index(self.char_map[self.py][self.px + 1])
        elif self.pd == U: 
            mv = [[-2, 0], [-1, 0], [-1, -1], [-1, 1], [0, -2], [0, -1], [0, 2], [0, 1], [2, 0], [1, 0], [1, -1], [1, 1]]
            tmp = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]] #FBRL
            self.forward = "WEBHK".index(self.char_map[self.py - 1][self.px])
        elif self.pd == D: 
            mv = [[2, 0], [1, 0], [1, 1], [1, -1], [0, 2], [0, 1], [0, -2], [0, -1], [-2, 0], [-1, 0], [-1, 1], [-1, -1]]
            tmp = [[1, 0], [-1, 0], [0, -1], [0, 1], [0, 0]] #FBRL
            self.forward = "WEBHK".index(self.char_map[self.py + 1][self.px])

        self.is_safe = [1, 1, 1, 1]
        for i in range(4):
            my = tmp[i][0] + self.py
            mx = tmp[i][1] + self.px
            if self.char_map[my][mx] == "W":
                self.is_safe[i] = 0
            elif self.char_map[my][mx] == "H" and self.hp > 20:
                pass
            elif self.char_map[my][mx] == "B":
                pass
            else:
                for j in range(5):
                    mmy = tmp[j][0] + my
                    mmx = tmp[j][1] + mx
                    if 0 <= mmy < MAX_COR and 0 <= mmx and mmx < MAX_COR:
                        if self.char_map[mmy][mmx] == "K" or (self.hp == 20 and self.char_map[mmy][mmx] == "H"):
                            self.is_safe[i] = 0
                            break

        for i in range(12):
            my = mv[i][0] + self.py
            mx = mv[i][1] + self.px
            if 0 <= my and my < MAX_COR and 0 <= mx and mx < MAX_COR:
                self.near[i] = types.index(self.char_map[my][mx])
            else:
                self.near[i] = -1

        
        self.danger = False
        if self.is_safe[0] == self.is_safe[1] == self.is_safe[2] == self.is_safe[3] == 0:
            self.danger = True
        
        self.b = b_cnt
        self.h = h_cnt
        self.k = k_cnt
        # bfs

        res = beefs(self.py, self.px, self.char_map, self.hp > 20)

        if len(res['to_go']) != 0:
            self.nxt = res['to_go'][0]

        self.to_go = [0 for _ in range(4)] #FBRL

        # y = 바라보고 있는 방향(+), 뒤(-)
        # x = 바라보고 있는 방향 오른쪽(+), 왼쪽 (-)
        if self.pd == L:
            if self.nxt == L: self.to_go[0] = 1
            elif self.nxt == R: self.to_go[1] = 1
            elif self.nxt == U: self.to_go[2] = 1
            elif self.nxt == D: self.to_go[3] = 1
        elif self.pd == R:
            if self.nxt == L: self.to_go[1] = 1
            elif self.nxt == R: self.to_go[0] = 1
            elif self.nxt == U: self.to_go[3] = 1
            elif self.nxt == D: self.to_go[2] = 1
        elif self.pd == U:
            if self.nxt == L: self.to_go[3] = 1
            elif self.nxt == R: self.to_go[2] = 1
            elif self.nxt == U: self.to_go[0] = 1
            elif self.nxt == D: self.to_go[1] = 1
        elif self.pd == D:
            if self.nxt == L: self.to_go[2] = 1
            elif self.nxt == R: self.to_go[3] = 1
            elif self.nxt == U: self.to_go[1] = 1
            elif self.nxt == D: self.to_go[0] = 1
 

    def features(self):
        result = [self.forward, self.hp > 20]
        result += self.is_safe
        result += self.to_go

        return result


def process_reward(state:State, next_state:State):
    # touched honeybee = 1.0
    if state.b > next_state.b: return 10.0
    # dead = -1.0
    if state.k > next_state.k: return -10.0
    if not state.is_safe[0] and not (state.py == next_state.py and state.px == next_state.px): return -10.0
    # dead = -1.0
    if next_state.hp < 10 : return -10.0
    # damaged = -0.2
    # if state.hp > next_state.hp: return (-0.1 * (100 - next_state.hp))
    
    if state.danger and state.pd != next_state.pd:
        return -1.0

    # no movement = -1.0
    if state.px == next_state.px and state.py == next_state.py and state.pd == next_state.pd: 
        if state.danger:
            return 1.0
        return -10.0

    rotate = U
    if state.pd == L and next_state.pd == U: rotate = R
    elif state.pd == U and next_state.pd == R: rotate = R
    elif state.pd == R and next_state.pd == D: rotate = R
    elif state.pd == D and next_state.pd == L: rotate = R
    if state.pd == L and next_state.pd == D: rotate = L
    elif state.pd == D and next_state.pd == R: rotate = L
    elif state.pd == R and next_state.pd == U: rotate = L
    elif state.pd == U and next_state.pd == L: rotate = L

    # bfs based reward(normal:-.02, backward:-.2, forward:+.2)
    if state.to_go[0] and state.pd == next_state.pd: return 0.2
    elif state.to_go[2]: # R
        if state.is_safe[2] and rotate == R: return 0.2
        elif not state.is_safe[2] and rotate == U: return 0.2
    elif state.to_go[3] or state.to_go[1]: # L or B
        if state.is_safe[3] and rotate == L: return 0.2
        elif not state.is_safe[3] and rotate == U: return 0.2

    if sum(state.to_go) == 0:
        if not (state.py == next_state.py and state.px == next_state.px) and state.is_safe[0]:
            return 0.5

    return -0.5


def beefs(start_y, start_x, map, hp):
    res = {'B': [], 'H': [], 'K': [], 'to_go': []} # LRUD
    vis = {}
    mv = [
        [0, -1], [0, 1], [-1, 0], [1, 0]
    ]
    q = deque()

    q.append((start_y, start_x, 0))
    vis[(start_y, start_x)] = True

    while q:
        cy, cx, cd = q.popleft()
        for i in range(4):
            my = cy + mv[i][0]
            mx = cx + mv[i][1]
            if map[my][mx] in "W": continue
            if (my, mx) not in vis:
                vis[(my, mx)] = True
                if map[my][mx] in "BHK":
                    if len(res[map[my][mx]]) == 0:
                        res[map[my][mx]].append((my, mx))
                if map[my][mx] not in "HK":
                   q.append((my, mx, cd + 1))

        if len(res['B']) == len(res['H']) == len(res['K']) == 1:
            break
    

    if len(res['B']) == 0:
        res['B'].append((start_y, start_x))
        return res

    vis = {}
    q = deque()
    q.append((start_y, start_x, []))
    while q:
        cy, cx, cur = q.popleft()
        for i in range(4):
            my = cy + mv[i][0]
            mx = cx + mv[i][1]
            if map[my][mx] in "WHK": continue
            if (my, mx) not in vis:
                vis[(my, mx)] = True
                nxt = cur + [i]
                if map[my][mx] == "B":
                    res['to_go'].append(nxt[0])
                    return res
                q.append((my, mx, nxt))

    return res
