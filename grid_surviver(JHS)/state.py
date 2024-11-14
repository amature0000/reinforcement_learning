from collections import deque
import numpy as np

L = 0
R = 1
U = 2
D = 3
MAX_COR = 34
CHANNELS = 8
WIDTH = 34
HEIGHT = 34

class State:
    def __init__(self, py=16, px=17, pd=R, hp=0, b=0, h=0, k=0):
        self.px = px
        self.py = py
        self.pd = pd
        self.hp = hp
        self.b = b
        self.h = h
        self.k = k
        self.input_data = np.zeros((CHANNELS, WIDTH, HEIGHT))  # 8개의 채널로 초기화
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
        for y in range(HEIGHT):
            for x in range(WIDTH):
                # 에이전트 위치 저장
                if char_map[y][x][0] == "A":
                    self.py, self.px = y, x
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

def process_reward(state:State, next_state:State):
    # touched honeybee = 1.0
    if state.b > next_state.b: return 1.0
    # damaged = -0.2
    if state.hp > next_state.hp: return -0.2
    # dead = -1.0
    if state.k > next_state.k: return -1.0
    # dead = -1.0
    if next_state.hp < 10 : return -1.0
    return 0.0