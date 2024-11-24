from collections import deque

import numpy as np



L = 0

U = 1

R = 2

D = 3

MAX_COR = 34

CHANNELS = 8
MAX_BEE = 50
MAX_HP = 100

OBS_WIDTH = 11 # 탐색 너비( 홀수여야 함. 플레이어 가운데.)

OBS_HALF = OBS_WIDTH//2

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



        self.grid_wall = np.zeros((OBS_WIDTH, OBS_WIDTH), dtype=np.int8)

        self.grid_bee = np.zeros((OBS_WIDTH, OBS_WIDTH), dtype=np.int8)

        self.grid_kbee = np.zeros((OBS_WIDTH, OBS_WIDTH), dtype=np.int8)

        self.grid_horn = np.zeros((OBS_WIDTH, OBS_WIDTH), dtype=np.int8)

        self.bee_dir = np.zeros(4, dtype=np.int8)

        self.agent_d = np.zeros(4, dtype=np.int8)

        # print(self.char_map)

        self.find_player_xy()

        # 0 1 2 3 4 |5| 6 7 8 9 10

        for y in range(MAX_COR):

            for x in range(MAX_COR):

                is_in = 1

                dx, dy = self.px+OBS_HALF - x, self.py+OBS_HALF - y

                if self.px-OBS_HALF > x or x > self.px+OBS_HALF or self.py-OBS_HALF > y or y > self.py+OBS_HALF:

                    is_in = 0

                if self.char_map[y][x] == "B": 

                    b_cnt += 1

                    if is_in: self.grid_bee[dy][dx] = 1

                    if dx>=OBS_HALF:

                        self.bee_dir[0] += 1 # L

                    else:

                        self.bee_dir[2] += 1 # R

                    if dy>=OBS_HALF:

                        self.bee_dir[1] += 1 # U

                    else:

                        self.bee_dir[3] += 1 # D

                elif self.char_map[y][x] == "H": 

                    h_cnt += 1

                    if is_in: self.grid_horn[dy][dx] = 1

                elif self.char_map[y][x] == "K": 

                    k_cnt += 1

                    if is_in: self.grid_kbee[dy][dx] = 1

                elif self.char_map[y][x] == "W":

                    if is_in: self.grid_wall[dy][dx] = 1

        self.b = b_cnt

        self.h = h_cnt

        self.k = k_cnt



    def find_player_xy(self):

        for y in range(MAX_COR):

            for x in range(MAX_COR):

                # 에이전트 위치 저장

                if self.char_map[y][x][0] == "A":

                    self.py, self.px = y, x

                    if self.char_map[y][x][1] == "L":

                        self.agent_d[L]=1

                    elif self.char_map[y][x][1] == "R":

                        self.agent_d[R]=1

                    elif self.char_map[y][x][1] == "U":

                        self.agent_d[U]=1

                    elif self.char_map[y][x][1] == "D":

                        self.agent_d[D]=1

                    return



    def features(self):

        tmp = (np.array([

            self.grid_wall,

            self.grid_bee,

            self.grid_kbee,

            self.grid_horn,

        ]), np.concatenate([self.agent_d, [a/MAX_BEE for a in self.bee_dir], [self.hp/MAX_HP]]))
        return tmp

    

    def features_dir(self):

        return self.agent_d





def process_reward(state:State, next_state:State):

    # touched honeybee

    if state.b > next_state.b: return 10.0

    # dead

    if state.k > next_state.k: return -10.0

    # dead

    if next_state.hp < 10 : return -10.0

    # damaged

    if state.hp > next_state.hp: return -2

    # no movement

    if state.px == next_state.px and state.py == next_state.py: return -1.0

    return 0
    """
    # touched honeybee

    if state.b > next_state.b: return 10.0

    # dead

    if state.k > next_state.k: return -500.0

    # dead

    if next_state.hp < 10 : return -500.0

    # damaged

    if state.hp > next_state.hp: return -100

    # no movement

    if state.px == next_state.px and state.py == next_state.py: return -1.0

    return 0
    """