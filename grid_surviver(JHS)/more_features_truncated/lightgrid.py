from knu_rl_env.grid_survivor.agent import GridSurvivorAgent
from knu_rl_env.grid_survivor.env import _BLUEPRINT_PATH
import os
import numpy as np
from collections import deque


_SYM_START = 'S'
_MAX_HIT_POINTS = 100
_ATTACK_HIT_POINTS = 20
_MAX_STPES = 1201

_STR2INT = {
    'E':0,
    'W':1,
    'B':2,
    'H':3,
    'K':4,
    'AR':0,
    'AD':0,
    'AL':0,
    'AU':0,
    # ---- csv to world ----
    '':0,
    'S':0
}

_INT2STR = {
    0: 'E',
    1: 'W',
    2: 'B',
    3: 'H',
    4: 'K',
    5: 'AR',
    6: 'AD',
    7: 'AL',
    8: 'AU'
}


_OBJ_B = 2
_OBJ_H = 3
_OBJ_K = 4

_DIR_TO_VEC = [
    np.array((1, 0)), # R
    np.array((0, 1)), # D
    np.array((-1, 0)), # L
    np.array((0, -1)), # U
    np.array((0, 0)),
]

import sys
class LightGridObject():
    def __init__(self, x, y, obj_type):
        self.pos = (x, y)
        self.obj_type = obj_type

class LightGridSurvivorEnv():
    def __init__(self):
        self.blueprint = np.loadtxt(_BLUEPRINT_PATH, dtype=str, delimiter=',').T
        self.start_pos = np.argwhere(self.blueprint == _SYM_START).flatten()
        self.start_dir = 0
        self.blueprint = self.blueprint
        self.width, self.height = self.blueprint.shape
        self._max_steps = _MAX_STPES
        self._reset()

    def _reset(self):
        self._agent_pos = tuple(self.start_pos)
        self._agent_dir = 0
        self._agent_front = np.array([0, 0])
        self._front_obj=None
        self._front_pos=(0, 0)

        self._objs = [deque() for _ in range(3)]
        self._honey_bees = deque()
        self._hornets = deque()
        self._killer_bees = deque()
        self._hit_points = _MAX_HIT_POINTS
        self._attack_hit_points = _ATTACK_HIT_POINTS
        self.action_cnt = 0
        self._generator = np.random.default_rng(None)

        self.world = np.zeros(self.blueprint.shape, dtype=np.int8)
        self._gen_world()
    
    def reset(self):
        self._reset()
        return self._get_obs(), {}

    def _gen_world(self):
        for y in range(self.height):
            for x in range(self.width):
                self.world[(x, y)] = _STR2INT[self.blueprint[(x, y)]]
                cur = self.world[(x, y)]
                if cur == _STR2INT['B']:
                    self._honey_bees.append(LightGridObject(x, y, _OBJ_B))
                elif cur == _STR2INT['H']:
                    self._hornets.append(LightGridObject(x, y, _OBJ_H))
                elif cur == _STR2INT['K']:
                    self._killer_bees.append(LightGridObject(x, y, _OBJ_K))
    
    def get_num_of_honeybees(self):
        return len(self._honey_bees)
    
    def step(self, action):
        is_game_over = False
        self.action_cnt += 1
        if action == GridSurvivorAgent.ACTION_FORWARD:
            self._front_obj = None
            self._front_pos = tuple(self._agent_pos + _DIR_TO_VEC[self._agent_dir])
            if self.world[self._front_pos] == 1:
                self._front_pos = self._agent_pos
                
            
            for obj in self._hornets:
                self._move_object(obj, 100, 'random')
            for obj in self._killer_bees:
                self._move_object(obj, 100, 'toward')
            for obj in self._honey_bees:
                self._move_object(obj, 100, 'stand')

            front_obj = self._front_obj
            is_saved = (front_obj and front_obj.obj_type == _OBJ_B)
            is_attacked = (front_obj and front_obj.obj_type == _OBJ_H)
            is_killed = (front_obj and front_obj.obj_type == _OBJ_K)

            self._agent_pos = self._front_pos
            if is_saved:
                self.world[self._front_pos] = 0
                self._honey_bees.remove(front_obj)
                is_game_over = len(self._honey_bees) <= 0
            elif is_attacked:
                self.world[self._front_pos] = 0
                self._hornets.remove(front_obj)
                self._hit_points = max(0, self._hit_points - self._attack_hit_points)
                is_game_over = self._hit_points <= 0
            elif is_killed:
                is_game_over = True
        elif action == GridSurvivorAgent.ACTION_LEFT:
            self._agent_dir = (self._agent_dir-1)%4
        else:
            self._agent_dir = (self._agent_dir+1)%4

        is_truncated = (self.action_cnt > _MAX_STPES)
        is_terminated = (is_game_over | is_truncated)
        return self._get_obs(), 0, is_terminated, is_truncated, {}
    
    def _move_object(self, obj, max_tries, mode):
        ox, oy = obj.pos
        positions = []

        for i in range(len(_DIR_TO_VEC)):
            dx, dy = _DIR_TO_VEC[i]
            nx, ny = ox + dx, oy + dy
            positions.append((nx, ny))
        positions = np.asarray(positions)
        weights = np.ones(len(positions))

        if mode == 'toward':
            ax, ay = self._agent_pos
            dist = np.array([np.abs(x - ax) + np.abs(y - ay) for x, y in positions])
            min_dist = np.min(dist)
            if min_dist < 10:
                weights[dist == min_dist] = 30
        elif mode == 'stand':
            weights[-1] = 10

        try:
            self._place_object(obj, max_tries, positions, weights)
        except RecursionError:
            pass
    
    def _place_object(self, obj, max_tries, positions, weights):
        if np.array_equal(obj.pos, self._front_pos):
            self._front_obj=obj
            return

        num_tries = 0
        is_same_pos = False
        probs = weights if weights is not None else np.ones(len(positions))
        probs = probs / np.sum(probs)
        while True:
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")
            num_tries += 1
            pos = tuple(self._generator.choice(positions, p=probs))

            if np.array_equal(obj.pos, pos):
                is_same_pos = True
                break
            if self.world[pos]:
                continue

            if np.array_equal(pos, self._agent_pos):
                continue
            
            break

        if not is_same_pos:
            if np.array_equal(pos, self._front_pos):
                self._front_obj = obj
            self.world[pos] = obj.obj_type
            self.world[obj.pos] = 0
            if obj is not None:
                obj.pos = pos

    def _get_obs(self):
        world_cp = np.full((self.width, self.height), 'E', dtype='U2')
        for y in range(self.height):
            for x in range(self.width):
                world_cp[(x, y)] = _INT2STR[self.world[(x, y)]]
        
        world_cp[self._agent_pos] = _INT2STR[5+self._agent_dir]
        return {
            'grid': world_cp,
            'hit_points': self._hit_points
        }
    def render(self, output=''):
        # 콘솔 출력 비우기
        # os.system('clear')
        for y in range(self.height):
            for x in range(self.width):
                if x == self._agent_pos[0] and y == self._agent_pos[1]:
                    tmp = '>'
                    if self._agent_dir==3:
                        tmp = '^'
                    elif self._agent_dir==2:
                        tmp = '<'
                    elif self._agent_dir==1:
                        tmp = 'v'
                    print('\033[41m'+tmp+'\033[0m', end=' ')
                elif self.world[(x, y)] == _STR2INT['W']:
                    print('\033[47m'+str(self.world[(x, y)])+'\033[0m', end=' ')
                elif self.world[(x, y)] == _STR2INT['E']:
                    print('\033[30m'+str(self.world[(x, y)])+'\033[0m', end=' ')
                elif self.world[(x, y)] == _STR2INT['B']:
                    print('\033[43m'+str(self.world[(x, y)])+'\033[0m', end=' ')
                elif self.world[(x, y)] == _STR2INT['K']:
                    print('\033[45m'+str(self.world[(x, y)])+'\033[0m', end=' ')
                elif self.world[(x, y)] == _STR2INT['H']:
                    print('\033[44m'+str(self.world[(x, y)])+'\033[0m', end=' ')
                else:
                    print(self.world[(x, y)], end=' ')
            print()
        print(f"actions: {self.action_cnt}/1201 bees: {self.get_num_of_honeybees()} health: {self._hit_points}")
        print(output)
        print('================================================================')

def make_light_grid_survivor(**kwargs):
    env = LightGridSurvivorEnv()

    return env