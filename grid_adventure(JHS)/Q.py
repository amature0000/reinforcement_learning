import numpy as np
import math
from collections import defaultdict
from utils import process_obs
import pickle
np.set_printoptions(precision=3)

def default_q():
    return np.zeros(6)

class QAgent:
    def __init__(self, num_actions=6, alpha=1, gamma=0.99, c=0.01):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.c = c

        # q테이블
        self.q_table = defaultdict(default_q)
        # 행동 방문 횟수
        self.action_counts = defaultdict(default_q)
        # 상태 방문 횟수 (미사용)
        self.state_counts = defaultdict(int)

    def save(self):
        with open('save.pkl', 'wb') as file:
            pickle.dump({
                'q_table': self.q_table,
                'action_counts': self.action_counts,
                'state_counts': self.state_counts
            }, file)
        print("save")

    def load(self):
        with open('save.pkl', 'rb') as file:
            data = pickle.load(file)
            self.q_table = data['q_table']
            self.action_counts = data['action_counts']
            self.state_counts = data['state_counts']
        print("load")

    def _state_to_key(self, state):
        return tuple(state)
    
    def select_action_while_train(self, state):
        state_p, _, _, _, _, _, _, _, _, _ = process_obs(state)
        #print(state_p)
        state_key = self._state_to_key(state_p)
        self.state_counts[state_key] += 1
        total_counts = self.state_counts[state_key]
        
        ucb_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            if self.action_counts[state_key][action] == 0:
                ucb_values[action] = float('inf')
            else:
                average_reward = self.q_table[state_key][action]
                exploration_term = self.c * math.sqrt(math.log(total_counts) / self.action_counts[state_key][action])
                ucb_values[action] = average_reward + exploration_term
        
        selected_action = np.argmax(ucb_values)
        #print(self.q_table[state_key], int(self.action_counts[state_key][selected_action]), ucb_values)
        return selected_action
    
    def update(self, state, action, reward, next_state, terminate, stupid):
        state_p, _, _, _, _, _, _, _, _, _ = process_obs(state)
        next_state_p, _, _, _, _, _, _, _, _, _ = process_obs(next_state)

        state_key = self._state_to_key(state_p)
        self.action_counts[state_key][action] += 1
        
        if terminate:
            target = reward
        if stupid:
            self.q_table[state_key][action] = -np.inf
            return
        else:
            next_state_key = self._state_to_key(next_state_p)
            next_step = np.max(self.q_table[next_state_key])
            target = reward + self.gamma * next_step
            if next_step == -np.inf:
                self.q_table[state_key][action] = -np.inf
                return
        current_q = self.q_table[state_key][action]
        if current_q == -np.inf:
            return
        self.q_table[state_key][action] = (1 - self.alpha) * current_q + self.alpha * target

    def select_action(self, state):
        state_p, _, _, _, _, _, _, _, _, _ = process_obs(state)
        state_key = self._state_to_key(state_p)
        
        return np.argmax(self.q_table[state_key])
    
    def reset(self):
        self.q_table.clear()
        self.action_counts.clear()
        self.state_counts.clear()
