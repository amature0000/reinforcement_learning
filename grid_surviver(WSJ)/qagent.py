import pickle
import numpy as np

def encode_state(state):
    forward, hp, safe_f, safe_b, safe_r, safe_l, togo_f, togo_b, togo_r, togo_l = state

    encoded_state = (forward << 9) | (hp << 8) | (safe_f << 7) | (safe_b << 6) | (safe_r << 5) | (safe_l << 4) | (togo_f << 3) | (togo_b << 2) | (togo_r << 1) | (togo_l)
    
    return encoded_state

class QLearningAgent:
    def __init__(self, n_actions, n_states, lr=1.0, gamma=0.0, epsilon=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.finish = False

    def print_q_value(self, state):
        for i in range(3):
            print(self.q_table[encode_state(state), i], end = ' ')
        print()

    def choose_action(self, state, ev):
        # Epsilon-greedy action selection
        if not ev and  not self.finish and np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            q_values = self.q_table[encode_state(state)]
            action = np.argmax(q_values)
        return action

    def learn(self, state, action, reward, next_state):
        q_predict = self.q_table[encode_state(state), action]
        q_target = reward # + self.gamma * np.max(self.q_table[encode_state(next_state)])
        # tmp = self.q_table[encode_state(state), action]
        self.q_table[encode_state(state), action] += self.lr * (q_target - q_predict)
        # if tmp != 0 and tmp != self.q_table[encode_state(state), action]:
        #     print("update : ", state, action, tmp, self.q_table[encode_state(state), action])
        self.epsilon = max(0.001, self.epsilon * 0.9999)
        # print("epsilon = ", self.epsilon)

    def save_model(self, file_name):
        with open(file_name, "wb") as fw:
            pickle.dump(self.q_table, fw)

    def load_model(self, file_name):
        self.finish = True
        with open(file_name, "rb") as fr:
            self.q_table = pickle.load(fr)