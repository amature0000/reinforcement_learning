import random
import numpy as np
import pickle
from state import State


class Agent:
    def __init__(self, alpha = 0.2, gamma = 0.99, epsilon = 0, decay = 0.9999, min_epsilon = 0.2, num_features = 16, possible_actions = 3):
        """
        alpha = learning rate
        gamma = discount rate
        epsilon = epsilon init value
        decay = epsilon decay rate
        min_epsilon = min epsilon value
        theta = FA parameter
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.theta = np.random.uniform(-0.01, 0.01, num_features)
        self.possible_actions = list(range(possible_actions))

    def save(self):
        with open('save.pkl', 'wb') as file:
            pickle.dump({
                'theta': self.theta,
                'epsilon': self.epsilon
            }, file)
    def load(self):
        with open('save.pkl', 'rb') as file:
            data = pickle.load(file)
            self.theta = data['theta']
            self.epsilon = data['epsilon']

    def compute_q(self, features):
        return np.dot(self.theta, features)
    def compute_future_q(self, state:State):
        return np.array([self.compute_q(state.process_features(a)) for a in self.possible_actions])

    def choose_action_while_train(self, state):
        if random.random() < self.epsilon:
            return np.random.choice(self.possible_actions)
        else:
            return self.choose_action(state)
    def choose_action(self, state):
        q_values = self.compute_future_q(state)
        return np.argmax(q_values)

    def update(self, state:State, action, reward, next_state, done):
        features = state.process_features(action)
        q_next = 0 if done else max(self.compute_future_q(next_state))

        td_error = reward + self.gamma * q_next - self.compute_q(features)
        
        self.theta += self.alpha * td_error * features
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)