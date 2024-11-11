import random
import numpy as np
from utils import process_obs, process_features


class Agent:
    def __init__(self, alpha = 0.2, gamma = 0.99, epsilon = 0.8, decay = 0.999, min_epsilon = 0.2, num_features = 13, possible_actions = 3):
        """
        alpha = learning rate
        gamma = discount rate
        epsilon = epsilon init value
        decay = epsilon decay rate
        min_epsilon = min epsilon value
        num_features = features' length
        possible_actions = actions set length
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.theta = np.random.uniform(-0.01, 0.01, num_features)
        self.possible_actions = possible_actions
    def save(self):
        pass
    def load(self):
        pass

    def compute_q(self, features):
        """
        calculate Qvalues
        """
        return np.dot(self.theta, features)

    def choose_action_while_train(self, state):
        """
        using epsilon-greedy to explorate
        """
        if random.random() < self.epsilon:
            return np.random.choice(self.possible_actions)
        else:
            features = np.array([process_features(state, a) for a in self.possible_actions])
            q_values = self.compute_q(features)
            return np.argmax(q_values)
        
    def choose_action(self, state):
        """
        deterministic action selection
        """
        pass

    def update(self, features, td_error):
        """
        update theta
        """
        self.theta += self.alpha * td_error * features