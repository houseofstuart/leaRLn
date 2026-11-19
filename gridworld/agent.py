from collections import defaultdict
import gymnasium as gym
import numpy as np


class GridWorldAgent:
    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float, epsilon_decay: float, final_epsilon: float, discount_factor: float = 0.95):
        
        self.env = env
        self.q_values = np.zeros((25, 4))
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progess
        self.training_error = []

    def get_action(self, obs) -> int:
        state = obs[0] * 5 + obs[1]
        # With probability epsiolon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # With probability (1 - epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[state]))

    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        state = obs[0] * 5 + obs[1]
        # Whats the best we can do from the next state
        # Zero if episode terminated (no future reward possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value
        # How wrong was our current estimate
        temporal_difference = target - self.q_values[state][action]

        # Update our estimate in the direction of error 
        # LR controls how big steps we take
        self.q_values[state][action] = (
            self.q_values[state][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)