import numpy as np
import random

class Maze:
    def __init__(self):
        self.width = 4
        self.height = 4
        self.n_states = self.width * self.height

        self.rewards = np.full(self.n_states, -1.0)
        self.rewards[3] = 40
        self.rewards[6] = -10
        self.rewards[7] = -10
        self.rewards[12] = 10
        self.rewards[13] = -2

        self.start_state = 14
        self.terminals = [3, 12]

        self.actions = {
            0: (-1, 0),  # left
            1: (1, 0),   # right
            2: (0, -1),  # up
            3: (0, 1),   # down
        }

    def is_terminal(self, state):
        return state in self.terminals

    def get_reward(self, state):
        return self.rewards[state]

    def step(self, state, action):
        if self.is_terminal(state):
            return state

        x = state % self.width
        y = state // self.width

        dx, dy = self.actions[action]
        new_x = max(0, min(self.width - 1, x + dx))
        new_y = max(0, min(self.height - 1, y + dy))

        return new_y * self.width + new_x





