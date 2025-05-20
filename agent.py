import numpy as np
from policy import Policy
from maze import Maze

class Agent:
    def __init__(self, maze, policy):
        self.maze = maze
        self.policy = policy
        self.value_function = np.zeros(maze.n_states)

    def act(self, state):
        action = self.policy.select_action(state)
        next_state = self.maze.step(state, action)
        reward = self.maze.get_reward(next_state)
        return state, action, next_state, reward
