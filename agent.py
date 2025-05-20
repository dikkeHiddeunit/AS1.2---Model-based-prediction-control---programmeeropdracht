import numpy as np

class Agent:
    """Een agent die een beleid volgt in een doolhofomgeving."""

    def __init__(self, maze, policy):
        self.maze = maze
        self.policy = policy
        self.value_function = np.zeros(maze.n_states)

    def act(self, state):
        """Voert een actie uit volgens het beleid en geeft transitie-informatie terug."""
        action = self.policy.select_action(state)
        next_state = self.maze.step(state, action)
        reward = self.maze.get_reward(state)  # correcte reward uit huidige toestand
        return state, action, next_state, reward
