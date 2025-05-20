import numpy as np

class Agent:
    """Een agent die een beleid volgt in een doolhofomgeving."""

    def __init__(self, maze, policy):
        """Initialiseert de agent met een doolhof en een beleid."""
        self.maze = maze
        self.policy = policy
        self.value_function = np.zeros(maze.n_states)

    def act(self, state):
        """
        Voert een actie uit volgens het beleid vanuit de gegeven toestand.
        
        Retourneert:
        - huidige toestand
        - gekozen actie
        - volgende toestand
        - beloning van de huidige toestand
        """
        action = self.policy.select_action(state)
        next_state = self.maze.step(state, action)
        reward = self.maze.get_reward(state) 
        return state, action, next_state, reward
