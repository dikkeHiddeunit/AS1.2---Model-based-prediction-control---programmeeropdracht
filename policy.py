import random

class Policy:
    def __init__(self, num_actions=4):
        self.num_actions = num_actions

    def select_action(self, state):
        return random.randint(0, self.num_actions - 1)


class DeterministicPolicy:
    def __init__(self, policy_dict):
        self.policy_dict = policy_dict

    def select_action(self, state):
        return self.policy_dict[state]