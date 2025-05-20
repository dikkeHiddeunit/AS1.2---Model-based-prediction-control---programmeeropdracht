import random


class DeterministicPolicy:
    """Een deterministisch beleid gedefinieerd door een beleidsdictionary."""

    def __init__(self, policy_dict):
        self.policy_dict = policy_dict

    def select_action(self, state):
        """Geeft de actie terug die hoort bij de toestand volgens het beleid."""
        #print(self.policy_dict)
        return self.policy_dict[state]
