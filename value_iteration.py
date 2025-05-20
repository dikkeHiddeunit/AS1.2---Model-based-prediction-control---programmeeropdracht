import numpy as np

class ValueIterationAgent:
    def __init__(self, states, actions, transition_probabilities, rewards, gamma=1, epsilon=1e-6):
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities  # dict: s -> a -> list of (prob, next_state)
        self.rewards = rewards  # dict: s -> a -> next_state -> reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.V = {s: 0 for s in states}
        self.policy = {s: None for s in states}

    def run(self, terminal_states):

        self.policy, self.V = self.value_iteration(terminal_states)
        print('poep', self.V.values())
        print(np.array(list(self.V.values())).reshape(4, 4))

    def value_iteration(self, termial_states):
        V = {s: 0 for s in self.states}

        while True:
            delta = 0
            for s in self.states:
                #check if state is terminal
                if s in termial_states:
                    continue
                v = V[s]
                max_value = float('-inf')
                for a in self.actions:
                    q = 0
                    # iterate over possible next states with probabilities
                    for prob, s_next in self.transition_probabilities.get(s, {}).get(a, []):
                        reward = self.rewards.get(s, {}).get(a, {}).get(s_next, 0)
                        print("s", s, "a", a, "s_next", s_next, "reward", reward)
                        q += prob * (reward + self.gamma * V[s_next])
                        #print(q)
                    if q > max_value:
                        max_value = q
                V[s] = max_value
                delta = max(delta, abs(v - V[s]))
                print("d", delta)

            if delta < self.epsilon:
                break

        policy = {}
        for s in self.states:
            best_a = None
            best_value = float('-inf')
            for a in self.actions:
                q = 0
                for prob, s_next in self.transition_probabilities.get(s, {}).get(a, []):
                    reward = self.rewards.get(s, {}).get(a, {}).get(s_next, 0)
                    q += prob * (reward + self.gamma * V[s_next])
                if q > best_value:
                    best_value = q
                    best_a = a
            policy[s] = best_a

        return policy, V

    def get_value_function(self):
        return self.V

    def get_policy(self):
        return self.policy
