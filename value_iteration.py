class ValueIterationAgent:
    def __init__(self, states, actions, transition_probabilities, rewards, gamma=0.9, theta=1e-6):
        self.states = states
        self.actions = actions
        self.P = transition_probabilities
        self.R = rewards
        self.gamma = gamma
        self.theta = theta
        self.V = {s: 0 for s in states}
        self.policy = {s: None for s in states}

    def run(self):
        while True:
            delta = 0
            for s in self.states:
                # Skip terminal states without actions
                if all(len(self.P[s][a]) == 0 for a in self.actions):
                    continue
                v = self.V[s]
                max_q = float('-inf')
                for a in self.actions:
                    q = 0
                    for prob, next_state in self.P[s][a]:
                        reward = self.R[s][a]
                        q += prob * (reward + self.gamma * self.V[next_state])
                    if q > max_q:
                        max_q = q
                self.V[s] = max_q
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break
        self.extract_policy()

    def extract_policy(self):
        for s in self.states:
            best_a = 0
            best_q = float('-inf')
            for a in self.actions:
                q = 0
                for prob, next_state in self.P[s][a]:
                    reward = self.R[s][a]
                    q += prob * (reward + self.gamma * self.V[next_state])
                if q > best_q:
                    best_q = q
                    best_a = a
            self.policy[s] = best_a

    def get_value_function(self):
        return self.V

    def get_policy(self):
        return self.policy
