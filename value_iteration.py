import numpy as np

class ValueIterationAgent:
    """Voert waarde-iteratie uit op basis van toestanden, acties, beloningen en transities."""

    def __init__(self, states, actions, transition_probs, rewards, gamma=1, epsilon=1e-6):
        self.states = states
        self.actions = actions
        self.P = transition_probs  # transitiekansen: P[s][a] = [(kans, volgende toestand)]
        self.R = rewards           # beloningen: R[s][a][s'] = waarde
        self.gamma = gamma
        self.epsilon = epsilon
        self.V = {s: 0 for s in states}
        self.policy = {s: None for s in states}

    def run(self, terminals):
        """Start de waarde-iteratie en update het beleid en de waarde-functie."""
        self.policy, self.V = self._value_iteration(terminals)
        #print(np.array(list(self.V.values())).reshape(4, 4)) 

    def _value_iteration(self, terminals):
        V = self.V.copy()

        while True:
            delta = 0

            for s in self.states:
                if s in terminals:
                    continue

                actie_waardes = []

                for a in self.actions:
                    totale_waarde = 0

                    transities = self.P.get(s, {}).get(a, [])
                    for transitie in transities:
                        prob, s_volgend = transitie

                        r = self.R.get(s, {}).get(a, {}).get(s_volgend, 0)
                        v_volgend = V[s_volgend]
                        bijdrage = prob * (r + self.gamma * v_volgend)

                        totale_waarde += bijdrage

                    actie_waardes.append((a, totale_waarde))

                beste_actie, hoogste_waarde = max(actie_waardes, key=lambda x: x[1], default=(None, 0))
                delta = max(delta, abs(V[s] - hoogste_waarde))
                V[s] = hoogste_waarde

            if delta < self.epsilon:
                break

        nieuw_beleid = {}
        for s in self.states:
            beste_actie = None
            beste_waarde = float('-inf')

            for a in self.actions:
                waarde = 0
                for prob, s1 in self.P.get(s, {}).get(a, []):
                    r = self.R.get(s, {}).get(a, {}).get(s1, 0)
                    waarde += prob * (r + self.gamma * V[s1])

                if waarde > beste_waarde:
                    beste_waarde = waarde
                    beste_actie = a

            nieuw_beleid[s] = beste_actie

        return nieuw_beleid, V

    def get_value_function(self):
        """Geeft de waarde-functie terug."""
        return self.V

    def get_policy(self):
        """Geeft het optimale beleid terug."""
        return self.policy
