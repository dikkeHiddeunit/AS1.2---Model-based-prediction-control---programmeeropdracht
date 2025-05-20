import numpy as np

class Maze:
    """
    Een eenvoudige 4x4 doolhofomgeving met vaste beloningen.
    
    - Toestanden worden genummerd van 0 t/m 15 (4x4 grid).
    - Elke toestand heeft een standaard beloning van -1, tenzij anders gedefinieerd.
    - Bepaalde toestanden zijn terminaal: hier stopt het spel.
    - De agent kan vier richtingen op bewegen: omhoog, omlaag, links, rechts.
    """

    def __init__(self):
        """
        Initialiseert de doolhofomgeving met breedte, hoogte, beloningen,
        begintoestand, terminale toestanden en mogelijke acties.
        """
        self.width = 4  
        self.height = 4 
        self.n_states = self.width * self.height 

        self.rewards = np.full(self.n_states, -1.0)

        # specifieke beloningen voor bepaalde toestanden
        self.rewards[3] = 40  
        self.rewards[6] = -10  
        self.rewards[7] = -10
        self.rewards[12] = 10 
        self.rewards[13] = -2  

        self.start_state = 14        
        self.terminals = [3, 12]   


        self.actions = {
            0: (-1, 0),   # Omhoog
            1: (1, 0),    # Omlaag
            2: (0, -1),   # Links
            3: (0, 1),    # Rechts
        }

        #print(self.rewards)

    def is_terminal(self, state):
        """
        Controleert of een toestand terminaal is.
        
        Parameters:
        - state (int): De toestand om te controleren.
        
        Retourneert:
        - bool: True als de toestand terminaal is, anders False.
        """
        return state in self.terminals

    def get_reward(self, state):
        """
        Haalt de beloning op van een specifieke toestand.
        
        Parameters:
        - state (int): De toestand waarvoor de beloning wordt opgevraagd.
        
        Retourneert:
        - float: De beloning van de toestand.
        """
        return self.rewards[state]

    def step(self, state, action):
        """
        Voert een actie uit vanuit een bepaalde toestand en geeft de volgende toestand.
        
        Parameters:
        - state (int): De huidige toestand.
        - action (int): De actie om uit te voeren (0=omhoog, 1=omlaag, 2=links, 3=rechts).
        
        Retourneert:
        - int: De volgende toestand na het uitvoeren van de actie.
        """
        # Als de toestand terminaal is, verandert er niets
        if self.is_terminal(state):
            return state

        # Bereken huidige coördinaten
        x = state % self.width
        y = state // self.width

        # Verkrijg de verplaatsing voor de actie
        dx, dy = self.actions[action]

        # Bereken nieuwe coördinaten met randcontrole
        new_x = max(0, min(self.width - 1, x + dx))
        new_y = max(0, min(self.height - 1, y + dy))

        # Converteer coördinaten terug naar toestandindex
        return new_y * self.width + new_x
