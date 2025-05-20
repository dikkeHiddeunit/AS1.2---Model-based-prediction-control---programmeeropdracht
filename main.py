from maze import Maze
from agent import Agent
from value_iteration import ValueIterationAgent
from policy import DeterministicPolicy

def build_transition_and_reward_matrices(maze):
    """
    Genereert transitie- en beloningsstructuren voor de doolhof.

    Retourneert:
        states: lijst van toestanden (int)
        actions: lijst van acties (int)
        P: dict van transities: P[s][a] = [(kans, s')]
        R: dict van beloningen: R[s][a][s'] = beloning
    """
    states = list(range(maze.n_states))
    actions = list(maze.actions.keys())
    P = {s: {a: [] for a in actions} for s in states}
    R = {s: {a: {} for a in actions} for s in states}

    for s in states:
        for a in actions:
            next_state = maze.step(s, a)
            reward = maze.get_reward(next_state)
            P[s][a].append((1.0, next_state)) 
            R[s][a][next_state] = reward

    return states, actions, P, R

def visualize_path_with_arrows(path, actions, maze):
    """
    Toont het gevolgde pad op een raster met pijlen.

    Argumenten:
        path: lijst van bezochte toestanden
        actions: bijbehorende acties
        maze: Maze-object

    Retourneert:
        Stringweergave van pad met pijlen en start/eindpunten
    """
    arrows = {0: '←', 1: '→', 2: '↑', 3: '↓'}
    grid = [[" ." for _ in range(maze.width)] for _ in range(maze.height)]

    for i in range(len(path) - 1):
        x = path[i] % maze.width
        y = path[i] // maze.width
        grid[y][x] = " " + arrows[actions[i]]

    x0, y0 = path[0] % maze.width, path[0] // maze.width
    grid[y0][x0] = " S"  # Start

    x_end, y_end = path[-1] % maze.width, path[-1] // maze.width
    grid[y_end][x_end] = " T" if maze.is_terminal(path[-1]) else " E"  # Terminal of eind

    return "\n".join("".join(row) for row in grid)

def main():
    """
    Initieert de doolhofomgeving, voert waarde-iteratie uit,
    laat de agent het optimale pad volgen en toont het resultaat.
    """
    maze = Maze()
    states, actions, P, R = build_transition_and_reward_matrices(maze)

    vi_agent = ValueIterationAgent(states, actions, P, R)
    terminal_states = [s for s in states if maze.is_terminal(s)]
    vi_agent.run(terminal_states)

    # Genereer beleid en koppel het aan de agent
    policy = DeterministicPolicy(vi_agent.get_policy())
    agent = Agent(maze, policy)

    # Simuleer pad vanaf de start
    state = maze.start_state
    path = [state]
    actions_taken = []

    for _ in range(20):  # max 20 stappen
        s, a, next_s, r = agent.act(state)
        path.append(next_s)
        actions_taken.append(a)
        state = next_s
        if maze.is_terminal(state):
            break

    print("Pad van de agent:", path)
    print("\nVisuele representatie met pijlen:\n")
    print(visualize_path_with_arrows(path, actions_taken, maze))

if __name__ == "__main__":
    main()
