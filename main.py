from maze import Maze
from agent import Agent
from value_iteration import ValueIterationAgent
from policy import DeterministicPolicy



def build_transition_and_reward_matrices(maze):
    states = list(range(maze.n_states))
    actions = list(maze.actions.keys())
    P = {s: {a: [] for a in actions} for s in states}
    R = {s: {a: 0 for a in actions} for s in states}

    for s in states:
        for a in actions:
            next_state = maze.step(s, a)
            reward = maze.get_reward(next_state)
            P[s][a].append((1.0, next_state))
            R[s][a] = reward
    return states, actions, P, R


def visualize_path_with_arrows(path, actions, maze):
    arrows = {0: '←', 1: '→', 2: '↑', 3: '↓'}
    grid = [[" ." for _ in range(maze.width)] for _ in range(maze.height)]

    for i in range(len(path) - 1):
        state = path[i]
        action = actions[i]
        x = state % maze.width
        y = state // maze.width
        grid[y][x] = " " + arrows[action]

    # Mark start
    x0 = path[0] % maze.width
    y0 = path[0] // maze.width
    grid[y0][x0] = " S"

    # Mark end
    x_end = path[-1] % maze.width
    y_end = path[-1] // maze.width
    grid[y_end][x_end] = " T" if maze.is_terminal(path[-1]) else " E"

    return "\n".join("".join(row) for row in grid)


def main():
    maze = Maze()
    states, actions, P, R = build_transition_and_reward_matrices(maze)

    vi_agent = ValueIterationAgent(states, actions, P, R)
    vi_agent.run()
    optimal_policy = vi_agent.get_policy()

    policy = DeterministicPolicy(optimal_policy)
    agent = Agent(maze, policy)

    state = maze.start_state
    path = [state]
    actions_taken = []

    for _ in range(20):
        s, a, ns, r = agent.act(state)
        path.append(ns)
        actions_taken.append(a)
        state = ns
        if maze.is_terminal(state):
            break

    print("Agent path:", path)
    print("\nVisual representation with arrows:\n")
    print(visualize_path_with_arrows(path, actions_taken, maze))


if __name__ == "__main__":
    main()

