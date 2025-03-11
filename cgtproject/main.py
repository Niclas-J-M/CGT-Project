import numpy as np
import matplotlib.pyplot as plt
from qlearning import QLearningAgent
from plots import RPSPlotting


def simulate_game(num_iters=500000, tau=0.1):
    """
    Simulate the repeated play of rock–paper–scissors between two Q-learning agents.
    
    num_iters: number of iterations (game rounds)
    lr0: initial learning rate scaling factor (which decreases over time)
    tau: temperature parameter for Boltzmann exploration
    """
    # Define actions: 0: Rock, 1: Paper, 2: Scissors
    # Define payoff matrix for player 1; player 2 is zero-sum so reward for player 2 is the negative.
    # Payoff matrix from the paper (row: player1's action, col: player2's action):
    #    Rock  Paper  Scissors
    # Rock   0     -1      1
    # Paper  1      0     -1
    # Sciss -1      1      0
    payoff_matrix = np.array([[0, -1, 1],
                              [1, 0, -1],
                              [-1, 1, 0]])
    
    # Create two Q-learning agents
    agent1 = QLearningAgent(n_actions=3, tau=tau)
    agent2 = QLearningAgent(n_actions=3, tau=tau)
    

    history_agent1 = []
    history_agent2 = []
    
    for n in range(1, num_iters + 1):
        # learning rate
        lr = (n + 100) ** (-0.9)

        # Tried to replicate the paper's results more
        # Decaying tau 
        # tau = 0.1 * np.exp(-0.000002 * n)  # Exponential decay

        # agent1.tau = tau
        # agent2.tau = tau

        action1, policy1 = agent1.select_action()
        action2, policy2 = agent2.select_action()
        
        # Determine rewards based on the game: player 1's reward from payoff_matrix; player 2 gets the negative.
        reward1 = payoff_matrix[action1, action2]
        reward2 = -reward1  # Zero-sum game
        
        # Update each agent's Q-value for the action taken.
        agent1.update(action1, reward1, lr, policy1)
        agent2.update(action2, reward2, lr, policy2)
        

        history_agent1.append(agent1.get_policy().copy())
        history_agent2.append(agent2.get_policy().copy())

        if n % 100000 == 0:
            print(f"Iteration {n}: π^1(R)={policy1[0]:.3f}, π^1(S)={policy1[2]:.3f}, π^1(P)={policy1[1]:.3f}")

    plotter = RPSPlotting(history_agent1, history_agent2)

    # Plot the evolution of agent 1's policy
    plotter.plot_policy_evolution(agent=1)

    # Scatter plot of strategy evolution
    plotter.scatter_strategy_evolution(omit_first=10000)

    # Sampled evolution plot
    plotter.plot_sampled_evolution(step=5000)
    


if __name__ == "__main__":
    simulate_game(num_iters=500000, tau=0.1)
