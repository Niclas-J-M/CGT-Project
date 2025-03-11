import numpy as np
import matplotlib.pyplot as plt
from qlearning import QLearningAgent


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
    
    # To record the evolution of the agents' policies over time
    history_agent1 = []
    history_agent2 = []
    
    for n in range(1, num_iters + 1):
        # Example learning rate schedule: decays slowly over time.
        lr = (n + 100) ** (-0.9)

        
        # Each agent selects an action and obtains its current policy distribution
        action1, policy1 = agent1.select_action()
        action2, policy2 = agent2.select_action()
        
        # Determine rewards based on the game: player 1's reward from payoff_matrix; player 2 gets the negative.
        reward1 = payoff_matrix[action1, action2]
        reward2 = -reward1  # Zero-sum game
        
        # Update each agent's Q-value for the action taken.
        agent1.update(action1, reward1, lr, policy1)
        agent2.update(action2, reward2, lr, policy2)
        
        # Record the policy probabilities for later analysis.
        history_agent1.append(agent1.get_policy().copy())
        history_agent2.append(agent2.get_policy().copy())

        # Print probabilities every 1000 iterations for debugging
        if n % 100000 == 0:
            print(f"Iteration {n}: π^1(R)={policy1[0]:.3f}, π^1(S)={policy1[2]:.3f}, π^1(P)={policy1[1]:.3f}")

    # Convert history to NumPy arrays for plotting
    #history_agent1 = np.array(history_agent1)
    
    # Convert history to NumPy arrays for plotting
    history_agent1 = np.array(history_agent1)
    history_agent2 = np.array(history_agent2)
    
    # Plot evolution of policy for Agent 1
    # plt.figure(figsize=(10, 6))
    # plt.plot(history_agent1[:, 0], label='Rock')
    # plt.plot(history_agent1[:, 1], label='Paper')
    # plt.plot(history_agent1[:, 2], label='Scissors')
    # plt.xlabel('Iteration')
    # plt.ylabel('Probability')
    # plt.title('Evolution of Agent 1 Policy (Q-learning with Smooth Best Response)')
    # plt.legend()
    # plt.show()
    
    # # For completeness, you could also plot Agent 2's policy evolution.
    # plt.figure(figsize=(10, 6))
    # plt.plot(history_agent2[:, 0], label='Rock')
    # plt.plot(history_agent2[:, 1], label='Paper')
    # plt.plot(history_agent2[:, 2], label='Scissors')
    # plt.xlabel('Iteration')
    # plt.ylabel('Probability')
    # plt.title('Evolution of Agent 2 Policy (Q-learning with Smooth Best Response)')
    # plt.legend()
    # plt.show()

    # Scatter plot π^1(S) (y-axis) vs π^1(R) (x-axis)
    plt.figure(figsize=(10, 6))
    plt.scatter(history_agent1[:, 0], history_agent1[:, 2], alpha=0.3, s=1)
    plt.xlabel(r'$\pi^1(R)$')
    plt.ylabel(r'$\pi^1(S)$')
    plt.title('Strategy Evolution of Player 1 in Rock-Paper-Scissors')
    plt.show()

    # Plot omitting first 10,000 iterations
    plt.figure(figsize=(10, 6))
    plt.scatter(history_agent1[10000:, 0], history_agent1[10000:, 2], alpha=0.3, s=1)
    plt.xlabel(r'$\pi^1(R)$')
    plt.ylabel(r'$\pi^1(S)$')
    plt.title('Strategy Evolution (Ignoring First 10,000 Iterations)')
    plt.show()

if __name__ == "__main__":
    simulate_game(num_iters=500000, tau=0.1)
