import numpy as np
import random
import matplotlib.pyplot as plt
from qlearning import QLearningAgent

def simulate_game_env(payoff_matrix, n_actions, rounds=500000, tau=0.1):
    """
    Standard simulation without recording the evolution history.
    """
    agent1 = QLearningAgent(n_actions=n_actions, tau=tau)
    agent2 = QLearningAgent(n_actions=n_actions, tau=tau)
    
    for n in range(1, rounds + 1):
        lr = (n + 100) ** (-0.9)  # Decaying learning rate.
        action1, policy1 = agent1.select_action()
        action2, policy2 = agent2.select_action()
        if isinstance(payoff_matrix, (tuple, list)):
            reward1 = payoff_matrix[0][action1, action2]
            reward2 = payoff_matrix[1][action1, action2]
        else:
            reward1 = payoff_matrix[action1, action2]
            reward2 = -reward1  # Zero-sum default.
        
        agent1.update(action1, reward1, lr, policy1)
        agent2.update(action2, reward2, lr, policy2)
    
    final_policy1 = agent1.get_policy()
    final_policy2 = agent2.get_policy()
    return final_policy1, final_policy2

def simulate_game_env_with_history(payoff_matrix, n_actions, rounds=500000, tau=0.1, record_interval=10000):
    """
    Simulate the game while recording the time-averaged policy (running average) history.
    
    This function computes the cumulative sum of each agent's softmax distributions and
    records the running average at every 'record_interval' rounds.
    """
    agent1 = QLearningAgent(n_actions=n_actions, tau=tau)
    agent2 = QLearningAgent(n_actions=n_actions, tau=tau)
    
    cum_policy1 = np.zeros(n_actions)
    cum_policy2 = np.zeros(n_actions)
    history1 = []  # To record the running averaged policy for agent 1.
    history2 = []  # To record the running averaged policy for agent 2.
    
    for n in range(1, rounds + 1):
        lr = (n + 100) ** (-0.9)
        action1, policy1 = agent1.select_action()
        action2, policy2 = agent2.select_action()
        
        if isinstance(payoff_matrix, (tuple, list)):
            reward1 = payoff_matrix[0][action1, action2]
            reward2 = payoff_matrix[1][action1, action2]
        else:
            reward1 = payoff_matrix[action1, action2]
            reward2 = -reward1
        
        # Update the agents.
        agent1.update(action1, reward1, lr, policy1)
        agent2.update(action2, reward2, lr, policy2)
        
        # Update cumulative policy sum.
        cum_policy1 += policy1
        cum_policy2 += policy2
        
        # Record the running average every 'record_interval' iterations.
        if n % record_interval == 0:
            avg_policy1 = cum_policy1 / n
            avg_policy2 = cum_policy2 / n
            # Save a copy for this time step.
            history1.append(avg_policy1.copy())
            history2.append(avg_policy2.copy())
    
    final_policy1 = agent1.get_policy()
    final_policy2 = agent2.get_policy()
    history1 = np.array(history1)
    history2 = np.array(history2)
    return final_policy1, final_policy2, history1, history2

def main():
    # Define game configurations.
    games = {
        "Matching Pennies": {
            "payoff_matrix": np.array([[1, -1],
                                       [-1, 1]]),
            "n_actions": 2
        },
        "Shapley's Game": {
            "payoff_matrix": (
                np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]]),
                np.array([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]])
            ),
            "n_actions": 3
        },
        "Stag Hunt": {
            "payoff_matrix": (
                np.array([[2, 0],
                          [3, 1]]),
                np.array([[2, 3],
                          [0, 1]])
            ),
            "n_actions": 2
        },
        "New Environment": {
            "payoff_matrix": (
                np.array([[2, 0],
                          [0, 1]]),
                np.array([[0, 1],
                          [2, 0]])
            ),
            "n_actions": 2
        },
        "Rock Paper Scissors": {
            "payoff_matrix": np.array([[0, -1, 1],
                                       [1, 0, -1],
                                       [-1, 1, 0]]),
            "n_actions": 3
        }
    }
    
    # Simulation parameters.
    num_runs = 5                    # Number of independent runs.
    total_rounds = 500000           # Total rounds per run.
    record_interval = 10000         # Record data every 10,000 rounds.
    tau = 0.1                       # Temperature parameter.
    
    game_name = "Shapley's Game"
    config = games[game_name]
    print(f"\nRunning multi-run Q-learning simulations for {game_name} with history recording")
    
    # Collect histories for each run.
    histories_agent1 = []  # List to store history arrays from each run.
    histories_agent2 = []  # If desired, you can do similar processing for agent 2.
    
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs} for {game_name}")
        final_p1, final_p2, history1, history2 = simulate_game_env_with_history(
            config["payoff_matrix"],
            config["n_actions"],
            rounds=total_rounds,
            tau=tau,
            record_interval=record_interval
        )
        histories_agent1.append(history1)
        histories_agent2.append(history2)
    
    # Convert list of histories to a numpy array: shape (num_runs, num_recordings, n_actions)
    histories_agent1 = np.stack(histories_agent1, axis=0)
    histories_agent2 = np.stack(histories_agent2, axis=0)
    
    # Compute the mean and standard deviation (for error bars) at each recorded time step.
    mean_history_agent1 = np.mean(histories_agent1, axis=0)  # (num_recordings, n_actions)
    std_history_agent1 = np.std(histories_agent1, axis=0)
    
    iterations = np.arange(record_interval, total_rounds + 1, record_interval)
    
    # Plot the averaged evolution with error bars for Player 1.
    plt.figure(figsize=(10, 6))
    for action in range(config["n_actions"]):
        plt.errorbar(
            iterations,
            mean_history_agent1[:, action],
            yerr=std_history_agent1[:, action],
            label=f'Action {action}',
            marker='o',
            linestyle='-'
        )
    plt.xlabel('Rounds')
    plt.ylabel('Time-Averaged Probability')
    plt.title("Evolution of Player 1's Time-Averaged Policy Across Runs")
    plt.legend()
    plt.savefig("Plots/multi_run_time_averaged_policy_player1_5runs.png")
    plt.close()
    
    print("Multi-run simulation completed. Plots have been saved.")

if __name__ == "__main__":
    main()
