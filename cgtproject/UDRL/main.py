import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from udrl import UDRLPolicy, train_udrl_policy, infer_action
from utils import generate_training_data, collect_trajectories_policy
from envs.matchingpennies import MatchingPenniesSimEnv
from envs.shapleys import ShapleySimEnv
from envs.stagHuntsgame import StagHuntSimEnv
from envs.NEW_env import NewGameSimultaneousEnv
from envs.RPS_env import RockPaperScissorsSimEnv

def experiment_on_env(env_name, env, state_dim, num_actions, desired_reward,
                      T=500, iterations=5, epochs=5, exploration_rate=0.3):
    """
    Run a UDRL self-play experiment on a given environment.
    For each iteration, the policy networks are trained using trajectories
    and then the current inferred probability distributions (from a reset state)
    are recorded. At the end, the final inferred policies are printed and returned,
    along with the complete evolution history.
    """
    # Create UDRL policy networks for both players.
    policy_net1 = UDRLPolicy(state_dim, hidden_dim=16, num_actions=num_actions)
    policy_net2 = UDRLPolicy(state_dim, hidden_dim=16, num_actions=num_actions)
    
    horizon = 1  # Fixed horizon.
    action_space = list(range(num_actions))
    
    # Lists to record the evolution of each player's policy.
    history1 = []
    history2 = []
    
    # Run self-play iterations.
    for it in range(iterations):
        print(f"  Iteration {it+1}/{iterations} for {env_name}")
        traj1, traj2 = collect_trajectories_policy(
            env, T, policy_net1, policy_net2, desired_reward, horizon,
            exploration_rate=exploration_rate, action_space=action_space
        )
        training_data1 = generate_training_data(traj1)
        training_data2 = generate_training_data(traj2)
        
        # Train both policy networks.
        train_udrl_policy(policy_net1, training_data1, num_epochs=epochs, batch_size=64, learning_rate=1e-3)
        train_udrl_policy(policy_net2, training_data2, num_epochs=epochs, batch_size=64, learning_rate=1e-3)
        
        # Record the evolution by doing inference after this iteration.
        state1, state2 = env.reset()
        _, probs1 = infer_action(policy_net1, state1, desired_reward, horizon)
        _, probs2 = infer_action(policy_net2, state2, desired_reward, horizon)
        history1.append(probs1.copy())
        history2.append(probs2.copy())
    
    # Final inference (optional, could be the same as the last iteration):
    state1, state2 = env.reset()
    _, final_probs1 = infer_action(policy_net1, state1, desired_reward, horizon)
    _, final_probs2 = infer_action(policy_net2, state2, desired_reward, horizon)
    
    print("\nFinal Inferred Probability Distributions for", env_name)
    print("  Player 1 probabilities:", final_probs1)
    print("  Player 2 probabilities:", final_probs2)
    
    return final_probs1, final_probs2, np.array(history1), np.array(history2)

def main():
    # Set random seeds for reproducibility.
    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)
    
    # Define the number of times to run each environment experiment.
    num_runs = 5
    
    # Define configurations for all environments.
    experiments = {
        "Matching Pennies": {
            "env": MatchingPenniesSimEnv(),
            "state_dim": 2,       # Two actions: one-hot length = 2.
            "num_actions": 2,
            "desired_reward": 0.0
        },
        "Shapley's Game": {
            "env": ShapleySimEnv(),
            "state_dim": 3,       # Three actions.
            "num_actions": 3,
            "desired_reward": 0.33
        },
        "Stag Hunt": {
            "env": StagHuntSimEnv(),
            "state_dim": 2,       # Two actions.
            "num_actions": 2,
            "desired_reward": 4.0
        },
        "New Environment": {
            "env": NewGameSimultaneousEnv(),
            "state_dim": 2,       # Two actions.
            "num_actions": 2,
            "desired_reward": 0.67
        },
        "Rock Paper Scissors": {
            "env": RockPaperScissorsSimEnv(),
            "state_dim": 3,       # Three actions.
            "num_actions": 3,
            "desired_reward": 0.0
        }
    }
    
    final_results = {}
    # To store policy evolution histories for each experiment.
    histories_over_runs = {}
    
    # Run each experiment num_runs times and average the results.
    for name, config in experiments.items():
        if name == "Shapley's Game":
            print("\n==========================")
            print(f"Running averaged experiment for {name}")
            print("==========================")
            probs1_list = []
            probs2_list = []
            run_histories1 = []  # To record per-run evolution for Player 1.
            run_histories2 = []  # Likewise for Player 2.
            for run in range(num_runs):
                print(f"\nRun {run+1}/{num_runs} for {name}")
                p1, p2, history1, history2 = experiment_on_env(
                    env_name=name,
                    env=config["env"],
                    state_dim=config["state_dim"],
                    num_actions=config["num_actions"],
                    desired_reward=config["desired_reward"],
                    T=500,
                    iterations=5,
                    epochs=5,
                    exploration_rate=0.3
                )
                probs1_list.append(p1)
                probs2_list.append(p2)
                run_histories1.append(history1)  # history1 is (iterations, n_actions)
                run_histories2.append(history2)
            # Average the final probability distributions over the runs.
            avg_probs1 = np.mean(probs1_list, axis=0)
            avg_probs2 = np.mean(probs2_list, axis=0)
            final_results[name] = {"Player1": avg_probs1, "Player2": avg_probs2}
            histories_over_runs[name] = {"Player1": np.stack(run_histories1, axis=0), "Player2": np.stack(run_histories2, axis=0)}
        
        # Print the summary of final probability distributions.
        print("\n==========================")
        print("Summary of Averaged Final Probability Distributions")
        print("==========================")
        for name, result in final_results.items():
            print(f"\n{name}:")
            print("  Player 1:", result["Player1"])
            print("  Player 2:", result["Player2"])
    
    # For each experiment, average the per-iteration history over runs and plot Player 1's evolution.
    for name, history_data in histories_over_runs.items():
        print(name)
        history_array = history_data["Player1"]  # shape: (num_runs, iterations, n_actions)
        mean_history = np.mean(history_array, axis=0)  # shape: (iterations, n_actions)
        std_history = np.std(history_array, axis=0)    # shape: (iterations, n_actions)
        iterations = np.arange(1, mean_history.shape[0] + 1)
        
        plt.figure(figsize=(10, 6))
        for action in range(mean_history.shape[1]):
            plt.errorbar(
                iterations,
                mean_history[:, action],
                yerr=std_history[:, action],
                marker='o',
                linestyle='-',
                label=f'Action {action}'
            )
        plt.xlabel('Iteration')
        plt.ylabel('Time-Averaged-Probability')
        plt.title(f"Evolution of Player 1's Policy for {name}")
        plt.legend()
        # Save the plot with a suitable filename (replace spaces with underscores)
        filename = f"Plots/averaged_policy_evolution_{name.replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Policy evolution plot saved as {filename}")

if __name__ == "__main__":
    main()
