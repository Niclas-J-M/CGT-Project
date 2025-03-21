import argparse
import random
import numpy as np
import torch
from udrl import UDRLPolicy, train_udrl_policy
from utils import collect_trajectory, generate_training_data, infer_action

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Run UDRL experiments with selectable environments."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="RPS",
        help="Environment to use: 'RPS' for Rock-Paper-Scissors or 'NEW' for NewGame."
    )
    parser.add_argument(
        "--DR",
        type=float,
        default=0.0,
        help="Desired Reward to use: 0 is Default"
    )
    parser.add_argument(
        "--T",
        type=int,
        default=1000,
        help="Number of rounds (trajectory length) to collect."
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Choose the environment and set its parameters.
    if args.env.upper() == "RPS":
        from RPS_env import RockPaperScissorsEnv
        env = RockPaperScissorsEnv()
        state_dim = 3    # One-hot vector for 3 actions.
        num_actions = 3
        action_space = list(range(num_actions))  # [0, 1, 2]
    elif args.env.upper() == "NEW":
        from NEW_env import NewGameEnv  # Ensure you have defined NewGameEnv similarly.
        env = NewGameEnv()
        state_dim = 2    # For example, a 2-dimensional state.
        num_actions = 2
        action_space = list(range(num_actions))  # [0, 1]
    else:
        raise ValueError("Invalid environment choice! Use 'RPS' or 'NEW'.")

    # Collect a trajectory using the appropriate action space.
    T = args.T
    trajectory = collect_trajectory(env, T, exploration_rate=0.5, action_space=action_space)
    print("Collected trajectory length:", len(trajectory))
    
    # Generate training data from the trajectory.
    training_data = generate_training_data(trajectory)
    print("Number of training samples:", len(training_data))
    
    # Create and train the UDRL policy network.
    policy_net = UDRLPolicy(state_dim, hidden_dim=16, num_actions=num_actions)
    train_udrl_policy(policy_net, training_data, num_epochs=10, batch_size=64, learning_rate=1e-3)
    
    # Use the trained policy for inference.
    current_state = env.reset()
    desired_reward = args.DR  # Change based on game
    horizon = 1         # Within one step.
    action, probs = infer_action(policy_net, current_state, desired_reward, horizon)
    print("Inferred action:", action, "with probabilities:", probs)

if __name__ == "__main__":
    main()
