import numpy as np
import random
import torch
from udrl import train_udrl_policy, UDRLPolicy
from RPS_env import RockPaperScissorsEnv
from utils import infer_action, generate_training_data, collect_trajectory

def main():
    # Set random seeds for reproducibility.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    env = RockPaperScissorsEnv()
    
    # Collect a trajectory.
    T = 1000  # number of rounds in an episode (can be adjusted)
    # Use a higher exploration rate during data collection.
    trajectory = collect_trajectory(env, T, exploration_rate=0.5)
    print("Collected trajectory length:", len(trajectory))
    
    # Generate training data from the trajectory
    training_data = generate_training_data(trajectory)
    print("Number of training samples:", len(training_data))
    
    # Create and train the UDRL policy network.
    # For RPS, state_dim = 3 (one-hot for opponent's last action), num_actions = 3.
    state_dim = 3
    hidden_dim = 16
    num_actions = 3
    policy_net = UDRLPolicy(state_dim, hidden_dim, num_actions)
    
    train_udrl_policy(policy_net, training_data, num_epochs=10, batch_size=64, learning_rate=1e-3)
    
    # Use the trained policy for inference.
    # Here we specify a command: e.g., "achieve 0.0 reward within 1 step".
    current_state = env.reset()
    desired_reward = 0.0  # command: aim for a neutral outcome (can be changed to prompt exploration)
    horizon = 1         # within one round
    action, probs = infer_action(policy_net, current_state, desired_reward, horizon)
    print("Inferred action:", action, "with probabilities:", probs)

if __name__ == "__main__":
    main()
