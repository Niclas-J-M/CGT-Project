import random
import torch
import numpy as np

# Data Collection (Trajectory Logging)
def collect_trajectory(env, T, exploration_rate=0.1, action_space=None):
    """
    Run the environment for T rounds and collect experiences.
    Each experience is a tuple: (state, action, reward, next_state)
    
    action_space: list or iterable of valid actions (e.g. [0,1,2] for RPS)
    """
    if action_space is None:
        raise ValueError("An action space must be provided.")
    
    trajectory = []
    state = env.reset()
    for t in range(T):
        # For initial data collection, choose a random action from the provided action_space.
        action = random.choice(action_space)
        next_state, reward, _ = env.step(action)
        trajectory.append((state, action, reward, next_state))
        state = next_state
    return trajectory

# Transform Trajectory into (Command -> Action) Training Pairs
def generate_training_data(trajectory):
    """
    For every pair (k, j) with k < j in a trajectory,
    create a training sample:
        input: (state_k, cumulative_reward from k to j, horizon=j-k)
        target: action taken at time k.
    """
    training_data = []
    T = len(trajectory)
    for k in range(T - 1):
        state_k, action_k, _, _ = trajectory[k]
        cumulative = 0.0
        for j in range(k+1, T):
            # Add the reward received at the (j-1)-th step.
            cumulative += trajectory[j-1][2]
            horizon = j - k
            training_data.append((state_k, cumulative, horizon, action_k))
    return training_data

# Inference using the Learned UDRL Policy
def infer_action(policy_net, state, desired_reward, horizon):
    """
    Given the current state and a command (desired_reward, horizon),
    produce a probability distribution over actions and sample one.
    """
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape: (1, state_dim)
    desired_reward_tensor = torch.tensor([[desired_reward]], dtype=torch.float32)
    horizon_tensor = torch.tensor([[horizon]], dtype=torch.float32)
    with torch.no_grad():
        logits = policy_net(state_tensor, desired_reward_tensor, horizon_tensor)
        probs = torch.softmax(logits, dim=1).numpy().flatten()
    action = np.random.choice(len(probs), p=probs)
    return action, probs