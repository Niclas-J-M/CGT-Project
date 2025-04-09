import random

# Data Collection (Trajectory Logging)
def collect_trajectories_policy(env, T, policy_net1, policy_net2, desired_reward, horizon, exploration_rate=0.1, action_space=None):
    """
    Collect simultaneous trajectories for both players using the current policies.
    With probability exploration_rate, the agent takes a random action; otherwise it uses its policy.
    Returns two trajectories (lists of tuples), one per player.
    Each tuple is: (state, action, reward, next_state)
    """
    if action_space is None:
        raise ValueError("An action space must be provided.")
    
    # Import the inference function here to avoid circular dependencies.
    from udrl import infer_action

    traj1 = []  # For player 1
    traj2 = []  # For player 2
    state1, state2 = env.reset()  # Get initial states for both players.
    
    for t in range(T):
        # For player 1: choose action from policy with probability 1 - exploration_rate, else random.
        if random.random() < exploration_rate:
            a1 = random.choice(action_space)
        else:
            a1, _ = infer_action(policy_net1, state1, desired_reward, horizon)
        
        # For player 2:
        if random.random() < exploration_rate:
            a2 = random.choice(action_space)
        else:
            a2, _ = infer_action(policy_net2, state2, desired_reward, horizon)
        
        (next_state1, reward1), (next_state2, reward2) = env.step(a1, a2)
        traj1.append((state1, a1, reward1, next_state1))
        traj2.append((state2, a2, reward2, next_state2))
        state1, state2 = next_state1, next_state2

    return traj1, traj2



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
