import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Rock-Paper-Scissors 
class RockPaperScissorsEnv:
    def __init__(self):
        # Payoff matrix: rows = agent action, columns = opponent action
        # For agent: Rock=0, Paper=1, Scissors=2
        self.payoff_matrix = np.array([[0, -1, 1],
                                       [1, 0, -1],
                                       [-1, 1, 0]])
    
    def reset(self):
        # For simplicity, we define the "state" as a one-hot encoding of the opponent's last action.
        # At reset, pick a random opponent action.
        opp_action = random.choice([0, 1, 2])
        return self._state_from_action(opp_action)
    
    def step(self, agent_action):
        # Opponent acts randomly.
        opp_action = random.choice([0, 1, 2])
        reward = self.payoff_matrix[agent_action, opp_action]
        next_state = self._state_from_action(opp_action)
        return next_state, reward, opp_action
    
    def _state_from_action(self, action):
        # One-hot encode the opponent's action (3-dimensional vector).
        state = np.zeros(3)
        state[action] = 1
        return state

# Data Collection (Trajectory Logging)
def collect_trajectory(env, T, exploration_rate=0.1):
    """
    Run the environment for T rounds and collect experiences.
    Each experience is a tuple: (state, action, reward, next_state)
    """
    trajectory = []
    state = env.reset()
    for t in range(T):
        # Here we use random exploration.
        if random.random() < exploration_rate:
            action = random.choice([0, 1, 2])
        else:
            # For initial data collection, we default to random actions.
            action = random.choice([0, 1, 2])
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

# UDRL Policy Network Definition and Training
class UDRLPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        """
        The network takes as input a concatenated vector of:
            - state (state_dim)
            - desired_reward (1)
            - horizon (1)
        and outputs logits for each action.
        """
        super(UDRLPolicy, self).__init__()
        input_dim = state_dim + 2  # state + desired_reward + horizon
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, state, desired_reward, horizon):
        # state: batch_size x state_dim
        # desired_reward, horizon: batch_size x 1
        x = torch.cat([state, desired_reward, horizon], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits  # raw logits for cross-entropy loss

def train_udrl_policy(policy_net, training_data, num_epochs=10, batch_size=64, learning_rate=1e-3):
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare lists for training data components.
    states = []
    desired_rewards = []
    horizons = []
    target_actions = []
    
    for sample in training_data:
        state, desired_reward, horizon, target_action = sample
        states.append(state)
        desired_rewards.append([desired_reward])  # shape: (1,)
        horizons.append([horizon])
        target_actions.append(target_action)
    
    # Convert lists to torch tensors.
    states = torch.tensor(np.array(states), dtype=torch.float32)
    desired_rewards = torch.tensor(np.array(desired_rewards), dtype=torch.float32)
    horizons = torch.tensor(np.array(horizons), dtype=torch.float32)
    target_actions = torch.tensor(np.array(target_actions), dtype=torch.long)
    
    dataset_size = states.shape[0]
    
    for epoch in range(num_epochs):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_states = states[indices]
            batch_desired_rewards = desired_rewards[indices]
            batch_horizons = horizons[indices]
            batch_targets = target_actions[indices]
            
            optimizer.zero_grad()
            logits = policy_net(batch_states, batch_desired_rewards, batch_horizons)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / (dataset_size/batch_size):.4f}")

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

