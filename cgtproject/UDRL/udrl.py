import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / (dataset_size/batch_size):.4f}")


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
