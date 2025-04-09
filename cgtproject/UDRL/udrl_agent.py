import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from utils import collect_trajectories_policy, generate_training_data
from udrl import UDRLPolicy

@dataclass
class Command:
    reward: float
    horizon: int

    def update(self, collected_reward: float, max_reward: float = None) -> None:
        # Update the command after collecting reward
        self.reward -= collected_reward
        if max_reward is not None:
            self.reward = min(self.reward, max_reward)
        self.horizon = max(self.horizon - 1, 1)

    def duplicate(self) -> "Command":
        return Command(self.reward, self.horizon)

class UDRLAgent:
    def __init__(self, env, state_dim, num_actions, device: str = None, learning_rate: float = 1e-3, max_reward: float = None):
        self.env = env
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Create the UDRL policy network and move it to device.
        self.policy_net = UDRLPolicy(state_dim, hidden_dim=16, num_actions=num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.max_reward = max_reward

        # Initialize a command; you may adjust these initial values.
        self.command = Command(reward=0.0, horizon=1)

    def update_policy(self, training_data, num_epochs: int = 5, batch_size: int = 64) -> None:
        dataset_size = len(training_data)
        for epoch in range(num_epochs):
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i : i + batch_size]
                batch = [training_data[idx] for idx in indices]
                # Each sample is (state, cumulative_reward, horizon, target_action)
                states = torch.tensor(np.array([s for (s, r, h, a) in batch]), dtype=torch.float32).to(self.device)
                desired_rewards = torch.tensor(np.array([[r] for (s, r, h, a) in batch]), dtype=torch.float32).to(self.device)
                horizons = torch.tensor(np.array([[h] for (s, r, h, a) in batch]), dtype=torch.float32).to(self.device)
                targets = torch.tensor(np.array([a for (s, r, h, a) in batch]), dtype=torch.long).to(self.device)

                self.optimizer.zero_grad()
                logits = self.policy_net(states, desired_rewards, horizons)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/(dataset_size/batch_size):.4f}")

    def learn(self, iterations: int, T: int, exploration_rate: float, action_space: list, desired_reward: float, epochs_per_iter: int = 5) -> None:
        """
        Iteratively collect trajectories using the evolving policy and update the policy.
        Here we assume a simultaneous environment where both players use the same policy_net.
        (For true multi-agent learning you might want separate networks.)
        """
        for it in range(iterations):
            print(f"\n--- Self-play Iteration {it+1}/{iterations} ---")
            # Collect trajectories for both players using the current policy.
            traj1, traj2 = collect_trajectories_policy(
                self.env,
                T,
                policy_net1=self.policy_net,
                policy_net2=self.policy_net,
                desired_reward=desired_reward,
                horizon=1,
                exploration_rate=exploration_rate,
                action_space=action_space,
            )
            print("Collected trajectories for both players.")

            # Generate training data from both trajectories.
            training_data1 = generate_training_data(traj1)
            training_data2 = generate_training_data(traj2)
            combined_training_data = training_data1 + training_data2
            print(f"Total training samples: {len(combined_training_data)}")

            # Update the policy network using the combined training data.
            self.update_policy(combined_training_data, num_epochs=epochs_per_iter)
    
    def infer_action(self, state, desired_reward: float, horizon: int):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        desired_reward_tensor = torch.tensor([[desired_reward]], dtype=torch.float32).to(self.device)
        horizon_tensor = torch.tensor([[horizon]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(state_tensor, desired_reward_tensor, horizon_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        action = np.random.choice(self.num_actions, p=probs)
        return action, probs
