import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from udrl import UDRLAgent

# Simple replay buffer to store (command, action, reward) tuples.
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def push(self, command, action, reward):
        self.buffer.append((command, action, reward))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Function to perform a batch update on the agent using a mini-batch from the replay buffer.
def update_agent_batch(agent, batch):
    # Prepare batch tensors:
    commands = torch.tensor([[x[0]] for x in batch], dtype=torch.float32, device=agent.device)
    targets = torch.tensor([x[1] for x in batch], dtype=torch.long, device=agent.device)
    # Forward pass through the network:
    outputs = agent.network(commands)  # shape: (batch_size, n_actions)
    loss = agent.criterion(outputs, targets)
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    return loss.item()

def simulate_game(num_iters=500000, batch_size=32, update_every=10):
    """
    Simulate repeated rock-paper-scissors where one agent learns via UDRL with a replay buffer,
    and the opponent always plays rock (action 0).

    With a fixed opponent, the best response for the learning agent is to play paper (action 1).
    """
    # Define payoff matrix for player 1.
    # Rows: learning agent's actions, Columns: opponent's actions.
    payoff_matrix = np.array([[0, -1, 1],   # Rock: vs Rock=0, vs Paper=-1, vs Scissors=1
                              [1, 0, -1],   # Paper: vs Rock=1, vs Paper=0, vs Scissors=-1
                              [-1, 1, 0]])  # Scissors: vs Rock=-1, vs Paper=1, vs Scissors=0

    # Create one UDRL agent with desired_return = 0.0.
    # (Even though the optimal best response here is paper, in a zero-sum game the Nash value is 0.)
    agent = UDRLAgent(desired_return=1.0, lr=0.01)
    
    replay_buffer = ReplayBuffer(capacity=10000)
    
    history_policy = []
    batch_losses = []
    # Initialize reward (to be updated each round)
    reward = 0

    for n in range(1, num_iters + 1):
        # Instead of using only the last reward, we add noise to broaden the command signal.
        noisy_command = reward + np.random.normal(0, 1.0)
        
        # Learning agent selects an action conditioned on the noisy command.
        action, policy = agent.select_action(command=noisy_command)
        
        # Fixed opponent always plays rock (action 0).
        opponent_action = 0
        
        # Determine reward for the learning agent.
        reward = payoff_matrix[action, opponent_action]
        
        # Store this experience in the replay buffer.
        replay_buffer.push(noisy_command, action, reward)
        
        history_policy.append(policy)
        
        # Periodically update the agent using a mini-batch sampled from the buffer.
        if n % update_every == 0 and len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = update_agent_batch(agent, batch)
            batch_losses.append(loss)
        
        if n % 100000 == 0:
            print(f"Iteration {n}: Latest policy: {policy}, Last reward: {reward}")
    
    # Plot the evolution of the learning agent's policy (based on the network's output with desired command 0.0).
    history_policy = np.array(history_policy)
    plt.figure(figsize=(8, 4))
    plt.plot(history_policy[-100:, 0], label="Rock")
    plt.plot(history_policy[-100:, 1], label="Paper")
    plt.plot(history_policy[-100:, 2], label="Scissors")
    plt.xlabel("Iteration (last 100 rounds)")
    plt.ylabel("Probability")
    plt.title("Learning Agent Policy Evolution (UDRL with Replay Buffer)")
    plt.legend()
    plt.show()
    
    # Optionally, plot the loss curve.
    plt.figure(figsize=(8, 4))
    plt.plot(batch_losses)
    plt.xlabel("Update Step")
    plt.ylabel("Batch Loss")
    plt.title("Training Loss")
    plt.show()

if __name__ == "__main__":
    simulate_game(num_iters=500000, batch_size=32, update_every=10)
