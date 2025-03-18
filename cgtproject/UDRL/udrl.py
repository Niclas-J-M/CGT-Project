import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class UDRLNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=3):
        """
        A simple feedforward network that maps a command (desired return)
        to a distribution over actions.
        """
        super(UDRLNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, command):
        # command is expected to be a tensor of shape (batch_size, 1)
        x = self.fc1(command)
        x = self.relu(x)
        x = self.fc2(x)
        # Softmax over output to get probabilities over actions
        return torch.softmax(x, dim=1)

class UDRLAgent:
    def __init__(self, desired_return=0.0, lr=0.01, device='cpu'):
        """
        desired_return: the command (target reward) used at test time.
                       For a zero-sum RPS game, 0.0 is a natural target.
        lr: learning rate for the network optimizer.
        """
        self.device = device
        self.desired_return = desired_return  # Test-time command
        self.network = UDRLNetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def get_policy(self, command=None):
        """
        Given a command (desired return), return the probability distribution
        over actions as computed by the network.
        If command is None, use self.desired_return.
        """
        if command is None:
            command = self.desired_return
        # Convert command to tensor of shape (1, 1)
        command_tensor = torch.tensor([[command]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.network(command_tensor).cpu().numpy().flatten()
        return probs

    def select_action(self, command=None):
        """
        Select an action based on the network's output.
        Returns the chosen action and the underlying probability distribution.
        """
        probs = self.get_policy(command)
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def update(self, command, action):
        """
        Update the network using the achieved reward as the command.
        In Upside Down RL, the training tuple is (command, action) where
        the command is typically the achieved return in that episode.
        We use cross-entropy loss to push the network's prediction (given the command)
        toward the taken action.
        
        command: a float (achieved reward from this round)
        action: an integer in {0, 1, 2}
        """
        # Prepare the input and target tensors
        command_tensor = torch.tensor([[command]], dtype=torch.float32).to(self.device)
        target = torch.tensor([action], dtype=torch.long).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.network(command_tensor)  # shape: (1, n_actions)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
