import numpy as np

class QLearningAgent:
    def __init__(self, n_actions, tau=0.1, init_Q=0.0):
        """
        n_actions: number of available actions
        tau: temperature parameter for Boltzmann (softmax) action selection
        init_Q: initial Q-value for each action
        """
        self.n_actions = n_actions
        self.tau = tau
        # self.Q = np.zeros(n_actions, dtype=float)
        self.Q = np.full(n_actions, init_Q, dtype=float)
    
    def get_policy(self):
        """
        Compute the Boltzmann (softmax) distribution over actions using current Q-values,
        with normalization to prevent overflow.
        """
        # Shift Q-values by subtracting the maximum Q-value for numerical stability.
        shifted_Q = self.Q - np.max(self.Q)
        exp_vals = np.exp(shifted_Q / self.tau)
        return exp_vals / np.sum(exp_vals)
    
    def select_action(self):
        """
        Select an action based on the softmax probabilities.
        Returns the selected action and the probability distribution.
        """
        policy = self.get_policy()
        action = np.random.choice(self.n_actions, p=policy)
        return action, policy
    
    def update(self, action, reward, lr, policy):
        """
        Update the Q-value for the selected action.
        The update rule is:
            Q(a) = Q(a) + lr * (reward - Q(a)) / π(a)
        where π(a) is the probability of the chosen action.
        """
        self.Q[action] += lr * (reward - self.Q[action]) / (policy[action])
