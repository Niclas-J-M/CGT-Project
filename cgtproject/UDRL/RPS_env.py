import numpy as np
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