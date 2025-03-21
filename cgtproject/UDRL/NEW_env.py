import numpy as np
import random

# New Game Environment with a 2x2 payoff matrix
class NewGameEnv:
    def __init__(self):
        # Payoff matrix: rows = agent action, columns = opponent action
        # Example: Agent: action 0 yields 2 against opponent 0, 0 against opponent 1; action 1 yields 0 vs. 0, and 1 vs. 1.
        self.payoff_matrix = np.array([[2, 0],
                                       [0, 1]])
    
    def reset(self):
        # Here, the state is the one-hot encoding of the opponent's last action.
        opp_action = random.choice([0, 1])
        return self._state_from_action(opp_action)
    
    def step(self, agent_action):
        # Opponent acts randomly.
        opp_action = random.choice([0, 1])
        reward = self.payoff_matrix[agent_action, opp_action]
        next_state = self._state_from_action(opp_action)
        return next_state, reward, opp_action
    
    def _state_from_action(self, action):
        # One-hot encode the opponent's action (2-dimensional vector).
        state = np.zeros(2)
        state[action] = 1
        return state
