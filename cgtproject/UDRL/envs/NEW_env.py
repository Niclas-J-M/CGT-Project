import numpy as np
import random

class NewGameSimultaneousEnv:
    def __init__(self):
        # Define payoff matrices for both players.
        # For player 1:
        self.payoff_matrix1 = np.array([[2, 0],
                                        [0, 1]])
        # For player 2:
        self.payoff_matrix2 = np.array([[0, 1],
                                        [2, 0]])
    
    def reset(self):
        # At the start, randomly pick an initial action for each player
        p1_action = random.choice([0, 1])
        p2_action = random.choice([0, 1])
        # Each player's state is the one-hot encoding of the opponent's last action.
        state1 = self._state_from_action(p2_action)  # Player1’s view: opponent (player2)’s action.
        state2 = self._state_from_action(p1_action)  # Player2’s view: opponent (player1)’s action.
        return state1, state2

    def step(self, p1_action, p2_action):
        # Compute rewards using the payoff matrices.
        reward1 = self.payoff_matrix1[p1_action, p2_action]
        reward2 = self.payoff_matrix2[p1_action, p2_action]
        # Next state for each player is the one-hot encoding of the opponent’s action.
        next_state1 = self._state_from_action(p2_action)
        next_state2 = self._state_from_action(p1_action)
        return (next_state1, reward1), (next_state2, reward2)
    
    def _state_from_action(self, action):
        # One-hot encode the action for 2 possible actions.
        state = np.zeros(2)
        state[action] = 1
        return state
