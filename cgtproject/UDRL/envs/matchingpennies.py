import numpy as np
import random

class MatchingPenniesSimEnv:
    """
    Matching Pennies Game:
      - Two players, each with two actions (0 and 1).
      - If both players choose the same action, player 1 wins (+1) and player 2 loses (-1).
      - If the actions differ, player 1 loses (-1) and player 2 wins (+1).
      - The game is zero-sum.
      - The state for each player is the one-hot encoding of the opponent's last action.
    """
    def __init__(self):
        self.payoff_matrix1 = np.array([[1, -1],
                                        [-1, 1]])
        self.payoff_matrix2 = -self.payoff_matrix1

    def reset(self):
        # Random initial actions to set up the starting states.
        p1_action = random.choice([0, 1])
        p2_action = random.choice([0, 1])
        state1 = self._state_from_action(p2_action)  # Player 1 observes p2's action.
        state2 = self._state_from_action(p1_action)  # Player 2 observes p1's action.
        return state1, state2

    def step(self, p1_action, p2_action):
        reward1 = self.payoff_matrix1[p1_action, p2_action]
        reward2 = self.payoff_matrix2[p1_action, p2_action]
        next_state1 = self._state_from_action(p2_action)
        next_state2 = self._state_from_action(p1_action)
        return (next_state1, reward1), (next_state2, reward2)

    def _state_from_action(self, action):
        # For two possible actions, the state is a one-hot encoded vector of length 2.
        state = np.zeros(2)
        state[action] = 1
        return state