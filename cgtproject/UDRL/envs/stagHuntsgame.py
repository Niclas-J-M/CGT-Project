import numpy as np
import random

class StagHuntSimEnv:
    """
    Stag Hunt Game:
      - Two players, each with two actions.
        Action 0: Stag
        Action 1: Hare
      - Typical payoffs:
          * Both choose Stag: Each receives a intermediate payoff (e.g., 2).
          * Both choose Hare: Each receives an low payoff (e.g., 1).
          * One chooses Stag while the other chooses Hare: The Stag hunter gets nothing (0) and the Hare hunter gets a moderate payoff (e.g., 3).
      - The state for each player is a one-hot encoding (of length 2) of the opponent’s last action.
    """
    def __init__(self):
        self.payoff_matrix1 = np.array([[2, 0],
                                        [3, 1]])
        self.payoff_matrix2 = np.array([[2, 3],
                                        [0, 1]])

    def reset(self):
        p1_action = random.choice([0, 1])
        p2_action = random.choice([0, 1])
        state1 = self._state_from_action(p2_action)
        state2 = self._state_from_action(p1_action)
        return state1, state2

    def step(self, p1_action, p2_action):
        reward1 = self.payoff_matrix1[p1_action, p2_action]
        reward2 = self.payoff_matrix2[p1_action, p2_action]
        next_state1 = self._state_from_action(p2_action)
        next_state2 = self._state_from_action(p1_action)
        return (next_state1, reward1), (next_state2, reward2)

    def _state_from_action(self, action):
        # For two actions, represent state with a one-hot encoded vector of length 2.
        state = np.zeros(2)
        state[action] = 1
        return state