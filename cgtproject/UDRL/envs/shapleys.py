import numpy as np
import random

class ShapleySimEnv:
    """
    Shapley's Game:
      - Two players, each with three actions (0, 1, 2).
      - The payoff matrices are defined so that, for example, Player 1's matrix is:
              Action 0     Action 1     Action 2
          0   (0,0)       (1,0)       (0,1)
          1   (0,1)       (0,0)       (1,0)
          2   (1,0)       (0,1)       (0,0)
      - Here, the first number in each pair is the payoff for Player 1 and the second is for Player 2.
      - The state for each player is the one-hot encoding (length 3) of the opponent's last action.
    """
    def __init__(self):
        # Define the payoff matrices based on the symmetric formulation of Shapley's Game.
        self.payoff_matrix1 = np.array([[0, 1, 0],
                                        [0, 0, 1],
                                        [1, 0, 0]])
        self.payoff_matrix2 = np.array([[0, 0, 1],
                                        [1, 0, 0],
                                        [0, 1, 0]])

    def reset(self):
        p1_action = random.choice([0, 1, 2])
        p2_action = random.choice([0, 1, 2])
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
        # With three actions, the state is a one-hot vector of length 3.
        state = np.zeros(3)
        state[action] = 1
        return state