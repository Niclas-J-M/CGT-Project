import numpy as np
import random

class RockPaperScissorsSimEnv:
    def __init__(self):
        # For player1:
        self.payoff_matrix1 = np.array([[0, -1, 1],
                                        [1, 0, -1],
                                        [-1, 1, 0]])
        # Zero-sum game: payoff for player2 is the negative of player1's.
        self.payoff_matrix2 = -self.payoff_matrix1

    def reset(self):
        # At the start, randomly choose an initial action for both players.
        p1_action = random.choice([0, 1, 2])
        p2_action = random.choice([0, 1, 2])
        # Each player's state is the one-hot encoding of the opponent's last action.
        state1 = self._state_from_action(p2_action)  # For player1: opponent's action.
        state2 = self._state_from_action(p1_action)  # For player2: opponent's action.
        return state1, state2

    def step(self, p1_action, p2_action):
        # Compute rewards.
        reward1 = self.payoff_matrix1[p1_action, p2_action]
        reward2 = self.payoff_matrix2[p1_action, p2_action]
        # Next states: each player observes the opponent's action.
        next_state1 = self._state_from_action(p2_action)
        next_state2 = self._state_from_action(p1_action)
        return (next_state1, reward1), (next_state2, reward2)

    def _state_from_action(self, action):
        state = np.zeros(3)
        state[action] = 1
        return state
