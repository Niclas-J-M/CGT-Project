# Multi-Agent Game Learning: UDRL vs. Q-learning

This project compares two reinforcement learning approaches—UDRL (Uncertain Deep Reinforcement Learning) and Q-learning—when applied to multi-agent games such as Shapley's Game, Matching Pennies, Stag Hunt, New Environment, and Rock Paper Scissors. The experiments focus on tracking the evolution of the agents’ policies over time and illustrate the intrinsic cyclic dynamics present in some competitive games. 

## Overview

- **UDRL Approach:**  
  Uses deep neural network–based policy networks. The code is implemented under the `UDRL` folder and includes the following:
  - `udrl.py` – Defines the UDRL policy network, training routines, and inference procedure.
  - `udrl_agent.py` – Contains the UDRL agent implementation that utilizes the policy network to collect trajectories, update the policy, and perform action inference.
  - `utils.py` – Provides helper functions for generating training data and collecting trajectories.

- **Q-learning Approach:**  
  A tabular variant of Q-learning with Boltzmann (softmax) action selection. The code is implemented under the `QLearning` folder:
  - `qlearning.py` – Contains the Q-learning agent class with methods for action selection and Q-value updates.

- **Experiment and Plotting Code:**  
  - `main.py` – The main script for running experiments. Depending on the configuration, it runs either UDRL self-play experiments or Q-learning experiments.
  - `plots.py` – Provides functions to plot the evolution of agents’ strategies and time-averaged policy distributions.


## Requirements

- Python 3.x
- NumPy
- PyTorch
- Matplotlib

Install the dependencies (if not already installed) via pip:

```bash
pip install numpy torch matplotlib
