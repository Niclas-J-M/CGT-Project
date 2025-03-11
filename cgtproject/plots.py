import matplotlib.pyplot as plt
import numpy as np

class RPSPlotting:
    def __init__(self, history_agent1, history_agent2):
        """
        Initializes the plotting class with policy histories of both agents.
        
        :param history_agent1: Numpy array containing the evolution of agent 1's policy.
        :param history_agent2: Numpy array containing the evolution of agent 2's policy.
        """
        self.history_agent1 = np.array(history_agent1)
        self.history_agent2 = np.array(history_agent2)

    def plot_policy_evolution(self, agent=1):
        """
        Plots the evolution of an agent's policy over iterations.
        
        :param agent: Specify 1 for agent 1, or 2 for agent 2.
        """
        history = self.history_agent1 if agent == 1 else self.history_agent2
        plt.figure(figsize=(10, 6))
        plt.plot(history[:, 0], label='Rock')
        plt.plot(history[:, 1], label='Paper')
        plt.plot(history[:, 2], label='Scissors')
        plt.xlabel('Iteration')
        plt.ylabel('Probability')
        plt.title(f'Evolution of Agent {agent} Policy (Q-learning with Smooth Best Response)')
        plt.legend()
        plt.show()

    def scatter_strategy_evolution(self, omit_first=0):
        """
        Plots a scatter plot of π^1(S) (y-axis) vs π^1(R) (x-axis).
        
        :param omit_first: Number of initial iterations to omit from the plot.
        """
        history = self.history_agent1[omit_first:]
        plt.figure(figsize=(10, 6))
        plt.scatter(history[:, 0], history[:, 2], alpha=0.3, s=1)
        plt.xlabel(r'$\pi^1(R)$')
        plt.ylabel(r'$\pi^1(S)$')
        plt.title(f'Strategy Evolution of Player 1 (Omitting First {omit_first} Iterations)')
        plt.show()

    def plot_sampled_evolution(self, step=5000):
        """
        Plots the sampled evolution of Player 1's strategy at fixed intervals.
        
        :param step: Interval at which to sample iterations for plotting.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history_agent1[::step, 0], self.history_agent1[::step, 2], marker='o', linestyle='-', markersize=0)
        plt.xlabel(r'$\pi^1(R)$')
        plt.ylabel(r'$\pi^1(S)$')
        plt.title(f'Strategy Evolution of Player 1 (Every {step} Iterations)')
        plt.show()
