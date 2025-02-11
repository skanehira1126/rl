from dataclasses import dataclass
from dataclasses import field

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class QtableHistory:
    epsilons: list[float] = field(init=False, default_factory=list)
    qtables: list[list] = field(init=False, default_factory=list)
    rewards: list[int | float] = field(init=False, default_factory=list)

    def log(self, epsilon: float, qtable: list[float], reward: int | float):
        """
        可視化のためのログを記録する
        """
        self.epsilons.append(epsilon)
        self.qtables.append(qtable.tolist()[0])
        self.rewards.append(reward)

    def plot_qtable_history(self):
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))
        axs = axs.flatten()

        self.plot_epsilon(axs[0])
        self.plot_rewards(axs[1])
        self.plot_qtables(axs[2])

        fig.tight_layout()
        plt.show()

    def plot_epsilon(self, ax=None):
        """
        epsilonを可視化する
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(range(1, len(self.epsilons) + 1), self.epsilons, marker=".")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_title("History of epsilon")
        ax.grid()

    def plot_rewards(self, ax=None):
        """
        各ステップの報酬を可視化する
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(range(1, len(self.rewards) + 1), self.rewards, marker=".")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rewards")
        ax.set_title("History of rewards")
        ax.grid()

    def plot_qtables(self, ax=None):
        """
        各ステップの報酬を可視化する
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 6))

        qtables = np.array(self.qtables)
        n_action = qtables.shape[1]
        for i in range(n_action):
            ax.plot(qtables[:, i].ravel(), label=f"Action: {i+1}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Action")
        ax.set_title("History of qtables")
        ax.grid()
        ax.legend()


@dataclass
class DQNHistory:
    epsilons: list[float] = field(init=False, default_factory=list)
    rewards: list[int | float] = field(init=False, default_factory=list)

    def log(self, epsilon: float, reward: int | float):
        """
        可視化のためのログを記録する
        """
        self.epsilons.append(epsilon)
        self.rewards.append(reward)

    def plot_history(self):
        fig, axs = plt.subplots(2, 1, figsize=(8, 12))
        axs = axs.flatten()

        self.plot_epsilon(axs[0])
        self.plot_rewards(axs[1])

        fig.tight_layout()
        plt.show()

    def plot_epsilon(self, ax=None):
        """
        epsilonを可視化する
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(range(1, len(self.epsilons) + 1), self.epsilons, marker=".")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Epsilon")
        ax.set_title("History of epsilon")
        ax.grid()

    def plot_rewards(self, ax=None):
        """
        各ステップの報酬を可視化する
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(range(1, len(self.rewards) + 1), self.rewards, marker=".")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rewards")
        ax.set_title("History of rewards")
        ax.grid()
