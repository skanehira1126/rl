from logging import getLogger

import numpy as np


logger = getLogger(__name__)


class QtableAgent:
    def __init__(
        self,
        n_state: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_decay: float = 1,
        random_state: int | None = None,
    ):
        """
        Parameters
        ----------
        n_state: int
            状態の数
        n_actions: int
            取りうる行動の数
        alpha: float
            学習率
        gamma: float
            割引率
        epsilon: float
            epsilon-greedyのパラメータ
        epsilon_decay: float
            epsilonの減衰率
        """
        self.n_state = n_state
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.rand_gen = np.random.default_rng(random_state)

        # Q-テーブル
        self.q_table = np.zeros((n_state, n_actions))

    def select_action(self, state_idx: int):
        """
        Parameters
        ----------
        state_idx: int
            状態の番号
        """
        if self.rand_gen.random() < self.epsilon:
            # ランダムな行動を選択（探索）
            action = self.rand_gen.integers(self.n_actions)
        else:
            # Qテーブルに基づく最適行動を選択（活用）
            action = self.select_best_action(state_idx)
        return action

    def select_best_action(self, state_idx: int):
        """
        一番良い期待値を選択する
        """
        return np.argmax(self.q_table[state_idx])

    def update_q_table(self, state, action, reward, next_state):
        """
        Qテーブルの更新

        遷移後の状態の一番良い行動の報酬を割引したものを加味したものをその行動の期待値とする

        Parameters
        ----------
        state: int
            状態のindex
        action: int
            行動のindex
        reward: float
            報酬
        next_state: int
            次の状態のindex
        """
        # 次の状態の最大期待値
        best_next_action = np.argmax(self.q_table[next_state])

        # 今回の報酬 + 次の状態の最大期待値の計算
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]

        # 今のテーブルとの差分
        td_error = td_target - self.q_table[state, action]
        # 差分を一定割引したものを追加
        self.q_table[state, action] += self.alpha * td_error

    def update_epsilon(self):
        logger.debug(f"Update epsilon from {self.epsilon} to {self.epsilon_decay * self.epsilon}")
        self.epsilon = self.epsilon * self.epsilon_decay

    def reset(self):
        """
        何かに使う？
        """
        pass
