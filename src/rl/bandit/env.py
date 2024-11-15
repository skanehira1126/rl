from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from logging import getLogger

import numpy as np


logger = getLogger(__name__)


@dataclass
class BetAction:
    idx: int
    bet_amount: int


class SlotLogic:
    """
    slotマシンシミュレーション
    """

    def __init__(
        self,
        n_action: int,
        probs: Sequence[float],
        auto_shuffle: bool = False,
    ):
        """
        Parameters
        ----------
        n_action: int
            取れる行動の数
        probs: float of sequence, optional
            各行動の当たる確率. Noneの場合はランダム
        auto_shuffle: bool, default False
            1行動ごとに確率をリセットする
        """
        self.n_action = n_action
        if len(probs) == n_action:
            self.probs = probs
        else:
            raise ValueError("probs size must be same n_action.")

        self.auto_shuffle = auto_shuffle

    @property
    def state(self):
        return self.probs

    def shuffle(self):
        self.probs = np.random.permutation(self.probs)

    def calculate_reward(self, action: BetAction):
        # 最初に指定した数以上のidxは何もしない
        # -> 少しの罰則
        if action.idx >= self.n_action:
            is_bet = False
            reward = -10
        else:
            is_bet = True
            reward = -action.bet_amount
            if self.is_winner(action):
                reward += action.bet_amount * 4

        # Change probabilies
        if self.auto_shuffle:
            self.shuffle()
        return reward, is_bet

    def is_winner(self, action: BetAction):
        return np.random.random() < self.probs[action.idx]


@dataclass
class State:
    """
    状態を管理するdataclass
    """

    pool: int
    n_trial_of_episode: int
    logic: SlotLogic
    current_trial: int = field(default=1, init=False)
    done: bool = field(default=False, init=False)

    def step(self, action: BetAction):
        """
        行動の結果を反映
        """
        logger.debug("Called step() of episode info")
        reward, is_bet = self.logic.calculate_reward(action)
        if is_bet:
            self.pool += reward

        # カウントを進める
        self.current_trial += 1

        # doneフラグの更新
        if self.current_trial == self.n_trial_of_episode:
            self.done = True
            reward += self.pool
        elif self.pool <= 0:
            self.done = True
            reward -= 500

        return reward

    @property
    def state(self):
        return np.concatenate(
            [self.logic.state, [self.pool / 100, self.n_trial_of_episode - self.current_trial]],
            dtype=np.float32,
        )


class SlotEnv:
    def __init__(self, pool: int, n_trial_of_eposode: int, logic: SlotLogic):
        # Episodeを管理するためのクラス
        # 環境初期化とともに利用
        self.initial_episode_info = State(
            pool,
            n_trial_of_eposode,
            logic,
        )
        self.episode_info = self.reset()

    @property
    def is_done(self):
        return self.episode_info.done

    def reset(self):
        """
        環境の初期化
        """
        logger.debug("Reset episode info")
        self.episode_info = replace(self.initial_episode_info)
        return self.episode_info

    def step(self, action: BetAction):
        """
        行動を選択して環境の状態を進める
        """
        # 報酬の計算
        reward = self.episode_info.step(action)

        # 次の状態と報酬、終了フラグを返す
        return self.episode_info.state, reward, self.is_done


if __name__ == "__main__":
    logic = SlotLogic(5, [0, 0.1, 0.15, 0.2, 0.3])
    env = SlotEnv(2000, 10, logic)

    while not env.is_done:
        action = BetAction(4, 100)
        state, reward, done = env.step(action)
        print(state, reward, done)
