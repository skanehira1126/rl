from collections import deque
from dataclasses import asdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Memory:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

    def astuple(self):
        return self.state, self.action, self.reward, self.next_state, self.done

    def asdict(self):
        return asdict(self)


class ReplayMemory:
    def __init__(self, maxlen: int, random_state: int | None = None):
        """
        Parameters
        ----------
        maxlen: int
            Memoryのサイズ
        """
        self.buffer = deque(maxlen=maxlen)
        self.rand_gen = np.random.default_rng(random_state)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        経験として記録する

        Parameters
        ----------
        state: np.ndarray
            現在の状態
        action: int
            実施した行動
        reward: float
            受け取った報酬
        next_state: np.ndarray
            行動後の状態
        done: bool
            エピソード終了フラグ
        """
        self.buffer.append(Memory(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Parameters
        ----------
        batch_size: int
            サンプリングして取得するサイズ
        """

        batch = self.rand_gen.choice(list(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*(x.astuple() for x in batch), strict=True)
        )
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
