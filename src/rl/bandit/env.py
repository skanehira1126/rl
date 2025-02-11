from dataclasses import asdict
from dataclasses import dataclass
from logging import getLogger

import numpy as np


logger = getLogger(__name__)


@dataclass
class Info:
    played_count: int
    hit_count: int

    def asdict(self):
        return asdict(self)


class SlotMachineEnv:
    """
    Attributes
    ----------
    n_actions: int
        slotの数
    """

    def __init__(self, n_actions: int, random_state: int | None = None):
        self.n_actions = n_actions
        self.rnd_gen = np.random.default_rng(random_state)

        self.setup_slots()

    def setup_slots(self):
        # 0 ~ 0.5の確率のスロットを生成
        self.probs = self.rnd_gen.random((self.n_actions,)) * 0.5
        self.played_count = np.zeros_like(self.probs)
        self.hit_count = np.zeros_like(self.probs)

        logger.info(f"Setup environment: probs={self.probs.tolist()}")

    def step(self, act_idx: int):
        """
        slotを選択して回す

        Parameters
        ----------
        act_idx: int
            回すスロットの番号

        Returns
        -------
        reward: int
            報酬
        done: bool
            終了判定
        info: dict
            補助情報
        """
        logger.info(f"Choice {act_idx}.")
        if not (0 <= act_idx < self.n_actions):
            raise ValueError(f"{act_idx} is out of range (0 to {self.n_actions - 1}).")

        self.played_count[act_idx] += 1
        if self.rnd_gen.random() <= self.probs[act_idx]:
            is_hit = True
            self.hit_count[act_idx] += 1
        else:
            is_hit = False

        reward = 1 if is_hit else 0
        done = False  # ゲームによってはstep側で終了判定することもある
        info = Info(self.played_count[act_idx], self.hit_count[act_idx])

        return reward, done, info


class MultiSetSlotEnv:
    def __init__(self, n_actions: int, n_slot_sets: int, random_state: int | None = None):
        self.n_actions = n_actions
        self.n_slot_sets = n_slot_sets
        self.rnd_gen = np.random.default_rng(random_state)

        self.slots = [
            SlotMachineEnv(
                n_actions, random_state=random_state if random_state is None else random_state + i
            )
            for i in range(self.n_slot_sets)
        ]

    @property
    def probs(self):
        return [env.probs for env in self.slots]

    def choice_slot(self):
        return self.rnd_gen.integers(self.n_slot_sets)

    def step(self, state: int, act_idx: int):
        """
        slotを選択して回す

        Parameters
        ----------
        state: int
            対象のスロットグループ
        act_idx: int
            回すスロットの番号

        Returns
        -------
        reward: int
            報酬
        done: bool
            終了判定
        info: dict
            補助情報
        """
        slot = self.slots[state]

        logger.info(f"Choice {act_idx}.")
        if not (0 <= act_idx < slot.n_actions):
            raise ValueError(f"{act_idx} is out of range (0 to {slot.n_actions - 1}).")

        slot.played_count[act_idx] += 1
        if slot.rnd_gen.random() <= slot.probs[act_idx]:
            is_hit = True
            slot.hit_count[act_idx] += 1
        else:
            is_hit = False

        reward = 1 if is_hit else 0
        done = False  # ゲームによってはstep側で終了判定することもある
        info = Info(slot.played_count[act_idx], slot.hit_count[act_idx])

        return reward, done, info
