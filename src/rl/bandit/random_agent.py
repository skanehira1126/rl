from logging import getLogger

import numpy as np


logger = getLogger(__name__)


class RandomAgent:
    def __init__(self, n_actions: int, random_state: int | None = None):
        self.n_actions = n_actions
        self.rand_gen = np.random.default_rng(random_state)

    def select_action(self):
        act_idx = self.rand_gen.integers(self.n_actions)
        logger.debug(f"Selected action: {act_idx}")
        return act_idx

    def reset(self):
        """
        何かに使う？
        """
        pass
