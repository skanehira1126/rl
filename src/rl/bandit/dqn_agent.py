import copy
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


logger = getLogger(__name__)


class DQN(nn.Module):
    def __init__(self, n_state, n_actions):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(n_state, 2 * n_state),
            nn.ReLU(),
            nn.Linear(2 * n_state, 2 * n_state),
            nn.ReLU(),
            nn.Linear(2 * n_state, n_actions),
        )

    def forward(self, x):
        return self.layer(x)


class DQNAgent:
    """
    いくつかのslotのセットがあり、ランダムに状態として与えられるとする.
    取りうる行動の数は、セット数 * slot数だけど、それぞれのセットをenvとして考える
    """

    def __init__(
        self,
        n_state: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        device: torch.device,
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
        device: torch.device
            利用device
        epsilon: float
            epsilon-greedyのパラメータ
        epsilon_decay: float
            epsilonの減衰率
        """
        self.n_state = n_state
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.rand_gen = np.random.default_rng(random_state)

        self.brain = DQN(n_state, n_actions).to(self.device)
        # optimizerの設定（Adamを使用）
        self.optimizer = optim.Adam(self.brain.parameters(), lr=alpha)

    def select_action(self, state: np.ndarray):
        """
        アクションを選ぶ
        """
        if self.rand_gen.random() < self.epsilon:
            action = self.rand_gen.integers(self.n_actions)
        else:
            action = self.select_best_action(state)

        return action

    def select_best_action(self, state: np.ndarray):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = int(self.brain(state_tensor).argmax().item())
        return action

    def update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> float:
        """
        1ステップ分の経験でネットワークを更新する

        Parameters
        ----------
        state: numpy.ndarray
            現在の状態
        action: int
            実施した行動のインデックス
        reward: float
            受け取った報酬
        next_state: numpy.ndarray
            次の状態（shape: (n_state, )）
        done: bool
            エピソード終了フラグ

        Returns
        -------
        loss: float
            この学習での損失.主に記録ようかな？
        """
        # 各要素をtensorに変換
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = (
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        reward_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(self.device)
        done_tensor = torch.tensor([int(done)], dtype=torch.float32).unsqueeze(0).to(self.device)

        # Calculation q-value
        q_values = self.brain(state_tensor)
        current_q = q_values[0, action].unsqueeze(0).unsqueeze(0)

        # 次の状態のQ値の計算
        # これは次の状態の想定期待値なので、勾配計算は必要ない
        with torch.no_grad():
            next_q_values = self.brain(next_state_tensor)
            max_next_q = torch.max(next_q_values)

            # episode終了時は次のstateの報酬は加味しない
            target_q = reward_tensor + self.gamma * max_next_q * (1 - done_tensor)

        # lossの計算
        loss = F.mse_loss(current_q, target_q)

        # optimizerでパラメータを更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_epsilon(self):
        """
        epsilonの更新
        """
        self.epsilon *= self.epsilon_decay

    def reset(self):
        """
        何かに使う？
        """
        pass


class DQNAgentWithTargetNetwork:
    """
    TargetNetworkがあるDQN
    """

    def __init__(
        self,
        n_state: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        device: torch.device,
        n_step_update_target_model: int,
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
        device: torch.device
            利用device
        n_step_update_target_model: int
            ターゲットネットワークを更新するために必要なstep数
        epsilon: float
            epsilon-greedyのパラメータ
        epsilon_decay: float
            epsilonの減衰率
        """
        self.n_state = n_state
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.rand_gen = np.random.default_rng(random_state)

        self.brain = DQN(n_state, n_actions).to(self.device)
        self.target = copy.deepcopy(self.brain)
        self.n_step_update_target_model = n_step_update_target_model
        self.cnt_update_target_model = 0
        # optimizerの設定（Adamを使用）
        self.optimizer = optim.Adam(self.brain.parameters(), lr=alpha)

    def select_action(self, state: np.ndarray):
        """
        アクションを選ぶ
        """
        if self.rand_gen.random() < self.epsilon:
            action = self.rand_gen.integers(self.n_actions)
        else:
            action = self.select_best_action(state)

        return action

    def select_best_action(self, state: np.ndarray):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = int(self.brain(state_tensor).argmax().item())
        return action

    def update(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> float:
        """
        1ステップ分の経験でネットワークを更新する

        Parameters
        ----------
        state: numpy.ndarray
            現在の状態
        action: int
            実施した行動のインデックス
        reward: float
            受け取った報酬
        next_state: numpy.ndarray
            次の状態（shape: (n_state, )）
        done: bool
            エピソード終了フラグ

        Returns
        -------
        loss: float
            この学習での損失.主に記録ようかな？
        """
        # 各要素をtensorに変換
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_tensor = (
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        reward_tensor = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(self.device)
        done_tensor = torch.tensor([int(done)], dtype=torch.float32).unsqueeze(0).to(self.device)

        # Calculation q-value
        q_values = self.brain(state_tensor)
        current_q = q_values[0, action].unsqueeze(0).unsqueeze(0)

        # 次の状態のQ値の計算
        # これは次の状態の想定期待値なので、勾配計算は必要ない
        with torch.no_grad():
            next_q_values = self.target(next_state_tensor)
            max_next_q = torch.max(next_q_values)

            # episode終了時は次のstateの報酬は加味しない
            target_q = reward_tensor + self.gamma * max_next_q * (1 - done_tensor)

        # lossの計算
        loss = F.mse_loss(current_q, target_q)

        # optimizerでパラメータを更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target modelの更新判定
        self.cnt_update_target_model += 1
        if self.cnt_update_target_model == self.n_step_update_target_model:
            logger.info("Update target model")
            self.target = copy.deepcopy(self.brain)
            self.cnt_update_target_model = 0

        return loss.item()

    def update_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        # それぞれのデータをtensorに変換
        state_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q値の計算
        q_values = self.brain(state_tensor)
        # 各サンプルにおける実行したアクションのQ値を取得する
        # actions は各サンプルで実行した行動のインデックスが入っている前提です
        current_q = q_values.gather(
            1, torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        ).squeeze(1)

        # 次状態のQ値の計算（ターゲットネットワークを使用）
        with torch.no_grad():
            next_q_values = self.target(next_state_tensor)
            max_next_q, _ = torch.max(next_q_values, dim=1)
            # done=True のサンプルについては、次状態の価値を0とする
            target_q = reward_tensor + self.gamma * max_next_q * (1 - done_tensor)

        # 損失の計算
        loss = F.mse_loss(current_q, target_q)

        # optimizerによるパラメータ更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ターゲットネットワークの更新
        self.cnt_update_target_model += 1
        if self.cnt_update_target_model >= self.n_step_update_target_model:
            logger.info("Update target model")
            self.target = copy.deepcopy(self.brain)
            self.cnt_update_target_model = 0

        return loss.item()

    def update_epsilon(self):
        """
        epsilonの更新
        """
        self.epsilon *= self.epsilon_decay

    def reset(self):
        """
        何かに使う？
        """
        pass
