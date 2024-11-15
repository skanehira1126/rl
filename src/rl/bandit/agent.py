from collections import deque
import random

import numpy as np
from numpy.typing import ArrayLike
import torch
import torch.nn as nn
import torch.optim as optim


class Agent:
    """
    Attributes
    ----------
    state_size: int
        DQNモデルの入力サイズ
    action_size: int
        DQNモデルが取りうる行動の数
    gamma: float
        将来の利益の割引率
    epsilon: float
        epsilon greedyのパラメータ
    epsilon_min: float
        epsilon greedyのパラメータ
    epsilon_decay: float
        epsilon greedyのパラメータ
    learning_rate: float
        DQNモデルの学習率
    memory: deque
        Experiment Reply用のコンテナ
    q_network: nn.Module
        意思決定を行うモデル
    target_network: nn.Module
        期待値計算のためのモデル
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        learning_rate: float = 0.0005,
        device: str = "auto",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=2000)  # 経験リプレイ用バッファ

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = "device"

        self.q_network = self._build_model()  # Qネットワーク
        self.target_network = self._build_model()  # ターゲットネットワーク
        self.update_target_network()

    def _build_model(self):
        """
        モデル構築

        これは外でもいいかも
        """
        # シンプルな全結合ネットワークを構築
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
        )
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model.to(self.device)

    def update_target_network(self):
        """
        ターゲットネットワークのパラメータを同期する
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state: ArrayLike) -> int:
        # Epsilon-Greedy法で行動選択
        if np.random.rand() <= self.epsilon:
            act_index = random.randrange(self.action_size)
        else:
            act_index = self.predict_q_network(state)
        return act_index

    def predict_q_network(self, state: ArrayLike):
        state = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.q_network(state.to(self.device))
        return int(torch.argmax(act_values).item())

    def remember(self, state, action, reward, next_state, done):
        # 経験リプレイ用のバッファに追加
        self.memory.append((state.tolist(), action, reward, next_state.tolist(), done))

    def replay(self, batch_size):
        # バッチサンプルを経験リプレイから抽出し、Qネットワークを更新
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        # テンソル化
        states = torch.tensor([s[0] for s in minibatch]).to(self.device)
        actions = torch.tensor([s[1] for s in minibatch]).to(self.device)
        rewards = torch.tensor([s[2] for s in minibatch]).to(self.device)
        next_states = torch.tensor([s[3] for s in minibatch]).to(self.device)
        add_next_q_value = torch.tensor([1 - s[4] for s in minibatch]).to(self.device)

        # 現在のQ値の計算
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 次の状態の最大Q値の計算
        next_q_values = self.target_network(next_states).max(1)[0]

        # ターゲットの計算
        target_q_values = rewards + (self.gamma * next_q_values * add_next_q_value)
        # 損失を計算し、最適化
        loss = nn.functional.huber_loss(current_q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        # Epsilonの減少
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
