import argparse
from logging import getLogger

import numpy as np
import torch
from tqdm.auto import tqdm

from rl.bandit.dqn_agent import DQNAgentWithTargetNetwork
from rl.bandit.env import MultiSetSlotEnv
from rl.bandit.history import DQNHistory
from rl.memory.memory import ReplayMemory


logger = getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_step", type=int)
    parser.add_argument("n_actions", type=int)
    parser.add_argument("n_states", type=int)
    parser.add_argument("n_step_update_model", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--epsilon-decay", type=float, default=1)
    parser.add_argument("--random-state", type=int)

    return parser


def main(
    n_step: int,
    n_actions: int,
    n_states: int,
    n_step_update_model: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    random_state: int | None = None,
):
    # 初期設定
    # stateは1つ
    agent = DQNAgentWithTargetNetwork(
        n_state=n_states,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        device=torch.device("mps") if torch.mps.is_available() else torch.device("cpu"),
        n_step_update_target_model=n_step_update_model,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        random_state=random_state,
    )

    env = MultiSetSlotEnv(n_actions, n_states, random_state=random_state)
    history = DQNHistory()
    memory = ReplayMemory(1000)

    # 進捗管理の変数
    total_reward = 0

    # 実行
    for i in tqdm(range(1, n_step + 1)):
        current_state_idx = env.choice_slot()
        next_state_idx = env.choice_slot()

        current_state = np.array([1 if i == current_state_idx else 0 for i in range(n_states)])
        next_state = np.array([1 if i == next_state_idx else 0 for i in range(n_states)])

        act_idx = agent.select_action(current_state)
        reward, done, _ = env.step(current_state_idx, act_idx)

        memory.push(
            state=current_state,
            action=act_idx,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        if len(memory) >= 24:
            states, actions, rewards, next_states, dones = memory.sample(24)
            agent.update_batch(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
            )
            agent.update_epsilon()
        total_reward += reward
        history.log(epsilon=agent.epsilon, reward=total_reward)

        # print(f"Step {i} / {n_step} Total reward: {total_reward} reward: {reward}")

    print("========== Result")
    print("Probabilities of slots")
    print(env.probs)

    print("=========== Trial 100 times")
    total_reward = 0
    for _ in tqdm(range(100)):
        current_state_idx = env.choice_slot()
        current_state = np.array([1 if i == current_state_idx else 0 for i in range(n_states)])

        act_index = agent.select_best_action(current_state)
        reward, done, _ = env.step(current_state_idx, act_index)

        total_reward += reward
    print(f"Total reward: {total_reward}")

    history.plot_history()


if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()

    main(**vars(args))
