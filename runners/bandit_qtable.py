import argparse
from logging import getLogger

from tqdm.auto import tqdm

from rl.bandit.env import SlotMachineEnv
from rl.bandit.qtable_agent import QtableAgent


logger = getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_step", type=int)
    parser.add_argument("n_actions", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--epsilon-decay", type=float, default=1)
    parser.add_argument("--random-state", type=int)

    return parser


def main(
    n_step: int,
    n_actions: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    random_state: int | None = None,
):
    # 初期設定
    # stateは1つ
    agent = QtableAgent(
        n_state=1,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        random_state=random_state,
    )
    state = 0
    env = SlotMachineEnv(n_actions, random_state=random_state)

    # 進捗管理の変数
    total_reward = 0

    # 実行
    for i in tqdm(range(1, n_step + 1)):
        act_idx = agent.select_action(state)
        reward, done, info = env.step(act_idx)

        agent.update_q_table(state=state, action=act_idx, next_state=0, reward=reward)
        agent.update_epsilon()
        total_reward += reward

        # print(f"Step {i} / {n_step} Total reward: {total_reward} reward: {reward}")

    print("========== Result")
    print("Q Table")
    print(agent.q_table)

    print("Probabilities of slots")
    print(env.probs)

    print("=========== Trial 100 times")
    total_reward = 0
    for _ in tqdm(range(100)):
        act_index = agent.select_best_action(state)
        reward, done, info = env.step(act_index)

        total_reward += reward
    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()

    main(**vars(args))
