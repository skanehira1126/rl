import argparse
from logging import getLogger

from rl.bandit.env import SlotMachineEnv
from rl.bandit.random_agent import RandomAgent


logger = getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_step", type=int)
    parser.add_argument("n_actions", type=int)
    parser.add_argument("--random-state", type=int)

    return parser


def main(n_step: int, n_actions: int, random_state: int | None = None):
    # 初期設定
    agent = RandomAgent(n_actions, random_state=random_state)
    env = SlotMachineEnv(n_actions, random_state=random_state)

    # 進捗管理の変数
    total_reward = 0

    # 実行
    for i in range(1, n_step + 1):
        act_idx = agent.select_action()
        reward, done, info = env.step(act_idx)
        total_reward += reward

        print(f"Step {i} / {n_step} Total reward: {total_reward} reward: {reward}")


if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()

    main(args.n_step, args.n_actions)
