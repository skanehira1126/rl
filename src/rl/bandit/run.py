from logging import getLogger

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from rl.bandit.agent import Agent
from rl.bandit.env import BetAction
from rl.bandit.env import SlotEnv
from rl.bandit.env import SlotLogic


logger = getLogger(__name__)

if __name__ == "__main__":
    # パラメータ
    state_size = 10 + 2  # pool, 残り挑戦回数
    action_size = 10
    initial_pool = 4000
    n_trial_of_episode = 80

    logic = SlotLogic(
        action_size,
        probs=[0, 0.025, 0.05, 0.075, 0.1, 0.015, 0.15, 0.25, 0.25, 0.8],
        auto_shuffle=True,
    )
    env = SlotEnv(initial_pool, n_trial_of_eposode=n_trial_of_episode, logic=logic)
    agent = Agent(state_size, action_size, gamma=0.5)  # 何もしない選択を追加

    num_episodes = 4000
    batch_size = 512

    # Agentの学習
    logger.info("Start training")
    pbar = tqdm(range(num_episodes))
    actions = {i: 0 for i in range(action_size + 1)}
    for _ in pbar:
        # 最初に環境をreset
        env.reset()

        while not env.is_done:
            # 行動を選択
            idx_action = agent.act(env.episode_info.state)
            action = BetAction(idx_action, 100)
            actions[idx_action] += 1

            # 環境のステップを実行
            next_state, reward, done = env.step(action)

            # メモリに経験を保存
            agent.remember(env.episode_info.state, idx_action, reward, next_state, done)

            # 経験リプレイから学習
            agent.replay(batch_size)

        pbar.set_postfix({"score": env.episode_info.pool, "epsilon": agent.epsilon})
        agent.decay_epsilon()

        # エピソード終了時にターゲットネットワークを更新
        agent.update_target_network()

    # シミュレーション
    bet_counts = []
    pools = []
    trial_idx = []
    for idx in range(30):
        env = SlotEnv(initial_pool, n_trial_of_episode, logic)

        bet_counts.append(0)
        pools.append(initial_pool)
        trial_idx.append(idx)

        bet_count = 1
        while not env.is_done:
            # 行動を選択
            idx_action = agent.predict_q_network(env.episode_info.state)
            action = BetAction(idx_action, 100)

            # 環境のステップを実行
            next_state, reward, done = env.step(action)

            bet_counts.append(bet_count)
            pools.append(env.episode_info.pool)
            trial_idx.append(idx)

            bet_count += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.lineplot(
        x=bet_counts,
        y=pools,
        # hue=trial_idx,
        ax=ax1,
    )
    ax1.grid()
    sns.lineplot(
        x=bet_counts,
        y=pools,
        hue=trial_idx,
        ax=ax2,
    )
    ax2.grid()
    plt.show()
