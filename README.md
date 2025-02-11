# rl
強化学習

## bandit問題
スロットマシンを選び、探索と獲得をバランスよく狙うゲーム.  
1. 探索を優先しすぎると最終的に獲得が下がる
2. 獲得を優先しすぎると実はより良い選択肢があったかもしれない

### 実行コマンド

```bash
# random
$ python3 runners/bandit_random.py 100 10

# Q Table
$ python3 runners/bandit_qtable.py 5000 10 \
    --alpha 0.2 --gamma 0.5 \
    --epsilon 0.9999 --epsilon-decay 0.99 \
    --random-state 20

# DQN 
$ python3 runners/bandit_dqn.py 15000 10 3 \
    --alpha 0.005 --gamma 0.1 \
    --epsilon 1 --epsilon-decay 0.9995 \
    --random-state 20 
```


