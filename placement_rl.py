import pandas as pd
import numpy as np
from time import time
import itertools

item = pd.DataFrame(data=[[1, 1], [6, 2], [18, 5], [22, 6], [28, 7]], columns=['Value', 'Weight'])
actions = list(range(len(item)))  # actions 每一個物品是一個action


def check_state(Q, knapsack, actions):
    """
    檢查輸入的背包狀態是否在Q table中, 若無則新增
    將該column中可執行的action設定為0, 其餘為Nan

    :param Q: Q table
    :param knapsack: 背包
    :param actions: 所有可執行的action
    :return: Q table
    """

    if str(knapsack) not in Q.index:  # knapsack表示状态, 例如现在包里有[1,2]
        # append new state to q table
        q_table_new = pd.Series([np.NAN] * len(actions), index=Q.columns, name=str(knapsack))
        # 下面是将能使用的状态设置为0, 不能使用的设置为NaN (这个很重要)
        for i in list(set(actions).difference(set(knapsack))):
            q_table_new[i] = 0
        return Q.append(q_table_new)
    else:
        return Q


def envReward(action, knapsack):
    """
    執行action, 返回reward, 下一步的狀態及done
    done表示是否完成(超過背包限制或所有物品已放完)

    :param action: 要執行的action
    :param knapsack: 目前的背包狀態
    :return: r, knapsack_, done
    """

    limit_w = 11
    knapsack_ = knapsack + [action]
    knapsack_.sort()
    knapsack_w = np.sum([item['Weight'][i] for i in knapsack_])  # 計算目前背包內的物品重量總和
    if knapsack_w > limit_w:
        r = -10
        completed = True
    else:
        if len(knapsack_) == len(item):
            r = 100
            completed = True
            return r, knapsack_, completed
        r = item['Value'][action]
        completed = False
    return r, knapsack_, completed


def mu_policy(Q, epsilon, nA, observation, actions):
    """
    epsilon-greedy的策略, 返回每一個動作執行的機率

    :param Q: Q table
    :param epsilon: epsilon
    :param nA: 所有動作的數量
    :param observation: 目前背包的狀態
    :param actions: 所有可執行的action
    :return: 每一個動作執行的概率, 一維陣列
    """
    # 尚未執行的action
    actions_list = list(set(actions).difference(set(observation)))
    # 輸入的背包狀態中, 所有不同action獲得的累計獎勵
    action_values = Q.loc[str(observation), :]
    # 使用action_values中最大的值
    greedy_action = action_values.idxmax()
    # 設定所有動作執行概率為0
    probabilities = np.zeros(nA)
    # 設定可執行動作執行概率為(1 / len(actions_list)) * epsilon
    for i in actions_list:
        probabilities[i] = (1 / len(actions_list)) * epsilon
    # greedy_action執行概率設定為(1 / len(actions_list)) * epsilon + (1 - epsilon)
    probabilities[greedy_action] = probabilities[greedy_action] + (1 - epsilon)
    return probabilities


def pi_policy(Q, observation):
    """
    greedy策略, 每次選擇能獲得最大獎勵的動作

    :param Q: Q table
    :param observation: 目前背包的狀態
    :return: 一維陣列, 每個動作出現的概率, 最大獎勵的動作為1
    """
    action_values = Q.loc[str(observation), :]
    best_action = action_values.idxmax()
    return np.eye(len(action_values))[best_action]


def qLearning(actions, num_episodes, discount_factor=1.0, alpha=0.7, epsilon=0.2):
    """
    Q Learning訓練

    :param actions: 所有可執行的action
    :param num_episodes: 訓練的迭代次數
    :param discount_factor: 衰減係數
    :param alpha: learning rate
    :param epsilon: epsilon值, 用於epsilon-greedy選擇當前最大獎勵action
    """
    # 環境中所有物品數量
    nA = len(actions)

    # 初始化Q table
    Q = pd.DataFrame(columns=actions)

    for i_episode in range(1, num_episodes + 1):
        # 開始一輪迭代
        # 開始時背包是空的
        knapsack = []

        # 新增Q table column
        Q = check_state(Q, knapsack, actions)

        # 從實際執行的policy中選擇action
        action = np.random.choice(nA, p=mu_policy(Q, epsilon, nA, knapsack, actions))
        for t in itertools.count():
            # 執行action, 返回reward, 下一步的狀態及是否完成(超過背包限制或所有物品已放完)
            reward, next_knapsack, done = envReward(action, knapsack)
            if done:
                Q.loc[str(knapsack), action] = reward
                break
            if t > 10:
                break

            # 更新Q table 下一步狀態的column
            Q = check_state(Q, next_knapsack, actions)
            # 更新Q table的value
            Q.loc[str(knapsack), action] = Q.loc[str(knapsack), action] + alpha * (
                        reward + discount_factor * Q.loc[str(next_knapsack), :].max() - Q.loc[str(knapsack), action])

            knapsack = next_knapsack
            # 選擇下一個action
            next_action = np.random.choice(nA, p=mu_policy(Q, epsilon, nA, next_knapsack, actions))
            action = next_action

        if i_episode % 50 == 0:
            print("\rEpisode {}/{}. | ".format(i_episode, num_episodes), end="")

    return Q


if __name__ == '__main__':
    # 訓練
    train_start_time = time()
    Q = qLearning(actions, num_episodes=1000, discount_factor=0.9, alpha=0.3, epsilon=0.1)
    train_finish_time = time()
    print(train_finish_time - train_start_time)
    print(Q)

    # 查看最终结果
    actionsList = []
    knapsack = []
    nA = len(actions)
    # 從實際執行的policy中選擇action
    action = np.random.choice(nA, p=pi_policy(Q, knapsack))
    t1 = time()
    for t in itertools.count():
        actionsList.append(action)
        # 執行action, 返回reward, 下一步的狀態及是否完成(超過背包限制或所有物品已放完)
        reward, next_knapsack, done = envReward(action, knapsack)
        if done:
            actionsList.pop()
            count = len(next_knapsack)
            if count >= 5:
                knapsack = next_knapsack
                break
            break
        else:
            # 選擇下一步動作
            next_action = np.random.choice(nA, p=pi_policy(Q, next_knapsack))
            action = next_action
            knapsack = next_knapsack
    t2 = time()
    print(t2 - t1)
    print(knapsack)

