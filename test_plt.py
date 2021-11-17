import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import itertools

item = pd.DataFrame(data=[[1, 1], [6, 2], [18, 5], [22, 6], [28, 7]], columns=['Value', 'Weight'])
actions = list(range(len(item)))  # 动作, 每一个物体是一个动作
limit_W = 11  # 背包重量的限制
gamma = 0.9


def envReward(action, knapsack):
    """返回下一步的state, reward和done
    """
    limit_W = 11
    knapsack_ = knapsack + [action]  # 得到新的背包里的东西, 现在是[2,3], 向里面增加物品[1], 得到新的状态[1,2,3]
    knapsack_.sort()
    knapsack_W = np.sum([item['Weight'][i] for i in knapsack_])  # 计算当前包内物品的总和
    if knapsack_W > limit_W:
        r = -10
        done = True
    else:
        r = item['Value'][action]
        done = False
    return r, knapsack_, done


def mu_policy(Q, epsilon, nA, observation, actions):
    """
    这是一个epsilon-greedy的策略, 返回每一个动作执行的概率, nA为动作的个数
    其中:
    - Q是q table, 为dataframe的格式;
    - nA是所有动作的个数
    """
    actionsList = list(set(actions).difference(set(observation)))  # 可以挑选的动作
    # 看到这个state之后, 采取不同action获得的累计奖励
    action_values = Q.loc[str(observation), :]
    # 使用获得奖励最大的那个动作
    greedy_action = action_values.idxmax()
    # 是的每个动作都有出现的可能性
    probabilities = np.zeros(nA)
    for i in actionsList:
        probabilities[i] = 1 / len(actionsList) * epsilon
    probabilities[greedy_action] = probabilities[greedy_action] + (1 - epsilon)
    return probabilities


def pi_policy(Q, observation):
    """
    这是greedy policy, 每次选择最优的动作.
    其中:
    - Q是q table, 为dataframe的格式;
    """
    action_values = Q.loc[str(observation), :]
    best_action = action_values.idxmax()  # 选择最优的动作
    return np.eye(len(action_values))[best_action]  # 返回的是两个动作出现的概率


def check_state(Q, knapsack, actions):
    """检查状态knapsack是否在q-table中, 没有则进行添加
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


def qLearning(actions, num_episodes, discount_factor=1.0, alpha=0.7, epsilon=0.2):
    # 环境中所有动作的数量
    nA = len(actions)
    # 初始化Q表
    Q = pd.DataFrame(columns=actions)
    # 记录reward和总长度的变化
    stats = {'episode_lengths':np.zeros(num_episodes + 1),
             'episode_rewards':np.zeros(num_episodes + 1),
             'final': []}
    for i_episode in range(1, num_episodes + 1):
        # 开始一轮游戏
        knapsack = []  # 开始的时候背包是空的
        Q = check_state(Q, knapsack, actions)
        action = np.random.choice(nA, p=mu_policy(Q, epsilon, nA, knapsack, actions))  # 从实际执行的policy, 选择action
        for t in itertools.count():
            reward, next_knapsack, done = envReward(action, knapsack)  # 执行action, 返回reward和下一步的状态
            Q = check_state(Q, next_knapsack, actions)
            next_action = np.random.choice(nA, p=mu_policy(Q, epsilon, nA, next_knapsack, actions))  # 选择下一步的动作
            # 更新Q
            Q.loc[str(knapsack), action] = Q.loc[str(knapsack), action] + alpha * (
                        reward + discount_factor * Q.loc[str(next_knapsack), :].max() - Q.loc[str(knapsack), action])
            # 计算统计数据(带有探索的策略)
            stats['episode_rewards'][i_episode] += reward  # 计算累计奖励
            stats['episode_lengths'][i_episode] = t  # 查看每一轮的时间
            stats['final'].append(reward)
            if done:
                break
            if t > 10:
                break
            knapsack = next_knapsack
            action = next_action
        if i_episode % 50 == 0:
            # 打印
            print("\rEpisode {}/{}. | ".format(i_episode, num_episodes), end="")
    return Q, stats


Q, stats = qLearning(actions, num_episodes=1000, discount_factor=0.9, alpha=0.3, epsilon=0.1)
# 查看最终结果
actionsList = []
knapsack = [] # 开始的时候背包是空的
nA = len(actions)
action = np.random.choice(nA, p=pi_policy(Q, knapsack)) # 从实际执行的policy, 选择action
for t in itertools.count():
    actionsList.append(action)
    reward, next_knapsack, done = envReward(action, knapsack) # 执行action, 返回reward和下一步的状态
    next_action = np.random.choice(nA, p=pi_policy(Q, next_knapsack)) # 选择下一步的动作
    if done:
        actionsList.pop()
        break
    else:
        action = next_action
        knapsack = next_knapsack
# plt.plot(stats['episode_rewards'])
reward = stats['final']
a = list()
for index, value in enumerate(reward):
    if index % 5 == 0:
        a.append(value)
plt.plot(reward)
plt.show()
