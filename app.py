# -*- coding: utf-8 -*-
"""

Knapsack problem
Maximise the total value under the maximal weight constraint
Reinforcement learning

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import itertools
item = pd.DataFrame(data=[[1, 1],
                          [6, 2],
                          [18, 5],
                          [22, 6],
                          [28, 7]],
                    columns=['Value', 'Weight'])

actions = list(range(len(item)))
limit_W = 11
gamma = 0.9
# 
'''
**************************************************************
1. Q learning
'''


class RLforKnapsack():
    def __init__(self, limit_W, actions):
        self.limit_W = limit_W  # maximal weight
        self.epsilon = 0.9  # e-greedy algorithm
        self.gamma = 0.9  # reward decay
        self.alpha = 0.8  # learning_rate
        self.actions = actions
        self.q_table = pd.DataFrame(columns=actions)
        self.done = False

    def check_state(self, knapsack):
        if str(knapsack) not in self.q_table.index:
            # append new state to q table
            q_table_new = pd.Series([np.NAN]*len(self.actions),
                                    index=self.q_table.columns,
                                    name=str(knapsack))
            # 0-1 knapsack
            for i in list(set(self.actions).difference(set(knapsack))):
                q_table_new[i] = 0
            self.q_table = self.q_table.append(q_table_new)

    def choose_action(self, knapsack):
        self.check_state(knapsack)
        state_action = self.q_table.loc[str(knapsack), :]
        # random state_action in case there are two or more maximum
        state_action = state_action.reindex(
                np.random.permutation(state_action.index)
                )
        if np.random.uniform() < self.epsilon:
            # choose best action
            action = state_action.idxmax()  # the first maximun
        else:
            # choose random action
            action = np.random.choice(
                    list(set(self.actions).difference(set(knapsack)))
                    )
        return action

    def greedy_action(self, knapsack):
        # testing
        # choose best action
        state_action = self.q_table.loc[str(knapsack), :]
        state_action = state_action.reindex(
                np.random.permutation(state_action.index)
                )
        action = state_action.idxmax()
        return action

    def take_action(self, knapsack, action):
        # take the item
        knapsack_ = knapsack + [action]
        knapsack_.sort()
        self.check_state(knapsack_)
        return knapsack_

    def rewardWithPenalty(self, knapsack_, action):
        # constraint
        knapsack_W = np.sum([item['Weight'][i] for i in knapsack_])
        if knapsack_W > self.limit_W:
            r = -10
            self.done = True
        else:
            r = item['Value'][action]
        return r

    def update_qvalue(self, knapsack, knapsack_, action):
        self.done = False
        reward = self.rewardWithPenalty(knapsack_, action)
        q_predict = self.q_table.loc[str(knapsack), action]
        if len(knapsack) != len(self.actions):
            q_target = reward + self.gamma * self.q_table.loc[
                    str(knapsack_), :].max()
        else:
            q_target = reward  # no item can be added
        self.q_table.loc[str(knapsack), action] += self.alpha * (
                q_target - q_predict)
        print("rl----")
        print(self.q_table)
        print(self.q_table.count())
        print("--------")
        return self.q_table, self.done


t1 = time()
plt.close('all')
RL = RLforKnapsack(limit_W=11, actions=actions)
for episode in range(100):
    print("episode--")
    print(episode)
    knapsack = []
    for step in range(5):
        print("step--")
        print(step)
        action = RL.choose_action(knapsack)
        print("action---")
        print(action)
        knapsack_ = RL.take_action(knapsack, action)
        q_table_RL, done = RL.update_qvalue(knapsack, knapsack_, action)
        knapsack = knapsack_
        if done:
		
            break
    plt.scatter(episode, q_table_RL.iloc[0, 3], c='r')
    plt.scatter(episode, q_table_RL.iloc[0, 4], c='b')
t2 = time()
plt.title([t2-t1, 'RL'])
plt.show()

#  Policy based on q table
knapsack = []
# 
action = RL.greedy_action(knapsack)
knapsack_ = RL.take_action(knapsack, action)
knapsack = knapsack_
np.sum([item['Weight'][i] for i in knapsack_])
print(np.sum([item['Weight'][i] for i in knapsack_]))
# # 
