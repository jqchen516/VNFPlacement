import pandas as pd
import numpy as np
from time import time
import itertools
import copy


class VNFPlacement():

    def __init__(self, item, number_of_node, limit_W):
        # 物品列表 dataframe
        self.item = item
        self.number_of_node = number_of_node
        # actions, 每一物品是一個動作
        self.actions = list()
        for place in range(self.number_of_node):
            self.actions = self.actions + [(place, item) for item in list(range(len(self.item)))]
        # 初始化Q table
        self.q_table = pd.DataFrame(columns=self.actions)
        self.limit_W = limit_W

    def check_state(self, q_table, placement, actions):
        """
        檢查輸入的背包狀態是否在Q table中, 若無則新增
        將該column中可執行的action設定為0, 其餘為Nan

        :param q_table: Q table
        :param placement: 背包
        :param actions: 所有可執行的action
        :return: Q table
        """

        if str(placement) not in q_table.index:
            q_table_new_row = pd.Series([np.NAN] * len(actions),
                                        index=q_table.columns,
                                        name=str(placement))
            # 可執行的action設為0, 其他為Nan
            already_placement = []
            for value in placement:
                for i in value:
                    already_placement = already_placement + [(x, i) for x in range(self.number_of_node)]
            for i in list(set(actions).difference(set(already_placement))):
                q_table_new_row[i] = 0
            return q_table.append(q_table_new_row)
        else:
            return q_table

    def env_reward(self, action, placement):
        """
        執行action, 返回reward, 下一步的狀態及done
        done表示是否完成(超過背包限制或所有物品已放完)

        :param action: 要執行的action
        :param placement: 目前的背包狀態
        :return: r, knapsack_, done
        """

        # 將輸入的action轉為(node, vnf)的格式
        node = action // len(self.item)
        vnf = action % len(self.item)
        placement_ = copy.deepcopy(placement)
        placement_[node] += [vnf]
        placement_[node].sort()
        count = 0
        for backpack in placement_:
            count += len(backpack)
            placement_weight = np.sum([self.item['Weight'][i] for i in backpack])
            # 超過背包限制
            if placement_weight > self.limit_W:
                r = -99
                done = True
                return r, placement_, done
        # 所有物品已放完
        if count >= len(self.item):
            r = 100
            done = True
            return r, placement_, done

        r = self.item['Value'][vnf]
        done = False
        return r, placement_, done

    def mu_policy(self, q_table, epsilon, nA, observation, actions):
        """
        epsilon-greedy的策略, 返回每一個動作執行的機率

        :param q_table: Q table
        :param epsilon: epsilon
        :param nA: 所有動作的數量
        :param observation: 目前背包的狀態
        :param actions: 所有可執行的action
        :return: 每一個動作執行的概率, 一維陣列
        """

        # 將目前背包狀態轉為已放置的action後跟所有action的列表比較, 取得尚未執行的action
        already_placement = []
        for value in observation:
            for i in value:
                already_placement = already_placement + [(x, i) for x in range(self.number_of_node)]
        actions_list = list(set(actions).difference(set(already_placement)))  # 可以挑选的动作
        # 列出目前背包狀態下所有action獲得的累計獎勵, 已放置的物品則為Nan
        action_values = q_table.loc[str(observation), :]
        # 使用獲得獎勵最大的action
        greedy_action = action_values.idxmax()
        # 執行epsilon-greedy取得所有可執行動作的執行機率
        probabilities = np.zeros(nA)
        for i in actions_list:
            node, vnf = i
            j = (node * len(self.item)) + vnf
            probabilities[j] = 1 / len(actions_list) * epsilon
        (node, vnf) = greedy_action
        greedy_action = (node * len(self.item)) + vnf
        probabilities[greedy_action] = probabilities[greedy_action] + (1 - epsilon)
        return probabilities

    def q_learning(self, num_episodes, discount_factor=1.0, alpha=0.7, epsilon=0.2):
        number_of_actions = len(self.actions)

        for i_episode in range(1, num_episodes + 1):
            # 開始一輪迭代
            # 開始時背包是空的
            placement = [[] for _ in range(self.number_of_node)]
            # 新增Q table column
            self.q_table = self.check_state(self.q_table, placement, self.actions)

            # 從實際執行的policy中選擇action
            action = np.random.choice(number_of_actions,
                                      p=self.mu_policy(self.q_table,
                                                       epsilon,
                                                       number_of_actions,
                                                       placement,
                                                       self.actions))
            for t in itertools.count():
                # 將action轉為(node, vnf)的格式
                node = action // len(self.item)
                vnf = action % len(self.item)
                action_index_name = (node, vnf)
                # 執行action, 返回reward, 下一步的狀態及是否完成(超過背包限制或所有物品已放完)
                reward, next_placement, done = self.env_reward(action, placement)
                if done:
                    self.q_table.loc[str(placement)][action_index_name] = reward
                    break
                self.q_table = self.check_state(self.q_table, next_placement, self.actions)
                self.q_table.loc[str(placement)][action_index_name] = \
                    self.q_table.loc[str(placement)][action_index_name] + \
                    alpha * (reward +
                             discount_factor * self.q_table.loc[str(next_placement), :].max() -
                             self.q_table.loc[str(placement)][action_index_name])
                # 未結束訓練, 更新放置狀態及動作
                placement = next_placement
                next_action = np.random.choice(number_of_actions,
                                               p=self.mu_policy(self.q_table,
                                                                epsilon,
                                                                number_of_actions,
                                                                next_placement,
                                                                self.actions))
                action = next_action
            if i_episode % 50 == 0:
                print("\rEpisode {}/{}. | ".format(i_episode, num_episodes), end="")
        print("\n")
        return self.q_table

    def pi_policy(self, observation):
        """
        greedy策略, 每次選擇能獲得最大獎勵的動作

        :param observation: 目前背包的狀態
        :return: 一維陣列, 每個動作出現的概率, 最大獎勵的動作為1其他為0
        """

        try:
            # 選擇目前狀態下累計獎勵最高的動作, 轉為(node, vnf)格式
            action_values = self.q_table.loc[str(observation), :]
            best_action = action_values.idxmax()
            node, vnf = best_action
            action = (node * len(self.item)) + vnf
            return np.eye(len(action_values))[action]
        except KeyError:
            return []

    def get_vnf_placement(self):
        # 開始時背包是空的
        knapsack = [[] for _ in range(self.number_of_node)]
        number_of_actions = len(self.actions)
        actions_list = []

        # 從實際執行的policy中選擇action
        action = np.random.choice(number_of_actions,
                                  p=self.pi_policy(knapsack))

        for t in itertools.count():
            actions_list.append(action)
            # 執行action, 返回reward, 下一步的狀態及是否完成(超過背包限制或所有物品已放完)
            reward, next_knapsack, done = self.env_reward(action, knapsack)
            if len(self.pi_policy(next_knapsack)) == 0:
                # 所有物品放完
                knapsack = next_knapsack
                break
            else:
                # 選擇下一步動作
                next_action = np.random.choice(number_of_actions,
                                               p=self.pi_policy(next_knapsack))
                if done:
                    # 物品超過背包限制
                    actions_list.pop()
                    break
                else:
                    # 未結束放置, 更新放置狀態及動作
                    action = next_action
                    knapsack = next_knapsack

        return knapsack


if __name__ == "__main__":
    item_list = [[1, 1], [6, 2], [18, 5], [22, 6], [28, 7], [20, 4], [29, 7], [40, 11]]
    item = pd.DataFrame(data=item_list, columns=['Value', 'Weight'])
    vnf_placement = VNFPlacement(item=item, number_of_node=6, limit_W=11)
    train_start_time = time()
    Q = vnf_placement.q_learning(num_episodes=1000, discount_factor=0.9, alpha=0.3, epsilon=0.1)
    train_finish_time = time()
    print("==============training time==========")
    print(train_finish_time - train_start_time)
    print("==============Q table================")

    get_result_start_time = time()
    vnf_placement_result = vnf_placement.get_vnf_placement()
    get_result_finish_time = time()
    print("==============result time==========")
    print(get_result_finish_time - get_result_start_time)
    print("==============result==========")
    print(vnf_placement_result)

