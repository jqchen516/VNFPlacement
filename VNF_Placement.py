import pandas as pd
import numpy as np
from time import time
import itertools
import copy
import matplotlib.pyplot as plt


class VNFPlacement():

    def __init__(self, item, number_of_node, limit_W):
        # self.node_state = [[0, 1, 3], [2, 4, 6, 7], [5, 8, 9, 10]]
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

    def gen_score(self, placement):
        score = 0
        count = 0
        for index, node_stat in enumerate(placement):
            count += len(node_stat)
            cpu_usage = np.sum([self.item['cpu'][i] for i in node_stat]) / self.limit_W['cpu']
            memory_usage = np.sum([self.item['memory'][i] for i in node_stat]) / self.limit_W['memory']
            bw_usage = np.sum([self.item['BW'][i] for i in node_stat]) / self.limit_W['BW']
            if cpu_usage > 1 or memory_usage > 1 or bw_usage > 1:
                score = -99
                done = "out of limit"
                return done, score

            node_score = int(cpu_usage * 10) + int(memory_usage * 10) + int(bw_usage * 10)

            # cpu_score = self.limit_W['cpu'] - np.sum([self.item['cpu'][i] for i in node_stat])
            # memory_score = self.limit_W['memory'] - np.sum([self.item['memory'][i] for i in node_stat])
            # node_score = cpu_score + memory_score
            score += node_score

        # 所有物品已放完
        if count >= len(self.item):
            for index, node_stat in enumerate(placement):
                # cpu_usage = np.sum([self.item['cpu'][i] for i in node_stat]) / self.limit_W['cpu']
                # memory_usage = np.sum([self.item['memory'][i] for i in node_stat]) / self.limit_W['memory']
                # bw_usage = np.sum([self.item['BW'][i] for i in node_stat]) / self.limit_W['BW']
                # if cpu_usage > 1 or memory_usage > 1 or bw_usage > 1:
                #     score = -99
                #     done = "out of limit"
                #     return done, score
                # node_score = int(cpu_usage * 10) + int(memory_usage * 10) + int(bw_usage * 10)

                # cpu_score = self.limit_W['cpu'] - np.sum([self.item['cpu'][i] for i in node_stat])
                # memory_score = self.limit_W['memory'] - np.sum([self.item['memory'][i] for i in node_stat])
                # node_score = cpu_score + memory_score
                # score += node_score

                # TODO 遷移成本
                # move = len(set(node_stat).intersection(self.node_state[index]))
                # move = len(self.node_state[index]) - len(set(node_stat).intersection(self.node_state[index]))
                # score += (10 * move)

                # 運行節點
                if len(node_stat) == 0:
                    score += 10
            done = "finish"
            return done, score
        # score = 50 * count
        done = "continue"
        return done, score

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
        done, reward = self.gen_score(placement_)

        return reward, placement_, done

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

        stats = {'episode_lengths': np.zeros(num_episodes + 1),
                 'episode_rewards': np.zeros(num_episodes + 1),
                 'final': [],
                 'total': [],
                 'placement': []}

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

                # if done == "continue" and done == "finish":
                stats['episode_rewards'][i_episode] += reward  # 计算累计奖励
                stats['episode_lengths'][i_episode] = t  # 查看每一轮的时间
                if done != "continue":
                    # stats['final'][i_episode] = 145
                    if done == "finish":
                        # stats['final'].append(self.q_table.loc[str(placement)][action_index_name] + \
                        #                       alpha * (reward +
                        #                                discount_factor * 100 -
                        #                                self.q_table.loc[str(placement)][action_index_name]))
                        stats['final'].append(self.q_table.loc[str(placement)][action_index_name] + \
                                              alpha * (reward +
                                                       discount_factor * 100 -
                                                       self.q_table.loc[str(placement)][action_index_name]))
                        stats['total'].append(stats['episode_rewards'][i_episode])
                        stats['placement'].append(next_placement)
                    self.q_table.loc[str(placement)][action_index_name] = \
                        self.q_table.loc[str(placement)][action_index_name] + \
                        alpha * (reward +
                                 discount_factor * 100 -
                                 self.q_table.loc[str(placement)][action_index_name])
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
            if i_episode % 500 == 0:
                print("\rEpisode {}/{}. | ".format(i_episode, num_episodes), end="")
        print("\n")
        return self.q_table, stats

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
            best_action_score = action_values.max()
            if isinstance(best_action, str):
                best_action = best_action[1:-1]
                best_action = tuple(map(int, best_action.split(', ')))
            (node, vnf) = best_action
            action = (node * len(self.item)) + vnf
            return np.eye(len(action_values))[action], best_action_score
        except KeyError:
            return [], 0

    def get_vnf_placement(self):
        # 開始時背包是空的
        knapsack = [[] for _ in range(self.number_of_node)]
        number_of_actions = len(self.actions)
        actions_list = []

        self.actions = list()
        for place in range(self.number_of_node):
            self.actions = self.actions + [(place, item) for item in list(range(len(self.item)))]

        # 從實際執行的policy中選擇action
        p, best_action = self.pi_policy(knapsack)
        action = np.random.choice(number_of_actions,
                                  p=p)
        final_score = 0
        for t in itertools.count():
            actions_list.append(action)
            # 執行action, 返回reward, 下一步的狀態及是否完成(超過背包限制或所有物品已放完)
            reward, next_knapsack, done = self.env_reward(action, knapsack)

            if done == "continue":
                p, best_action = self.pi_policy(next_knapsack)
                # 選擇下一步動作
                try:
                    next_action = np.random.choice(number_of_actions,
                                                   p=p)
                    final_score = best_action
                except:
                    print(number_of_actions)
                    print(p)

                # 未結束放置, 更新放置狀態及動作
                action = next_action
                knapsack = next_knapsack
            elif done == "out of limit":
                print(done)
                break
            else:
                knapsack = next_knapsack
                break

        return knapsack, final_score, done


def experiment(num_episodes, discount_factor, alpha, epsilon, filename):
    vnf_resource_list = [[4, 4, 100], [2, 4, 100], [1, 10, 100],
                         [2, 6, 100], [3, 4, 100], [2, 4, 100],
                         [1, 1, 100], [2, 1, 100], [2, 2, 100],
                         [2, 2, 100], [1, 6, 100]]
    # vnf_resource_list = [[4, 4, 100], [2, 4, 100], [1, 10, 100],
    #                      [2, 6, 100], [3, 4, 100], [2, 4, 100]]

    item = pd.DataFrame(data=vnf_resource_list, columns=['cpu', 'memory', 'BW'])
    node_resource = {
        'cpu': 8,
        'memory': 16,
        'BW': 1000
    }

    vnf_placement = VNFPlacement(item=item, number_of_node=5, limit_W=node_resource)
    train_start_time = time()
    Q, stats = vnf_placement.q_learning(num_episodes=num_episodes,
                                        discount_factor=discount_factor,
                                        alpha=alpha,
                                        epsilon=epsilon)

    train_finish_time = time()
    # print("==============training time==========")
    training_time = train_finish_time - train_start_time
    # print(training_time)
    # print("==============Q table================")
    # print(Q)

    get_result_start_time = time()
    vnf_placement_result, final_score, done = vnf_placement.get_vnf_placement()
    get_result_finish_time = time()
    # print("==============result time==========")
    inference_time = get_result_finish_time - get_result_start_time
    # print(inference_time)
    # print("==============result==========")
    # print(vnf_placement_result)
    # print(final_score)
    done, placement_score = vnf_placement.gen_score(vnf_placement_result)
    with open('/Users/chenjianqun//Downloads/rl/experiment1_results/{filename}'.format(filename=filename), 'a') as f:
        f.write(str(training_time) + ', ' + str(inference_time) + ', ' + str(final_score) + ',' + str(
            vnf_placement_result) + ',' + str(placement_score) + ',' + done + '\n')


def experiment_debug(num_episodes, discount_factor, alpha, epsilon):
    vnf_resource_list = [[4, 4, 100], [2, 4, 100], [1, 10, 100],
                         [2, 6, 100], [3, 4, 100], [2, 4, 100],
                         [1, 1, 100], [2, 1, 100], [2, 2, 100],
                         [2, 2, 100], [1, 6, 100]]

    # vnf_resource_list = [[1, 2, 10], [2, 2, 10], [4, 1, 10], [5, 4, 10], [2, 5, 10],
    #                      [2, 3, 10], [1, 4, 10], [5, 5, 10], [5, 2, 10], [4, 3, 10],
    #                      [2, 1, 10], [4, 4, 10], [3, 3, 10], [3, 5, 10], [3, 4, 10],
    #                      [4, 1, 10], [4, 1, 10], [1, 3, 10], [5, 1, 10], [4, 3, 10],
    #                      [5, 5, 10], [3, 3, 10], [2, 3, 10], [2, 1, 10], [3, 1, 10],
    #                      [3, 5, 10], [2, 2, 10], [5, 4, 10], [1, 4, 10], [3, 4, 10]]

    # vnf_resource_list = [[1, 2, 10], [2, 2, 10], [4, 1, 10], [5, 4, 10], [2, 5, 10],
    #                      [2, 3, 10], [1, 4, 10], [5, 5, 10], [5, 2, 10], [4, 3, 10],
    #                      [2, 1, 10], [4, 4, 10], [3, 3, 10], [3, 5, 10], [3, 4, 10],
    #                      [8, 9, 10], [9, 8, 10], [7, 7, 10], [8, 6, 10], [7, 6, 10],
    #                      [6, 8, 10], [9, 6, 10], [7, 8, 10], [9, 9, 10], [6, 7, 10],
    #                      [8, 6, 10], [8, 7, 10], [8, 6, 10], [6, 6, 10], [6, 7, 10]]

    # vnf_resource_list = [[7, 9, 10], [9, 8, 10], [7, 7, 10], [9, 6, 10], [9, 6, 10],
    #                      [8, 6, 10], [6, 8, 10], [7, 8, 10], [6, 6, 10], [8, 6, 10],
    #                      [9, 6, 10], [6, 9, 10], [6, 7, 10], [6, 8, 10], [9, 8, 10],
    #                      [8, 9, 10], [9, 8, 10], [7, 7, 10], [8, 6, 10], [7, 6, 10],
    #                      [6, 8, 10], [9, 6, 10], [7, 8, 10], [9, 9, 10], [6, 7, 10],
    #                      [8, 6, 10], [8, 7, 10], [8, 6, 10], [6, 6, 10], [6, 7, 10]]

    # vnf_resource_list = [[8, 9, 10], [9, 8, 10], [7, 7, 10], [8, 6, 10], [7, 6, 10],
    #                      [6, 8, 10], [9, 6, 10], [7, 8, 10], [9, 9, 10], [6, 7, 10],
    #                      [8, 6, 10], [8, 7, 10], [8, 6, 10], [6, 6, 10], [6, 7, 10],
    #                      [10, 20, 10], [10, 20, 10], [10, 10, 10], [10, 10, 10], [10, 10, 10],
    #                      [10, 10, 10], [10, 20, 10], [10, 10, 10], [10, 20, 10], [10, 10, 10],
    #                      [10, 10, 10], [10, 10, 10], [10, 20, 10], [10, 10, 10], [10, 20, 10]]

    # vnf_resource_list = [[10, 20, 10], [10, 20, 10], [10, 10, 10], [10, 10, 10], [10, 10, 10],
    #                      [10, 10, 10], [10, 10, 10], [10, 10, 10], [10, 10, 10], [10, 20, 10],
    #                      [10, 20, 10], [10, 10, 10], [10, 10, 10], [10, 20, 10], [10, 20, 10],
    #                      [10, 20, 10], [10, 20, 10], [10, 10, 10], [10, 10, 10], [10, 10, 10],
    #                      [10, 10, 10], [10, 20, 10], [10, 10, 10], [10, 20, 10], [10, 10, 10],
    #                      [10, 10, 10], [10, 10, 10], [10, 20, 10], [10, 10, 10], [10, 20, 10]]

    item = pd.DataFrame(data=vnf_resource_list, columns=['cpu', 'memory', 'BW'])
    node_resource = {
        'cpu': 8,
        'memory': 16,
        'BW': 1000
    }

    vnf_placement = VNFPlacement(item=item, number_of_node=5, limit_W=node_resource)
    train_start_time = time()
    Q, stats = vnf_placement.q_learning(num_episodes=num_episodes,
                                        discount_factor=discount_factor,
                                        alpha=alpha,
                                        epsilon=epsilon)

    train_finish_time = time()
    print("==============training time==========")
    training_time = train_finish_time - train_start_time
    print(training_time)
    print("==============Q table================")
    print(Q)
    Q.to_csv("q_table.csv")

    get_result_start_time = time()
    vnf_placement_result, final_score, done = vnf_placement.get_vnf_placement()
    get_result_finish_time = time()
    print("==============result time==========")
    inference_time = get_result_finish_time - get_result_start_time
    print(inference_time)
    print("==============result==========")
    print(vnf_placement_result)
    print(final_score)
    print(done)
    done, placement_score = vnf_placement.gen_score(vnf_placement_result)
    print(placement_score)
    # print(stats['final'])
    reward = stats['final']
    # reward = stats['total']
    # a = list()
    # for index, value in enumerate(reward):
    #     if index % 3 == 0:
    #         a.append(value)
    print(reward)
    print(len(reward))
    # plt.plot(reward, linewidth=0.8)
    # plt.show()


if __name__ == "__main__":
    # FILE_NAME = '5nodetest.csv'
    # with open('/Users/chenjianqun//Downloads/rl/experiment1_results/{filename}'.format(filename=FILE_NAME), 'w') as f:
    #     f.write("training_time, inference_time, final_score, placement, placement_score, state\n")
    # for i in range(20):
    #     experiment(num_episodes=1000, discount_factor=0.9, alpha=0.1, epsilon=0.05, filename=FILE_NAME)
    #     print("\rEpisode {}/{}. | ".format(i, 1000), end="")

    experiment_debug(num_episodes=10000, discount_factor=0.7, alpha=0.3, epsilon=0.4)

    # vnf_resource_list = [[4, 4, 100], [2, 4, 100], [1, 10, 100],
    #                      [2, 6, 100], [3, 4, 100], [2, 4, 100],
    #                      [1, 1, 100], [2, 1, 100], [2, 2, 100],
    #                      [2, 2, 100], [1, 6, 100]]
    # vnf_resource_list = [[4, 4, 100], [2, 4, 100], [1, 10, 100],
    #                      [2, 6, 100], [3, 4, 100], [2, 4, 100],
    #                      [1, 1, 100], [2, 1, 100]]
    # item = pd.DataFrame(data=vnf_resource_list, columns=['cpu', 'memory', 'BW'])
    # node_resource = {
    #     'cpu': 8,
    #     'memory': 16,
    #     'BW': 1000
    # }
    #
    # a = VNFPlacement(item=item, number_of_node=4, limit_W=node_resource)
    # q_table = pd.read_csv("q_table.csv", index_col=[0])
    # a.q_table = q_table
    # print(q_table)
    # # actions = list()
    # # for place in range(4):
    # #     actions = actions + [(place, item) for item in list(range(len(item)))]
    # # a.actions = actions
    #
    # vnf_placement_result, final_score, done = a.get_vnf_placement()
    # print(vnf_placement_result, final_score, done)
son = {
    "Nodes": [
        {
            "<node_name>": {
                "CPU": "8",
                "Memory": "16MB",
                "Bandwidth": "1GB/s"
            }
        }
    ],
    "VNFs": [
        {
            "<vnf_id>": {
                "CPU": "1",
                "Memory": "1MB",
                "Bandwidth": "100MB/s"
            }
        }
    ]
}
