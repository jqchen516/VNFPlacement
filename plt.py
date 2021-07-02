# -*- coding: UTF-8 -*-
import csv
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


print(matplotlib.__file__)
x = list()
min_label = list()
max_label = list()
avg_label = list()
for i in ['50', '100', '200', '500', '1000']:
    FILE_NAME = '{v}timealpha1.csv'.format(v=i)
    score = list()
    with open('/Users/chenjianqun//Downloads/rl/experiment2_results/{filename}'.format(filename=FILE_NAME)) as f:
        rows = f.readlines()
        for row in rows:
            try:
                final_score = float(row.split(',')[-2])
                score.append(final_score)
            except:
                print(row)
    max_score = max(score)
    min_score = min(score)
    avg_value = 0 if len(score) == 0 else sum(score) / len(score)
    x.append(i)
    min_label.append(min_score)
    max_label.append(max_score)
    avg_label.append(avg_value)
print(avg_label)
print(min_label)
print(max_label)
# plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
# # plt.rcParams['axes.unicode_minus'] = False
#
# plt.plot(x, min_label, 'r-^', label='Minimum')
# plt.plot(x, max_label, 'g-*', label='Maximum')
# plt.plot(x, avg_label, 'b-o', label='Average')
# plt.grid(axis="y", linestyle="-.")
# # plt.title("0.1學習率", fontsize=12) #圖表標題
# plt.xlabel("Q-Learning訓練迭代次數", fontsize=12) #x軸標題
# plt.ylabel("最終決策結果之評分", fontsize=12) #y軸標題
# plt.legend(loc = "best", fontsize=8)
# plt.show()
