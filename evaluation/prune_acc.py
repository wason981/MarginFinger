import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes


prune_ratio = [0,5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
accuracy = [87.34,87.16,86.88,85.95,85.55,85.49,84.69,84.41,83.94,83.5,83.34,82.24,77.2,60.73,18.62]
similarity = [97.20,91.10,  85.00, 80.20,  77.80,76.60, 74.60, 74.50, 72.00, 71.80, 70.20, 62.50, 45.30, 35.60, 17.30]
irr_simi=    [41.8, 41.8, 41.8, 41.8, 41.8, 41.8, 41.8, 41.8, 41.8, 41.8, 41.8, 41.8, 41.8, 41.8, 41.8]

# accuracy=[73.94,73.58,71.08,49.28,08.18,1,1,1,1,1,1,1,1,1,1]
# similarity = [90.30,80.30,65.40,26.50,3.10,1,1,1,1,1,1,1,1,1,1]
# irr_simi=  [19.2,19.2,19.2,19.2,19.2,19.2,19.2,19.2,19.2,19.2,19.2,19.2,19.2,19.2,19.2]
# for threshold in range(0,101):
#     threshold=threshold/100
#     robustness_scores.append((sum(torch.tensor(pro_acc)>threshold)+
#                               sum(torch.tensor(lab_acc)>threshold)+
#                               sum(torch.tensor(adv_acc)>threshold)+
#                               sum(torch.tensor(fp_acc)>threshold)+
#                               sum(torch.tensor(ft_acc)>threshold)+
#                               sum(torch.tensor(tl_acc)>threshold)
#                               )/105)
#     uniqueness_scores.append(sum(torch.tensor(irr_acc)<threshold)/len(irr_acc))
# thresholds = [i / 100 for i in range(101)]

# fig = plt.figure()
# ax = fig.subplots()
# ax2 = ax.twinx()
# plt.plot(prune_ratio, accuracy, label=Accuracy, color='#0099CC')
# plt.plot(prune_ratio, similarity, label='Similarity', color='#FF6666')  # 同上
# # max_values = [min(x, y) for x, y in zip(robustness_scores, uniqueness_scores)]
# # aruc = np.trapz(max_values, thresholds)
#
# # plt.fill_between(thresholds, max_values, facecolor='#CCCCCC', lw=.1, zorder=2)
# # plt.title('(e) Accuracy and model’s accuracy change
# # during pruning.CIFAR-10 SAC-m(ARUC=.3f)'  aruc)
# # plt.title('(f)Tiny-ImageNet SAC-w(ARUC=.3f)'  aruc)
# ax.set_xlabel('Ratio of Neurons Pruned ')
#
# x_major_locator = plt.MultipleLocator(0.1)
# y_major_locator = plt.MultipleLocator(0.1)
#
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(y_major_locator)
#
# plt.legend(loc='upper right')
# plt.savefig('ARUC_CIFAR_10_SAC_m.png')
# plt.show()

# 创建图像和坐标轴对象
# fig, ax1 = plt.subplots()
#
# # 绘制第一条曲线（左边坐标轴）
# ax1.plot(prune_ratio, accuracy, label='Accuracy %', color='#0099CC')
# ax1.set_xlabel('Ratio of Neurons Pruned %')
# ax1.set_ylabel('Accuracy')
#
# # 创建第二个坐标轴对象
# ax2 = ax1.twinx()
# # 绘制第二条曲线（右边坐标轴）
# ax2.plot(prune_ratio, similarity, label='Similarity %', color='#FF6666')
# ax2.set_ylabel('Similarity')
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
# # 显示图形
# plt.show()
############################################################################################################
fig = plt.figure()
ax = HostAxes(fig, [0.1, 0.1, 0.7, 0.7])  #用[left, bottom, weight, height]的方式定义axes, 0 <= l,b,w,h <= 1
kw = dict(linewidth = 2,markerfacecolor='none',markersize = 4)

#parasite addtional axes, share x
ax02 = ParasiteAxes(ax, sharex=ax)
ax03 = ParasiteAxes(ax, sharex=ax)

#append axes
ax.parasites.append(ax02)
ax.parasites.append(ax03)

# ax03_axisline = ax02.get_grid_helper().new_fixed_axis
# ax03.axis['right4'] = ax03_axisline(loc='right', axes=ax03, offset=(60,0))


fig.add_axes(ax)
h1, = ax.plot(prune_ratio, accuracy, linewidth=2,color='#0099CC',label="Accuracy")
h2, = ax02.plot(prune_ratio, similarity, linewidth=2,color='#FF6666',label="Similarity")
h2, = ax02.plot(prune_ratio, irr_simi, linewidth=2,color='#FF6666',label="Irrlevant Similarity",linestyle='--')
# h3 = ax03.plot(prune_ratio,irr_simi,linewidth=2,color='#FF6666',label="Irrlevant Similarity",linestyle='--')

#invisible right axis of ax
ax.axis['right'].set_visible(False)
ax.axis['top'].set_visible(True)
ax02.axis['right'].set_visible(True)
ax02.axis['right'].major_ticklabels.set_visible(True)
ax02.axis['right'].label.set_visible(True)

#set label for axis
ax.set_ylabel('Accuracy %')
ax.set_xlabel('Ratio of Neurons Pruned %')
ax02.set_ylabel('Similarity %')
# ax03.set_ylabel('y3-中文')

#set xlim for yaxis
# ax.set_ylim(-150,150)
# ax02.set_ylim(0,150)
# ax03.set_ylim(0,500)

ax.legend()

#name axies, xticks colors
# ax.axis['left'].label.set_color('r')
# ax02.axis['right'].label.set_color('g')
# ax03.axis['right4'].label.set_color('b')

# ax.axis['left'].major_ticks.set_color('r')
# ax02.axis['right'].major_ticks.set_color('g')
# ax03.axis['right4'].major_ticks.set_color('b')

# ax.axis['left'].major_ticklabels.set_color('r')
# ax02.axis['right'].major_ticklabels.set_color('g')
# ax03.axis['right4'].major_ticklabels.set_color('b')

# ax.axis['left'].line.set_color('r')
# ax02.axis['right'].line.set_color('g')
# ax03.axis['right4'].line.set_color('b')
# plt.savefig('acc_sim_Tiny_ImageNet.png')
plt.savefig('acc_sim_CIFAR_10.png')
plt.show()
