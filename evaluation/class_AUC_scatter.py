import matplotlib.pyplot as plt
import numpy as np

# 假设你的数据是一个10x6的NumPy数组
x=range(10)
y = np.random.rand(10, 6)
# 创建一个散点图
plt.figure(figsize=(10, 6))

colors = ['b', 'g','r','c','m','y','k','gray','brown','orange']  # 建立颜色列表
labels = ['CLASS1', 'CLASS2', 'CLASS3', 'CLASS4', 'CLASS5', 'CLASS6', 'CLASS7', 'CLASS8', 'CLASS9','CLASS10',]

# 为每一行的数据画散点图
for i in range(y.shape[0]):
    plt.scatter(range(6), y[i, :], label=labels[i],marker='o')

avg=np.random.rand(1,6)
mfa=np.random.rand(1,6)
plt.scatter(range(6), avg, label='MF(AVG)',marker='s',color='r')
plt.scatter(range(6), mfa, label='MF-A',marker='*',color='k')



plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)  # 设置图像边界
# 设置横纵坐标轴标签
plt.xlabel('Threat')
plt.ylabel('AUC')
plt.xticks(range(6), ['MEP', 'MEL', 'MEA', 'FP', 'FTL', 'TL'])
# 设置图例
# plt.legend()
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.title('AUC of different class\' samples in MF')  # 设置图形标题

# 显示图形
plt.show()
plt.savefig('class_MF.png')