import matplotlib.pyplot as plt

# 数据
sizes = [1, 2, 3, 3, 60, 10, 1, 0, 10, 10]
# labels = ['1%', '2%', '3%', '3%', '60%', '10%', '1%', '0%', '10%', '10%']
# colors = ['red', 'orange', 'yellow', 'gold', 'lightblue', 'lightgreen', 'pink', 'white', 'purple', 'brown']
colors = ['steelblue', 'tomato','firebrick','skyblue','mediumaquamarine','orange','darksalmon','gray','brown','palevioletred']  # 建立颜色列表
labels = ['CLASS1', 'CLASS2', 'CLASS3', 'CLASS4', 'CLASS5', 'CLASS6', 'CLASS7', 'CLASS8', 'CLASS9','CLASS10',]

# 绘制饼状图
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

# 设置标题
plt.title('Proportion of Generated Samples')

# 显示图形
plt.axis('equal')  # 保证饼状图是正圆的
plt.show()
