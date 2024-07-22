import json
import matplotlib.pyplot as plt
import os

# 假设所有的info文件都在同一个目录下
info_files = [f for f in os.listdir('.') if f.endswith('_info.json')]
file_prefixes = [f.replace('_info.json', '') for f in info_files]

# 准备一个空的列表来存储所有文件的battle_won_mean数据
all_battle_won_means = []

# 读取每个JSON文件并提取数据
for file_name in info_files:
    with open(file_name, 'r') as file:
        data = json.load(file)
        battle_won_mean = data['test_dead_allies_mean']
        all_battle_won_means.append(battle_won_mean)

# 创建一个曲线图
plt.figure(figsize=(10, 5))

# 绘制每个文件的battle_won_mean曲线，去掉描点
for i, battle_won_mean in enumerate(all_battle_won_means):
    # 使用file_prefixes数组中的内容作为标签
    plt.plot(battle_won_mean, label=file_prefixes[i], marker=None)

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('Dead Allies Mean Over Time')
plt.xlabel('Index')
plt.ylabel('Dead Allies Mean')

# 显示网格
plt.grid(True)

# 显示图形
plt.show()