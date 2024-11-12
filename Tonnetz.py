import librosa
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib as mpl

# 设置WAV文件所在目录
base_directory = r'C:\Users\Administrator\Desktop\xietiao'
target_folder = 'glutamate'  # 将这里改为您想要处理的特定文件夹名称

directory = os.path.join(base_directory, target_folder)

# 修改存储结果的字典
features = {
    'file_number': [],
    'Tonnetz': [],
    'Tonnetz_Highest': []
}

# 检查目标文件夹是否存在
if not os.path.isdir(directory):
    print(f"错误：指定的文件夹 '{target_folder}' 不存在。")
else:
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            
            # 提取文件名中的数字部分
            number = int(re.search(r'\d+', filename).group())
            
            # 加载WAV文件
            y, sr = librosa.load(file_path, sr=None)
            
            # 计算tonnetz特征
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            
            # 计算tonnetz特征
            tonnetz_basic = np.mean(tonnetz)  # 基础tonnetz平均值
            tonnetz_highest = np.max(tonnetz)  # 最高音
            
            # 存储结果
            features['file_number'].append(number)
            features['Tonnetz'].append(tonnetz_basic)
            features['Tonnetz_Highest'].append(tonnetz_highest)
            
            print(f'{filename} 的处理结果已存储。')

    # 根据文件编号排序结果
    sorted_indices = np.argsort(features['file_number'])
    sorted_numbers = np.array(features['file_number'])[sorted_indices]
    
    # 更新需要分析的特征列表
    feature_names = [
        'Tonnetz',
        'Tonnetz_Highest'
    ]
    
    # 设置matplotlib使用支持中文的字体
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    
    # 分别为每个特征创建单独的图表
    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        sorted_values = np.array(features[feature])[sorted_indices]
        
        # 计算皮尔逊相关系数
        x = sorted_numbers
        y = sorted_values
        correlation_coef = np.corrcoef(x, y)[0, 1]
        
        # 绘制散点图和拟合直线
        plt.scatter(x, y, alpha=0.6, label='Tonnetz Highest Value')
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        plt.plot(x, y_pred, color='red', label=f'Fitting line (Pearson Correlation Coefficient = {correlation_coef:.4f})')
        plt.xlabel('kcat/km', fontsize=10)
        plt.ylabel(feature, fontsize=10)
        plt.title(f'{feature} Scatter Plot', fontsize=12)
        plt.legend(fontsize=8)
        plt.grid(True)
    
    plt.tight_layout()  # 自动调整布局
    plt.show()
    
    # 输出各特征的皮尔逊相关系数
    print("各特征的皮尔逊相关系数：")
    for feature in feature_names:
        sorted_values = np.array(features[feature])[sorted_indices]
        x = sorted_numbers
        y = sorted_values
        correlation_coef = np.corrcoef(x, y)[0, 1]
        print(f"{feature} 的皮尔逊相关系数: {correlation_coef:.4f}")
