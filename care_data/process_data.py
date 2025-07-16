import pandas as pd
import os
import re

# 定义输入和输出文件名
input_filename = 'protein_train.csv'
output_filename = 'enzyme_sequences.csv'

def filter_ec_number(ec_string):
    """
    处理EC编号字符串。
    - 如果只有一个EC编号，保留它。
    - 如果有多个EC编号，检查它们的第一部分是否都相同。
      - 如果相同, 保留第一个EC编号。
      - 如果不同, 返回None，以便后续删除该行。
    """
    if not isinstance(ec_string, str):
        return None

    # 按分号或空白符分割，并清除空字符串
    ec_numbers = [ec for ec in re.split(r'[;\s]+', ec_string.strip()) if ec]

    if not ec_numbers:
        return None

    if len(ec_numbers) == 1:
        return ec_numbers[0]
    
    # 多个EC编号的情况
    try:
        # 获取每个EC编号的第一部分
        first_components = [ec.split('.')[0] for ec in ec_numbers]

        # 如果所有第一部分都相同，则保留第一个EC编号
        if len(set(first_components)) == 1:
            return ec_numbers[0]
        else:
            # 否则，丢弃该条目
            return None
    except IndexError:
        # 处理没有'.'的格式错误的EC编号
        return None

# 检查输入文件是否存在
if not os.path.exists(input_filename):
    print(f"错误: 输入文件 '{input_filename}' 不存在。请确保文件在当前目录下。")
else:
    # 加载数据集
    # 使用 on_bad_lines='skip' 来跳过格式不正确的行
    df = pd.read_csv(input_filename, on_bad_lines='skip')

    # 检查所需列是否存在
    if 'EC number' in df.columns and 'Sequence' in df.columns:
        # 创建一个新的DataFrame，只包含'EC number'和'Sequence'列
        # .copy()可以避免SettingWithCopyWarning
        processed_df = df[['EC number', 'Sequence']].copy()

        # 应用自定义函数处理'EC number'列
        processed_df['EC number'] = processed_df['EC number'].apply(filter_ec_number)
        
        # 删除处理后'EC number'为空的行或'Sequence'为空的行
        processed_df.dropna(subset=['EC number', 'Sequence'], inplace=True)

        # 提取EC编号的第一类
        processed_df['EC_class'] = processed_df['EC number'].str.split('.').str[0]
        
        # 确保EC_class是字符串类型，以便后续筛选
        processed_df['EC_class'] = processed_df['EC_class'].astype(str)

        print("开始为每个EC类别采样并生成文件...")

        # 为EC1到EC7的每个类别分别采样6000个并保存
        for i in range(1, 8):
            class_str = str(i)
            # 筛选出当前类别的数据
            class_df = processed_df[processed_df['EC_class'] == class_str]

            # 检查样本数量
            if len(class_df) >= 6000:
                # 随机抽取6000个样本
                # 使用 random_state 确保每次运行结果一致
                sampled_df = class_df.sample(n=6000, random_state=42)
                
                # 定义输出文件名
                output_filename = f'ec_class_{class_str}.csv'
                
                # 准备保存的数据（只含EC号和序列）
                df_to_save = sampled_df[['EC number', 'Sequence']]
                
                # 保存到CSV
                df_to_save.to_csv(output_filename, index=False)
                
                print(f"成功创建文件: '{output_filename}'，包含 6000 个样本。")
            else:
                print(f"EC类别 {class_str} 的样本不足6000个 (只有 {len(class_df)} 个)，无法生成文件。")

    else:
        print(f"错误: 输入文件 '{input_filename}' 中缺少 'EC number' 或 'Sequence' 列。") 