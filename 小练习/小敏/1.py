import pandas as pd
import os

folder_path = r'Y:\Desktop\temp\沐歆2号'  # 请替换为实际文件夹路径
output_file = r'Y:\Desktop\temp\沐歆2号\output.xlsx'

# 获取文件夹中的所有Excel文件
all_files = [file for file in os.listdir(folder_path) if (file.endswith('.xlsx') or file.endswith('.xls')) and not file.startswith('~$')]

all_data = []  # 用于存放所有符合条件的数据

# 遍历每个Excel文件并筛选数据
for file in all_files:
    try:
        file_path = os.path.join(folder_path, file)
        xls = pd.ExcelFile(file_path)
        
        # 遍历每个sheet
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 过滤出包含"gfpt"的备注
            mask = df['备注'].str.contains('gfpt', case=False, na=False) # 这里假设列名为“项目备注”
            filtered_data = df[mask].copy()  # 使用.copy()确保在原始DataFrame上没有任何修改
            
            # 为每个筛选出的项目添加“补充说明”
            filtered_data['补充'] = (f"来源文件: {file}, "
                                     f"页数: {sheet_name}, "
                                     f"行数: " + (filtered_data.index + 2).astype(str))
            # 注意：行数+2是因为Excel的行数从1开始计数，而DataFrame的index从0开始。同时，因为有表头行，所以我们需要加2而不是加1。

            all_data.append(filtered_data)
            
    except PermissionError:
        print(f"Permission denied for file: {file}. Skipping this file.")

# 将所有数据合并为一个DataFrame
final_data = pd.concat(all_data, ignore_index=True)

# 将数据保存到一个新的Excel文件
final_data.to_excel(output_file, index=False, engine='openpyxl')
