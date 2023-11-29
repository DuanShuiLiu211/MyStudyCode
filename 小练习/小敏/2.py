import os

import pandas as pd

# 替换为您本地的文件夹路径
folder_path = r"Y:\Desktop\temp"

# 替换为您希望保存Excel文件的实际文件路径
output_file_path = r"Y:\Desktop\temp\合并后的文件.xlsx"

# 获取文件夹中所有的Excel文件
excel_files = [
    os.path.join(folder_path, file)
    for file in os.listdir(folder_path)
    if file.endswith(".xlsx")
]

all_sheets = []

# 遍历每个文件并读取每个工作表
for file_path in excel_files:
    with pd.ExcelFile(file_path) as xls:
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            all_sheets.append(df)

# 合并所有工作表
combined_df = pd.concat(all_sheets, ignore_index=True)

# 删除具有相同'业务流水号'的行
combined_df.drop_duplicates(subset="业务流水号", keep="first", inplace=True)

# 将'交易时间'列转换为datetime类型，以便正确排序
combined_df["交易时间"] = pd.to_datetime(combined_df["交易时间"])

# 按'交易时间'排序
combined_df.sort_values(by="交易时间", ascending=True, inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# 查看合并和排序后的DataFrame
print(combined_df)

# 保存最终的DataFrame为Excel文件
combined_df.to_excel(output_file_path, index=False, sheet_name="合并后的数据")
