import pandas as pd

if __name__ == "__main__":
    file_dir_1 = r'/Users/WangHao/Desktop/TODO/Data/data_divide.xlsx'
    pd_sheet_1 = pd.read_excel(file_dir_1)
    pd_data_1 = pd_sheet_1.values[:, 1]
    list_data_1 = sorted(list(pd_data_1))
    set_data_1 = set(list_data_1)

    file_dir_2 = r'/Volumes/昊大侠/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/数据/短轴动态狭窄率/result/狭窄率图表_0916/rvm/data_divide.xlsx'
    pd_sheet_2 = pd.read_excel(file_dir_2)
    pd_data_2 = pd_sheet_2.values[:, 1]
    list_data_2 = sorted(list(pd_data_2))
    set_data_2 = set(list_data_2)

    print(f"diffrent data {set_data_1 ^ set_data_2}")

    print("运行完成")
