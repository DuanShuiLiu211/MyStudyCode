import pandas as pd
import os


if __name__ == "__main__":
    file_dir_1 = r'/Volumes/昊大侠/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/数据/短轴动态狭窄率/result/狭窄率图表_0916/results/rvm'
    pd_sheet_1 = os.listdir(file_dir_1)
    list_data_1 = sorted(pd_sheet_1)
    set_data_1 = set(list_data_1)

    file_dir_2 = r'/Volumes/昊大侠/工作/实习相关/微创卜算子医疗科技有限公司/陈嘉懿组/数据/短轴动态狭窄率/result/狭窄率图表_0916/results/transbts'
    pd_sheet_2 = os.listdir(file_dir_2)
    # pd_sheet_2.remove('.DS_Store')
    # pd_sheet_2.remove('._.DS_Store')
    list_data_2 = sorted(pd_sheet_2)
    set_data_2 = set(list_data_2)

    print(f"diffrent data {set_data_1 ^ set_data_2}")

    print("运行完成")
