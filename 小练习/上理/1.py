import platform

import numpy as np
import torch

array = np.array([1, 2, 3, 4])
print(array)
chance = np.random.randint(100) + 1
print(chance)

tensor = torch.empty(5, 3)
print(tensor)

if platform.system() == "Darwin":
    from torch.backends import mps

    print(mps.is_built())  # 判断机器是否安装了mps
    print(mps.is_available())  # 判断是否安装了mps
else:
    from torch.backends import cudnn

    print(torch.cuda.is_available())  # 判断机器是否安装了cuda
    print(cudnn.is_available())  # 判断是否安装了cuDNN
    print(torch.cuda.device_count())  # 机器的cuda设备数
    print(torch.cuda.current_device())  # 当前设备的索引
    print(torch.cuda.device(0))  # 当前设备设置为索引0
    print(torch.cuda.get_device_name(0))  # 索引0设备的名称

print("运行完成")
