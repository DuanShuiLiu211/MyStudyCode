import numpy as np

a = [1, 2, 3, 4]
aArray = np.array([1, 2, 3])

print(a)
a.append(100)
print(a)
b = [99, 88, 77]
a[1:2] = b
print(a)

import torch

x = torch.empty(5, 3)
print(x)

import torch

print(torch.cuda.is_available())  # 返回True则说明已经安装了cuda

from torch.backends import cudnn  # 判断是否安装了cuDNN

print(cudnn.is_available())

print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.empty(3, 3))
print(5 // 2)

chance = np.random.randint(100) + 1
print(chance)

if not 0:
    print('运行了')
