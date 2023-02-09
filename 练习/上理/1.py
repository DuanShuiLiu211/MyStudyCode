import numpy as np
<<<<<<< HEAD
array = np.array([1, 2, 3, 4])
print(array)
chance = np.random.randint(100) + 1
print(chance)

import torch
tensor = torch.empty(5, 3)
print(tensor)

import platform
if platform.system() == 'Darwin':
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

print('运行完成')
=======

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
>>>>>>> a98794fef118e4fbd47d0348edb5f8b3154dd000
