import math

import numpy as np
import torch
import torch.nn as nn

# 封装一个参数作为需要学习的张量
w = torch.nn.Parameter(torch.empty(2, 3))

# 1. 均匀分布 - u(a, b)
# torch.nn.init.uniform_(tensor, a=0, b=1)
nn.init.uniform_(w)
# tensor([[ 0.0578,  0.3402,  0.5034],
#         [ 0.7865,  0.7280,  0.6269]])

# 2. 正态分布 - N(mean, std)
# torch.nn.init.normal_(tensor, mean=0, std=1)
nn.init.normal_(w)
# tensor([[ 0.3326,  0.0171, -0.6745],
#        [ 0.1669,  0.1747,  0.0472]])

# 3. 常数 - value
# torch.nn.init.constant_(tensor, value)
nn.init.constant_(w, 0.3)
# tensor([[ 0.3000,  0.3000,  0.3000],
#         [ 0.3000,  0.3000,  0.3000]])

# 4. 全1分布
# torch.nn.init.ones_(tensor)
torch.nn.init.ones_(w)
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

# 5. 全0分布
# torch.nn.init.zeros_(tensor)
torch.nn.init.zeros_(w)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

# 6. 对角线为1， 其它为0
# torch.nn.init.eye_(tensor)
nn.init.eye_(w)
# tensor([[ 1.,  0.,  0.],
#         [ 0.,  1.,  0.]])

# 7. xavier_uniform 初始化
# torch.nn.init.xavier_uniform_(tensor, gain=1)
# From - Understanding the difficulty of training deep feedforward neural networks - Bengio 2010
# calculate_gain 返回默认增益值
a = nn.init.calculate_gain("relu")
# 1.414
nn.init.xavier_uniform_(w, gain=a)
# tensor([[ 1.3374,  0.7932, -0.0891],
#         [-1.3363, -0.0206, -0.9346]])

# 8. xavier_normal 初始化
# torch.nn.init.xavier_normal_(tensor, gain=1)
nn.init.xavier_normal_(w)
# tensor([[-0.1777,  0.6740,  0.1139],
#         [ 0.3018, -0.2443,  0.6824]])

# 9. kaiming_uniform 初始化
# From - Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He Kaiming 2015
# torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
nn.init.kaiming_uniform_(w, mode="fan_in", nonlinearity="relu")
# tensor([[ 0.6426, -0.9582, -1.1783],
#         [-0.0515, -0.4975,  1.3237]])

# 10. kaiming_normal 初始化
# torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")
# tensor([[ 0.2530, -0.4382,  1.5995],
#         [ 0.0544,  1.6392, -2.0752]])

# 11. 正交矩阵 - (semi)orthogonal matrix
# From - Exact solutions to the nonlinear dynamics of learning in deep linear neural networks - Saxe 2013
# torch.nn.init.orthogonal_(tensor, gain=1)
nn.init.orthogonal_(w)
# tensor([[ 0.5786, -0.5642, -0.5890],
#         [-0.7517, -0.0886, -0.6536]])

# 12. 稀疏矩阵 - sparse matrix
# 非零元素采用正态分布 N(0, 0.01) 初始化.
# From - Deep learning via Hessian-free optimization - Martens 2010
# torch.nn.init.sparse_(tensor, sparsity, std=0.01)
nn.init.sparse_(w, sparsity=0.1)
# tensor(1.00000e-03 *
#        [[-0.3382,  1.9501, -1.7761],
#         [ 0.0000,  0.0000,  0.0000]])

# 13. Dirac delta 函数初始化，仅适用于 {3, 4, 5}-维的 torch.Tensor
# torch.nn.init.dirac_(tensor)
b = torch.empty(3, 16, 5, 5)  # 是根据输入的size信息生成张量的方法,其dtype默认torch.folat32
nn.init.dirac_(b)
c = torch.tensor(
    np.array([1, 2, 3.0])
)  # torch.tensor()则是复制输入的数据再生成张量的方法,其dtype默认输入数据的dtype

# 举例
# 1 一层一层单独设置
# conv 与 bn
conv = nn.Conv2d(1, 3, kernel_size=1)
# init.kaiming_uniform_(self.weight, a=math.sqrt(5))  ## nn.Conv2d()的weight的默认的初始化方式
# fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
# bound = 1 / math.sqrt(fan_in)
# init.uniform_(self.bias, -bound, bound)  ## nn.Conv2d()的bias的默认的初始化方式
nn.init.kaiming_normal_(conv.weight, mode="fan_in")
nn.init.constant_(conv.bias, 0.0)

bn = (nn.BatchNorm2d(3),)
# init.ones_(self.weight)  ## nn.BatchNorm2d()的weight的默认的初始化方式
# init.zeros_(self.bias)  ## nn.BatchNorm2d()的bias的默认的初始化方式
nn.init.normal_(bn[0].weight, mean=1.0, std=0.02)
nn.init.constant_(bn[0].bias, 0.0)


# 2 在模型中遍历设置
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.double_conv(x)


dc = DoubleConv(1, 4, 2)
