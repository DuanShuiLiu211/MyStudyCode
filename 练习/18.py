#encoding: utf-8
import torch
import numpy as np

# a[:, :, 0] == a[..., 0]
a = torch.rand([2, 3, 4])
b = a[:, :, 0]
c = a[..., 0]

# torch.Tensor() == torch.FloatTensor()是一个类只能传入一个数据
d = torch.Tensor([1, 2])
print(d.type())
e = torch.FloatTensor([1, 2])
print(e.type())
# torch.tensor()是一个函数输有一个位置参数 data 和三个赋值参数 dtype device requires_grad
# 根据原始数据类型生成相应的torch.LongTensor, torch.FloatTensor, torch.DoubleTensor数据类型
f = torch.tensor([1, 2])
print(f.type())
h = torch.tensor(np.zeros(2, dtype=np.float64))
print(h.type())
# 现版本的 torch.autograd.Variable(),就是封装成一个torch.FloatTensor,因此和torch.Torch()方法一样
j = torch.autograd.Variable(torch.ones(1))
print(j.type())

# tensor.data 与 tensor.detach() 区别与联系
k = torch.tensor([1., 2., 3.], requires_grad=True)
out1 = k.sigmoid()
print(out1)
l = out1.data  # l的数据与out1共享，并且.data过程不被记录
l.zero_()
print('将l归0后', out1)

out1.sum().backward()  # 反向传播值仅是标量不能非scalar
print(k.grad)

q = torch.tensor([1., 2., 3.], requires_grad=True)
out2 = q.sigmoid()
print(out2)
w = out2.detach()  # w的数据与out2共享,并且.detach()过程被in-place记录
w.zero_()  # 直接修改w等同修改out2
print('将w归0后', out2)

out2.sum().backward()  # out2在计算图中数据被篡改in-place将报告给autograd使方向传播打断
print(q.grad)
pass
