import torch
import torch.nn.functional as F

# solfmax 计算
# 取决于dim,0为列向,1为行向
x = torch.Tensor([[1, 2, 3, 4], [1, 3, 4, 5], [3, 4, 5, 6]])
x1 = F.softmax(x, dim=0)
x2 = F.softmax(x, dim=1)

print("\n", x, x.size(), x.shape, x.is_contiguous())
print("\n", x1, x2, sep="\n\n", end="\n\n")


# 列表中的多个张量转成一个张量
# .stack() 方法在列表中张量的基础上给列表顺序增加一个张量维度
tensor_list = [x1, x2]
x3 = torch.stack(tensor_list)
x4 = torch.cat(tensor_list)
print("\n", x3, x4, sep="\n\n", end="\n\n")

# 张量的形状改变
# tensor 调用过transpose,permute等方法的话会使该 tensor 在内存中变得不再连续
# 浅拷贝不为新变量开辟新的内存空间，仅拷贝id,深拷贝开辟新的内存空间
# .view() 方法只能改变内存连续的张量,否则需要先调用.contiguous()方法,改变形状的时候,总的数据个数不能变
# .reshape() 方法不受内存连续的张量限制,改变形状的时候,总的数据个数不能变
# .resize_() 不受以上两点限制,会直接修改原地址张量再返回,其他两种则是返回张量再修改它
# 以上方向变形过程都遵循，从左往右，从上往下，从外往里
x5 = x1.view(-1, 6)
x6 = x2.reshape(-1, 6)

print("\n", x5, x6, sep="\n\n", end="\n\n")

x7 = x.permute(1, 0)
x8 = x7
print(id(x), id(x7), id(x8))
print("\n", x, x7, x7.is_contiguous(), x8.resize_(2, 4), sep="\n\n", end="\n\n")
