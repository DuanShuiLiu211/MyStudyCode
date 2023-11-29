import torch
from torch.autograd import Variable

# 用来导出指定张量的梯度
# torch.Tensor.register_hook(hook), hook(grad) -> Variable or None
# hook不应该修改它的输入，但是它可以返回一个替代当前梯度的新梯度

# 用来导出指定子模块的输入输出张量
# torch.nn.Module.register_forward_hook(hook), hook(module, input, output) -> None
# hook 不应该修改但可以保留使用 input 和 output 的值

# 用来导出或修改指定子模块的输入张量的梯度
# torch.nn.Module.register_backward_hook(hook), hook(module, grad_input, grad_output) -> Tensor or None
# 如果 module 有多个输入输出的话，那么 grad_input, grad_output 将会是个 tuple 可以选择性的返回关于输入的梯度，这个返回的梯度在后续的计算中会替代 grad_input
# grad_input 保存的是钩到的模块输入值的梯度，grad_output 保存的是钩到的模块 forward 方法输出值的梯度

# hook 返回一个句柄 handle ，它有一个方法 handle.remove() 可以用这个方法将 hook 从 module 移除

# 举例：register_hook
x = Variable(torch.randn(2, 2, 2, 2), requires_grad=True)
print("x value is \n", x)
y = x + torch.tensor(3.0)  # y 不是叶子节点 grad 计算结束并不会保留
print("y value is \n", y)
z = torch.pow(y, 0.5)  # 两个矩阵指数点乘
print("z value is \n", z)
g = torch.nn.MaxPool2d(2, 2)(z)
print("g value is \n", g)
k = torch.mean(g)
print("z value is \n", z)
lr = torch.tensor(1e-3)

x.register_hook(lambda grad: print("x grad is \n", grad))
y.register_hook(lambda grad: print("y grad is \n", grad))
k.backward()
x.data -= lr * x.grad.data
print("new x is\n", x)

v = torch.tensor([0, 0, 0], requires_grad=True, dtype=torch.float32)
h = v.register_hook(lambda grad: grad * 2)
v.backward(
    torch.tensor([1, 1, 1], dtype=torch.float32)
)  # 先计算 v 的梯度，再通过 v 的 hook 获得双倍梯度
h.remove()
