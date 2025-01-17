import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # 第一个输入向量的指针。
    y_ptr,  # 第二个输入向量的指针。
    output_ptr,  # 输出向量的指针。
    n_elements,  # 向量的大小。
    BLOCK_SIZE: tl.constexpr,  # 每个程序应该处理的元素数量 `tl.constexpr` 标记的参数在编译期间被确定，从而可以进行优化。
):
    # 有多个程序处理不同的数据。
    pid = tl.program_id(axis=0)  # 我们使用 1D launch 网格，因此 axis 是 0。
    # 该程序将处理与初始数据偏移的输入。
    # 例如，如果您有长度为 256 的向量和块大小为 64，程序
    # 将分别访问元素[0:64, 64:128, 128:192, 192:256]。
    # 请注意，偏移量是指针的列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建一个掩码以防止内存操作超出范围。
    mask = offsets < n_elements
    # 从 DRAM 加载 x 和 y，以掩盖掉输入不是块大小的倍数的任何额外元素。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 将 x + y 写回到 DRAM。
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    # 我们需要预先分配输出。
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # SPMD即单程序多数据启动网格表示并行运行的内核实例数。
    # 它类似于CUDA启动网格。它可以是Tuple[int]，或者是Callable(metaparameters) -> Tuple[int]。
    # 当前可以使用一个1D网格，其大小是块的数量：
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # 注意：
    #  - `torch.tensor` 对象都隐式地转换为指向其第一个元素的指针。
    #  - `triton.jit` 装饰的函数可以通过一个启动网格索引来获得一个可调用的GPU内核。
    #  - 将元参数作为关键字参数传递。
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # 我们返回一个指向z的句柄，但是，由于`torch.cuda.synchronize()`尚未被调用，内核此时仍在异步运行。
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output_torch = x + y
    output_triton = add(x, y)
    torch.cuda.synchronize()
    print(output_torch)
    print(output_triton)
    print(
        f"在torch和triton之间的最大差异是 "
        f"{torch.max(torch.abs(output_torch - output_triton))}"
    )
