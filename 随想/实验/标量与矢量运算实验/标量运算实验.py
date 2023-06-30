import time
import numpy as np
from numba import jit
import cython
import torch
import tensorflow as tf


def execute_time(func):
    def func_new(*args, **kwargs):
        time_start = time.time_ns()
        func(*args, **kwargs)
        time_end = time.time_ns()
        sum_time = (time_end - time_start) / 1e9
        print(f"{func.__name__}运行总时间{sum_time}秒")
    return func_new


# 加法
@execute_time
def python_add(a=1, b=2, m=1e6, mode="for"):
    m = int(m)
    c = 0
    i = 0
    if mode == "for":
        for _ in range(m):
            c = a + b
    else:
        while i < m:
            c = a + b
            i += 1
    return c


@execute_time
def numpy_add(a=np.array(1), b=np.array(2), m=int(1e6), mode="for"):
    m = np.array(m, dtype=np.int64)
    c = np.array(0)
    i = np.array(0)
    if mode == "for":
        for _ in range(m):
            c = np.add(a, b)
    else:
        while i < m:
            c = np.add(a, b)
            i = np.add(i, np.array(1))
    return c


# 减法
@execute_time
def python_sub(a=1, b=2, m=1e6, mode="for"):
    m = int(m)
    c = 0
    i = 0
    if mode == "for":
        for _ in range(m):
            c = a - b
    else:
        while i < m:
            c = a - b
            i += 1
    return c


@execute_time
def numpy_sub(a=np.array(1), b=np.array(2), m=int(1e6), mode="for"):
    m = np.array(m, dtype=np.int64)
    c = np.array(0)
    i = np.array(0)
    if mode == "for":
        for _ in range(m):
            c = np.subtract(a, b)
    else:
        while i < m:
            c = np.subtract(a, b)
            i = np.add(i, np.array(1))
    return c


# 乘法
@execute_time
def python_mul(a=1, b=2, m=1e6, mode="for"):
    m = int(m)
    c = 0
    i = 0
    if mode == "for":
        for _ in range(m):
            c = a * b
    else:
        while i < m:
            c = a * b
            i += 1
    return c


@execute_time
def numpy_mul(a=np.array(1), b=np.array(2), m=int(1e6), mode="for"):
    m = np.array(m, dtype=np.int64)
    c = np.array(0)
    i = np.array(0)
    if mode == "for":
        for _ in range(m):
            c = np.multiply(a, b)
    else:
        while i < m:
            c = np.multiply(a, b)
            i = np.add(i, np.array(1))
    return c


# 除法
@execute_time
def python_div(a=1, b=2, m=1e6, mode="for"):
    m = int(m)
    c = 0
    i = 0
    if mode == "for":
        for _ in range(m):
            c = a / b
    else:
        while i < m:
            c = a / b
            i += 1
    return c


@execute_time
def numpy_div(a=np.array(1), b=np.array(2), m=int(1e6), mode="for"):
    m = np.array(m, dtype=np.int64)
    c = np.array(0)
    i = np.array(0)
    if mode == "for":
        for _ in range(m):
            c = np.divide(a, b)
    else:
        while i < m:
            c = np.divide(a, b)
            i = np.add(i, np.array(1))
    return c


@execute_time
@jit(nopython=True)
# Function is compiled to machine code when called the first time
# Numba 可以加速循环但是循环状态必须是 int32 int64 uint64
# Numba 可以加速 NumPy function
# Numba 可以加速 NumPy broadcasting
# 对代码中的变量类型有要求，当无法静态确定函数的返回类型时无法正常编译代码，例如，返回类型取决于仅在运行时可用的值的情况
def numba_div(a=1, b=2, m=int(1e6), mode="for"):
    i = 0
    c = 0
    if mode == "for":
        for _ in range(m):
            c = np.divide(a, b)
            c = np.tanh(c)
    else:
        while i < m:
            c = np.divide(a, b)
            i = np.add(i, np.array(1))
    return c


@execute_time
@cython.cfunc
def cython_div(a=1, b=2, m=int(1e6), mode="for"):
    a: cython.int = a
    b: cython.int = b
    m: cython.int = m
    i: cython.int = 0
    c: cython.int = 0
    if mode == "for":
        for i in range(m):
            c = a / b
    else:
        while i < m:
            c = a / b
            i += 1
    return c


@execute_time
def torch_div(a=torch.tensor(1), b=torch.tensor(2), m=int(1e6), mode="for"):
    m = torch.tensor(m, dtype=torch.int64)
    c = torch.tensor(0)
    i = torch.tensor(0)
    if mode == "for":
        for _ in range(m):
            c = torch.divide(a, b)
    else:
        while i < m:
            c = torch.divide(a, b)
            i = torch.add(i, torch.tensor(1))
    return c


@execute_time
def torch_div_gpu(a=torch.tensor(1, device='mps'), b=torch.tensor(2, device='mps'), m=int(1e6), mode="for"):
    m = torch.tensor(m, dtype=torch.int64)
    c = torch.tensor(0)
    i = torch.tensor(0)
    if mode == "for":
        for _ in range(m):
            c = torch.divide(a, b)
    else:
        while i < m:
            c = torch.divide(a, b)
            i = torch.add(i, torch.tensor(1))
    return c


@execute_time
def tensorflow_div(a=tf.constant(1), b=tf.constant(2), m=int(1e6), mode="for"):
    m = tf.constant(m, dtype=tf.int64)
    c = tf.constant(0)
    i = tf.constant(0)
    if mode == "for":
        for _ in range(m):
            c = tf.divide(a, b)
    else:
        while i < m:
            c = tf.divide(a, b)
            i = tf.add(i, tf.constant(1))
    return c


if __name__ == "__main__":
    # 标量计算
    python_add()  # 运行总时间0.020576秒
    numpy_add()  # 运行总时间0.273121秒
    python_sub()  # 运行总时间0.02032秒
    numpy_sub()  # 运行总时间0.278431秒
    python_mul()  # 运行总时间0.021092秒
    numpy_mul()  # 运行总时间0.272244秒
    python_div()  # 运行总时间0.023047秒
    numpy_div()  # 运行总时间0.452664秒
    numba_div()  # 运行总时间0.169949秒
    numba_div()  # 运行总时间0.000141秒
    cython_div()  # 运行总时间0.019132秒
    torch_div()  # 运行总时间1.878211秒
    torch_div_gpu()  # 运行总时间24.878211秒
    with tf.device('cpu:0'):
        tensorflow_div()  # 运行总时间24.869914秒
    with tf.device('gpu:0'):
        tensorflow_div()  # 运行总时间24.809914秒

    # 使用纯c编写 运行总时间0.001738秒
