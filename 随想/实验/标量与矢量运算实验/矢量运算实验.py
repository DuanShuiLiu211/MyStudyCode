import time
import numpy as np
from numba import jit, njit
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


@execute_time
def python_dot(a, b):
    if len(a) != len(b[0]):
        raise ValueError('shape not matched')
    n, p, m = len(a), len(a[0]), len(b[0])
    c = [[0 for i in range(n)] for j in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0
            for k in range(p):
                s += a[i][k] * b[k][j]
            c[i][j] = s
    return c

@execute_time
@cython.cfunc
def cython_dot(a, b):
    if len(a) != len(b[0]):
        raise ValueError('shape not matched')
    n: cython.int = len(a)
    p: cython.int = len(a[0])
    m: cython.int = len(b[0])
    c = [[0 for _ in range(n)] for _ in range(m)]
    i: cython.int = 0
    j: cython.int = 0
    for i in range(m):
        for j in range(n):
            s = 0
            for k in range(p):
                s += a[i][k] * b[k][j]
            c[i][j] = s
    return c

@execute_time
def numpy_dot(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    c = np.matmul(a, b)
    return c


@execute_time
@jit(nopython=True)
def numba_dot_1(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    n, p, m = a.shape[0], a.shape[1], b.shape[1]
    c = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            s = 0
            for k in range(p):
                s += a[i, k] * b[k, j]
            c[i, j] = s
    return c


@execute_time
@njit
def numba_dot_2(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    c = np.dot(a, b)
    return c


@execute_time
def torch_dot(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    c = torch.matmul(a, b)
    return c


@execute_time
def tensorflow_dot(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    c = tf.matmul(a, b)
    return c


if __name__ == "__main__":
    a = [[0 for i in range(1000)] for j in range(500)]
    b = [[0 for i in range(500)] for j in range(1000)]
    # python_dot(a, b)

    a = np.random.rand(1000, 500)
    b = np.random.rand(500, 1000)
    numpy_dot(a, b)
    
    a = np.random.rand(1000, 500)
    b = np.random.rand(500, 1000)
    numba_dot_1(a, b)
    numba_dot_1(a, b)
    numba_dot_2(a, b)
    numba_dot_2(a, b)
    
    a = [[0 for i in range(1000)] for j in range(500)]
    b = [[0 for i in range(500)] for j in range(1000)]
    # cython_dot(a, b)

    a = torch.ones(1000, 500)
    b = torch.ones(500, 1000)
    torch_dot(a, b)
    
    a = torch.ones((1000, 500), device='mps')
    b = torch.ones((500, 1000), device='mps')
    torch_dot(a, b)

    a = tf.random.normal((1000, 500))
    b = tf.random.normal((500, 1000))
    tensorflow_dot(a, b)
    
    with tf.device('gpu:0'):
        a = tf.random.normal((1000, 500))
        b = tf.random.normal((500, 1000))
        tensorflow_dot(a, b)
