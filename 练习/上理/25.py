import numpy as np


def fft2d_divideloop(inputs):
    m, n = inputs.shape
    # fft 需要保证序列是2^n, 否则补零
    if 2**np.ceil(np.log2(m)) != m:
        inputs = np.pad(inputs, ((0, int(2**np.ceil(np.log2(m))-m)), (0, 0)), 'constant', constant_values=(0, 0))
    if 2**np.ceil(np.log2(n)) != n:
        inputs = np.pad(inputs, ((0, 0), (0, int(2**np.ceil(np.log2(n))-n))), 'constant', constant_values=(0, 0))
    m, n = inputs.shape
    result = np.zeros((m, n), dtype=np.complex64) 
    for v in range(n):
        result[:, v] = fft1d_divideloop(inputs[:, v])    
    for u in range(m):
        result[u, :] = fft1d_divideloop(result[u, :])

    return result

def fft1d_divideloop(inputs):
    l = len(inputs)
    ll = int(np.log2(l))
    # 通过二进制编码重排数据两两组成一对计算元
    vector = np.array(reset_data(inputs), dtype=np.complex128)
    for i in range(ll):
        m = 2**(i+1)
        ym = np.exp(-1.j * 2 * np.pi / m)
        for j in range(0, l, m):
            y = 1
            for k in range(m//2):
                c0 = vector[k+j] 
                c1 = vector[k+j+m//2] * y
                vector[k+j] = c0 + c1
                vector[k+j+m//2] = c0 - c1
                y *= ym
                
    return vector
    
def reset_data(inputs):
    l = len(inputs)
    outputs = np.zeros(l, dtype=np.complex128)
    for i in range(l):
        j = int(reverse_bin(i, l), 2)
        outputs[j] = inputs[i]
        
    return outputs    

def reverse_bin(numbers, length):
    bin_str = bin(numbers)[2:]
    if len(bin_str) <= np.log2(length):
        bin_str = ("0"*(int(np.log2(length)-len(bin_str))) + bin_str)[::-1]
    
    return bin_str

a = np.arange(1,10,1).reshape(3,3)
a1 = fft2d_divideloop(a)
a2 = np.fft.fft2(a)