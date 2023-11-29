import numpy as np

# 蒙特卡罗仿真求解从均匀分布的[0, 1, 2]取得[0, 0, 1, 2]的期望
n = 10000
i = 0
a = np.zeros(n)
while i < n:
    b = np.zeros(3)
    while b[0] < 2 or b[1] < 1 or b[2] < 1:
        j = np.random.choice(a=[0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])
        b[j] += 1
    a[i] = np.sum(b)
    i += 1

e = np.mean(a)
print(e)
