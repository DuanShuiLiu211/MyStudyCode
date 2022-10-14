import tifffile
import random
import numpy as np


a = tifffile.imread(
    "/Users/WangHao/Desktop/Paper ImT UNet/data_0913/MT/N1_LD/sequence-as-stack-MT1.N1.LD-BP-250.tif"
)

c = [i for i in range(len(a)) if i % 5 == 0]
# random.shuffle(c)
b = a[c, ...]

tifffile.imsave(
    "/Users/WangHao/Desktop/Paper ImT UNet/data_0913/MT/N1_LD/sequence-as-stack-MT1.N1.LD-BP-250_r5000.tif",
    b
)
print('运行完成')
