import random

import tifffile

input = tifffile.imread(
    "/Users/WangHao/Desktop/Paper ImT UNet/data_0913/PreImage/Averaged shifted histograms_11.tif"
)

index = [i for i in range(len(input)) if i % 10 == 0]
random.shuffle(index)
index = list(range(8, 23))
output = input[index, ...]

tifffile.imsave(
    "/Users/WangHao/Desktop/Paper ImT UNet/data_0913/PreImage/Averaged shifted histograms_11_2.tif",
    output,
)
print("运行完成")
