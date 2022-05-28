import numpy as np
import torch
from PIL import Image

x = torch.empty(5, 3)

print(x)

'''
from torch import empty

x = empty(5, 3)
print(x)
'''

print(16 * 16 * 1088)
print(32 * 32 * 512)

img = Image.open('W:\桌面\工作\专利\数据\MERGE.tif')
# 1
re_img = np.asarray(img)
a = re_img.max()
b = re_img.min()
image = Image.fromarray(np.uint8(re_img))
image.show()
# 2
nrm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
c = nrm_img.max()
d = nrm_img.min()
image1 = Image.fromarray(np.uint8(255 * nrm_img))
image1.show()