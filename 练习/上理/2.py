import numpy as np
import torch
from PIL import Image

x = torch.empty(5, 3)
print(x)
print(16 * 16 * 1088)
print(32 * 32 * 512)

img = Image.open('数据/图片1.png')
# 1
img = np.asarray(img)
print(img.max(), img.min())
image = Image.fromarray(np.uint8(img))
image.show()
# 2
norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
print(norm_img.max(), norm_img.min())
image = Image.fromarray(np.uint8(255 * norm_img))
image.show()