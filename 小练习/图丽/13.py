import numpy as np
import cv2
import PIL.Image as Image

# 生成随机图像数据
a = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

# 使用 OpenCV 进行图像缩放
b = cv2.resize(a, (5, 5), interpolation=cv2.INTER_LINEAR)

# 使用 PIL 进行图像缩放并转换回 NumPy 数组
c = Image.fromarray(a).resize((5, 5), resample=Image.BILINEAR)
c = np.array(c)

# 打印两个缩放后图像的差异
print(c - b)
