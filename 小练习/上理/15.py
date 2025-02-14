import cv2 as cv
import numpy as np

path = np.fromfile(r"数据/图片1.png")
src = cv.imdecode(path, -1)  # 此法可读取中文路径图片，读取后为RGB模式
# src = cv.cvtColor(src, cv.COLOR_RGB2BGR)
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# sigma = 5、15、25
blur_img = cv.GaussianBlur(src, (0, 0), 15)
usm = cv.addWeighted(src, 1.5, blur_img, -0.5, 0)
cv.imshow("mask image", usm)

h, w = src.shape[:2]
result = np.zeros([h, w * 2, 3], dtype=src.dtype)
result[0:h, 0:w, :] = src
result[0:h, w : 2 * w, :] = usm
cv.putText(result, "original image", (10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.putText(result, "sharpen image", (w + 10, 30), cv.FONT_ITALIC, 1.0, (0, 0, 255), 2)
cv.imshow("sharpen_image", result)
cv.imwrite("E:/result.png", usm)

cv.waitKey(5)
cv.destroyAllWindows()
