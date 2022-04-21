import cv2
import numpy as np


def cv_imread(filepath):
    cv_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    # cv2.imread读取的是bgr
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img


if __name__ == '__main__':
    path = 'W:\\桌面\\图改\\1.jpg'
    img = cv_imread(path)
    cv2.namedWindow('Example', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Example', img)
    # cv2.imshow按照rgb来显示图像
    k = cv2.waitKey(0)
    # 这样是保存到了和当前运行目录下
    cv2.imencode('.jpg', img)[1].tofile('1.jpg')

