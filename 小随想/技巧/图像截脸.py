import os.path
from glob import glob

import cv2
import numpy as np


def detect(
    filename,
    cascade_file="W:\\Study Flies\\Pycharm\\Test1\\faces\\lbpcascade_animeface.xml",
):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.imread(filename) # cv2.imread读取中文路径的文件会None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray,
        # detector options
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    for i, (x, y, w, h) in enumerate(faces):
        j = 0
        face = image[y : y + h, x : x + w, :]
        face = cv2.resize(face, (160, 160))
        save_filename = "%s-%d.png" % (os.path.basename(filename).split(".")[0], j)
        cv2.imwrite("faces/" + save_filename, face)
        j += 1


if __name__ == "__main__":
    if os.path.exists("faces") is False:
        os.makedirs("faces")
    file_list = glob("W:\\桌面\\工作\\数据\\灰度\\*.png")
    for filename in file_list:
        detect(filename)
